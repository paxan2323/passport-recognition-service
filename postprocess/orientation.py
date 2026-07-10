"""Автовыбор ориентации страницы (0/90/180/270°).

Два режима (Settings.orientation_mode):
- probe (default): сначала дешёвый MRZ-probe (классификатор + нижняя
  полоса MRZ на каждом повороте), затем один полный пайплайн;
- full: перебор с полным OCR на каждом повороте (медленно, запасной).

MRZ-чексуммы ICAO — надёжный сигнал ориентации: doc_conf одинаков
на 0° и 180°, а number_ok/birth_date_ok — нет.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from docreader.schemas import PageResult
from postprocess.mrz import parse_mrz
from postprocess.mrz_ocr import MrzOcr
from postprocess.passport import _read_mrz, refine_passport

logger = logging.getLogger(__name__)

ROTATIONS: tuple[Optional[int], ...] = (
    None,
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180,
    cv2.ROTATE_90_COUNTERCLOCKWISE,
)
ROTATION_DEGREES: tuple[int, ...] = (0, 90, 180, 270)

# probe: number_ok + birth_date_ok
_PROBE_CONFIDENT_SCORE = 8.0


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """BGR-изображение с учётом EXIF Orientation (фото с телефона)."""
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
        rgb = np.array(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def rotate_image(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    if rotation_deg == 0:
        return image
    flag = ROTATIONS[ROTATION_DEGREES.index(rotation_deg)]
    if flag is None:
        return image
    return cv2.rotate(image, flag)


def _pick_passport_classification(classified) -> Optional[object]:
    passports = [d for d in classified if d.doc_type == "passport"]
    if not passports:
        return None
    return max(passports, key=lambda d: d.confidence)


def mrz_probe_score(reader, image: np.ndarray, mrz_ocr: Optional[MrzOcr]) -> float:
    """Дешёвая оценка ориентации: YOLO-классификатор + MRZ-полоска."""
    if mrz_ocr is None:
        return -1.0

    classified = reader._classifier.classify(image)
    doc = _pick_passport_classification(classified)
    if doc is None:
        return -1.0

    line1, line2 = mrz_ocr.read_document(doc.crop)
    mrz = parse_mrz(line1, line2)

    score = float(doc.confidence) * 0.1
    if mrz.number_ok:
        score += 5.0
    if mrz.birth_date_ok:
        score += 3.0
    if mrz.optional_ok:
        score += 1.0
    if len(mrz.raw_line2) >= 36:
        score += 0.3
    return score


def probe_orientation(
    reader,
    image: np.ndarray,
    mrz_ocr: Optional[MrzOcr],
) -> tuple[int, float]:
    """Подбирает угол по MRZ-probe; на 0° с валидной MRZ — без перебора."""
    score0 = mrz_probe_score(reader, image, mrz_ocr)
    if score0 >= _PROBE_CONFIDENT_SCORE:
        return 0, score0

    best_deg, best_score = 0, score0
    for deg in ROTATION_DEGREES[1:]:
        score = mrz_probe_score(reader, rotate_image(image, deg), mrz_ocr)
        if score > best_score:
            best_deg, best_score = deg, score

    return best_deg, best_score


def score_passport_page(doc, mrz_ocr: Optional[MrzOcr]) -> float:
    """Полная оценка (режим full): MRZ + все поля после OCR."""
    if doc is None or doc.doc_type != "passport":
        return -1.0

    mrz = _read_mrz(doc, mrz_ocr)
    refined = refine_passport(doc, mrz_ocr)

    score = float(doc.doc_confidence)
    if mrz.number_ok:
        score += 3.0
    if mrz.birth_date_ok:
        score += 2.0
    if mrz.optional_ok:
        score += 0.5

    for key in ("surname", "name", "patronymic", "series_number", "birth_date", "sex"):
        field = refined.get(key)
        if field and field.value:
            score += 0.35
            if field.confirmed:
                score += 1.0

    score += len(doc.zones) * 0.01
    return score


def is_orientation_confident(doc, mrz_ocr: Optional[MrzOcr]) -> bool:
    """0° достаточно для режима full (ранний выход)."""
    if doc is None or doc.doc_type != "passport":
        return False
    mrz = _read_mrz(doc, mrz_ocr)
    if not (mrz.number_ok and mrz.birth_date_ok):
        return False
    surname = refine_passport(doc, mrz_ocr).get("surname")
    return bool(surname and surname.value and surname.confirmed)


def _process_full_rotation_search(
    reader,
    image: np.ndarray,
    mrz_ocr: Optional[MrzOcr],
    pick_doc,
) -> tuple[PageResult, np.ndarray, int]:
    """Запасной режим: полный OCR на каждом повороте."""
    page = reader.process(image, return_crops=True)
    doc = pick_doc(page)
    if is_orientation_confident(doc, mrz_ocr):
        return page, image, 0

    best_page, best_image, best_deg, best_score = page, image, 0, score_passport_page(doc, mrz_ocr)

    for deg, rot in zip(ROTATION_DEGREES[1:], ROTATIONS[1:]):
        rotated = cv2.rotate(image, rot)
        try:
            candidate = reader.process(rotated, return_crops=True)
        except Exception:
            logger.warning("Pipeline failed at rotation %s°", deg, exc_info=True)
            continue
        score = score_passport_page(pick_doc(candidate), mrz_ocr)
        if score > best_score:
            best_page, best_image, best_deg, best_score = candidate, rotated, deg, score

    if best_deg:
        logger.info("Orientation (full): %s° score=%.2f", best_deg, best_score)
    return best_page, best_image, best_deg


def process_with_rotation(
    reader,
    image: np.ndarray,
    mrz_ocr: Optional[MrzOcr],
    *,
    pick_doc,
    auto_rotate: bool = True,
    mode: str = "probe",
) -> tuple[PageResult, np.ndarray, int]:
    """Возвращает (page, выровненное_изображение, угол_°)."""
    if not auto_rotate:
        return reader.process(image, return_crops=True), image, 0

    if mode == "full":
        return _process_full_rotation_search(reader, image, mrz_ocr, pick_doc)

    deg, probe_score = probe_orientation(reader, image, mrz_ocr)
    aligned = rotate_image(image, deg)
    page = reader.process(aligned, return_crops=True)
    if deg:
        logger.info("Orientation probe: %s° mrz_score=%.2f", deg, probe_score)
    return page, aligned, deg
