"""Детекторное чтение прописки: YOLO-OBB локализует зоны, затем OCR.

Альтернатива эвристическому postprocess/registration.py (полностраничный
OCR + regex с ручными фиксами). Здесь зоны штампа находит обученный
OBB-детектор (training/train_detector.py на синтетике из
training/regpage_generator.py), поэтому адрес собирается из чистых
покополевых кропов без хардкода под конкретные регионы.

Детектор — отдельный файл весов; без обученных весов не используется.
Оба ридера возвращают адрес строкой — их можно объединять (детектор как
основной, эвристика как фолбэк).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from docreader.preprocessing import crop_obb_region

from postprocess import normalize
from postprocess.passport import RefinedField

logger = logging.getLogger(__name__)

# повороты для перебора ориентации (штамп на фото часто повёрнут на 90°)
_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# синхронно с CLASS_NAMES в training/regpage_generator.py
ZONE_AUTHORITY = "reg_authority"
ZONE_DATE = "reg_date"
ZONE_ADDRESS = "reg_address"
ZONE_STAMP = "reg_stamp"


@dataclass
class RegistrationResult:
    authority: Optional[str] = None
    date: Optional[str] = None
    address_lines: list[str] = field(default_factory=list)
    # компоненты после словарной коррекции (гейзеттир)
    region: Optional[str] = None
    city: Optional[str] = None
    street: Optional[str] = None

    @property
    def address(self) -> Optional[str]:
        """Скорректированный адрес (регион/город/улица); иначе сырые строки."""
        from postprocess.gazetteer import format_address

        corrected = format_address(
            {"region": self.region, "city": self.city, "street": self.street}
        )
        if corrected:
            return corrected
        return ", ".join(self.address_lines) if self.address_lines else None

    def as_dict(self) -> dict:
        return {
            "authority": self.authority,
            "date": self.date,
            "address": self.address,
            "region": self.region,
            "city": self.city,
            "street": self.street,
            "address_lines": self.address_lines,
        }


class RegistrationDetectorReader:
    def __init__(self, weights_path: str, ocr_engine=None, confidence: float = 0.25):
        from ultralytics import YOLO

        self._model = YOLO(weights_path)
        self._conf = confidence
        self._ocr = ocr_engine or self._build_default_ocr()
        logger.info("RegistrationDetectorReader loaded: %s", weights_path)

    @staticmethod
    def _build_default_ocr():
        from postprocess.retry_ocr import RetryOcrEngine
        from postprocess.sorted_ocr import SortedOcrEngine

        return RetryOcrEngine(SortedOcrEngine.from_config(gpu=False))

    def _detect(self, image: np.ndarray):
        predictions = self._model.predict(image, conf=self._conf, verbose=False)
        if not predictions:
            return None
        det = predictions[0]
        if det.obb is None or len(det.obb) == 0:
            return None
        return det

    def _orientation_score(self, det) -> float:
        """Суммарная уверенность по текстовым зонам (не штамп) — метрика
        правильной ориентации: на верном повороте детектор увереннее."""
        names = det.names
        classes = det.obb.cls.cpu().numpy().astype(int)
        confs = det.obb.conf.cpu().numpy()
        score = 0.0
        for cls_idx, conf in zip(classes, confs):
            if names[cls_idx] != ZONE_STAMP:
                score += float(conf)
        return score

    def read(self, page_bgr: np.ndarray, auto_rotate: bool = True) -> RegistrationResult:
        if auto_rotate:
            best_img, best_det, best_score = page_bgr, None, -1.0
            for rot in _ROTATIONS.values():
                img = cv2.rotate(page_bgr, rot) if rot is not None else page_bgr
                det = self._detect(img)
                score = self._orientation_score(det) if det is not None else -1.0
                if score > best_score:
                    best_img, best_det, best_score = img, det, score
            page_bgr, det = best_img, best_det
        else:
            det = self._detect(page_bgr)

        result = RegistrationResult()
        if det is None:
            return result

        names = det.names
        polys = det.obb.xyxyxyxy.cpu().numpy()      # (n, 4, 2)
        classes = det.obb.cls.cpu().numpy().astype(int)

        address_items: list[tuple[float, str]] = []
        authority_items: list[tuple[float, str]] = []
        for poly, cls_idx in zip(polys, classes):
            zone = names[cls_idx]
            if zone == ZONE_STAMP:
                continue
            crop = crop_obb_region(page_bgr, poly.reshape(-1))
            if crop is None or crop.size == 0:
                continue
            text = self._ocr.recognize(crop).text
            y_center = float(poly[:, 1].mean())

            if zone == ZONE_ADDRESS:
                if norm := normalize.normalize_text_field(text):
                    address_items.append((y_center, norm))
            elif zone == ZONE_AUTHORITY:
                if norm := normalize.normalize_text_field(text):
                    authority_items.append((y_center, norm))
            elif zone == ZONE_DATE and result.date is None:
                result.date = normalize.normalize_any_date(text)

        authority_items.sort(key=lambda item: item[0])
        result.authority = " ".join(text for _, text in authority_items) or None
        address_items.sort(key=lambda item: item[0])
        result.address_lines = [text for _, text in address_items]

        # словарная коррекция адреса (регион/город/улица) + дата прописью
        from postprocess.gazetteer import refine_address_lines

        # дата может быть распознана как строка адреса («14 июля 2015 г.»)
        if result.date is None:
            for line in result.address_lines:
                if d := normalize.normalize_any_date(line):
                    result.date = d
                    break
        refined = refine_address_lines(result.address_lines)
        result.region = refined["region"]
        result.city = refined["city"]
        result.street = refined["street"]
        return result

    def read_as_field(self, page_bgr: np.ndarray) -> Optional[RefinedField]:
        """Адрес как RefinedField — совместимо с выводом refine_passport."""
        result = self.read(page_bgr)
        if not result.address:
            return None
        return RefinedField(result.address, "detector", confirmed=False)
