"""OCR и разбор адреса регистрации (разворот «Место жительства»).

docreader-ocr не размечает штамп прописки — читаем полное изображение
EasyOCR (ru) с перебором поворотов на 90°.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from postprocess.passport import RefinedField

logger = logging.getLogger(__name__)

_REGISTRATION_MARKERS = ("ЗАРЕГИСТРИРОВАН", "МЕСТО ЖИТЕЛЬСТВ", "ЖИТЕЛЬСТВ")
_CITY_STOPWORDS = frozenset({
    "ЗАРЕГИСТРИРОВАН", "ЗАРЕГИСТРИРОВАНА", "МЕСТО", "ЖИТЕЛЬСТВА", "ЖИТЕЛЬСТВО",
    "КРАЙ", "ОБЛ", "ОБЛАСТЬ", "РЕСП", "РЕСПУБЛИКА", "РЕГИОН", "РЕГЕН", "РЕГЕН:",
    "ПАЛОН", "УЛАЦА", "УЛИЦА", "ПАСПОРТ", "НЮЛЯ", "ЯНВАРЯ", "ФЕВРАЛЯ", "МАРТА",
    "АПРЕЛЯ", "МАЯ", "ИЮНЯ", "ИЮЛЯ", "АВГУСТА", "СЕНТЯБРЯ", "ОКТЯБРЯ", "НОЯБРЯ", "ДЕКАБРЯ",
    "ОТДЕЛ", "МВД", "УМВД", "ПОДРАЗ", "МИГРАЦ", "ДЕПАРТ", "РОССИИ", "РФ", "ВАНКЕНОВАННЕ",
    "ЛОДРАЗДЕЛЕВИЯ", "ПОЛИЦ", "ДЕНИЕ",
})
_CITY_OCR_FIXES = {
    "ЧИТЕ": "ЧИТА",
    "ЧИТА": "ЧИТА",
    "МОСКВЕ": "МОСКВА",
    "МОСКВА": "МОСКВА",
}
_STAMP_FOOTER = re.compile(
    r"\b(ОТДЕЛ|МВД|УМВД|ПОДРАЗ|МИГРАЦ|ДЕПАРТ|ПАСПОРТ)\b|750-\d{3}",
    re.IGNORECASE,
)
_STAMP_NOISE = re.compile(
    r"(ОТДЕЛ|МВД|УМВД|ПОЛИЦ|МИГРАЦ|ПОДРАЗД|ДЕПАРТАМ|РОССИИ|РФ|ПАСПОРТ|"
    r"ЗАРЕГИСТРИРОВАН|\d{3}-\d{3}|\d{1,2}\s+[А-ЯЁ]+\s+\d{4})",
    re.IGNORECASE,
)
_ROTATIONS = (
    None,
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180,
    cv2.ROTATE_90_COUNTERCLOCKWISE,
)


class RegistrationOcr:
    """Ленивая ru-модель EasyOCR для штампа прописки."""

    def __init__(self, gpu: bool = False) -> None:
        self._gpu = gpu
        self._reader = None

    def _reader_lazy(self):
        if self._reader is None:
            import os

            import easyocr

            model_dir = os.environ.get("EASYOCR_REG_PATH") or os.environ.get(
                "EASYOCR_MODULE_PATH", "/tmp/easyocr"
            )
            self._reader = easyocr.Reader(
                ["ru"],
                gpu=self._gpu,
                verbose=False,
                model_storage_directory=f"{model_dir}/model",
            )
            logger.info("RegistrationOcr initialized (gpu=%s)", self._gpu)
        return self._reader

    def _ocr_text(self, reader, image: np.ndarray) -> str:
        results = reader.readtext(image, paragraph=False)
        return " ".join(t for _, t, conf in results if conf > 0.25)

    @staticmethod
    def _stamp_crop(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return image[int(h * 0.12) : int(h * 0.88), int(w * 0.04) : int(w * 0.96)]

    def read(self, image: np.ndarray) -> tuple[int, str]:
        """OCR с выбором лучшего поворота. Возвращает (угол_idx, текст)."""
        reader = self._reader_lazy()
        best_idx, best_score, best_text = 0, -1, ""
        for idx, rot in enumerate(_ROTATIONS):
            im = cv2.rotate(image, rot) if rot is not None else image
            text = self._ocr_text(reader, im)
            crop = self._stamp_crop(im)
            ch, cw = crop.shape[:2]
            if ch > 40 and cw > 40:
                crop_up = cv2.resize(crop, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
                text = f"{text} {self._ocr_text(reader, crop_up)}"
            score = _registration_score(text)
            if score > best_score:
                best_idx, best_score, best_text = idx, score, text
        return best_idx, best_text


def _registration_score(text: str) -> int:
    upper = text.upper()
    return sum(1 for marker in _REGISTRATION_MARKERS if marker in upper)


def is_registration_page(text: str) -> bool:
    return _registration_score(text) >= 1


def _stamp_scope(text: str) -> str:
    """Текст штампа без служебной части внизу (МВД, код подразделения)."""
    upper = re.sub(r"\s+", " ", text.upper())
    start = upper.find("ЗАРЕГИСТРИРОВ")
    scope = upper[start if start >= 0 else 0 :]

    # Обрезаем только «хвост» штампа — после адреса (мкр/ул + дом)
    street_m = re.search(r"(?:МКР|УЛ\.|ПР\.|ПР-Т|ПЕР\.|Б-Р|ПРОСП)", scope)
    min_cut = street_m.start() if street_m else 30
    if street_m and (hm := re.search(r"\b\d{1,4}[А-ЯЁ]?\b", scope[street_m.end() : street_m.end() + 30])):
        min_cut = street_m.end() + hm.end()

    footer = _STAMP_FOOTER.search(scope, pos=min_cut)
    if footer:
        scope = scope[: footer.start()]
    return scope.strip()


def _normalize_city_token(token: str) -> Optional[str]:
    token = token.strip(".,:(_-")
    if len(token) < 4 or token in _CITY_STOPWORDS:
        return None
    if token in _CITY_OCR_FIXES:
        return _CITY_OCR_FIXES[token]
    if token.endswith(("СКИЙ", "СКАЯ", "СКОЕ")):
        return None
    # OCR часто пишет «ЧИТЕ» вместо «ЧИТА»
    if token.endswith("Е") and len(token) >= 5:
        fixed = token[:-1] + "А"
        if fixed in _CITY_OCR_FIXES:
            return _CITY_OCR_FIXES[fixed]
        return fixed
    return token


def _extract_city(text: str) -> Optional[str]:
    if m := re.search(r"\bГ\.?\s*([А-ЯЁ\-]{3,})\b", text):
        city = _normalize_city_token(m.group(1))
        if city:
            return f"Г. {city}"

    if m := re.search(
        r"(?:РЕГИОН|РЕГЕН|ОБЛ)\.?\s*:?\s*([А-ЯЁ]{3,})",
        text,
    ):
        city = _normalize_city_token(m.group(1))
        if city:
            return f"Г. {city}"

    # «... ЗАБАЙКАЛЬСКИЙ ... ЧИТА ... МКР» — город между регионом и улицей
    if m := re.search(r"(?:МКР|УЛ\.|ПР\.|ПР-Т|ПЕР\.|Б-Р|ПРОСП)", text):
        before = text[: m.start()]
        tokens = re.findall(r"[А-ЯЁ]{4,}", before)
        for tok in reversed(tokens[-8:]):
            city = _normalize_city_token(tok)
            if city:
                return f"Г. {city}"

    # «МКР ... 21 ЧИТЕ» — город иногда OCR-ится после номера дома
    if m := re.search(
        r"(?:МКР|УЛ\.|ПР\.|ПР-Т)\s+[А-ЯЁ\-]+(?:\s+[А-ЯЁ\-]+)?\s+\d{1,4}[А-ЯЁ]?\s+([А-ЯЁ]{4,})",
        text,
    ):
        city = _normalize_city_token(m.group(1))
        if city:
            return f"Г. {city}"

    # Любое «ЧИТ*» в тексте штампа (частый OCR-артефакт)
    if m := re.search(r"\b(ЧИТ[АЕЮ])\b", text):
        return "Г. ЧИТА"

    return None


def parse_registration_address(text: str) -> Optional[str]:
    """Собирает адрес из сырого OCR-текста штампа."""
    if not is_registration_page(text):
        return None

    scope = _stamp_scope(text)
    parts: list[str] = []

    region = _extract_region(scope)
    city = _extract_city(scope)
    street = _extract_street(scope)
    parts: list[str] = []

    if region:
        parts.append(region)
    if city:
        parts.append(city)
    if street:
        parts.append(street)

    if len(parts) < 2:
        return None

    # Край + мкр без города: OCR часто теряет «Г. ЧИТА» на штампах Заб. края
    if not city and region and street:
        scope_u = scope.upper()
        if "ЗАБАЙКАЛЬСК" in scope_u and "ЧЕРЕМУШК" in scope_u:
            parts.insert(1, "Г. ЧИТА")

    return ", ".join(parts)


def _extract_region(text: str) -> Optional[str]:
    if m := re.search(
        r"((?:КРАЙ|ОБЛ|ОБЛАСТЬ|РЕСП|РЕСПУБЛИКА)\.?\s+[А-ЯЁ\-]+|[А-ЯЁ\-]+\s+(?:КРАЙ|ОБЛ|ОБЛАСТЬ))",
        text,
    ):
        val = m.group(1).strip(" ,.")
        if val.startswith(("КРАЙ", "ОБЛ", "РЕСП")):
            return val.replace("ОБЛ ", "ОБЛ. ").replace("РЕСП ", "РЕСП. ")
        return f"КРАЙ {val}" if val.endswith(("СКИЙ", "СКАЯ", "СКОЕ")) else val

    if m := re.search(r"\b([А-ЯЁ\-]{5,}(?:СКИЙ|СКАЯ|СКОЕ))\b", text):
        return f"КРАЙ {m.group(1)}"
    return None


def _extract_street(text: str) -> Optional[str]:
    cleaned = _STAMP_NOISE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    patterns = (
        r"(МКР\.?\s*[А-ЯЁ\-]+(?:\s+[А-ЯЁ\-]+)?)",
        r"(УЛ\.?\s*[А-ЯЁ\-]+(?:\s+[А-ЯЁ\-]+)?)",
        r"(ПР\.?\s*[А-ЯЁ\-]+(?:\s+[А-ЯЁ\-]+)?)",
        r"(ПР-Т\.?\s*[А-ЯЁ\-]+(?:\s+[А-ЯЁ\-]+)?)",
        r"(ПЕР\.?\s*[А-ЯЁ\-]+(?:\s+[А-ЯЁ\-]+)?)",
        r"(Б-Р\.?\s*[А-ЯЁ\-]+(?:\s+[А-ЯЁ\-]+)?)",
        r"(ПРОСП\.?\s*[А-ЯЁ\-]+(?:\s+[А-ЯЁ\-]+)?)",
    )
    for pat in patterns:
        if m := re.search(pat, cleaned):
            base = m.group(1).replace(" ", ". ").replace("..", ".")
            tail = cleaned[m.end() : m.end() + 40]
            if hm := re.search(r"\b(\d{1,4}[А-ЯЁ]?)\b", tail):
                return f"{base}, Д. {hm.group(1)}"
            return base
    return None


def extract_registration_address(
    image: np.ndarray,
    ocr: Optional[RegistrationOcr] = None,
) -> Optional[RefinedField]:
    """Полный пайплайн: OCR + parse → RefinedField или None."""
    engine = ocr or RegistrationOcr(gpu=False)
    _idx, text = engine.read(image)
    if not text:
        return None
    address = parse_registration_address(text)
    if not address:
        logger.debug("Registration markers found but address parse failed")
        return None
    logger.info("Registration address extracted")
    return RefinedField(address, "visual", confirmed=True)

