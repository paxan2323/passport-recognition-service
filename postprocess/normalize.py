"""Нормализация и форматная валидация полей паспорта.

Каждая функция возвращает нормализованное значение или None,
если текст не проходит валидацию (тогда поле пойдёт на арбитраж
с MRZ / повторное распознавание).
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Optional

# Латинские двойники кириллицы: OCR иногда отдаёт их в русских полях
_LAT_TO_CYR = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
})


def normalize_name(text: str) -> Optional[str]:
    """ФИО: только кириллица, дефис и пробел, минимум 2 буквы."""
    if not text:
        return None
    cleaned = text.upper().translate(_LAT_TO_CYR)
    cleaned = re.sub(r"[^А-ЯЁ\- ]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    if len(re.sub(r"[^А-ЯЁ]", "", cleaned)) < 2:
        return None
    return cleaned


def normalize_date(text: str, extract: bool = True) -> Optional[str]:
    """Дата: строгий разбор 8 цифр, при неудаче — экстракция из шума.

    На реальных фото в кроп зоны затекают соседние поля, и верная дата
    часто содержится ВНУТРИ шумного предсказания («...94242 2.11.1983»).
    Экстракция ищет канонический паттерн Д[Д].ММ.ГГГГ (точки обязательны —
    это защита от ложных сборок из случайных цифр) и принимает только
    однозначный календарно-валидный кандидат.
    """
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        day, month, year = int(digits[:2]), int(digits[2:4]), int(digits[4:])
        try:
            return dt.date(year, month, day).strftime("%d.%m.%Y")
        except ValueError:
            return None
    if not extract:
        return None

    candidates: set[str] = set()
    for m in re.finditer(r"(?<!\d)(\d{1,2})[.,]\s?(\d{2})[.,]\s?(\d{4})(?!\d)", text):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1900 <= year <= 2099:
            continue
        try:
            candidates.add(dt.date(year, month, day).strftime("%d.%m.%Y"))
        except ValueError:
            continue
    if len(candidates) == 1:
        return candidates.pop()
    return None


# основы месяцев в родительном падеже (штампы: «14 июля 2015 г.»)
_MONTH_STEMS = {
    "ЯНВАР": 1, "ФЕВРАЛ": 2, "МАРТ": 3, "АПРЕЛ": 4, "МА": 5, "ИЮН": 6,
    "ИЮЛ": 7, "АВГУСТ": 8, "СЕНТЯБР": 9, "ОКТЯБР": 10, "НОЯБР": 11, "ДЕКАБР": 12,
}


def normalize_textual_date(text: str) -> Optional[str]:
    """Дата прописью: «14 июля 2015 г.» -> «14.07.2015» (устойчиво к OCR)."""
    if not text:
        return None
    upper = text.upper()
    m = re.search(r"\b(\d{1,2})\s+([А-ЯЁ]{3,})\s+(\d{4})", upper)
    if not m:
        return None
    day, month_word, year = int(m.group(1)), m.group(2), int(m.group(3))
    month = None
    for stem, num in _MONTH_STEMS.items():
        if month_word.startswith(stem):
            month = num
            break
    if month is None:
        return None
    try:
        return dt.date(year, month, day).strftime("%d.%m.%Y")
    except ValueError:
        return None


def normalize_any_date(text: str) -> Optional[str]:
    """Дата в любом формате: числовая или прописью."""
    return normalize_date(text) or normalize_textual_date(text)


def normalize_series_number(text: str, extract: bool = True) -> Optional[str]:
    """Серия и номер: ровно 10 цифр -> 'СС СС НННННН'; из шума — экстракцией."""
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 10:
        return f"{digits[:2]} {digits[2:4]} {digits[4:]}"
    if not extract:
        return None

    candidates = {
        f"{m.group(1)} {m.group(2)} {m.group(3)}"
        for m in re.finditer(r"(?<!\d)(\d{2})\s?(\d{2})\s?(\d{6})(?!\d)", text)
    }
    if len(candidates) == 1:
        return candidates.pop()
    return None


def normalize_issue_code(text: str, extract: bool = True) -> Optional[str]:
    """Код подразделения: ровно 6 цифр -> 'XXX-XXX'; из шума — только с дефисом."""
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 6:
        return f"{digits[:3]}-{digits[3:]}"
    if not extract:
        return None

    # без строгого дефиса любые 6 цифр подряд дали бы ложные коды
    candidates = {
        f"{m.group(1)}-{m.group(2)}"
        for m in re.finditer(r"(?<!\d)(\d{3})-(\d{3})(?!\d)", text)
    }
    if len(candidates) == 1:
        return candidates.pop()
    return None


def normalize_sex(text: str) -> Optional[str]:
    """Пол: 'МУЖ.' / 'ЖЕН.'."""
    if not text:
        return None
    upper = text.upper().translate(_LAT_TO_CYR)
    if "ЖЕН" in upper or upper.strip(". ").startswith("Ж"):
        return "ЖЕН."
    if "МУЖ" in upper or upper.strip(". ").startswith("М"):
        return "МУЖ."
    return None


def normalize_text_field(text: str) -> Optional[str]:
    """Место рождения / кем выдан: кириллица, цифры, базовая пунктуация."""
    if not text:
        return None
    cleaned = text.upper().translate(_LAT_TO_CYR)
    cleaned = re.sub(r"[^А-ЯЁ0-9\-.,№ ]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # "Г.КРАСНОДАР" / "Г КРАСНОДАР" -> "Г. КРАСНОДАР"
    cleaned = re.sub(
        r"\b(Г|ГОР|ПОС|С|Д|СТ|П|ПГТ)\.(?=[А-ЯЁ])",
        r"\1. ",
        cleaned,
    )
    cleaned = re.sub(r"^(Г|ГОР|ПОС|С|Д|СТ)\s+", r"\1. ", cleaned)
    if len(re.sub(r"[^А-ЯЁ]", "", cleaned)) < 3:
        return None
    return cleaned
