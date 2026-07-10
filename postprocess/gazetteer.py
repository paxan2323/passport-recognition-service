"""Словарная коррекция адресов (прописка / место рождения).

Фаззи-сверка распознанных токенов со справочниками:
- регионы РФ (85 субъектов) и города — из hflabs/city (CC BY-SA),
  postprocess/data/city.csv;
- типы улиц/населённых пунктов — нормализация сокращений.

Исправляет типичный OCR-шум: «ЧИТЕ» → «ЧИТА», «ЗАБАЙКАЛЬСКИ» →
«ЗАБАЙКАЛЬСКИЙ КРАЙ», выкидывает мусорные строки без словарных
совпадений. Улицы полного гейзеттира не имеют (нужен ФИАС), поэтому
для них — только нормализация типа и коррекция по известным основам.
"""

from __future__ import annotations

import csv
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

_CITY_CSV = Path(__file__).parent / "data" / "city.csv"

# сокращение типа региона -> каноническая форма (регион в им. падеже)
_REGION_TYPE_FULL = {
    "край": "КРАЙ", "обл": "ОБЛАСТЬ", "Респ": "РЕСПУБЛИКА",
    "г": "Г.", "АО": "АО", "Аобл": "АВТОНОМНАЯ ОБЛАСТЬ",
}

# нормализация типов улиц / населённых пунктов
STREET_TYPES = {
    "УЛ": "УЛ.", "УЛИЦА": "УЛ.", "ПР": "ПР-КТ", "ПРКТ": "ПР-КТ",
    "ПРОСП": "ПР-КТ", "ПРОСПЕКТ": "ПР-КТ", "ПЕР": "ПЕР.", "ПЕРЕУЛОК": "ПЕР.",
    "БР": "Б-Р", "БУЛЬВАР": "Б-Р", "Ш": "Ш.", "ШОССЕ": "Ш.",
    "НАБ": "НАБ.", "ПРОЕЗД": "ПРОЕЗД", "ТУП": "ТУП.", "МКР": "МКР.",
    "МКРН": "МКР.", "МИКРОРАЙОН": "МКР.",
}
SETTLEMENT_TYPES = {
    "Г": "Г.", "ГОР": "Г.", "ПОС": "ПОС.", "П": "П.", "ПГТ": "ПГТ",
    "С": "С.", "СТ": "СТ.", "СТЦА": "СТ-ЦА", "СТАНИЦА": "СТ-ЦА",
    "Д": "Д.", "ДЕР": "Д.", "Х": "Х.", "ХУТОР": "Х.", "АУЛ": "АУЛ",
    "МКР": "МКР.",
}

_CITY_THRESHOLD = 82
_REGION_THRESHOLD = 80


@lru_cache(maxsize=1)
def _load() -> tuple[frozenset[str], dict[str, str]]:
    """Возвращает (множество городов, {ядро_региона: каноническая_форма})."""
    if not _CITY_CSV.is_file():
        logger.warning("Справочник %s не найден — коррекция отключена", _CITY_CSV)
        return frozenset(), {}

    cities: set[str] = set()
    regions: dict[str, str] = {}
    with open(_CITY_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            city = (row.get("city") or "").strip().upper()
            if city:
                cities.add(city)
            region = (row.get("region") or "").strip().upper()
            rtype = (row.get("region_type") or "").strip()
            if region and region not in regions:
                regions[region] = _canonical_region(region, rtype)
    logger.info("Гейзеттир: %d городов, %d регионов", len(cities), len(regions))
    return frozenset(cities), regions


def _canonical_region(region: str, rtype: str) -> str:
    full = _REGION_TYPE_FULL.get(rtype, rtype.upper())
    if rtype == "Респ":
        return f"РЕСПУБЛИКА {region}"
    if rtype == "г":
        return f"Г. {region}"
    return f"{region} {full}".strip()


def correct_city(text: str, threshold: int = _CITY_THRESHOLD) -> Optional[str]:
    """Ближайший город из справочника или None."""
    cities, _ = _load()
    if not cities or not text:
        return None
    token = re.sub(r"[^А-ЯЁ\-]", "", text.upper())
    if len(token) < 3:
        return None
    match = process.extractOne(token, cities, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0]
    return None


def correct_region(text: str, threshold: int = _REGION_THRESHOLD) -> Optional[str]:
    """Ближайший регион (каноническая форма) или None."""
    _, regions = _load()
    if not regions or not text:
        return None
    token = re.sub(r"[^А-ЯЁ\-]", "", text.upper())
    if len(token) < 4:
        return None
    match = process.extractOne(token, regions.keys(), scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return regions[match[0]]
    return None


def normalize_settlement_type(token: str) -> Optional[str]:
    return SETTLEMENT_TYPES.get(re.sub(r"[^А-ЯЁ]", "", token.upper()))


def normalize_street_type(token: str) -> Optional[str]:
    return STREET_TYPES.get(re.sub(r"[^А-ЯЁ]", "", token.upper()))


def refine_address_lines(lines: list[str]) -> dict:
    """Собирает чистый адрес из шумных строк OCR через гейзеттир.

    Возвращает {region, city, street, raw_kept} — компоненты, которые
    удалось подтвердить словарём (город/регион) или структурой (тип+название
    улицы). Строки без единого совпадения отбрасываются как мусор.
    """
    region: Optional[str] = None
    city: Optional[str] = None
    street: Optional[str] = None
    kept: list[str] = []

    for line in lines:
        tokens = re.findall(r"[А-ЯЁ\-]{2,}", line.upper())
        matched = False

        # регион — по любому длинному токену
        if region is None:
            for tok in tokens:
                if len(tok) >= 6 and (r := correct_region(tok)):
                    region, matched = r, True
                    break

        # улица/микрорайон — тип + следующее название (+ номер дома)
        if street is None:
            street = _extract_street(line)
            if street:
                matched = True

        # город: сначала после маркера «Г.»/«ГОР.» (пониженный порог —
        # маркер уже подтверждает, что дальше город), потом по любому токену
        if city is None:
            marker = re.search(r"\b(?:Г|ГОР)\.?\s+([А-ЯЁ][А-ЯЁ\-]{2,})", line.upper())
            if marker and (c := correct_city(marker.group(1), threshold=68)):
                city, matched = c, True
        if city is None:
            for tok in tokens:
                if len(tok) >= 4 and tok not in STREET_TYPES and tok not in SETTLEMENT_TYPES:
                    if (c := correct_city(tok)):
                        city, matched = c, True
                        break

        if matched:
            kept.append(line)

    return {"region": region, "city": city, "street": street, "raw_kept": kept}


def _extract_street(line: str) -> Optional[str]:
    upper = line.upper()
    m = re.search(
        r"\b(УЛ|УЛИЦА|ПР|ПРОСП|ПРОСПЕКТ|ПЕР|БР|Б-Р|Ш|НАБ|ПРОЕЗД|ТУП|МКР|МКРН)\.?\s+([А-ЯЁ][А-ЯЁ\-]{2,}(?:\s+[А-ЯЁ\-]{2,})?)",
        upper,
    )
    if not m:
        return None
    stype = normalize_street_type(m.group(1)) or m.group(1)
    name = m.group(2).strip()
    result = f"{stype} {name}"
    house = re.search(r"\bД\.?\s*(\d{1,4}[А-ЯЁ]?)", upper[m.end():])
    if house:
        result += f", Д. {house.group(1)}"
    return result


def format_address(refined: dict) -> Optional[str]:
    parts = [refined.get("region"), refined.get("city"), refined.get("street")]
    parts = [p for p in parts if p]
    return ", ".join(parts) if parts else None
