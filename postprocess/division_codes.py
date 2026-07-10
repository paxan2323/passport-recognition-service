"""Справочник «Кем выдан» по коду подразделения.

Данные: https://github.com/hflabs/fms-unit (собран HFLabs/DaData,
CC BY-SA). Файл postprocess/data/fms_unit.csv, формат: code,name.
На один код приходится несколько исторических наименований —
выбираем то, что лучше всего совпадает с распознанным текстом.
"""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

_CSV_PATH = Path(__file__).parent / "data" / "fms_unit.csv"


@lru_cache(maxsize=1)
def _load() -> dict[str, list[str]]:
    if not _CSV_PATH.is_file():
        logger.warning("Справочник %s не найден — lookup отключён", _CSV_PATH)
        return {}
    by_code: dict[str, list[str]] = {}
    with open(_CSV_PATH, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            code = row.get("code", "").strip()
            name = row.get("name", "").strip()
            if code and name:
                by_code.setdefault(code, []).append(name)
    logger.info("Справочник подразделений: %d кодов", len(by_code))
    return by_code


def lookup_issued_by(issue_code: str, ocr_hint: Optional[str] = None) -> Optional[str]:
    """Официальное «Кем выдан» по коду подразделения.

    Если есть распознанный текст поля (ocr_hint) — берём кандидата,
    наиболее похожего на него; иначе первый (актуальный) вариант.
    """
    candidates = _load().get(issue_code)
    if not candidates:
        return None
    if ocr_hint:
        return max(candidates, key=lambda c: fuzz.token_set_ratio(ocr_hint.upper(), c.upper()))
    return candidates[0]
