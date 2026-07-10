"""Уточнение места рождения после OCR.

Не сводит всё к городу: село/деревня/посёлок сохраняются как есть,
гейзеттир применяется только к городам (маркер Г./ГОР.) и регионам.
"""

from __future__ import annotations

import re
from typing import Optional

from postprocess.gazetteer import (
    SETTLEMENT_TYPES,
    correct_city,
    correct_region,
    normalize_settlement_type,
)

_CITY_MARKERS = frozenset({"Г", "ГОР"})
_VILLAGE_MARKERS = frozenset({"С", "Д", "ПОС", "П", "ПГТ", "СТ", "СТ-ЦА", "Х", "АУЛ"})
_REGION_HINTS = ("КРАЙ", "ОБЛ", "ОБЛАСТЬ", "РЕСП", "РЕСПУБЛИКА", "АО", "АОБЛ")
_SETTLEMENT_RE = re.compile(
    r"^(Г|ГОР|ПОС|С|Д|СТ|П|ПГТ|СТ-ЦА|АУЛ|Х)\.\s*(.+)$",
    re.IGNORECASE,
)


def refine_birth_place(text: str) -> tuple[Optional[str], bool, str]:
    """Возвращает (значение, подтверждено_словарём, примечание)."""
    if not text:
        return None, False, ""

    segments = [s.strip() for s in re.split(r"[,;]", text) if s.strip()]
    if not segments:
        segments = [text.strip()]

    parts: list[str] = []
    confirmed = False
    notes: list[str] = []

    for seg in segments:
        refined, ok, note = _refine_segment(seg)
        if refined:
            parts.append(refined)
            confirmed = confirmed or ok
            if note:
                notes.append(note)

    if not parts:
        return text, False, ""
    return ", ".join(parts), confirmed, "; ".join(notes)


def score_birth_place_candidate(raw: str, normalized: Optional[str], confidence: float) -> float:
    """Ранжирует дубликаты зоны birthCity (дата/мусор — вниз)."""
    if not normalized:
        return -1.0

    score = float(confidence)
    if _looks_like_date(raw) or _looks_like_date(normalized):
        score -= 2.0

    digits = len(re.sub(r"\D", "", raw))
    letters = len(re.sub(r"[^А-ЯЁа-яё]", "", raw, flags=re.IGNORECASE))
    if digits and digits >= max(letters, 1):
        score -= 1.5

    if _SETTLEMENT_RE.match(normalized):
        score += 0.12

    _, gaz_ok, _ = refine_birth_place(normalized)
    if gaz_ok:
        score += 0.2

    score += min(len(normalized), 48) * 0.0025
    return score


def _refine_segment(segment: str) -> tuple[str, bool, str]:
    upper = segment.upper().strip()
    if not upper:
        return "", False, ""

    m = _SETTLEMENT_RE.match(upper)
    if m:
        stype, rest = m.group(1).upper(), m.group(2).strip()
        canon = normalize_settlement_type(stype) or f"{stype}."
        if stype in _CITY_MARKERS:
            fixed, ok = _refine_city_name(rest, marked_as_city=True)
            note = f"OCR: {rest}" if ok and fixed.upper() != rest else ""
            return f"{canon} {fixed}", ok, note
        # село, деревня, посёлок — название не подменяем городом из справочника
        return f"{canon} {rest}", False, ""

    if any(h in upper for h in _REGION_HINTS):
        region, ok = _refine_region(upper)
        if region:
            return region, ok, f"регион: {segment}" if ok else ""

    fixed, ok = _refine_city_name(upper, marked_as_city=False)
    if ok:
        return fixed, True, f"OCR: {segment}" if fixed.upper() != upper else ""

    return segment, False, ""


def _refine_region(text: str) -> tuple[Optional[str], bool]:
    for tok in re.findall(r"[А-ЯЁ\-]{4,}", text.upper()):
        if tok in SETTLEMENT_TYPES or tok in _REGION_HINTS:
            continue
        if region := correct_region(tok):
            return region, region.upper() not in text.upper()
    return None, False


def _refine_city_name(name: str, *, marked_as_city: bool) -> tuple[str, bool]:
    """Коррекция города; без маркера Г./ГОР. — строже порог."""
    token = re.sub(r"[^А-ЯЁ\- ]", "", name.upper()).strip()
    if len(re.sub(r"[^А-ЯЁ]", "", token)) < 3:
        return name, False

    threshold = 82 if marked_as_city else 88
    candidates: list[str] = [token]
    if " " in token:
        candidates.append(token.replace(" ", "-"))
        candidates.append(token.split()[-1])

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if corrected := correct_city(cand, threshold=threshold):
            if corrected != cand and _safe_city_fix(cand, corrected):
                return corrected, True
        if len(cand) >= 5 and (corrected := correct_city(cand, threshold=max(threshold - 4, 78))):
            if corrected != cand and _is_truncation_fix(cand, corrected):
                return corrected, True

    return name, False


def _safe_city_fix(original: str, corrected: str) -> bool:
    """Не подменять произвольное слово похожим городом."""
    o = re.sub(r"[^А-ЯЁ]", "", original.upper())
    c = re.sub(r"[^А-ЯЁ]", "", corrected.upper())
    if o == c:
        return False
    if c.startswith(o) or o.startswith(c[: max(3, len(c) - 2)]):
        return True
    return len(o) >= 5 and len(c) >= 5


def _is_truncation_fix(original: str, corrected: str) -> bool:
    o = re.sub(r"[^А-ЯЁ]", "", original.upper())
    c = re.sub(r"[^А-ЯЁ]", "", corrected.upper())
    return c.startswith(o) and len(c) > len(o)


def _looks_like_date(text: str) -> bool:
    from postprocess import normalize

    return bool(normalize.normalize_date(text, extract=True))
