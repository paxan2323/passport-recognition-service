"""Арбитраж полей паспорта: визуальные зоны vs MRZ vs справочник.

Правила:
- числовые поля (серия/номер, дата рождения, дата выдачи, код) берутся
  из MRZ, если сошлась контрольная цифра; иначе из визуальной зоны,
  прошедшей форматную валидацию;
- ФИО берётся из визуальной зоны, если это валидная кириллица, и
  сверяется с транслитерацией MRZ; при нечитаемой зоне — из MRZ;
- «Кем выдан» подставляется из справочника по коду подразделения.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from docreader.schemas import DocumentResult, ZoneResult

from postprocess import normalize
from postprocess.birth_place import refine_birth_place, score_birth_place_candidate
from postprocess.division_codes import lookup_issued_by
from postprocess.mrz import MrzResult, parse_mrz
from postprocess.mrz_ocr import MrzOcr

logger = logging.getLogger(__name__)

# Возможные имена зон детектора -> каноническое имя поля.
# Первыми идут фактические классы passport.pt (birthCity, dateOfBirth, ...).
_ZONE_ALIASES: dict[str, tuple[str, ...]] = {
    "surname": ("surname", "familia", "last_name"),
    "name": ("firstname", "name", "first_name"),
    "patronymic": ("middlename", "patronymic", "middle_name"),
    "sex": ("sex", "gender", "pol"),
    "birth_date": ("dateOfBirth", "birth_date", "birthdate"),
    "birth_place": ("birthCity", "birth_place", "birthplace"),
    "series_number": ("passport_num", "series_number", "series_num", "number"),
    "issue_date": ("issued_date", "issue_date", "date_of_issue"),
    "issue_code": ("issued_code", "issue_code", "division_code"),
    "issued_by": ("issued_by", "issued", "authority"),
    "mrz_line1": ("mrz_line1", "mrz1"),
    "mrz_line2": ("mrz_line2", "mrz2"),
}

_NAME_MATCH_THRESHOLD = 92
# Прежний порог (75) пропускал реальные ошибки OCR как "подтверждённые"
# (напр. визуальное "ЦМИТРИЕВН" против MRZ-транслитерации "ДМИТРИЕВНА"
# даёт fuzzy ratio ~89% — по старому порогу считалось совпадением).
# Это загрязняло обучающие пары, собираемые как "confirmed" из
# collect_from_s3.py/collect_dataset.py, что и привело к регрессии
# распознавалки при дообучении на реальных S3-данных (03-04.07.2026).


@dataclass
class RefinedField:
    value: Optional[str]
    source: str          # visual | mrz | lookup | none
    confirmed: bool      # подтверждено вторым источником или чексуммой
    note: str = ""

    def as_dict(self) -> dict:
        data = {"value": self.value, "source": self.source, "confirmed": self.confirmed}
        if self.note:
            data["note"] = self.note
        return data


def _zones(doc: DocumentResult, canonical: str) -> list[ZoneResult]:
    """Все зоны поля (детектор может найти дубликаты), лучшие первыми."""
    aliases = _ZONE_ALIASES[canonical]
    matched = [zone for zone in doc.zones if zone.name in aliases]
    matched.sort(key=lambda z: z.confidence, reverse=True)
    return matched


def _zone(doc: DocumentResult, canonical: str) -> Optional[ZoneResult]:
    zones = _zones(doc, canonical)
    return zones[0] if zones else None


def _zone_text(doc: DocumentResult, canonical: str) -> str:
    zone = _zone(doc, canonical)
    return zone.text if zone else ""


def _pick_birth_place_raw(doc: DocumentResult) -> str:
    """Лучшая зона birthCity среди дубликатов (не дата/мусор)."""
    best_raw = ""
    best_score = -1.0
    for zone in _zones(doc, "birth_place"):
        raw = zone.text or ""
        norm = normalize.normalize_text_field(raw)
        score = score_birth_place_candidate(raw, norm, zone.confidence)
        if score > best_score:
            best_score = score
            best_raw = raw
    return best_raw


def _read_mrz(doc: DocumentResult, mrz_ocr: Optional[MrzOcr]) -> MrzResult:
    line1_zone, line2_zone = _zone(doc, "mrz_line1"), _zone(doc, "mrz_line2")
    line1 = line1_zone.text if line1_zone else ""
    line2 = line2_zone.text if line2_zone else ""

    if mrz_ocr is not None:
        if line1_zone is not None and line1_zone.crop_image is not None:
            line1 = mrz_ocr.read_line(line1_zone.crop_image) or line1
        if line2_zone is not None and line2_zone.crop_image is not None:
            line2 = mrz_ocr.read_line(line2_zone.crop_image) or line2
        if not (line1 or line2) and doc.doc_crop is not None:
            # Детектор passport.pt не размечает MRZ — читаем её
            # напрямую из нижней части кропа документа.
            line1, line2 = mrz_ocr.read_document(doc.doc_crop)

    # Визуальная серия/номер как подсказка для чексумм-коррекции MRZ:
    # 10 цифр «SS SS NNNNNN» -> 9 цифр поля MRZ (без 4-й цифры серии)
    number_hint = None
    for zone in _zones(doc, "series_number"):
        visual = normalize.normalize_series_number(zone.text)
        if visual:
            digits = visual.replace(" ", "")
            number_hint = digits[:3] + digits[4:]
            break
    return parse_mrz(line1, line2, number_hint=number_hint)


def _pick_name(
    visual_raw: str,
    mrz_value: Optional[str],
    mrz_reliable: bool = False,
) -> RefinedField:
    visual = normalize.normalize_name(visual_raw)
    mrz_clean = mrz_value if (mrz_value and "?" not in mrz_value) else None

    if visual and mrz_clean:
        score = fuzz.ratio(visual, mrz_clean)
        if score >= _NAME_MATCH_THRESHOLD:
            return RefinedField(visual, "visual", True)
        if mrz_reliable:
            # чексуммы строки 2 сошлись -> MRZ прочитана качественно,
            # при конфликте доверяем ей, а не шумной визуальной зоне
            return RefinedField(mrz_clean, "mrz", False, f"визуальная зона: {visual}")
        return RefinedField(visual, "visual", False, f"расходится с MRZ: {mrz_clean}")
    if visual:
        return RefinedField(visual, "visual", False, "MRZ не читается")
    if mrz_clean:
        return RefinedField(mrz_clean, "mrz", False, "видимая зона не читается")
    if mrz_value:
        return RefinedField(mrz_value, "mrz", False, "MRZ с ошибками ('?')")
    return RefinedField(None, "none", False)


def _pick_checked(
    visual_value: Optional[str],
    mrz_value: Optional[str],
    mrz_ok: bool,
) -> RefinedField:
    """Арбитраж поля, у которого в MRZ есть контрольная цифра."""
    if mrz_ok and visual_value and visual_value == mrz_value:
        return RefinedField(visual_value, "visual", True)
    if mrz_ok and mrz_value:
        note = f"визуальная зона: {visual_value}" if visual_value else "визуальная зона не читается"
        return RefinedField(mrz_value, "mrz", True, note)
    if visual_value and mrz_value == visual_value:
        # чексумма не сошлась, но два источника совпали
        return RefinedField(visual_value, "visual", True, "подтверждено MRZ без чексуммы")
    if visual_value:
        return RefinedField(visual_value, "visual", False)
    if mrz_value:
        return RefinedField(mrz_value, "mrz", False, "чексумма не сошлась")
    return RefinedField(None, "none", False)


def _pick_series_number(doc: DocumentResult, mrz: MrzResult) -> RefinedField:
    # Серия/номер напечатаны на странице дважды — берём первую зону,
    # прошедшую форматную валидацию (ровно 10 цифр)
    visual = None
    for zone in _zones(doc, "series_number"):
        visual = normalize.normalize_series_number(zone.text)
        if visual:
            break
    mrz_full = mrz.series_number if (mrz.series_number and len(mrz.series_number.replace(" ", "")) == 10) else None

    if mrz.number_ok and visual:
        visual_digits = visual.replace(" ", "")
        mrz_digits = mrz_full.replace(" ", "") if mrz_full else (mrz.doc9 or "")
        if visual_digits == mrz_digits:
            return RefinedField(visual, "visual", True)
        # Источники расходятся. Одна чексумма может «сойтись» и на битом
        # чтении (пропуск цифры), поэтому MRZ побеждает только если
        # подтверждена и вторая чексумма (необязательного поля).
        if mrz.optional_ok and mrz_full:
            return RefinedField(mrz_full, "mrz", True, f"визуальная зона: {visual}")
        return RefinedField(visual, "visual", False, f"расходится с MRZ: {mrz_full or mrz.doc9}")
    if mrz.number_ok and mrz_full:
        return RefinedField(mrz_full, "mrz", True, "визуальная зона не читается")
    return _pick_checked(visual, mrz_full, False)


def refine_passport(doc: DocumentResult, mrz_ocr: Optional[MrzOcr] = None) -> dict[str, RefinedField]:
    """Возвращает финальные поля паспорта после валидации и арбитража."""
    mrz = _read_mrz(doc, mrz_ocr)
    if mrz.problems:
        logger.info("MRZ: %s", "; ".join(mrz.problems))
    # чексуммы строки 2 сошлись -> строка 1 (ФИО) почти наверняка тоже чистая
    mrz_reliable = mrz.number_ok and mrz.birth_date_ok

    fields: dict[str, RefinedField] = {
        "surname": _pick_name(_zone_text(doc, "surname"), mrz.surname, mrz_reliable),
        "name": _pick_name(_zone_text(doc, "name"), mrz.name, mrz_reliable),
        "patronymic": _pick_name(_zone_text(doc, "patronymic"), mrz.patronymic, mrz_reliable),
        "series_number": _pick_series_number(doc, mrz),
        "birth_date": _pick_checked(
            normalize.normalize_date(_zone_text(doc, "birth_date")),
            mrz.birth_date, mrz.birth_date_ok,
        ),
        "issue_date": _pick_checked(
            normalize.normalize_date(_zone_text(doc, "issue_date")),
            mrz.issue_date, mrz.optional_ok,
        ),
        "issue_code": _pick_checked(
            normalize.normalize_issue_code(_zone_text(doc, "issue_code")),
            mrz.issue_code, mrz.optional_ok,
        ),
    }

    visual_sex = normalize.normalize_sex(_zone_text(doc, "sex"))
    if visual_sex:
        fields["sex"] = RefinedField(visual_sex, "visual", visual_sex == mrz.sex)
    elif mrz.sex:
        fields["sex"] = RefinedField(mrz.sex, "mrz", False, "видимая зона не читается")
    else:
        fields["sex"] = RefinedField(None, "none", False)

    birth_norm = normalize.normalize_text_field(_pick_birth_place_raw(doc))
    birth_place, birth_confirmed, birth_note = refine_birth_place(birth_norm or "")
    fields["birth_place"] = RefinedField(
        birth_place,
        "visual" if birth_place else "none",
        birth_confirmed,
        birth_note,
    )

    fields["issued_by"] = _pick_issued_by(doc, fields["issue_code"])
    fields["mrz"] = RefinedField(
        f"{mrz.raw_line1} | {mrz.raw_line2}".strip(" |"),
        "mrz",
        mrz.number_ok and mrz.birth_date_ok,
        "; ".join(mrz.problems),
    )
    return fields


def _pick_issued_by(doc: DocumentResult, issue_code: RefinedField) -> RefinedField:
    visual = normalize.normalize_text_field(_zone_text(doc, "issued_by"))
    if issue_code.value:
        official = lookup_issued_by(issue_code.value, ocr_hint=visual)
        if official:
            confirmed = bool(visual) and fuzz.token_set_ratio(visual, official) >= 70
            return RefinedField(official, "lookup", confirmed, f"OCR: {visual}" if visual else "")
    if visual:
        return RefinedField(visual, "visual", False, "код не найден в справочнике")
    return RefinedField(None, "none", False)
