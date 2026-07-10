"""Map postprocess RefinedField to API JSON schema."""

from __future__ import annotations

from typing import Any, Optional

from docreader.schemas import DocumentResult
from postprocess.mrz import MrzResult
from postprocess.passport import RefinedField, _read_mrz
from postprocess.mrz_ocr import MrzOcr
from service.core.engine import RecognitionOutcome
from service.core.schemas import (
    DocumentBlock,
    ErrorDetail,
    FieldSource,
    FieldValue,
    MrzBlock,
    MrzChecksums,
    RecognizeError,
    RecognizeSuccess,
    Warning,
    WarningCode,
)

# Field weights for overall_confidence (critical fields weigh more)
_FIELD_WEIGHTS: dict[str, float] = {
    "series": 1.5,
    "number": 1.5,
    "surname": 1.2,
    "name": 1.2,
    "birth_date": 1.0,
    "patronymic": 0.8,
    "sex": 0.5,
    "birth_place": 0.5,
    "issue_date": 0.8,
    "issue_code": 0.8,
    "issued_by": 0.6,
    "registration_address": 0.3,
}

_ZONE_CONFIDENCE: dict[str, tuple[str, ...]] = {
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
}


def _zone_confidence(doc: DocumentResult, aliases: tuple[str, ...]) -> float:
    best = 0.0
    for zone in doc.zones:
        if zone.name in aliases and zone.confidence > best:
            best = zone.confidence
    return round(best, 4)


def _source_enum(source: str) -> FieldSource:
    try:
        return FieldSource(source)
    except ValueError:
        return FieldSource.NONE


def _split_series_number(value: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not value:
        return None, None
    parts = value.replace("  ", " ").split()
    if len(parts) >= 3:
        return f"{parts[0]} {parts[1]}", parts[2]
    digits = value.replace(" ", "")
    if len(digits) == 10:
        return f"{digits[:2]} {digits[2:4]}", digits[4:]
    return value, None


def _field_confidence(
    canonical: str, doc: DocumentResult, field: RefinedField
) -> float:
    if field.value is None:
        return 0.0
    if field.confirmed:
        return min(1.0, _zone_confidence(doc, _ZONE_CONFIDENCE.get(canonical, ())) + 0.05)
    conf = _zone_confidence(doc, _ZONE_CONFIDENCE.get(canonical, ()))
    if conf > 0:
        if canonical == "birth_place" and not field.confirmed:
            conf = min(conf, 0.9)
        return conf
    if field.source in ("mrz", "lookup"):
        return 0.95 if field.confirmed else 0.85
    return 0.5


def _build_mrz_block(mrz: MrzResult) -> MrzBlock:
    return MrzBlock(
        line1=mrz.raw_line1,
        line2=mrz.raw_line2,
        valid=mrz.number_ok and mrz.birth_date_ok,
        checksums=MrzChecksums(
            number=mrz.number_ok,
            birth_date=mrz.birth_date_ok,
            optional=mrz.optional_ok,
        ),
    )


def _overall_confidence(fields: dict[str, FieldValue]) -> float:
    total_w = 0.0
    weighted = 0.0
    for name, weight in _FIELD_WEIGHTS.items():
        fv = fields.get(name)
        if fv is None:
            continue
        total_w += weight
        weighted += weight * fv.confidence
    if total_w == 0:
        return 0.0
    return round(weighted / total_w, 4)


def map_success(
    outcome: RecognitionOutcome,
    request_id: str,
    processing_time_ms: int,
    mrz_ocr: Optional[MrzOcr] = None,
) -> RecognizeSuccess:
    doc = outcome.doc
    refined = outcome.refined
    mrz = outcome.mrz
    if mrz is None and doc is not None:
        mrz = _read_mrz(doc, mrz_ocr)

    fields: dict[str, FieldValue] = {}
    series_val, number_val = _split_series_number(
        refined.get("series_number", RefinedField(None, "none", False)).value
    )

    for canonical, rf in refined.items():
        if canonical == "series_number":
            continue
        conf = _field_confidence(canonical, doc, rf) if doc else (0.85 if rf.confirmed else 0.5)
        fields[canonical] = FieldValue(
            value=rf.value,
            confidence=round(conf, 4),
            source=_source_enum(rf.source),
            confirmed=rf.confirmed,
        )

    fields["series"] = FieldValue(
        value=series_val,
        confidence=_field_confidence("series_number", doc, refined.get("series_number", RefinedField(None, "none", False))) if doc else 0.0,
        source=_source_enum(refined.get("series_number", RefinedField(None, "none", False)).source),
        confirmed=refined.get("series_number", RefinedField(None, "none", False)).confirmed,
    )
    fields["number"] = FieldValue(
        value=number_val,
        confidence=fields["series"].confidence,
        source=fields["series"].source,
        confirmed=fields["series"].confirmed,
    )

    reg_rf = refined.get("registration_address")
    if reg_rf and reg_rf.value:
        conf = _field_confidence("registration_address", doc, reg_rf) if doc else 0.85
        fields["registration_address"] = FieldValue(
            value=reg_rf.value,
            confidence=round(max(conf, 0.75), 4),
            source=_source_enum(reg_rf.source),
            confirmed=reg_rf.confirmed,
        )
    else:
        fields["registration_address"] = FieldValue(
            value=None,
            confidence=0.0,
            source=FieldSource.NONE,
            confirmed=False,
        )

    warnings = [
        Warning(code=WarningCode(w["code"]), field=w.get("field"), message=w["message"])
        for w in outcome.warnings
    ]
    if not reg_rf or not reg_rf.value:
        warnings.append(Warning(
            code=WarningCode.FIELD_NOT_FOUND,
            field="registration_address",
            message="Разворот с пропиской не обнаружен",
        ))

    doc_confidence = doc.doc_confidence if doc else 0.0
    return RecognizeSuccess(
        request_id=request_id,
        processing_time_ms=processing_time_ms,
        document=DocumentBlock(
            confidence=round(doc_confidence, 4) if doc else 0.0,
            fields=fields,
            mrz=_build_mrz_block(mrz) if mrz else MrzBlock(),
            overall_confidence=_overall_confidence(fields),
            warnings=warnings,
        ),
    )


def map_error(
    error_code: str,
    message: str,
    request_id: str,
    details: Optional[dict[str, Any]] = None,
) -> RecognizeError:
    return RecognizeError(
        request_id=request_id,
        error=ErrorDetail(code=error_code, message=message, details=details),
    )
