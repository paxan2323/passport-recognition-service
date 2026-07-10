"""RecognitionEngine abstraction and docreader adapter."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from docreader.schemas import DocumentResult, PageResult
from postprocess.mrz import MrzResult
from postprocess.mrz_ocr import MrzOcr
from postprocess.passport import RefinedField, refine_passport
from service.core.config import get_settings
from service.core.errors import (
    DOCUMENT_NOT_FOUND,
    LOW_QUALITY,
    NOT_A_PASSPORT,
    OCR_FAILED,
    ServiceError,
)

from service.core.quality import QualityMetrics, assess_doc_crop

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("series_number", "surname", "name", "birth_date")


@dataclass
class RecognitionOutcome:
    success: bool
    error: Optional[ServiceError] = None
    error_details: Optional[dict[str, Any]] = None
    doc: Optional[DocumentResult] = None
    refined: dict[str, RefinedField] = field(default_factory=dict)
    mrz: Optional[MrzResult] = None
    quality: Optional[QualityMetrics] = None
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)


class RecognitionEngine(ABC):
    @abstractmethod
    def warmup(self) -> None:
        """Load models."""

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def recognize(self, image_bytes: bytes, min_quality: str = "normal") -> RecognitionOutcome:
        ...


class DocReaderEngine(RecognitionEngine):
    """Adapter wrapping docreader-ocr + postprocess."""

    def __init__(self) -> None:
        self._reader = None
        self._mrz_ocr: Optional[MrzOcr] = None
        self._reg_ocr = None
        self._reg_detector = None
        self._ready = False

    def warmup(self) -> None:
        from postprocess import MrzOcr, build_reader

        logger.info("Loading OCR models...")
        t0 = time.perf_counter()
        self._reader = build_reader()
        self._mrz_ocr = MrzOcr(gpu=False)
        self._ready = True
        logger.info("Models loaded in %.1fs", time.perf_counter() - t0)

    def is_ready(self) -> bool:
        return self._ready and self._reader is not None

    def recognize(self, image_bytes: bytes, min_quality: str = "normal") -> RecognitionOutcome:
        if not self.is_ready():
            raise RuntimeError("Engine not ready")

        settings = get_settings()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        import cv2

        from postprocess.orientation import decode_image

        image = decode_image(image_bytes)
        if image is None:
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return RecognitionOutcome(
                success=False,
                error=DOCUMENT_NOT_FOUND,
                error_details={"reason": "decode_failed"},
            )

        t_classify = time.perf_counter()
        from postprocess.orientation import process_with_rotation

        page, image, rotation_deg = process_with_rotation(
            self._reader,
            image,
            self._mrz_ocr,
            pick_doc=self._pick_passport_doc,
            auto_rotate=settings.auto_rotate_passport,
            mode=settings.orientation_mode,
        )
        timings["classify_detect_ocr_ms"] = (time.perf_counter() - t_classify) * 1000
        if rotation_deg:
            timings["orientation_deg"] = rotation_deg

        doc = self._pick_passport_doc(page)
        reg_field: Optional[RefinedField] = None
        if not self._looks_like_main_passport(doc, settings):
            t_reg = time.perf_counter()
            reg_field = self._try_registration(image)
            timings["registration_ms"] = (time.perf_counter() - t_reg) * 1000
        else:
            timings["registration_ms"] = 0.0
            logger.debug("Skipping registration OCR: main passport spread detected")

        if not page.documents:
            if reg_field:
                return self._registration_only_outcome(
                    reg_field, None, timings, t0, warnings=[]
                )
            return RecognitionOutcome(
                success=False,
                error=DOCUMENT_NOT_FOUND,
                timings=timings,
            )

        if doc is None:
            first = page.documents[0]
            if reg_field:
                return self._registration_only_outcome(
                    reg_field, first, timings, t0, warnings=[],
                    quality=assess_doc_crop(first.doc_crop),
                )
            return RecognitionOutcome(
                success=False,
                error=NOT_A_PASSPORT,
                error_details={
                    "detected": first.doc_type,
                    "confidence": first.doc_confidence,
                    "candidates": [
                        {"type": d.doc_type, "confidence": d.doc_confidence}
                        for d in page.documents
                    ],
                },
                doc=first,
                quality=assess_doc_crop(first.doc_crop),
                timings=timings,
            )

        quality = assess_doc_crop(doc.doc_crop)

        if doc.doc_confidence < settings.min_doc_confidence:
            return RecognitionOutcome(
                success=False,
                error=NOT_A_PASSPORT,
                error_details={"detected": doc.doc_type, "confidence": doc.doc_confidence},
                doc=doc,
                quality=quality,
                timings=timings,
            )

        if min_quality != "low" and quality.blur_score < settings.min_quality_blur:
            return RecognitionOutcome(
                success=False,
                error=LOW_QUALITY,
                error_details={
                    "blur_score": quality.blur_score,
                    "glare_ratio": quality.glare_ratio,
                    "width": quality.width,
                },
                doc=doc,
                quality=quality,
                timings=timings,
            )

        if min_quality != "low" and quality.glare_ratio > settings.min_quality_glare_ratio:
            return RecognitionOutcome(
                success=False,
                error=LOW_QUALITY,
                error_details={
                    "blur_score": quality.blur_score,
                    "glare_ratio": quality.glare_ratio,
                    "width": quality.width,
                },
                doc=doc,
                quality=quality,
                timings=timings,
            )

        t_pp = time.perf_counter()
        refined = refine_passport(doc, self._mrz_ocr)
        timings["postprocess_ms"] = (time.perf_counter() - t_pp) * 1000

        if reg_field:
            refined["registration_address"] = reg_field

        # Parse MRZ for response block
        mrz_field = refined.pop("mrz", None)
        mrz = self._extract_mrz(doc, mrz_field)

        warnings: list[dict[str, Any]] = []
        if rotation_deg:
            warnings.append({
                "code": "ORIENTATION_CORRECTED",
                "field": None,
                "message": f"Изображение повёрнуто на {rotation_deg}° для распознавания",
            })
        if quality.low_resolution:
            warnings.append({
                "code": "LOW_RESOLUTION",
                "field": None,
                "message": f"Разрешение документа {quality.width}px по ширине < 800px",
            })

        missing_required = [
            f for f in REQUIRED_FIELDS if not refined.get(f) or not refined[f].value
        ]
        has_registration = bool(
            refined.get("registration_address") and refined["registration_address"].value
        )
        if missing_required and any(refined.get(f) and refined[f].value for f in REQUIRED_FIELDS):
            warnings.append({
                "code": "PARTIAL_DOCUMENT",
                "field": ",".join(missing_required),
                "message": "Не все обязательные поля распознаны",
            })

        if missing_required and not any(refined.get(f) and refined[f].value for f in REQUIRED_FIELDS):
            if has_registration:
                timings["total_ms"] = (time.perf_counter() - t0) * 1000
                return RecognitionOutcome(
                    success=True,
                    doc=doc,
                    refined=refined,
                    mrz=mrz,
                    quality=quality,
                    timings=timings,
                    warnings=warnings,
                )
            return RecognitionOutcome(
                success=False,
                error=OCR_FAILED,
                error_details={"missing_fields": missing_required},
                doc=doc,
                refined=refined,
                mrz=mrz,
                quality=quality,
                timings=timings,
                warnings=warnings,
            )

        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return RecognitionOutcome(
            success=True,
            doc=doc,
            refined=refined,
            mrz=mrz,
            quality=quality,
            timings=timings,
            warnings=warnings,
        )

    @staticmethod
    def _pick_passport_doc(page: PageResult):
        """На сканах разворота классификатор часто находит несколько документов
        (passport + ложные attestat/diplom). Берём лучший passport, не первый."""
        passports = [d for d in page.documents if d.doc_type == "passport"]
        if not passports:
            return None
        return max(passports, key=lambda d: (len(d.zones), d.doc_confidence))

    _MAIN_PASSPORT_ZONES = frozenset({
        "passport_num", "surname", "firstname", "middlename",
        "dateOfBirth", "birthCity", "sex",
        "issued_date", "issued_code", "issued_by",
        "mrz_line1", "mrz_line2",
    })

    @classmethod
    def _looks_like_main_passport(cls, doc: Optional[DocumentResult], settings) -> bool:
        """Основной разворот — есть зоны паспорта; прописку не гоняем."""
        if doc is None or doc.doc_type != "passport":
            return False
        if doc.doc_confidence < settings.min_doc_confidence:
            return False
        return any(z.name in cls._MAIN_PASSPORT_ZONES for z in doc.zones)

    def _try_registration(self, image: np.ndarray) -> Optional[RefinedField]:
        settings = get_settings()
        reg_weights = self._resolve_regpage_weights(settings)
        if reg_weights is not None:
            try:
                if self._reg_detector is None:
                    from postprocess.registration_detector import RegistrationDetectorReader

                    self._reg_detector = RegistrationDetectorReader(
                        str(reg_weights), confidence=0.35
                    )
                    logger.info("Registration detector: %s", reg_weights)
                field = self._reg_detector.read_as_field(image)
                if field:
                    return field
            except Exception as exc:
                logger.warning("Registration detector failed: %s", exc)

        try:
            from postprocess.registration import RegistrationOcr, extract_registration_address

            if self._reg_ocr is None:
                self._reg_ocr = RegistrationOcr(gpu=False)
            return extract_registration_address(image, self._reg_ocr)
        except Exception as exc:
            logger.warning("Registration OCR failed: %s", exc)
            return None

    @staticmethod
    def _resolve_regpage_weights(settings):
        from pathlib import Path

        if settings.regpage_weights_path:
            path = Path(settings.regpage_weights_path)
            if path.is_file():
                return path

        try:
            from docreader.hub import get_cache_dir

            default = get_cache_dir() / "regpage_obb.pt"
            if default.is_file():
                return default
        except Exception:
            pass
        return None

    def _registration_only_outcome(
        self,
        reg_field: RefinedField,
        doc: Optional[DocumentResult],
        timings: dict[str, float],
        t0: float,
        warnings: list[dict[str, Any]],
        quality: Optional[QualityMetrics] = None,
    ) -> RecognitionOutcome:
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return RecognitionOutcome(
            success=True,
            doc=doc,
            refined={"registration_address": reg_field},
            quality=quality,
            timings=timings,
            warnings=warnings,
        )

    def _extract_mrz(
        self, doc: DocumentResult, mrz_field: Optional[RefinedField]
    ) -> MrzResult:
        from postprocess.passport import _read_mrz  # noqa: PLC2701

        return _read_mrz(doc, self._mrz_ocr)


def create_engine() -> RecognitionEngine:
    return DocReaderEngine()
