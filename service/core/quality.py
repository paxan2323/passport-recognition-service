"""Image quality checks (blur, glare, resolution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from service.core.config import get_settings
from service.core.errors import LOW_QUALITY, ServiceError


@dataclass
class QualityMetrics:
    blur_score: float
    glare_ratio: float
    width: int
    height: int
    low_resolution: bool = False


class QualityError(Exception):
    def __init__(self, error: ServiceError, metrics: QualityMetrics):
        self.error = error
        self.metrics = metrics


def assess_quality(image_bytes: bytes, min_quality: str = "normal") -> QualityMetrics:
    """Assess blur/glare on decoded image. Raises QualityError if below threshold."""
    settings = get_settings()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise QualityError(
            LOW_QUALITY,
            QualityMetrics(blur_score=0, glare_ratio=1.0, width=0, height=0),
        )

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = float((gray > 235).mean())

    low_res = w < 800
    metrics = QualityMetrics(
        blur_score=blur_score,
        glare_ratio=glare_ratio,
        width=w,
        height=h,
        low_resolution=low_res,
    )

    if min_quality == "low":
        return metrics

    if blur_score < settings.min_quality_blur or glare_ratio > settings.min_quality_glare_ratio:
        raise QualityError(LOW_QUALITY, metrics)

    return metrics


def assess_doc_crop(crop: Optional[np.ndarray]) -> QualityMetrics:
    """Quality on document crop after classification."""
    if crop is None or crop.size == 0:
        return QualityMetrics(blur_score=0, glare_ratio=1.0, width=0, height=0, low_resolution=True)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape[:2]
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = float((gray > 235).mean())
    return QualityMetrics(
        blur_score=blur_score,
        glare_ratio=glare_ratio,
        width=w,
        height=h,
        low_resolution=w < 800,
    )
