"""OCR-движок с ретраями низкоконфидентных зон.

Оборачивает штатный TextRecognizer: если базовое распознавание дало
пустой текст или confidence ниже порога, кроп прогоняется через серию
препроцессингов (апскейл, CLAHE, подавление бликов, бинаризация)
и берётся результат с максимальной уверенностью.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import cv2
import numpy as np

from docreader.ocr.base import BaseOcrEngine, OcrResult

logger = logging.getLogger(__name__)


class RetryOcrEngine(BaseOcrEngine):
    def __init__(
        self,
        base: BaseOcrEngine,
        confidence_threshold: float = 0.6,
        max_upscaled_side: int = 1600,
    ):
        self._base = base
        self._threshold = confidence_threshold
        self._max_side = max_upscaled_side

    def recognize(self, image: np.ndarray) -> OcrResult:
        best = self._base.recognize(image)
        if best.text and best.confidence >= self._threshold:
            return best

        variants: list[tuple[str, Callable[[np.ndarray], Optional[np.ndarray]]]] = [
            ("upscale2x", lambda img: self._upscale(img, 2.0)),
            ("deglare", self._deglare),
            ("upscale3x+clahe", lambda img: self._clahe(self._upscale(img, 3.0))),
            ("binarize", self._binarize),
        ]
        for name, make in variants:
            variant = make(image)
            if variant is None:
                continue
            candidate = self._base.recognize(variant)
            if candidate.text and candidate.confidence > best.confidence:
                logger.debug(
                    "retry '%s': %.2f -> %.2f (%r)",
                    name, best.confidence, candidate.confidence, candidate.text,
                )
                best = candidate
            if best.confidence >= self._threshold:
                break
        return best

    def _upscale(self, image: np.ndarray, factor: float) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        if max(h, w) * factor > self._max_side:
            factor = self._max_side / max(h, w)
        if factor <= 1.05:
            return None
        return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _clahe(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if image is None:
            return None
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        l_channel = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l_channel)
        return cv2.cvtColor(cv2.merge((l_channel, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _deglare(image: np.ndarray) -> Optional[np.ndarray]:
        """Закрашивает пересвеченные участки (блики ламинации)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 235).astype(np.uint8) * 255
        ratio = mask.mean() / 255.0
        if ratio < 0.01 or ratio > 0.5:
            return None
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    def _binarize(self, image: np.ndarray) -> Optional[np.ndarray]:
        upscaled = self._upscale(image, 3.0)
        gray = cv2.cvtColor(upscaled if upscaled is not None else image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
