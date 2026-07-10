"""Сборка DocReader с ретраями OCR."""

from __future__ import annotations

from typing import Optional

import numpy as np

from docreader import DocReader
from docreader.config import PipelineConfig

from postprocess.retry_ocr import RetryOcrEngine
from postprocess.sorted_ocr import SortedOcrEngine

_original_crop = None


def set_crop_shrink(shrink: float) -> None:
    """Сжимает OBB-рамки зон к центроиду перед кропом (0.0 — выключено).

    На реальных фото в кропы затекают фрагменты соседних полей; сжатие
    рамки на несколько процентов срезает края. Патчится точка вызова в
    docreader.pipeline (crop_obb_region импортирован туда по имени).
    """
    import docreader.pipeline as dr_pipeline
    from docreader.preprocessing import crop_obb_region as original

    global _original_crop
    if _original_crop is None:
        _original_crop = original

    if shrink <= 0:
        dr_pipeline.crop_obb_region = _original_crop
        return

    base = _original_crop

    def shrinking_crop(image, obb_points, zone_name=""):
        pts = np.array(obb_points, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0)
        pts = center + (pts - center) * (1.0 - shrink)
        return base(image, pts.reshape(-1), zone_name=zone_name)

    dr_pipeline.crop_obb_region = shrinking_crop


def build_reader(
    config: Optional[PipelineConfig] = None,
    retry_confidence_threshold: float = 0.6,
    crop_shrink: float = 0.0,
) -> DocReader:
    """DocReader с распознавателем, читающим фрагменты в правильном порядке
    (SortedOcrEngine), обёрнутым в RetryOcrEngine для низкоконфидентных зон."""
    config = config or PipelineConfig()
    device = config.resolve_device()

    set_crop_shrink(crop_shrink)
    base_ocr = SortedOcrEngine.from_config(config, gpu=(device != "cpu"))
    retry_ocr = RetryOcrEngine(base_ocr, confidence_threshold=retry_confidence_threshold)
    return DocReader(config=config, ocr_engine=retry_ocr)
