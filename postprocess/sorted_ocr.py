"""OCR-движок с правильным пространственным порядком фрагментов.

Штатный TextRecognizer docreader склеивает фрагменты readtext в том
порядке, в каком их вернул EasyOCR — а внутренний детектор строк
EasyOCR нередко отдаёт куски одной строки не слева-направо. Из-за
этого рушатся числовые поля: "17.04.1985" -> "17 1985 04." (после
удаления нецифр даёт неверный порядок цифр и невалидную дату).

Здесь фрагменты группируются в строки по вертикали и сортируются
слева-направо внутри строки, строки — сверху-вниз. Модель распознавания
та же (кастомная сеть docreader), меняется только порядок склейки.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from docreader.ocr.base import BaseOcrEngine, OcrResult

logger = logging.getLogger(__name__)


def _order_fragments(results: list) -> tuple[str, float]:
    """Склеивает фрагменты readtext в порядке чтения (строки → слева-направо)."""
    items = []
    for box, text, conf in results:
        if not text:
            continue
        ys = [point[1] for point in box]
        xs = [point[0] for point in box]
        items.append({
            "text": text,
            "conf": conf,
            "y": sum(ys) / len(ys),
            "x": min(xs),
            "h": max(ys) - min(ys),
        })
    if not items:
        return "", 0.0

    # группировка в строки: фрагмент попадает в строку, если его центр по Y
    # близок к центру строки (порог — доля высоты фрагмента)
    items.sort(key=lambda it: it["y"])
    lines: list[list[dict]] = []
    for it in items:
        placed = False
        for line in lines:
            line_y = sum(f["y"] for f in line) / len(line)
            if abs(line_y - it["y"]) < max(it["h"] * 0.6, 10):
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])

    lines.sort(key=lambda line: sum(f["y"] for f in line) / len(line))
    parts: list[str] = []
    confs: list[float] = []
    for line in lines:
        line.sort(key=lambda f: f["x"])
        parts.append(" ".join(f["text"] for f in line))
        confs.extend(f["conf"] for f in line)

    text = " ".join(parts).strip()
    confidence = sum(confs) / len(confs) if confs else 0.0
    return text, confidence


class SortedOcrEngine(BaseOcrEngine):
    """EasyOCR-движок docreader со склейкой в правильном порядке чтения.

    Сначала пробует прямое распознавание всего кропа одной строкой,
    минуя внутренний детектор строк EasyOCR: кропы зон — почти всегда
    одна строка, а детектор на них и ошибался (пустые ответы на
    коротких словах, дробление дат на переставленные куски).
    Многострочные зоны (кем выдан, место рождения) дают низкий
    confidence на прямом пути и уходят в fallback readtext+сортировка.
    """

    def __init__(self, reader, direct_threshold: float = 0.55):
        self._reader = reader
        self._direct_threshold = direct_threshold

    def _recognize_direct(self, image: np.ndarray) -> OcrResult:
        """Весь кроп как одна строка, без детектора строк."""
        results = self._reader.recognize(image, horizontal_list=None, free_list=None)
        if not results:
            return OcrResult(text="", confidence=0.0)
        _, text, confidence = results[0]
        return OcrResult(text=text.strip(), confidence=float(confidence))

    def recognize(self, image: np.ndarray) -> OcrResult:
        direct = self._recognize_direct(image)
        if direct.text and direct.confidence >= self._direct_threshold:
            return direct

        results = self._reader.readtext(image)
        if not results:
            return direct if direct.text else OcrResult(text="", confidence=0.0)
        text, confidence = _order_fragments(results)
        if direct.text and direct.confidence > confidence:
            return direct
        return OcrResult(text=text, confidence=confidence)

    @classmethod
    def from_config(cls, config=None, gpu: bool = False) -> "SortedOcrEngine":
        """Строит движок с той же кастомной сетью, что и docreader."""
        import easyocr

        from docreader.config import PipelineConfig
        from docreader.hub import ensure_model

        cfg = config or PipelineConfig()
        easyocr_dir = ensure_model(cfg.ocr_model_archive)
        reader = easyocr.Reader(
            lang_list=cfg.ocr_lang,
            gpu=gpu,
            download_enabled=cfg.ocr_download_enabled,
            model_storage_directory=str(easyocr_dir / cfg.ocr_model_subdir),
            user_network_directory=str(easyocr_dir / cfg.ocr_network_subdir),
            recog_network=cfg.ocr_recog_network,
            verbose=False,
        )
        return cls(reader)
