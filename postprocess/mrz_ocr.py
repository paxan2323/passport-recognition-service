"""Отдельное распознавание MRZ латинской моделью EasyOCR.

Кастомная кириллическая сеть docreader читает MRZ плохо (это латиница
и '<'), поэтому кропы MRZ-зон перечитываются стандартной английской
моделью EasyOCR с жёстким allowlist алфавита MRZ.

При первом запуске EasyOCR докачает en-модель (~100 МБ) в ~/.EasyOCR.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MRZ_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
_TARGET_HEIGHT = 64


class MrzOcr:
    def __init__(self, gpu: bool = False):
        import os

        import easyocr

        model_dir = os.environ.get("EASYOCR_MODULE_PATH", "/tmp/easyocr")
        self._reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            verbose=False,
            model_storage_directory=os.path.join(model_dir, "model"),
        )
        logger.info("MrzOcr initialized (gpu=%s)", gpu)

    def read_line(self, crop: np.ndarray) -> str:
        """Читает одну строку MRZ, возвращает склеенный текст без пробелов."""
        image = self._prepare(crop)
        results = self._reader.readtext(image, allowlist=MRZ_ALLOWLIST, paragraph=False)
        if not results:
            return ""
        # Фрагменты склеиваем слева направо: в MRZ пробелов не бывает
        results.sort(key=lambda item: min(point[0] for point in item[0]))
        return "".join(text for _, text, _ in results)

    def read_document(self, doc_crop: np.ndarray) -> tuple[str, str]:
        """Читает MRZ из нижней части кропа документа.

        Детектор passport.pt не выделяет MRZ отдельной зоной, но она
        всегда занимает нижнюю полосу разворота. Фрагменты группируются
        в строки по вертикали; возвращаются две нижние MRZ-подобные
        строки (line1, line2).
        """
        h = doc_crop.shape[0]
        strip = doc_crop[int(h * 0.72):, :]
        if strip.size == 0:
            return "", ""

        results = self._reader.readtext(strip, allowlist=MRZ_ALLOWLIST, paragraph=False)
        lines: list[tuple[float, list[tuple[float, str]]]] = []  # (y, [(x, text)])
        for box, text, _conf in results:
            if not text.strip():
                continue
            y_center = sum(point[1] for point in box) / len(box)
            x_left = min(point[0] for point in box)
            box_h = max(point[1] for point in box) - min(point[1] for point in box)
            for line in lines:
                if abs(line[0] - y_center) < max(box_h * 0.7, 10):
                    line[1].append((x_left, text))
                    break
            else:
                lines.append((y_center, [(x_left, text)]))

        merged = []
        for y_center, fragments in sorted(lines, key=lambda item: item[0]):
            fragments.sort(key=lambda item: item[0])
            text = "".join(fragment for _, fragment in fragments)
            if len(text) >= 15:  # отсекаем случайные обрывки
                merged.append(text)
        if not merged:
            return "", ""
        if len(merged) == 1:
            # единственная строка: решаем, первая это или вторая
            if merged[0][:2].upper().startswith("P"):
                return merged[0], ""
            return "", merged[0]
        return merged[-2], merged[-1]

    @staticmethod
    def _prepare(crop: np.ndarray) -> np.ndarray:
        h = crop.shape[0]
        if 0 < h < _TARGET_HEIGHT:
            scale = _TARGET_HEIGHT / h
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop
