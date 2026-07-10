"""Проверка end-to-end чтения прописки: детектор + распознавалка.

Запускать в GPU-venv:
    training\\.venv-gpu\\Scripts\\python.exe training\\verify_registration.py <img> <weights>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def main() -> int:
    image_path = Path(sys.argv[1])
    weights = sys.argv[2]

    from postprocess.registration_detector import RegistrationDetectorReader

    image = cv2.imread(str(image_path))
    reader = RegistrationDetectorReader(weights, confidence=0.35)

    # 1) сырые детекции + оверлей
    det = reader._model.predict(image, conf=0.35, verbose=False)[0]
    overlay = image.copy()
    colors = {0: (0, 180, 0), 1: (0, 140, 255), 2: (255, 80, 0), 3: (0, 0, 220)}
    n = 0
    if det.obb is not None:
        polys = det.obb.xyxyxyxy.cpu().numpy()
        classes = det.obb.cls.cpu().numpy().astype(int)
        for poly, cls in zip(polys, classes):
            cv2.polylines(overlay, [poly.astype(int)], True, colors.get(int(cls), (255, 255, 255)), 2)
            n += 1
    out = image_path.parent / f"_verify_{image_path.stem}.png"
    cv2.imwrite(str(out), overlay)

    # 2) собранный результат
    result = reader.read(image)
    print(f"Детекций: {n}")
    print(f"Оверлей: {out}")
    print("Результат чтения прописки:")
    for key, value in result.as_dict().items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
