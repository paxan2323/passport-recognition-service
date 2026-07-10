"""Обучение YOLO-OBB детектора зон страницы регистрации (прописки).

Данные готовит regpage_generator.py (синтетика с автоматической
OBB-разметкой). Дообучается от предобученного yolo11n-obb.

Запуск:
    python training\\train_detector.py --smoke              # проверка (CPU, пара эпох)
    python training\\train_detector.py --epochs 100         # полное обучение (GPU)

Готовый детектор подключается в пайплайн как новый тип документа:
    from docreader.config import PipelineConfig
    cfg = PipelineConfig()
    cfg.detector_weights["prilozhenie"] = "prilozhenie.pt"  # или свой ключ
и файл весов кладётся в ~/.cache/docreader/models/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DATA = HERE / "data" / "regpage" / "data.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Обучение OBB-детектора страницы регистрации")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n-obb.pt", help="Базовая OBB-модель")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="'' = авто (cpu/0)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"Нет {args.data}. Сначала: python training\\regpage_generator.py", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    epochs = 2 if args.smoke else args.epochs
    imgsz = 512 if args.smoke else args.imgsz
    name = "regpage_smoke" if args.smoke else "regpage"

    device = args.device
    if not device:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"Старт: model={args.model}, epochs={epochs}, imgsz={imgsz}, device={device}")
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=device,
        project=str(HERE / "runs"),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    best = HERE / "runs" / name / "weights" / "best.pt"
    print(f"\nГотово. Веса: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
