"""Безопасная подмена детектора прописки с проверкой качества.

Валидирует новый best.pt на regpage_val (held-out), подменяет
только если mAP50 не хуже порога (иначе прошлое обучение могло
сломаться из-за плохих данных/переобучения и мы не хотим тихо
выкатить худший детектор в ночном прогоне без присмотра).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODELS_DIR = Path.home() / ".cache" / "docreader" / "models"
DEPLOYED = MODELS_DIR / "regpage_obb.pt"
BACKUP = MODELS_DIR / "regpage_obb.backup_pre_rotation.pt"
CANDIDATE = Path(__file__).parent / "runs" / "regpage" / "weights" / "best.pt"
VAL_DATA = Path(__file__).parent / "data" / "regpage_split.yaml"
MIN_MAP50 = 0.85


def main() -> int:
    if not CANDIDATE.is_file():
        print(f"Нет кандидата {CANDIDATE} — подмена отменена", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    print("=== Валидация нового детектора на held-out (regpage_val) ===")
    model = YOLO(str(CANDIDATE))
    metrics = model.val(data=str(VAL_DATA), verbose=False)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    print(f"mAP50: {map50:.3f}, mAP50-95: {map5095:.3f}")

    if map50 < MIN_MAP50:
        print(f"\n!!! mAP50 {map50:.3f} < порога {MIN_MAP50} — подмена ОТМЕНЕНА, старый детектор оставлен.")
        return 1

    if DEPLOYED.is_file():
        shutil.copyfile(DEPLOYED, BACKUP)
    shutil.copyfile(CANDIDATE, DEPLOYED)
    print(f"\nOK: детектор подменён (бэкап: {BACKUP.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
