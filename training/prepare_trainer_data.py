"""Конвертация датасетов (labels.tsv) в формат трейнера EasyOCR.

Собирает один или несколько датасетов (синтетика + реальные кропы)
в training/EasyOCR/trainer/all_data/{ru_train,ru_val}: изображения
плюс labels.csv (filename,words). Метки фильтруются по алфавиту сети.

Запуск:
    python training\\prepare_trainer_data.py training\\data\\synth_v1 training\\data\\real
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.synth_generator import load_charset

TRAINER_DATA = Path(__file__).parent / "EasyOCR" / "trainer" / "all_data"
MAX_LABEL_LENGTH = 75


def prepare(sources: list[Path], val_fraction: float, seed: int, limit: int) -> None:
    charset = load_charset()
    rows: list[tuple[Path, str]] = []
    skipped = 0
    for source in sources:
        labels_path = source / "labels.tsv"
        if not labels_path.is_file():
            print(f"! пропускаю {source}: нет labels.tsv", file=sys.stderr)
            continue
        with open(labels_path, encoding="utf-8") as fh:
            for line in fh:
                rel_path, _, label = line.rstrip("\n").partition("\t")
                if not rel_path or not label:
                    continue
                if len(label) > MAX_LABEL_LENGTH or not set(label) <= charset:
                    skipped += 1
                    continue
                image_path = source / rel_path
                if image_path.is_file():
                    rows.append((image_path, label))

    if not rows:
        print("Нет данных.", file=sys.stderr)
        return

    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    val_count = max(1, int(len(rows) * val_fraction))
    splits = {"ru_val": rows[:val_count], "ru_train": rows[val_count:]}

    for split_name, split_rows in splits.items():
        split_dir = TRAINER_DATA / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        with open(split_dir / "labels.csv", "w", encoding="utf-8", newline="") as csv:
            csv.write("filename,words\n")
            for index, (image_path, label) in enumerate(split_rows):
                name = f"{index:06d}{image_path.suffix.lower()}"
                shutil.copyfile(image_path, split_dir / name)
                csv.write(f"{name},{label}\n")
        print(f"{split_name}: {len(split_rows)} примеров")
    print(f"Отфильтровано (алфавит/длина): {skipped}")
    print(f"Данные: {TRAINER_DATA}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Подготовка данных для трейнера EasyOCR")
    parser.add_argument("sources", nargs="+", type=Path, help="Каталоги с labels.tsv")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Ограничить общий объём (для smoke-прогонов)")
    args = parser.parse_args()
    prepare(args.sources, args.val_fraction, args.seed, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
