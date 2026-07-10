"""Сбор обучающего датасета из реальных фото с MRZ-авторазметкой.

Прогоняет папку с фотографиями через пайплайн. Поля, подтверждённые
арбитражем (чексуммы MRZ / совпадение источников), сохраняются как
готовые обучающие пары «кроп зоны -> текст» в images/ + labels.tsv.
Неподтверждённые кропы и текстовые поля без эталона (место рождения,
кем выдан) уходят в hard/ на ручную проверку: hard/review.tsv содержит
подсказку из OCR/справочника, которую надо руками сверить с кропом.

Запуск (инкрементальный — уже обработанные фото пропускаются):
    python training\\collect_dataset.py D:\\photos --out training\\data\\real
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# до импорта mrz_ocr: локально en-модель уже лежит в ~/.EasyOCR
os.environ.setdefault("EASYOCR_MODULE_PATH", str(Path.home() / ".EasyOCR"))

import cv2

# Поля с объективным эталоном (чексуммы MRZ, совпадение источников)
AUTO_FIELDS = (
    "surname", "name", "patronymic", "birth_date",
    "issue_date", "issue_code", "series_number", "sex",
)
# Поля без эталона — всегда на ручную проверку
REVIEW_FIELDS = ("birth_place", "issued_by")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(input_dir: Path, out_dir: Path, recursive: bool) -> None:
    from postprocess import MrzOcr, build_reader, refine_passport
    from postprocess.passport import _zones  # внутренний, но проект наш

    images_dir = out_dir / "images"
    hard_dir = out_dir / "hard" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.tsv"
    review_path = out_dir / "hard" / "review.tsv"
    manifest_path = out_dir / "processed.txt"

    processed: set[str] = set()
    if manifest_path.is_file():
        processed = set(manifest_path.read_text(encoding="utf-8").splitlines())

    pattern = "**/*" if recursive else "*"
    photos = sorted(
        p for p in input_dir.glob(pattern)
        if p.suffix.lower() in IMAGE_SUFFIXES and p.name not in processed
    )
    if not photos:
        print("Новых фото не найдено.")
        return

    print(f"К обработке: {len(photos)} фото")
    reader = build_reader()
    mrz_ocr = MrzOcr(gpu=False)

    stats = {"auto": 0, "hard": 0, "docs": 0}
    with open(labels_path, "a", encoding="utf-8", newline="") as labels, \
         open(review_path, "a", encoding="utf-8", newline="") as review, \
         open(manifest_path, "a", encoding="utf-8", newline="") as manifest:
        for photo in photos:
            try:
                result = reader.process(str(photo), return_crops=True)
            except Exception as exc:  # битые файлы не должны ронять сбор
                print(f"  ! {photo.name}: {exc}")
                continue

            for doc_idx, doc in enumerate(result.documents):
                if doc.doc_type != "passport":
                    continue
                stats["docs"] += 1
                refined = refine_passport(doc, mrz_ocr)
                stem = f"{photo.stem}_{doc_idx}"

                for field in AUTO_FIELDS + REVIEW_FIELDS:
                    fld = refined.get(field)
                    if fld is None or not fld.value:
                        continue
                    zones = _zones(doc, field)
                    is_auto = fld.confirmed and field in AUTO_FIELDS
                    for zone_idx, zone in enumerate(zones):
                        if zone.crop_image is None or zone.crop_image.size == 0:
                            continue
                        name = f"{stem}_{field}_{zone_idx}.jpg"
                        if is_auto:
                            cv2.imwrite(str(images_dir / name), zone.crop_image)
                            labels.write(f"images/{name}\t{fld.value}\n")
                            stats["auto"] += 1
                        else:
                            cv2.imwrite(str(hard_dir / name), zone.crop_image)
                            review.write(
                                f"images/{name}\t{fld.value}\t{fld.source}\t{fld.note}\n"
                            )
                            stats["hard"] += 1

            manifest.write(photo.name + "\n")
            print(f"  + {photo.name}")

    print(
        f"\nПаспортов: {stats['docs']}; авторазметка: {stats['auto']} пар; "
        f"на ручную проверку (hard): {stats['hard']}"
    )
    print(f"Датасет: {labels_path}\nОчередь проверки: {review_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор датасета с MRZ-авторазметкой")
    parser.add_argument("input", type=Path, help="Папка с фотографиями паспортов")
    parser.add_argument("--out", type=Path, default=Path("training/data/real"))
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Нет папки {args.input}", file=sys.stderr)
        return 1
    collect(args.input, args.out, args.recursive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
