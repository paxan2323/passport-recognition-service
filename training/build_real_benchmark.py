"""Реальный held-out бенчмарк из MRZ-подтверждённых паспортов.

Выбирает N исходных паспортов, уже прошедших авторазметку (богатых
подтверждёнными полями), заново скачивает их из S3 и переразмечает
свежим пайплайном (строгий порог сверки ФИО, прямое распознавание,
чексумм-коррекция) в training/data/real_bench. Затем ИСКЛЮЧАЕТ эти
паспорта из обучающего набора (real_s3_main), чтобы бенчмарк был
честным held-out: пары удаляются из labels.tsv и images/, а ключи
остаются в manifest коллектора — повторно в обучение не попадут.

Запуск (GPU-venv):
    python training\\build_real_benchmark.py --s3-config <path> [--count 300]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("EASYOCR_MODULE_PATH", str(Path.home() / ".EasyOCR"))

import cv2

FIELDS = (
    "surname", "name", "patronymic", "birth_date",
    "issue_date", "issue_code", "series_number", "sex",
)
_PAIR_RE = re.compile(
    r"^images/(?P<src>.+)_(?P<doc>\d+)_(?P<field>"
    + "|".join(FIELDS) + r")_(?P<zone>\d+)\.jpg$"
)

TRAIN_DIR = Path("training/data/real_s3_main")
BENCH_DIR = Path("training/data/real_bench")


def select_stems(count: int, min_fields: int) -> list[str]:
    fields_by_src: dict[str, set[str]] = defaultdict(set)
    for line in (TRAIN_DIR / "labels.tsv").read_text(encoding="utf-8").splitlines():
        path = line.split("\t", 1)[0]
        m = _PAIR_RE.match(path)
        if m:
            fields_by_src[m.group("src")].add(m.group("field"))
    rich = sorted(src for src, flds in fields_by_src.items() if len(flds) >= min_fields)
    import random

    random.Random(42).shuffle(rich)
    return rich[:count]


def map_stems_to_keys(stems: list[str]) -> dict[str, str]:
    stem_to_key: dict[str, str] = {}
    for key in (TRAIN_DIR / "manifest.txt").read_text(encoding="utf-8").splitlines():
        stem_to_key.setdefault(Path(key).stem, key)
    return {s: stem_to_key[s] for s in stems if s in stem_to_key}


def rebuild_bench(mapping: dict[str, str], s3_config: Path) -> dict:
    from postprocess import MrzOcr, build_reader, refine_passport
    from postprocess.passport import _zones
    from training.collect_from_s3 import make_s3_client, process_main_with_rotation

    images_dir = BENCH_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = BENCH_DIR / "labels.tsv"

    client, bucket = make_s3_client(s3_config)
    reader = build_reader()
    mrz_ocr = MrzOcr(gpu=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="bench_"))

    stats = {"processed": 0, "pairs": 0, "errors": 0}
    per_field = {f: 0 for f in FIELDS}
    with open(labels_path, "w", encoding="utf-8", newline="") as fh:
        for stem, key in mapping.items():
            local = tmp_dir / Path(key).name
            try:
                client.download_file(bucket, key, str(local))
                image = cv2.imread(str(local))
                if image is None:
                    raise ValueError("cv2.imread -> None")
                result = process_main_with_rotation(reader, image)
                for doc_idx, doc in enumerate(result.documents):
                    if doc.doc_type != "passport":
                        continue
                    refined = refine_passport(doc, mrz_ocr)
                    for field in FIELDS:
                        fld = refined.get(field)
                        if fld is None or not fld.value or not fld.confirmed:
                            continue
                        for zone_idx, zone in enumerate(_zones(doc, field)):
                            if zone.crop_image is None or zone.crop_image.size == 0:
                                continue
                            name = f"{stem}_{doc_idx}_{field}_{zone_idx}.jpg"
                            cv2.imwrite(str(images_dir / name), zone.crop_image)
                            fh.write(f"images/{name}\t{fld.value}\n")
                            stats["pairs"] += 1
                            per_field[field] += 1
                stats["processed"] += 1
            except Exception as exc:
                stats["errors"] += 1
                print(f"  ! {key}: {exc}", file=sys.stderr)
            finally:
                local.unlink(missing_ok=True)
            if stats["processed"] % 50 == 0:
                print(f"  {stats['processed']}/{len(mapping)}: pairs={stats['pairs']}")
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    stats["per_field"] = per_field
    return stats


def exclude_from_training(stems: list[str]) -> int:
    stems_set = set(stems)
    lines = (TRAIN_DIR / "labels.tsv").read_text(encoding="utf-8").splitlines()
    kept, removed = [], 0
    for line in lines:
        m = _PAIR_RE.match(line.split("\t", 1)[0])
        if m and m.group("src") in stems_set:
            img = TRAIN_DIR / line.split("\t", 1)[0]
            img.unlink(missing_ok=True)
            removed += 1
        else:
            kept.append(line)
    (TRAIN_DIR / "labels.tsv").write_text("\n".join(kept) + "\n", encoding="utf-8")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Построение реального бенчмарка")
    parser.add_argument("--s3-config", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--min-fields", type=int, default=4)
    args = parser.parse_args()

    stems = select_stems(args.count, args.min_fields)
    mapping = map_stems_to_keys(stems)
    print(f"Отобрано паспортов: {len(stems)}, найдено ключей: {len(mapping)}")

    stats = rebuild_bench(mapping, args.s3_config)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    removed = exclude_from_training(list(mapping.keys()))
    (BENCH_DIR / "bench_keys.txt").write_text(
        "\n".join(mapping.values()) + "\n", encoding="utf-8"
    )
    print(f"Исключено пар из обучающего набора: {removed}")
    print(f"Бенчмарк: {BENCH_DIR / 'labels.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
