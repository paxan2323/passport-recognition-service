"""Живой A/B пайплайна на бенчмарк-паспортах (полный цикл, не кропы).

Переобрабатывает исходные фото 300 бенчмарк-паспортов (bench_keys.txt)
полным продакшн-пайплайном (детекция зон -> кроп -> OCR -> refine с MRZ)
и сравнивает финальные значения полей с эталонными метками бенчмарка.
Используется для A/B параметров пайплайна (например --crop-shrink).

Запуск (GPU-venv):
    python training\\eval_pipeline_bench.py --s3-config <path> [--crop-shrink 0.08]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("EASYOCR_MODULE_PATH", str(Path.home() / ".EasyOCR"))

import cv2

BENCH = Path(__file__).parent / "data" / "real_bench"
FIELDS = (
    "surname", "name", "patronymic", "birth_date",
    "issue_date", "issue_code", "series_number", "sex",
)
_PAIR_RE = re.compile(
    r"^images/(?P<src>.+)_(?P<doc>\d+)_(?P<field>" + "|".join(FIELDS) + r")_\d+\.jpg$"
)


def load_truth() -> dict[str, dict[str, str]]:
    truth: dict[str, dict[str, str]] = defaultdict(dict)
    for line in (BENCH / "labels.tsv").read_text(encoding="utf-8").splitlines():
        path, _, label = line.partition("\t")
        m = _PAIR_RE.match(path)
        if m and label:
            truth[m.group("src")][m.group("field")] = label
    return truth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-config", type=Path, required=True)
    parser.add_argument("--crop-shrink", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from postprocess import MrzOcr, build_reader, refine_passport
    from training.collect_from_s3 import make_s3_client, process_main_with_rotation

    truth = load_truth()
    keys = (BENCH / "bench_keys.txt").read_text(encoding="utf-8").splitlines()
    if args.limit:
        keys = keys[: args.limit]

    client, bucket = make_s3_client(args.s3_config)
    reader = build_reader(crop_shrink=args.crop_shrink)
    mrz_ocr = MrzOcr(gpu=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="ab_"))

    per_field = defaultdict(lambda: {"n": 0, "ok": 0})
    processed = 0
    for key in keys:
        stem = Path(key).stem
        expected = truth.get(stem)
        if not expected:
            continue
        local = tmp_dir / Path(key).name
        try:
            client.download_file(bucket, key, str(local))
            image = cv2.imread(str(local))
            if image is None:
                continue
            result = process_main_with_rotation(reader, image)
            values: dict[str, str] = {}
            for doc in result.documents:
                if doc.doc_type != "passport":
                    continue
                refined = refine_passport(doc, mrz_ocr)
                for field in FIELDS:
                    fld = refined.get(field)
                    if fld is not None and fld.value and field not in values:
                        values[field] = fld.value
            for field, label in expected.items():
                per_field[field]["n"] += 1
                per_field[field]["ok"] += int(values.get(field) == label)
            processed += 1
            if processed % 50 == 0:
                print(f"  {processed} паспортов...")
        except Exception as exc:
            print(f"  ! {key}: {exc}", file=sys.stderr)
        finally:
            local.unlink(missing_ok=True)

    print(f"\ncrop_shrink={args.crop_shrink}, паспортов: {processed}")
    print(f"{'Поле':15} {'N':>5} {'accuracy':>9}")
    total_n = total_ok = 0
    for field in FIELDS:
        s = per_field[field]
        if not s["n"]:
            continue
        total_n += s["n"]
        total_ok += s["ok"]
        print(f"{field:15} {s['n']:>5} {s['ok']/s['n']:>8.1%}")
    print("-" * 32)
    print(f"{'ИТОГО':15} {total_n:>5} {total_ok/total_n:>8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
