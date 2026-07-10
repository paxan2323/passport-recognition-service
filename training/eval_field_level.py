"""Полевой eval на реальном бенчмарке: сырой OCR vs после нормализации.

Для каждого кропа бенчмарка известен тип поля (из имени файла) — прогоняем
продакшн-стек OCR и применяем полевую нормализацию с экстракцией и без.
Показывает, сколько полей «спасает» экстракция значений из шумных
предсказаний (затекание соседних зон в кроп).

Запуск: python training\\eval_field_level.py [--limit N]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
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
    r"^images/.+_\d+_(?P<field>" + "|".join(FIELDS) + r")_\d+\.jpg$"
)


def normalize_for_field(field: str, text: str, extract: bool):
    from postprocess import normalize

    if field in ("birth_date", "issue_date"):
        return normalize.normalize_date(text, extract=extract)
    if field == "series_number":
        return normalize.normalize_series_number(text, extract=extract)
    if field == "issue_code":
        return normalize.normalize_issue_code(text, extract=extract)
    if field == "sex":
        return normalize.normalize_sex(text)
    return normalize.normalize_name(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from postprocess.retry_ocr import RetryOcrEngine
    from postprocess.sorted_ocr import SortedOcrEngine

    engine = RetryOcrEngine(SortedOcrEngine.from_config(gpu=True))

    rows = []
    for line in (BENCH / "labels.tsv").read_text(encoding="utf-8").splitlines():
        path, _, label = line.partition("\t")
        m = _PAIR_RE.match(path)
        if m and label:
            rows.append((BENCH / path, m.group("field"), label))
    if args.limit:
        rows = rows[: args.limit]

    stats = defaultdict(lambda: {"n": 0, "raw": 0, "norm_off": 0, "norm_on": 0, "rescued": 0})
    for image_path, field, label in rows:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        text = engine.recognize(image).text
        s = stats[field]
        s["n"] += 1
        s["raw"] += int(text == label)
        off = normalize_for_field(field, text, extract=False)
        on = normalize_for_field(field, text, extract=True)
        s["norm_off"] += int(off == label)
        s["norm_on"] += int(on == label)
        if off != label and on == label:
            s["rescued"] += 1

    total = {"n": 0, "raw": 0, "norm_off": 0, "norm_on": 0, "rescued": 0}
    print(f"{'Поле':15} {'N':>5} {'raw':>7} {'norm':>7} {'norm+extract':>13} {'спасено':>8}")
    for field in FIELDS:
        s = stats[field]
        if not s["n"]:
            continue
        for k in total:
            total[k] += s[k]
        print(
            f"{field:15} {s['n']:>5} {s['raw']/s['n']:>6.1%} {s['norm_off']/s['n']:>6.1%} "
            f"{s['norm_on']/s['n']:>12.1%} {s['rescued']:>8}"
        )
    n = total["n"]
    print("-" * 60)
    print(
        f"{'ИТОГО':15} {n:>5} {total['raw']/n:>6.1%} {total['norm_off']/n:>6.1%} "
        f"{total['norm_on']/n:>12.1%} {total['rescued']:>8}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
