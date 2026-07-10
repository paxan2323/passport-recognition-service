"""Оценка распознавалки на размеченном датасете (labels.tsv).

Метрики: точность полного совпадения и CER. Используется для
baseline текущей сети и сравнения с кандидатами после дообучения.

Запуск:
    python training\\eval_recognizer.py training\\data\\synth_preview --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from rapidfuzz.distance import Levenshtein


def evaluate(dataset_dir: Path, limit: int, show_worst: int, unsorted: bool = False) -> int:
    labels_path = dataset_dir / "labels.tsv"
    if not labels_path.is_file():
        print(f"Нет файла {labels_path}", file=sys.stderr)
        return 1

    rows = []
    with open(labels_path, encoding="utf-8") as fh:
        for line in fh:
            path, _, label = line.rstrip("\n").partition("\t")
            if path and label:
                rows.append((dataset_dir / path, label))
    if limit:
        rows = rows[:limit]

    from docreader.config import PipelineConfig

    config = PipelineConfig()
    if unsorted:
        from docreader.hub import ensure_model
        from docreader.ocr.easyocr_engine import TextRecognizer

        easyocr_dir = ensure_model(config.ocr_model_archive)
        ocr = TextRecognizer(
            lang=config.ocr_lang,
            gpu=False,
            model_storage_directory=str(easyocr_dir / config.ocr_model_subdir),
            user_network_directory=str(easyocr_dir / config.ocr_network_subdir),
            recog_network=config.ocr_recog_network,
            download_enabled=config.ocr_download_enabled,
        )
    else:
        from postprocess.sorted_ocr import SortedOcrEngine

        ocr = SortedOcrEngine.from_config(config, gpu=False)

    exact = 0
    total_edit = 0
    total_chars = 0
    results = []
    for image_path, label in rows:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        predicted = ocr.recognize(image).text
        edit = Levenshtein.distance(predicted, label)
        exact += int(predicted == label)
        total_edit += edit
        total_chars += len(label)
        results.append((edit / max(len(label), 1), label, predicted))

    n = len(results)
    cer = total_edit / max(total_chars, 1)
    print(f"Примеров: {n}")
    print(f"Точность (полное совпадение): {exact / n:.1%}")
    print(f"CER: {cer:.1%}")

    if show_worst:
        print(f"\nХудшие {show_worst}:")
        results.sort(key=lambda r: r[0], reverse=True)
        for rate, label, predicted in results[:show_worst]:
            print(f"  {rate:.0%}  {label!r} -> {predicted!r}")
    return {"n": n, "accuracy": exact / n, "cer": cer}


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка OCR на размеченном датасете")
    parser.add_argument("dataset", type=Path, help="Каталог с labels.tsv")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число примеров")
    parser.add_argument("--worst", type=int, default=10, help="Показать N худших")
    parser.add_argument("--unsorted", action="store_true", help="Без сортировки фрагментов (штатный TextRecognizer)")
    args = parser.parse_args()
    result = evaluate(args.dataset, args.limit, args.worst, unsorted=args.unsorted)
    return 0 if isinstance(result, dict) else result


if __name__ == "__main__":
    raise SystemExit(main())
