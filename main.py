"""Распознавание российских документов: docreader-ocr + постобработка.

Для паспортов поверх сырого OCR применяется валидация полей,
арбитраж с MRZ (по контрольным цифрам) и справочник «Кем выдан»
по коду подразделения.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from postprocess import MrzOcr, build_reader, refine_passport

FIELD_LABELS = {
    "surname": "Фамилия",
    "name": "Имя",
    "patronymic": "Отчество",
    "birth_date": "Дата рождения",
    "sex": "Пол",
    "birth_place": "Место рождения",
    "series_number": "Серия и номер",
    "issue_date": "Дата выдачи",
    "issue_code": "Код подразделения",
    "issued_by": "Кем выдан",
    "mrz": "MRZ",
}


def process_image(image_path: Path, show_raw: bool = False) -> None:
    reader = build_reader()
    mrz_ocr = MrzOcr(gpu=False)

    result = reader.process(str(image_path), return_crops=True)
    if not result.documents:
        print("Документы не найдены.")
        return

    for index, doc in enumerate(result.documents, start=1):
        print(f"\nДокумент {index}: {doc.doc_type} (confidence={doc.doc_confidence:.2f})")

        if show_raw:
            print("--- Сырой OCR ---")
            print(json.dumps(doc.fields, ensure_ascii=False, indent=2))

        if doc.doc_type == "passport":
            refined = refine_passport(doc, mrz_ocr)
            print("--- После постобработки ---")
            for key, label in FIELD_LABELS.items():
                fld = refined.get(key)
                if fld is None:
                    continue
                mark = "✓" if fld.confirmed else " "
                note = f"  // {fld.note}" if fld.note else ""
                print(f"{mark} {label:18} {fld.value or '—':40} [{fld.source}]{note}")
        else:
            print(json.dumps(doc.fields, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Распознавание российских документов через docreader-ocr")
    parser.add_argument("image", type=Path, help="Путь к фото документа (jpg/png)")
    parser.add_argument("--raw", action="store_true", help="Показать сырой OCR до постобработки")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробные логи")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.image.is_file():
        print(f"Файл не найден: {args.image}", file=sys.stderr)
        return 1

    process_image(args.image, show_raw=args.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
