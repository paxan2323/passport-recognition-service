"""Сквозной тест MRZ: рендер строк -> EasyOCR(en) -> парсер.

Публичные сэмплы паспортов идут с замазанной MRZ, поэтому полоса
генерируется синтетически. Запуск (докачает en-модель EasyOCR):
.venv\\Scripts\\python.exe tests\\test_mrz_e2e.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from postprocess.mrz import compute_check_digit, parse_mrz
from postprocess.mrz_ocr import MrzOcr


def build_lines() -> tuple[str, str]:
    line1 = "PNRUSPIVEN9<<ANDREQ<ANDREEVI3".ljust(44, "<")
    doc9, birth = "032057902", "000625"
    optional14 = "9" + "200708" + "230002" + "<"
    line2 = (
        doc9 + compute_check_digit(doc9)
        + "RUS" + birth + compute_check_digit(birth)
        + "M" + "<<<<<<" + "<"
        + optional14 + compute_check_digit(optional14)
    )
    composite_body = line2[0:10] + line2[13:20] + line2[21:43]
    return line1, line2 + compute_check_digit(composite_body)


def render_mrz_strip(line1: str, line2: str) -> np.ndarray:
    font = ImageFont.truetype(r"C:\Windows\Fonts\cour.ttf", 28)
    image = Image.new("RGB", (44 * 18 + 40, 110), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 15), line1, font=font, fill="black")
    draw.text((20, 60), line2, font=font, fill="black")
    return np.array(image)[:, :, ::-1].copy()  # RGB -> BGR


def main() -> int:
    line1, line2 = build_lines()
    strip = render_mrz_strip(line1, line2)

    ocr = MrzOcr(gpu=False)
    read1 = ocr.read_line(strip[:55])
    read2 = ocr.read_line(strip[55:])
    print(f"эталон 1: {line1}\nOCR    1: {read1}")
    print(f"эталон 2: {line2}\nOCR    2: {read2}")

    mrz = parse_mrz(read1, read2)
    print(mrz)

    failures = []
    for field_name, expected in [
        ("surname", "ПИВЕНЬ"),
        ("name", "АНДРЕЙ"),
        ("patronymic", "АНДРЕЕВИЧ"),
        ("series_number", "03 20 579029"),
        ("birth_date", "25.06.2000"),
        ("sex", "МУЖ."),
        ("issue_date", "08.07.2020"),
        ("issue_code", "230-002"),
    ]:
        actual = getattr(mrz, field_name)
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            failures.append(field_name)
        print(f"{status} {field_name}: {actual!r} (ожидалось {expected!r})")

    for flag in ("number_ok", "birth_date_ok", "optional_ok"):
        print(f"{'PASS' if getattr(mrz, flag) else 'FAIL'} {flag}")
        if not getattr(mrz, flag):
            failures.append(flag)

    print("\n" + ("OK" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
