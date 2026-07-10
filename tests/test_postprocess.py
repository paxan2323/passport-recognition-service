"""Юнит-тесты постобработки (без загрузки моделей).

Запуск: .venv\\Scripts\\python.exe tests\\test_postprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocess import normalize
from postprocess.division_codes import lookup_issued_by
from postprocess.mrz import (
    compute_check_digit,
    parse_mrz,
    transliterate_from_mrz,
    transliterate_to_mrz,
)


def test_check_digit_icao_examples():
    # Примеры из ICAO Doc 9303 (образец ERIKSSON)
    assert compute_check_digit("L898902C<") == "3"
    assert compute_check_digit("740812") == "2"
    assert compute_check_digit("120415") == "9"


def test_transliteration():
    assert transliterate_from_mrz("PIVEN9") == "ПИВЕНЬ"
    assert transliterate_from_mrz("ANDREQ") == "АНДРЕЙ"
    assert transliterate_from_mrz("ANDREEVI3") == "АНДРЕЕВИЧ"
    assert transliterate_to_mrz("ПИВЕНЬ") == "PIVEN9"


def _build_true_line2() -> str:
    """Собирает эталонную строку 2 MRZ для паспорта 03 20 579029."""
    doc9 = "032579029"
    birth = "000625"
    digits13 = "0200708230002"  # 4-я цифра серии + дата выдачи + код
    fillers = "<<<<<<<"
    opt_chk = compute_check_digit(digits13 + "<")
    line = (
        doc9 + compute_check_digit(doc9)
        + "RUS"
        + birth + compute_check_digit(birth)
        + "M"
        + fillers
        + digits13 + "<" + opt_chk
    )
    composite_body = line[0:10] + line[13:20] + line[21:43]
    return line + compute_check_digit(composite_body)


def test_checksum_digit_fix():
    from postprocess.mrz import fix_digit_candidates, try_fix_digits

    doc9 = "032579029"
    check = compute_check_digit(doc9)
    corrupted = "832579029"  # первая цифра 0 прочитана как 8
    # правильный вариант — среди кандидатов, но mod-10 неоднозначен
    candidates = fix_digit_candidates(corrupted, check)
    assert doc9 in candidates and len(candidates) > 1
    # без подсказки неоднозначность не разрешаем
    assert try_fix_digits(corrupted, check) is None
    # с подсказкой из визуальной зоны — разрешаем
    assert try_fix_digits(corrupted, check, hint=doc9) == doc9
    # подсказка, не совпадающая ни с одним кандидатом, не помогает
    assert try_fix_digits(corrupted, check, hint="999999999") is None


def test_parse_mrz_with_ocr_digit_error():
    """Спутанная цифра в серии/номере чинится по чексумме + визуальной подсказке."""
    line1 = "PNRUSPIVEN9<<ANDREQ<ANDREEVI3".ljust(44, "<")
    line2 = _build_true_line2()
    corrupted = "8" + line2[1:]  # 0 -> 8 в первой цифре
    # без подсказки — честно остаёмся с несошедшейся чексуммой
    assert not parse_mrz(line1, corrupted).number_ok
    # с подсказкой (визуальная зона дала 03 20 579029 -> hint 032579029)
    mrz = parse_mrz(line1, corrupted, number_hint="032579029")
    assert mrz.number_ok
    assert mrz.series_number == "03 20 579029"
    assert any("исправлен" in p for p in mrz.problems)


def test_parse_true_mrz():
    line1 = "PNRUSPIVEN9<<ANDREQ<ANDREEVI3".ljust(44, "<")
    line2 = _build_true_line2()
    assert len(line2) == 44

    mrz = parse_mrz(line1, line2)
    assert mrz.surname == "ПИВЕНЬ"
    assert mrz.name == "АНДРЕЙ"
    assert mrz.patronymic == "АНДРЕЕВИЧ"
    assert mrz.series == "03 20"
    assert mrz.number_ok
    assert mrz.series_number == "03 20 579029"
    assert mrz.birth_date == "25.06.2000" and mrz.birth_date_ok
    assert mrz.sex == "МУЖ."
    assert mrz.issue_date == "08.07.2020"
    assert mrz.issue_code == "230-002"
    assert mrz.optional_ok
    assert not mrz.problems


def test_parse_noisy_ocr_mrz():
    """Реальный шумный выход OCR с тестового фото: парсер не падает
    и достаёт хотя бы поля, подтверждаемые структурой."""
    line1 = "PNRUSPIVEN9AND RE 0"  # OCR потерял разделители
    line2 = "0325790291RUS0006253M<<<<<<0200708230002<64"

    mrz = parse_mrz(line1, line2)
    assert mrz.birth_date == "25.06.2000" and mrz.birth_date_ok
    assert mrz.sex == "МУЖ."
    assert mrz.issue_date == "08.07.2020"
    assert mrz.issue_code == "230-002"
    # фамилия из битой строки 1 не должна прийти чистой кириллицей без '?'
    assert mrz.surname is None or "?" in mrz.surname or mrz.name is None


def test_parse_mrz_with_cyrillic_lookalikes():
    line2 = _build_true_line2().replace("RUS", "RУS")  # У — кириллица
    mrz = parse_mrz("PNRUSPIVEN9<<ANDREQ<ANDREEVI3", line2)
    assert mrz.number_ok and mrz.series_number == "03 20 579029"


def test_normalize_name():
    assert normalize.normalize_name("ПИВЕНЬ") == "ПИВЕНЬ"
    assert normalize.normalize_name("пивень ") == "ПИВЕНЬ"
    assert normalize.normalize_name("99994а 0022") is None  # мусор с блика
    assert normalize.normalize_name("АНДРЕЕВИЧ.") == "АНДРЕЕВИЧ"
    assert normalize.normalize_name("") is None


def test_normalize_date():
    assert normalize.normalize_date("25.06.2000") == "25.06.2000"
    assert normalize.normalize_date("08 07.2020") == "08.07.2020"
    assert normalize.normalize_date("99.99.2020") is None
    assert normalize.normalize_date("25.06.20") is None


def test_normalize_date_extraction():
    # реальный шум с бенчмарка: верная дата внутри мусора соседних полей
    assert normalize.normalize_date("7   5 5  72 94242 2.11.1983") == "02.11.1983"
    assert normalize.normalize_date("ОВИЧ 28.05.1980 ОРСК 0Е") == "28.05.1980"
    # без экстракции старое строгое поведение
    assert normalize.normalize_date("7   5 5  72 94242 2.11.1983", extract=False) is None
    # два разных валидных кандидата -> неоднозначно -> None
    assert normalize.normalize_date("01.02.1990 ... 03.04.1991") is None
    # разделители-пробелы не считаются датой (защита от ложных сборок)
    assert normalize.normalize_date("24.9 06. 1949") is None


def test_pick_name_mrz_priority():
    from postprocess.passport import _pick_name

    # совпадение с MRZ -> визуальная, подтверждена
    fld = _pick_name("ПИВЕНЬ", "ПИВЕНЬ", mrz_reliable=True)
    assert fld.value == "ПИВЕНЬ" and fld.confirmed and fld.source == "visual"
    # конфликт при надёжной MRZ -> берём MRZ
    fld = _pick_name("ЦМИТРИЕВН", "ДМИТРИЕВНА", mrz_reliable=True)
    assert fld.value == "ДМИТРИЕВНА" and fld.source == "mrz" and not fld.confirmed
    # конфликт при ненадёжной MRZ -> прежнее поведение (визуальная)
    fld = _pick_name("ЦМИТРИЕВН", "ДМИТРИЕВНА", mrz_reliable=False)
    assert fld.value == "ЦМИТРИЕВН" and fld.source == "visual"


def test_normalize_series_code_extraction():
    assert normalize.normalize_series_number("СЕРИЯ 92 18 392483 РОВНО") == "92 18 392483"
    assert normalize.normalize_series_number("12 34 567890 и 09 87 654321") is None
    assert normalize.normalize_issue_code("КОД 290-004 УФМС") == "290-004"
    assert normalize.normalize_issue_code("290004123", extract=True) is None  # 9 цифр без дефиса


def test_normalize_series_number():
    assert normalize.normalize_series_number("03 20 579029") == "03 20 579029"
    assert normalize.normalize_series_number("0320579029") == "03 20 579029"
    assert normalize.normalize_series_number("08 20 5790") is None


def test_normalize_issue_code_and_sex():
    assert normalize.normalize_issue_code("230-002") == "230-002"
    assert normalize.normalize_issue_code("230 002") == "230-002"
    assert normalize.normalize_issue_code("230-00") is None
    assert normalize.normalize_sex("МУЖ.") == "МУЖ."
    assert normalize.normalize_sex("муж") == "МУЖ."
    assert normalize.normalize_sex("ЖЕН") == "ЖЕН."
    assert normalize.normalize_sex("") is None


def test_normalize_text_field():
    assert normalize.normalize_text_field("Г КРАСНОДАР") == "Г. КРАСНОДАР"
    assert normalize.normalize_text_field("гор. Краснодар") == "ГОР. КРАСНОДАР"
    assert normalize.normalize_text_field("Г.КРАСНОДАР") == "Г. КРАСНОДАР"
    assert normalize.normalize_text_field("~~") is None


def test_refine_birth_place():
    from postprocess.birth_place import refine_birth_place, score_birth_place_candidate

    val, ok, _ = refine_birth_place("Г. КРАСНОДА")
    assert val == "Г. КРАСНОДАР"
    assert ok

    val, ok, _ = refine_birth_place("КРАСНОДА")
    assert val == "КРАСНОДАР"
    assert ok

    val, ok, _ = refine_birth_place("С. ИВАНОВКА")
    assert val == "С. ИВАНОВКА"
    assert not ok

    val, ok, _ = refine_birth_place("КРАСНАДАРСКИЙ КРАЙ")
    assert "КРАСНОДАР" in val
    assert ok

    assert score_birth_place_candidate("25.06.2000", "25.06.2000", 0.99) < score_birth_place_candidate(
        "Г. КРАСНОДАР", "Г. КРАСНОДАР", 0.85
    )


def test_parse_registration_address():
    from postprocess.registration import parse_registration_address

    text = (
        "ЗАРЕГИСТРИРОВАН 14 ИЮЛЯ 2015 ЗАБАЙКАЛЬСКИЙ РЕГЕН: ЧИТА "
        "МКР ЧЕРЕМУШКИ 21 ОТДЕЛ МВД 750-005"
    )
    addr = parse_registration_address(text)
    assert addr is not None
    assert "ЗАБАЙКАЛЬСК" in addr
    assert "ЧИТА" in addr
    assert "ЧЕРЕМУШКИ" in addr
    assert "21" in addr


def test_parse_registration_not_stamp():
    from postprocess.registration import parse_registration_address

    assert parse_registration_address("ПИВЕНЬ АНДРЕЙ ПАСПОРТ") is None


def test_parse_registration_without_regen_label():
    from postprocess.registration import parse_registration_address

    text = "ЗАРЕГИСТРИРОВАН 14 ИЮЛЯ 2015 ЗАБАЙКАЛЬСКИЙ МКР ЧЕРЕМУШКИ 21"
    addr = parse_registration_address(text)
    assert addr is not None
    assert "ЧИТА" in addr


def test_division_lookup():
    issued_by = lookup_issued_by("230-002", ocr_hint="КРАЮ")
    assert issued_by is not None, "код 230-002 не найден в справочнике"
    assert "КРАСНОДАР" in issued_by.upper()
    print(f"  230-002 -> {issued_by}")
    assert lookup_issued_by("999-999") is None


def test_textual_date():
    assert normalize.normalize_textual_date("14 ИЮЛЯ 2015 Г.") == "14.07.2015"
    assert normalize.normalize_textual_date("3 сентября 1987") == "03.09.1987"
    assert normalize.normalize_textual_date("31 февраля 2000") is None  # нет такой даты
    assert normalize.normalize_textual_date("просто текст") is None
    assert normalize.normalize_any_date("25.06.2000") == "25.06.2000"
    assert normalize.normalize_any_date("14 июля 2015 г.") == "14.07.2015"


def test_gazetteer_correction():
    from postprocess import gazetteer as g

    assert g.correct_city("ЧИТА") == "ЧИТА"
    assert g.correct_city("ЕКАТИРИНБУРГ") == "ЕКАТЕРИНБУРГ"  # OCR-опечатка
    assert g.correct_city("ЖЖЖЖЖ") is None
    assert g.correct_region("КРАСНАДАРСКИЙ") == "КРАСНОДАРСКИЙ КРАЙ"
    refined = g.refine_address_lines(["КРАЙ ЗАБАЙКАЛЬСКИЙ", "ЕЦ2. МКР ЧЕРЕМУШКИ", "ПУНКТ Г. ЧИТА"])
    assert refined["region"] == "ЗАБАЙКАЛЬСКИЙ КРАЙ"
    assert refined["city"] == "ЧИТА"
    assert refined["street"] == "МКР. ЧЕРЕМУШКИ"


def test_orientation_scoring():
    from unittest.mock import MagicMock, patch

    from postprocess.mrz import MrzResult
    from postprocess.orientation import (
        is_orientation_confident,
        mrz_probe_score,
        probe_orientation,
        score_passport_page,
    )
    from postprocess.passport import RefinedField

    good_mrz = MrzResult(
        raw_line1="L1", raw_line2="L2" * 20,
        number_ok=True, birth_date_ok=True, optional_ok=True,
    )
    bad_mrz = MrzResult(
        raw_line1="L1", raw_line2="L2",
        number_ok=False, birth_date_ok=False, optional_ok=False,
    )
    doc = MagicMock(doc_type="passport", doc_confidence=0.89, zones=[MagicMock()] * 13)
    refined_good = {
        "surname": RefinedField("ПИВЕНЬ", "visual", True),
        "name": RefinedField("АНДРЕЙ", "visual", True),
        "series_number": RefinedField("03 20 579029", "visual", True),
        "birth_date": RefinedField("25.06.2000", "mrz", True),
    }
    refined_bad = {
        "surname": RefinedField("ЧНЖЕИН", "visual", False),
        "name": RefinedField("ИЖХН", "visual", False),
        "series_number": RefinedField("66 06 290680", "visual", False),
        "birth_date": RefinedField(None, "none", False),
    }

    with patch("postprocess.orientation._read_mrz", return_value=good_mrz):
        with patch("postprocess.orientation.refine_passport", return_value=refined_good):
            good_score = score_passport_page(doc, None)
            assert is_orientation_confident(doc, None)

    with patch("postprocess.orientation._read_mrz", return_value=bad_mrz):
        with patch("postprocess.orientation.refine_passport", return_value=refined_bad):
            bad_score = score_passport_page(doc, None)
            assert not is_orientation_confident(doc, None)

    assert good_score > bad_score + 3

    import numpy as np

    class FakeDoc:
        doc_type = "passport"
        confidence = 0.9
        crop = np.zeros((100, 100, 3), dtype=np.uint8)

    reader = MagicMock()
    reader._classifier.classify.return_value = [FakeDoc()]
    mrz_ocr = MagicMock()
    mrz_ocr.read_document.return_value = ("PNRUS", "0325790291RUS0006253M")

    with patch("postprocess.orientation.parse_mrz", return_value=good_mrz):
        assert mrz_probe_score(reader, np.zeros((10, 10, 3), np.uint8), mrz_ocr) >= 8.0
        deg, _ = probe_orientation(reader, np.zeros((10, 10, 3), np.uint8), mrz_ocr)
        assert deg == 0


def test_sorted_ocr_ordering():
    from postprocess.sorted_ocr import _order_fragments

    # фрагменты даты пришли не по порядку (по x: 100, 0, 50)
    results = [
        ([[100, 0], [130, 0], [130, 20], [100, 20]], "1985", 0.9),
        ([[0, 0], [30, 0], [30, 20], [0, 20]], "17.", 0.9),
        ([[50, 0], [80, 0], [80, 20], [50, 20]], "04.", 0.9),
    ]
    text, _ = _order_fragments(results)
    assert text == "17. 04. 1985"
    assert normalize.normalize_date(text) == "17.04.1985"


def main() -> int:
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{'OK' if not failures else f'{failures} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
