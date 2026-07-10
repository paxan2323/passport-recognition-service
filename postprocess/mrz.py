"""Парсинг и валидация MRZ внутреннего паспорта РФ.

Формат — ICAO Doc 9303 (тип PN), транслитерация — национальная
(приказ МВД, используется во внутренних паспортах с 2011 года).

Контрольные цифры позволяют математически проверить серию/номер,
дату рождения и необязательное поле (последняя цифра номера +
дата выдачи + код подразделения).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_CHECK_WEIGHTS = (7, 3, 1)

# Транслитерация MRZ внутреннего паспорта -> кириллица
MRZ_TO_CYRILLIC = {
    "A": "А", "B": "Б", "V": "В", "G": "Г", "D": "Д", "E": "Е",
    "2": "Ё", "J": "Ж", "Z": "З", "I": "И", "Q": "Й", "K": "К",
    "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р",
    "S": "С", "T": "Т", "U": "У", "F": "Ф", "H": "Х", "C": "Ц",
    "3": "Ч", "W": "Ш", "4": "Щ", "7": "Ъ", "Y": "Ы", "9": "Ь",
    "6": "Э", "8": "Ю", "5": "Я",
}
CYRILLIC_TO_MRZ = {v: k for k, v in MRZ_TO_CYRILLIC.items()}

# Кириллические двойники латиницы: OCR часто путает их в MRZ
_CYR_LOOKALIKES = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "U", "Х": "X",
    "«": "<", "﻿": "",
})


def compute_check_digit(value: str) -> str:
    """Контрольная цифра ICAO 9303: веса 7-3-1, '<'=0, A-Z=10-35."""
    total = 0
    for i, ch in enumerate(value):
        if ch == "<":
            digit = 0
        elif ch.isdigit():
            digit = int(ch)
        elif "A" <= ch <= "Z":
            digit = ord(ch) - 55
        else:
            digit = 0
        total += digit * _CHECK_WEIGHTS[i % 3]
    return str(total % 10)


def clean_mrz_line(raw: str) -> str:
    """Приводит OCR-строку MRZ к алфавиту [A-Z0-9<]."""
    line = raw.upper().translate(_CYR_LOOKALIKES)
    return re.sub(r"[^A-Z0-9<]", "", line)


# Частые OCR-путаницы цифр (наблюдаемая -> чем могла быть на самом деле)
_DIGIT_CONFUSIONS = {
    "0": "869", "1": "74", "2": "7", "3": "89", "4": "19",
    "5": "68", "6": "589", "7": "12", "8": "0356", "9": "3460",
}


def fix_digit_candidates(digits: str, expected_check: str, tail: str = "") -> set[str]:
    """Все варианты одиночной замены по таблице путаниц, проходящие чексумму."""
    candidates: set[str] = set()
    for i, ch in enumerate(digits):
        for alt in _DIGIT_CONFUSIONS.get(ch, ""):
            fixed = digits[:i] + alt + digits[i + 1:]
            if compute_check_digit(fixed + tail) == expected_check:
                candidates.add(fixed)
    return candidates


def try_fix_digits(
    digits: str,
    expected_check: str,
    tail: str = "",
    hint: Optional[str] = None,
) -> Optional[str]:
    """Восстанавливает одну неверно прочитанную цифру по контрольной цифре.

    Перебираются только правдоподобные OCR-путаницы (0<->8, 9<->6, ...).
    Чексумма mod-10 сама по себе неоднозначна (несколько кандидатов могут
    пройти), поэтому исправление принимается если:
    - ровно один кандидат прошёл чексумму, либо
    - среди кандидатов есть совпадающий с hint (независимое чтение того же
      значения из визуальной зоны страницы) — два источника сошлись.
    """
    candidates = fix_digit_candidates(digits, expected_check, tail)
    if len(candidates) == 1:
        return candidates.pop()
    if hint is not None and hint in candidates:
        return hint
    return None


def transliterate_from_mrz(value: str) -> str:
    """MRZ-имя -> кириллица; нераспознанный символ -> '?'."""
    return "".join(MRZ_TO_CYRILLIC.get(ch, "?") for ch in value)


def transliterate_to_mrz(value: str) -> str:
    """Кириллическое имя -> MRZ-форма (для сравнения с MRZ)."""
    return "".join(CYRILLIC_TO_MRZ.get(ch, ch) for ch in value.upper() if ch != " ")


def _format_date(yymmdd: str, *, birth: bool = False) -> Optional[str]:
    """YYMMDD -> ДД.ММ.ГГГГ. Для дат рождения век выбирается эвристикой."""
    import datetime as dt

    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    current_yy = dt.date.today().year % 100
    century = 1900 if (birth and yy > current_yy) else 2000
    try:
        return dt.date(century + yy, mm, dd).strftime("%d.%m.%Y")
    except ValueError:
        return None


@dataclass
class MrzResult:
    """Разобранная MRZ. *_ok — сошлась ли контрольная цифра поля."""

    surname: Optional[str] = None
    name: Optional[str] = None
    patronymic: Optional[str] = None

    series: Optional[str] = None        # "03 20"
    number9: Optional[str] = None       # 6 цифр номера (позиции 4–9 поля doc9)
    doc9: Optional[str] = None          # 9 цифр «3 серии + 6 номера» из MRZ
    number_ok: bool = False             # чексумма поля серия+номер (9 цифр)
    number_last_digit: Optional[str] = None  # 10-я цифра, из необязательного поля

    birth_date: Optional[str] = None    # "ДД.ММ.ГГГГ"
    birth_date_ok: bool = False

    sex: Optional[str] = None           # "МУЖ." / "ЖЕН."

    issue_date: Optional[str] = None    # "ДД.ММ.ГГГГ"
    issue_code: Optional[str] = None    # "230-002"
    optional_ok: bool = False           # чексумма необязательного поля

    raw_line1: str = ""
    raw_line2: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def series_number(self) -> Optional[str]:
        """Полные серия и номер: 'СС СС НННННН'."""
        if self.series is None or self.number9 is None:
            return None
        return f"{self.series} {self.number9}"


def parse_mrz(
    line1_raw: str,
    line2_raw: str,
    number_hint: Optional[str] = None,
) -> MrzResult:
    """Разбирает обе строки MRZ, устойчиво к OCR-шуму.

    Поля извлекаются якорными регулярками (а не жёсткими позициями),
    достоверность каждого числового поля подтверждается его
    контрольной цифрой.

    number_hint — 9 цифр поля «серия+номер», прочитанные из визуальной
    зоны страницы: используется для дизамбигуации при восстановлении
    спутанной цифры по чексумме.
    """
    result = MrzResult(
        raw_line1=clean_mrz_line(line1_raw),
        raw_line2=clean_mrz_line(line2_raw),
    )
    _parse_line1(result)
    _parse_line2(result, number_hint=number_hint)
    return result


def _parse_line1(result: MrzResult) -> None:
    match = re.search(r"PNRUS([A-Z0-9<]+)", result.raw_line1)
    if not match:
        result.problems.append("строка 1: не найден префикс PNRUS")
        return

    parts = match.group(1).split("<<")
    surname = parts[0].strip("<")
    if surname:
        result.surname = transliterate_from_mrz(surname)
    if len(parts) > 1:
        given = [p for p in parts[1].split("<") if p]
        if given:
            result.name = transliterate_from_mrz(given[0])
        if len(given) > 1:
            result.patronymic = transliterate_from_mrz(given[1])


def _parse_line2(result: MrzResult, number_hint: Optional[str] = None) -> None:
    line = result.raw_line2
    match = re.search(r"(\d{9})(\d)RUS(\d{6})(\d)([MF])", line)
    if not match:
        result.problems.append("строка 2: не найдена основная группа полей")
        return

    doc9, doc_check, birth, birth_check, sex = match.groups()

    if compute_check_digit(doc9) != doc_check:
        fixed = try_fix_digits(doc9, doc_check, hint=number_hint)
        if fixed is not None:
            doc9 = fixed
            result.problems.append("строка 2: серия/номер исправлены по чексумме")
    result.doc9 = doc9
    # 9 цифр: 3 первые серии + 6 номера; 4-я цифра серии — в необязательном поле
    result.number9 = doc9[3:9]
    result.number_ok = compute_check_digit(doc9) == doc_check
    if not result.number_ok:
        result.problems.append("строка 2: чексумма серии/номера не сошлась")

    if compute_check_digit(birth) != birth_check:
        fixed = try_fix_digits(birth, birth_check)
        if fixed is not None and _format_date(fixed, birth=True) is not None:
            birth = fixed
            result.problems.append("строка 2: дата рождения исправлена по чексумме")
    result.birth_date = _format_date(birth, birth=True)
    result.birth_date_ok = (
        result.birth_date is not None
        and compute_check_digit(birth) == birth_check
    )
    if not result.birth_date_ok:
        result.problems.append("строка 2: чексумма даты рождения не сошлась")

    result.sex = "МУЖ." if sex == "M" else "ЖЕН."

    # Необязательное поле: 10-я цифра номера + дата выдачи + код подразделения
    tail = line[match.end():]
    opt = re.search(r"(\d{13})", tail)
    if not opt:
        result.problems.append("строка 2: не найдено необязательное поле")
        return

    digits13 = opt.group(1)

    after = tail[opt.end():].lstrip("<")
    expected = after[0] if (after and after[0].isdigit()) else None
    if expected is not None:
        result.optional_ok = expected in (
            compute_check_digit(digits13 + "<"),
            compute_check_digit(digits13),
        )
        if not result.optional_ok:
            fixed = try_fix_digits(digits13, expected, tail="<")
            if fixed is not None and _format_date(fixed[1:7]) is not None:
                digits13 = fixed
                result.optional_ok = True
                result.problems.append("строка 2: необязательное поле исправлено по чексумме")

    result.number_last_digit = digits13[0]
    result.series = f"{doc9[:2]} {doc9[2]}{result.number_last_digit}"
    result.issue_date = _format_date(digits13[1:7])
    result.issue_code = f"{digits13[7:10]}-{digits13[10:13]}"

    if not result.optional_ok:
        result.problems.append("строка 2: чексумма необязательного поля не сошлась")
