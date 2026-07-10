"""Генератор синтетических кропов полей паспорта РФ для дообучения CRNN.

Выход — формат deep-text-recognition-benchmark: папка images/ и labels.tsv
(путь<TAB>текст). Алфавит меток проверяется по charset кастомной сети
(custom_example.yaml), несовместимые символы не генерируются.

Запуск:
    python training\\synth_generator.py --count 1000 --out training\\data\\synth
    python training\\synth_generator.py --count 100000 --out D:\\datasets\\synth

Дополнительные шрифты кладите в training\\fonts\\*.ttf (моноширинные
с кириллицей); системные Courier New / Lucida Console / Consolas
подхватываются автоматически.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

IMG_HEIGHT = 64  # imgH кастомной сети (custom_example.yaml)

_CHARSET_YAML = (
    Path.home() / ".cache" / "docreader" / "models" / "easyocr"
    / "user_network" / "custom_example.yaml"
)

_SYSTEM_FONTS = [
    r"C:\Windows\Fonts\cour.ttf",     # Courier New
    r"C:\Windows\Fonts\courbd.ttf",   # Courier New Bold
    r"C:\Windows\Fonts\lucon.ttf",    # Lucida Console
    r"C:\Windows\Fonts\consola.ttf",  # Consolas
]

MRZ_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# --- словари-затравки (расширяются процедурной генерацией) ---

_MALE_NAMES = [
    "АЛЕКСАНДР", "АЛЕКСЕЙ", "АНАТОЛИЙ", "АНДРЕЙ", "АНТОН", "АРТЕМ", "БОРИС",
    "ВАДИМ", "ВАЛЕРИЙ", "ВАСИЛИЙ", "ВИКТОР", "ВИТАЛИЙ", "ВЛАДИМИР", "ВЛАДИСЛАВ",
    "ГЕННАДИЙ", "ГЕОРГИЙ", "ГРИГОРИЙ", "ДАНИИЛ", "ДЕНИС", "ДМИТРИЙ", "ЕВГЕНИЙ",
    "ЕГОР", "ИВАН", "ИГОРЬ", "ИЛЬЯ", "КИРИЛЛ", "КОНСТАНТИН", "ЛЕОНИД", "МАКСИМ",
    "МИХАИЛ", "НИКИТА", "НИКОЛАЙ", "ОЛЕГ", "ПАВЕЛ", "ПЕТР", "РОМАН", "РУСЛАН",
    "СЕРГЕЙ", "СТАНИСЛАВ", "СТЕПАН", "ТИМОФЕЙ", "ФЕДОР", "ЭДУАРД", "ЮРИЙ", "ЯРОСЛАВ",
]
_FEMALE_NAMES = [
    "АЛЕКСАНДРА", "АЛИНА", "АЛЛА", "АНАСТАСИЯ", "АННА", "ВАЛЕНТИНА", "ВЕРА",
    "ВИКТОРИЯ", "ГАЛИНА", "ДАРЬЯ", "ЕКАТЕРИНА", "ЕЛЕНА", "ЕЛИЗАВЕТА", "ИРИНА",
    "КСЕНИЯ", "ЛАРИСА", "ЛЮБОВЬ", "ЛЮДМИЛА", "МАРГАРИТА", "МАРИНА", "МАРИЯ",
    "НАДЕЖДА", "НАТАЛЬЯ", "НИНА", "ОКСАНА", "ОЛЬГА", "ПОЛИНА", "СВЕТЛАНА",
    "СОФИЯ", "ТАМАРА", "ТАТЬЯНА", "ЮЛИЯ", "ЯНА",
]
_SURNAME_STEMS = [
    "ИВАН", "ПЕТР", "СИДОР", "КУЗНЕЦ", "СМИРН", "ПОПОВ", "ВАСИЛЬ", "СОКОЛ",
    "МИХАЙЛ", "НОВИК", "ФЕДОР", "МОРОЗ", "ВОЛК", "АЛЕКСЕ", "ЛЕБЕД", "СЕМЕН",
    "ЕГОР", "ПАВЛ", "КОЗЛ", "СТЕПАН", "НИКОЛА", "ОРЛ", "АНДРЕ", "МАКАР",
    "НИКИТ", "ЗАХАР", "ЗАЙЦ", "СОЛОВЬ", "БОРИС", "ЯКОВЛ", "ГРИГОРЬ", "РОМАН",
    "ВОРОБЬ", "БЕЛ", "ЧЕРН", "КРАСН", "ПИВ", "ГОНЧАР", "БОНДАР", "КРАВЦ",
    "ШЕВЧ", "МЕЛЬНИК", "ТКАЧ", "ПОЛЯК", "ВИНОГРАД", "БОГДАН", "ТАРАС",
]
_CITIES = [
    "МОСКВА", "САНКТ-ПЕТЕРБУРГ", "КРАСНОДАР", "НОВОСИБИРСК", "ЕКАТЕРИНБУРГ",
    "КАЗАНЬ", "НИЖНИЙ НОВГОРОД", "ЧЕЛЯБИНСК", "САМАРА", "ОМСК", "РОСТОВ-НА-ДОНУ",
    "УФА", "КРАСНОЯРСК", "ВОРОНЕЖ", "ПЕРМЬ", "ВОЛГОГРАД", "САРАТОВ", "ТЮМЕНЬ",
    "ТОЛЬЯТТИ", "ИЖЕВСК", "БАРНАУЛ", "ИРКУТСК", "ХАБАРОВСК", "ЯРОСЛАВЛЬ",
    "ВЛАДИВОСТОК", "ТОМСК", "ОРЕНБУРГ", "КЕМЕРОВО", "РЯЗАНЬ", "АСТРАХАНЬ",
]
_REGIONS = [
    "КРАСНОДАРСКОМУ КРАЮ", "МОСКОВСКОЙ ОБЛАСТИ", "РОСТОВСКОЙ ОБЛАСТИ",
    "СВЕРДЛОВСКОЙ ОБЛАСТИ", "РЕСПУБЛИКЕ ТАТАРСТАН", "ПЕРМСКОМУ КРАЮ",
    "НОВОСИБИРСКОЙ ОБЛАСТИ", "САМАРСКОЙ ОБЛАСТИ", "ЧЕЛЯБИНСКОЙ ОБЛАСТИ",
    "Г. МОСКВЕ", "Г. САНКТ-ПЕТЕРБУРГУ", "ВОРОНЕЖСКОЙ ОБЛАСТИ",
    "СТАВРОПОЛЬСКОМУ КРАЮ", "ИРКУТСКОЙ ОБЛАСТИ", "ПРИМОРСКОМУ КРАЮ",
]

# --- адресный домен (прописка / место рождения) ---

_ADDR_REGIONS = [
    "КРАСНОДАРСКИЙ КРАЙ", "ПЕРМСКИЙ КРАЙ", "СТАВРОПОЛЬСКИЙ КРАЙ",
    "ПРИМОРСКИЙ КРАЙ", "МОСКОВСКАЯ ОБЛ.", "РОСТОВСКАЯ ОБЛ.",
    "СВЕРДЛОВСКАЯ ОБЛ.", "САМАРСКАЯ ОБЛАСТЬ", "ВОРОНЕЖСКАЯ ОБЛ.",
    "РЕСП. ТАТАРСТАН", "РЕСП. БАШКОРТОСТАН", "УДМУРТСКАЯ РЕСП.",
]
_DISTRICTS = [
    "ЛЕНИНСКИЙ Р-Н", "СОВЕТСКИЙ Р-Н", "ОКТЯБРЬСКИЙ Р-Н", "КИРОВСКИЙ Р-Н",
    "ЦЕНТРАЛЬНЫЙ Р-Н", "ПРИМОРСКИЙ Р-Н", "ДИНСКОЙ Р-Н", "АБИНСКИЙ Р-Н",
    "СЕВЕРСКИЙ РАЙОН", "ТЕМРЮКСКИЙ РАЙОН",
]
_SETTLEMENT_PREFIXES = [
    "Г.", "ГОР.", "ПОС.", "П.", "ПГТ", "С.", "СТ-ЦА", "Х.", "Д.", "АУЛ",
]
_SETTLEMENT_NAMES = [
    "СОЛНЕЧНЫЙ", "СЕВЕРНАЯ", "ДИНСКАЯ", "ИЛЬСКИЙ", "АФИПСКИЙ", "ЯБЛОНОВСКИЙ",
    "КРЫМСК", "АБИНСК", "ТИМАШЕВСК", "КОРЕНОВСК", "БЕРЕЗОВКА", "ИВАНОВКА",
    "НОВОСЕЛОВКА", "ПЕТРОВСКОЕ", "АЛЕКСАНДРОВКА",
]
_STREET_TYPES = [
    "УЛ.", "УЛИЦА", "ПР-КТ", "ПРОСП.", "ПЕР.", "Б-Р", "Ш.", "НАБ.",
    "ПРОЕЗД", "ТУП.",
]
_STREET_NAMES = [
    "ЛЕНИНА", "МИРА", "САДОВАЯ", "ЦЕНТРАЛЬНАЯ", "МОЛОДЕЖНАЯ", "ШКОЛЬНАЯ",
    "ЛЕСНАЯ", "СОВЕТСКАЯ", "ГАГАРИНА", "ПУШКИНА", "КИРОВА", "ПОБЕДЫ",
    "КРАСНАЯ", "ОКТЯБРЬСКАЯ", "ЗЕЛЕНАЯ", "НОВАЯ", "ЖЕЛЕЗНОДОРОЖНАЯ",
    "КОМСОМОЛЬСКАЯ", "ПЕРВОМАЙСКАЯ", "СТАВРОПОЛЬСКАЯ", "СЕВЕРНАЯ",
    "40 ЛЕТ ПОБЕДЫ", "ИМ. ТУРГЕНЕВА", "КРАСНЫХ ПАРТИЗАН",
]


def load_charset() -> set[str]:
    with open(_CHARSET_YAML, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return set(config["character_list"]) | {" "}


def load_fonts(fonts_dir: Path) -> list[str]:
    paths = [p for p in _SYSTEM_FONTS if Path(p).is_file()]
    if fonts_dir.is_dir():
        paths.extend(str(p) for p in sorted(fonts_dir.glob("*.ttf")))
        paths.extend(str(p) for p in sorted(fonts_dir.glob("*.otf")))
    if not paths:
        raise RuntimeError("Не найдено ни одного шрифта")
    return paths


# --- генерация текстов ---

class TextSampler:
    def __init__(self, rng: random.Random, include_mrz: bool):
        self._rng = rng
        kinds = [
            (self.name, 0.13), (self.surname, 0.13), (self.patronymic, 0.10),
            (self.date, 0.13), (self.series_number, 0.09), (self.issue_code, 0.07),
            (self.place, 0.08), (self.issued_by, 0.07), (self.sex, 0.04),
            (self.address, 0.16),
        ]
        if include_mrz:
            kinds.append((self.mrz_fragment, 0.10))
        self._kinds = [k for k, _ in kinds]
        self._weights = [w for _, w in kinds]

    def sample(self) -> str:
        return self._rng.choices(self._kinds, weights=self._weights, k=1)[0]()

    def name(self) -> str:
        return self._rng.choice(_MALE_NAMES + _FEMALE_NAMES)

    def surname(self) -> str:
        rng = self._rng
        stem = rng.choice(_SURNAME_STEMS)
        male = rng.random() < 0.5
        suffix = rng.choice(
            ["ОВ", "ЕВ", "ИН", "СКИЙ", "ЦКИЙ", "ЕНКО", "УК", "ЫХ", "ЕНЬ"]
        )
        if not male and suffix in {"ОВ", "ЕВ", "ИН"}:
            suffix += "А"
        elif not male and suffix in {"СКИЙ", "ЦКИЙ"}:
            suffix = suffix[:-2] + "АЯ"
        return stem + suffix

    def patronymic(self) -> str:
        base = self._rng.choice(_MALE_NAMES)
        if base.endswith(("Й", "Ь")):
            base = base[:-1]
            suffix = "ЕВИЧ" if self._rng.random() < 0.5 else "ЕВНА"
        else:
            suffix = "ОВИЧ" if self._rng.random() < 0.5 else "ОВНА"
        return base + suffix

    def date(self) -> str:
        rng = self._rng
        day, month = rng.randint(1, 28), rng.randint(1, 12)
        year = rng.randint(1940, 2025)
        return f"{day:02d}.{month:02d}.{year}"

    def series_number(self) -> str:
        rng = self._rng
        return f"{rng.randint(0, 99):02d} {rng.randint(0, 99):02d} {rng.randint(0, 999999):06d}"

    def issue_code(self) -> str:
        return f"{self._rng.randint(0, 999):03d}-{self._rng.randint(0, 999):03d}"

    def place(self) -> str:
        city = self._rng.choice(_CITIES)
        prefix = self._rng.choice(["ГОР. ", "Г. ", "Г.", "", "ПОС. ", "С. "])
        return prefix + city

    def issued_by(self) -> str:
        rng = self._rng
        head = rng.choice(["ГУ МВД РОССИИ ПО", "МВД ПО", "ОТДЕЛОМ УФМС РОССИИ ПО", "УМВД РОССИИ ПО"])
        text = f"{head} {rng.choice(_REGIONS)}"
        # зона многострочная, кропы приходят и обрезками
        if rng.random() < 0.35:
            words = text.split()
            start = rng.randint(0, max(0, len(words) - 2))
            text = " ".join(words[start:start + rng.randint(1, 4)])
        return text

    def sex(self) -> str:
        return self._rng.choice(["МУЖ.", "ЖЕН.", "МУЖ", "ЖЕН"])

    def address_region(self) -> str:
        """Регион в именительном падеже (для органа регистрации)."""
        return self._rng.choice(_ADDR_REGIONS)

    def address(self) -> str:
        """Строка прописки/адреса: регион, район, нас. пункт, улица, дом.

        Штамп регистрации многострочный, поэтому генерируем и полные
        цепочки, и отдельные компоненты (как режет детектор/кроп).
        """
        rng = self._rng
        parts = []
        if rng.random() < 0.45:
            parts.append(rng.choice(_ADDR_REGIONS))
        if rng.random() < 0.35:
            parts.append(rng.choice(_DISTRICTS))
        if rng.random() < 0.7:
            parts.append(f"{rng.choice(_SETTLEMENT_PREFIXES)} {rng.choice(_SETTLEMENT_NAMES + _CITIES)}")
        if rng.random() < 0.3:
            parts.append(f"МКР. {rng.choice(_SETTLEMENT_NAMES + [str(rng.randint(1, 12))])}")
        if rng.random() < 0.75:
            parts.append(f"{rng.choice(_STREET_TYPES)} {rng.choice(_STREET_NAMES)}")
            house = f"Д. {rng.randint(1, 199)}"
            if rng.random() < 0.3:
                house += f" КОРП. {rng.randint(1, 5)}"
            if rng.random() < 0.6:
                house += f" КВ. {rng.randint(1, 300)}"
            parts.append(house)
        if not parts:
            parts = [f"УЛ. {rng.choice(_STREET_NAMES)} Д. {rng.randint(1, 99)}"]

        text = rng.choice([", ", " "]).join(parts)
        # обрезки длинных строк — как при кропе зоны по частям
        words = text.split()
        if len(words) > 4 and rng.random() < 0.4:
            start = rng.randint(0, len(words) - 2)
            text = " ".join(words[start:start + rng.randint(2, 5)])
        return text

    def mrz_fragment(self) -> str:
        rng = self._rng
        length = rng.randint(8, 30)
        return "".join(
            "<" if rng.random() < 0.25 else rng.choice(MRZ_CHARS)
            for _ in range(length)
        )


# --- рендер и аугментации ---

def guilloche_background(width: int, height: int, rng: random.Random) -> Image.Image:
    """Розоватый фон с волнистой защитной сеткой, как на странице паспорта."""
    base = (
        rng.randint(225, 248),
        rng.randint(205, 230),
        rng.randint(200, 228),
    )
    image = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(image)
    line_color = (
        base[0] - rng.randint(15, 45),
        base[1] - rng.randint(25, 60),
        base[2] - rng.randint(20, 55),
    )
    step = rng.randint(4, 9)
    amplitude = rng.uniform(1.0, 4.0)
    period = rng.uniform(20.0, 60.0)
    phase_step = rng.uniform(0.5, 2.0)
    for row, y0 in enumerate(range(-step, height + step, step)):
        points = [
            (x, y0 + amplitude * np.sin(x / period + row * phase_step))
            for x in range(0, width + 8, 8)
        ]
        draw.line(points, fill=line_color, width=1)
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, 4, (height, width, 3))
    arr = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_text(text: str, font_path: str, rng: random.Random) -> np.ndarray:
    font_size = rng.randint(30, 48)
    font = ImageFont.truetype(font_path, font_size)
    char_widths = [max(1, int(font.getlength(ch))) for ch in text]
    spacing = rng.randint(0, 6)
    total_width = sum(char_widths) + spacing * (len(text) - 1)
    pad_x, pad_y = rng.randint(8, 30), rng.randint(6, 20)
    width, height = total_width + 2 * pad_x, font_size + 2 * pad_y

    image = guilloche_background(width, height, rng)
    draw = ImageDraw.Draw(image)
    ink = rng.randint(15, 70)
    color = (ink, ink, ink + rng.randint(0, 15))
    x = pad_x
    y_base = pad_y
    for ch, ch_width in zip(text, char_widths):
        # эффект машинописи: каждый символ чуть гуляет по вертикали и тону
        jitter_y = y_base + rng.randint(-2, 2)
        jitter_ink = tuple(min(255, c + rng.randint(-10, 25)) for c in color)
        draw.text((x, jitter_y), ch, font=font, fill=jitter_ink)
        x += ch_width + spacing
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = image.shape[:2]

    # перспектива + лёгкий поворот
    if rng.random() < 0.7:
        shift = w * 0.02
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32(
            [[rng.uniform(-shift, shift), rng.uniform(-3, 3)] for _ in range(4)]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )

    # блик ламинации: светлое эллиптическое пятно
    if rng.random() < 0.45:
        overlay = np.zeros((h, w), dtype=np.float32)
        center = (rng.randint(0, w), rng.randint(0, h))
        axes = (rng.randint(w // 6, max(w // 2, w // 6 + 1)), rng.randint(h // 3, h))
        cv2.ellipse(overlay, center, axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(w, h) / 8)
        strength = rng.uniform(60, 170)
        image = np.clip(
            image.astype(np.float32) + overlay[..., None] * strength, 0, 255
        ).astype(np.uint8)

    # расфокус или смаз
    roll = rng.random()
    if roll < 0.45:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=rng.uniform(0.4, 1.6))
    elif roll < 0.6:
        k = rng.choice([3, 5, 7])
        kernel = np.zeros((k, k), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        image = cv2.filter2D(image, -1, kernel)

    # яркость/контраст
    alpha, beta = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # JPEG-артефакты
    if rng.random() < 0.7:
        quality = rng.randint(35, 90)
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # приводим к высоте сети
    scale = IMG_HEIGHT / image.shape[0]
    return cv2.resize(image, (max(16, int(image.shape[1] * scale)), IMG_HEIGHT))


# --- основной цикл ---

def generate(count: int, out_dir: Path, seed: int, include_mrz: bool) -> None:
    rng = random.Random(seed)
    charset = load_charset()
    fonts = load_fonts(Path(__file__).parent / "fonts")
    sampler = TextSampler(rng, include_mrz)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.tsv"

    print(f"Шрифты: {len(fonts)}, алфавит сети: {len(charset)} символов")
    skipped = 0
    with open(labels_path, "w", encoding="utf-8", newline="") as labels:
        written = 0
        while written < count:
            text = sampler.sample()
            if not set(text) <= charset:
                skipped += 1
                continue
            image = augment(render_text(text, rng.choice(fonts), rng), rng)
            name = f"img_{written:06d}.jpg"
            cv2.imwrite(
                str(images_dir / name), image,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            labels.write(f"images/{name}\t{text}\n")
            written += 1
            if written % 5000 == 0:
                print(f"  {written}/{count}")
    print(f"Готово: {count} примеров в {out_dir} (отброшено вне алфавита: {skipped})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Генератор синтетики для дообучения OCR паспортов")
    parser.add_argument("--count", type=int, default=1000, help="Сколько примеров сгенерировать")
    parser.add_argument("--out", type=Path, default=Path("training/data/synth"), help="Каталог вывода")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mrz", action="store_true", help="Не генерировать MRZ-фрагменты")
    args = parser.parse_args()

    generate(args.count, args.out, args.seed, include_mrz=not args.no_mrz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
