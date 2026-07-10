"""Генератор синтетических страниц регистрации (прописки) с OBB-разметкой.

Рендерит страницу «Место жительства» российского паспорта со штампами
регистрации и адресами, сразу зная координаты каждого поля. Аугментация
(перспектива/поворот) применяется одним преобразованием и к картинке,
и к угловым точкам рамок — поэтому разметка остаётся точной.

Выход — формат ultralytics OBB:
    images/img_000000.jpg
    labels/img_000000.txt   (class x1 y1 x2 y2 x3 y3 x4 y4, нормировано 0..1)
    data.yaml               (конфиг датасета для обучения)

Зоны (классы):
    0 reg_authority — кем зарегистрирован (орган)
    1 reg_date      — дата регистрации
    2 reg_address   — строка адреса (регион/город/улица/дом); строк несколько
    3 reg_stamp     — круглый штамп (OCR пропускается)

Запуск:
    python training\\regpage_generator.py --count 3000 --out training\\data\\regpage
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.synth_generator import (
    TextSampler,
    guilloche_background,
    load_charset,
    load_fonts,
)

CLASS_NAMES = ["reg_authority", "reg_date", "reg_address", "reg_stamp"]
CLASS_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Печатные подписи формы (статичные, зонами не размечаются)
_FORM_HEADERS = [
    "МЕСТО ЖИТЕЛЬСТВА", "ОТМЕТКИ О РЕГИСТРАЦИИ",
    "ЗАРЕГИСТРИРОВАН ПО МЕСТУ ЖИТЕЛЬСТВА",
]
_AUTHORITY_TEMPLATES = [
    "ОУФМС РОССИИ ПО {reg}",
    "ОТДЕЛ УФМС РОССИИ ПО {reg}",
    "ОВД {reg}",
    "ТП УФМС РОССИИ",
]


class Box:
    __slots__ = ("cls", "pts")

    def __init__(self, cls: str, x0: float, y0: float, x1: float, y1: float):
        self.cls = cls
        # 4 угла по часовой: TL, TR, BR, BL
        self.pts = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        )


def _draw_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    x: int,
    y: int,
    fill,
) -> tuple[int, int, int, int]:
    """Рисует строку, возвращает её bbox (x0,y0,x1,y1)."""
    draw.text((x, y), text, font=font, fill=fill)
    bbox = draw.textbbox((x, y), text, font=font)
    return bbox


def build_page(
    rng: random.Random,
    sampler: TextSampler,
    fonts: list[str],
) -> tuple[Image.Image, list[Box]]:
    width = rng.randint(650, 780)
    height = rng.randint(900, 1060)
    image = guilloche_background(width, height, rng)
    draw = ImageDraw.Draw(image)
    boxes: list[Box] = []

    label_font = ImageFont.truetype(rng.choice(fonts), rng.randint(20, 26))
    ink = rng.randint(20, 70)
    text_color = (ink, ink, ink + rng.randint(0, 20))
    faint = tuple(min(255, c + 90) for c in text_color)  # печатные подписи бледнее

    # заголовок формы (не зона)
    draw.text((rng.randint(40, 90), rng.randint(20, 45)),
              rng.choice(_FORM_HEADERS), font=label_font, fill=faint)

    y = rng.randint(80, 130)
    blocks = rng.randint(1, 2)
    for _ in range(blocks):
        stamp_font = ImageFont.truetype(rng.choice(fonts), rng.randint(24, 34))
        left = rng.randint(50, 110)

        # орган регистрации (1-2 строки)
        authority = rng.choice(_AUTHORITY_TEMPLATES).format(reg=sampler.address_region())
        for chunk in _wrap(authority, rng.randint(22, 32)):
            bbox = _draw_line(draw, chunk, stamp_font, left, y, text_color)
            boxes.append(Box("reg_authority", *bbox))
            y += stamp_font.size + rng.randint(4, 12)

        # адрес (1-3 строки), каждая строка — своя зона
        for _ in range(rng.randint(1, 3)):
            line = sampler.address()
            for chunk in _wrap(line, rng.randint(24, 34)):
                bbox = _draw_line(draw, chunk, stamp_font, left, y, text_color)
                boxes.append(Box("reg_address", *bbox))
                y += stamp_font.size + rng.randint(4, 12)

        # дата регистрации
        date = sampler.date()
        date_x = rng.randint(left, left + 120)
        bbox = _draw_line(draw, date, stamp_font, date_x, y, text_color)
        boxes.append(Box("reg_date", *bbox))
        y += stamp_font.size + rng.randint(20, 45)

        # круглый штамп (иногда), перекрывает часть текста
        if rng.random() < 0.6:
            _draw_stamp(draw, boxes, width, height, y, rng)

        y += rng.randint(20, 60)
        if y > height - 160:
            break

    return image, boxes


def _draw_stamp(draw, boxes, width, height, y_hint, rng) -> None:
    radius = rng.randint(55, 90)
    cx = rng.randint(width // 2, width - radius - 30)
    cy = min(height - radius - 30, y_hint + rng.randint(-20, 40))
    color = (rng.randint(120, 190), rng.randint(30, 70), rng.randint(30, 70))
    for dr in (0, 8):
        draw.ellipse([cx - radius + dr, cy - radius + dr, cx + radius - dr, cy + radius - dr],
                     outline=color, width=2)
    boxes.append(Box("reg_stamp", cx - radius, cy - radius, cx + radius, cy + radius))


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _rotate90(arr: np.ndarray, boxes: list[Box], k: int) -> np.ndarray:
    """Поворот на k*90° (изображение + точки рамок). k: 1=CW, 2=180, 3=CCW."""
    h, w = arr.shape[:2]
    if k == 1:
        arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
        for b in boxes:
            x, y = b.pts[:, 0].copy(), b.pts[:, 1].copy()
            b.pts[:, 0], b.pts[:, 1] = (h - 1) - y, x
    elif k == 2:
        arr = cv2.rotate(arr, cv2.ROTATE_180)
        for b in boxes:
            b.pts[:, 0], b.pts[:, 1] = (w - 1) - b.pts[:, 0], (h - 1) - b.pts[:, 1]
    elif k == 3:
        arr = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for b in boxes:
            x, y = b.pts[:, 0].copy(), b.pts[:, 1].copy()
            b.pts[:, 0], b.pts[:, 1] = y, (w - 1) - x
    return arr


def augment_page(
    image: Image.Image,
    boxes: list[Box],
    rng: random.Random,
) -> tuple[np.ndarray, list[Box]]:
    """Перспектива+поворот к картинке и к рамкам одним преобразованием."""
    arr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    # поворот страницы на 0/90/180/270 — фото штампа часто повёрнуто
    k = rng.randint(0, 3)
    if k:
        arr = _rotate90(arr, boxes, k)

    h, w = arr.shape[:2]

    # случайная перспектива
    m = min(w, h) * 0.05
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-m, m), rng.uniform(-m, m)] for _ in range(4)])
    matrix = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(arr, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    for box in boxes:
        pts = np.concatenate([box.pts, np.ones((4, 1), dtype=np.float32)], axis=1)
        warped = (matrix @ pts.T).T
        box.pts = warped[:, :2] / warped[:, 2:3]

    # фотометрия (рамки не двигает)
    if rng.random() < 0.5:
        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=rng.uniform(0.4, 1.4))
    alpha, beta = rng.uniform(0.75, 1.2), rng.uniform(-20, 25)
    arr = cv2.convertScaleAbs(arr, alpha=alpha, beta=beta)
    if rng.random() < 0.4:  # общий блик
        overlay = np.zeros((h, w), np.float32)
        cv2.ellipse(overlay, (rng.randint(0, w), rng.randint(0, h)),
                    (rng.randint(w // 4, w // 2), rng.randint(h // 6, h // 3)),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(w, h) / 8)
        arr = np.clip(arr.astype(np.float32) + overlay[..., None] * rng.uniform(40, 120), 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        _, enc = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(45, 92)])
        arr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return arr, boxes


def _yolo_obb_line(box: Box, w: int, h: int) -> str | None:
    pts = box.pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1) / w
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1) / h
    # отбрасываем вырожденные (уехавшие за край) рамки
    area = cv2.contourArea(pts.astype(np.float32))
    if area < 1e-4:
        return None
    coords = " ".join(f"{v:.6f}" for v in pts.flatten())
    return f"{CLASS_IDX[box.cls]} {coords}"


def write_data_yaml(out_dir: Path) -> None:
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images",
        "val: images",  # для smoke; для полного обучения сделать отдельный val
        "",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(count: int, out_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    charset = load_charset()
    fonts = load_fonts(Path(__file__).parent / "fonts")
    sampler = TextSampler(rng, include_mrz=False)

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        image, boxes = build_page(rng, sampler, fonts)
        arr, boxes = augment_page(image, boxes, rng)
        h, w = arr.shape[:2]

        label_lines = []
        for box in boxes:
            # текстовые зоны с символами вне алфавита сети пропускаем как метки OCR,
            # но для детекции они всё равно валидны — оставляем рамку
            line = _yolo_obb_line(box, w, h)
            if line is not None:
                label_lines.append(line)
        if not label_lines:
            continue

        name = f"img_{i:06d}"
        cv2.imwrite(str(images_dir / f"{name}.jpg"), arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (labels_dir / f"{name}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{count}")

    write_data_yaml(out_dir)
    print(f"Готово: {count} страниц в {out_dir}; классы: {CLASS_NAMES}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Генератор страниц регистрации с OBB-разметкой")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=Path("training/data/regpage"))
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    generate(args.count, args.out, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
