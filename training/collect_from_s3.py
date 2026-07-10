"""Сбор обучающих данных прямо из S3-бакета с реальными паспортами.

Режим main: скачивает главные развороты, прогоняет через пайплайн
с перебором поворотов (0/90/180/270 — реальные фото часто повёрнуты),
сохраняет MRZ-подтверждённые пары «кроп зоны -> текст» (сильный сигнал:
контрольные цифры ICAO 9303 математически проверяют корректность).
Скачанный файл удаляется сразу после обработки (PII не копится на диске).

Режим registration: диагностика детектора прописки на реальных страницах
(без сохранения кропов как обучающих пар — для адреса нет независимого
источника истины вроде MRZ, поэтому это только health-check: сколько зон
находит детектор на живых фото, а не тренировочные данные).

Инкрементально: manifest.txt хранит обработанные ключи, повторный запуск
продолжает с места остановки. --time-limit-hours останавливает чисто по
истечении времени (для ночного прогона с гарантированным дедлайном).

Запуск:
    python training\\collect_from_s3.py main --s3-config <path> --out training\\data\\real_s3_main --time-limit-hours 5
    python training\\collect_from_s3.py registration --s3-config <path> --out training\\data\\real_s3_reg_report --time-limit-hours 1.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("EASYOCR_MODULE_PATH", str(Path.home() / ".EasyOCR"))

import cv2
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("collect_from_s3")

_ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)

AUTO_FIELDS = (
    "surname", "name", "patronymic", "birth_date",
    "issue_date", "issue_code", "series_number", "sex",
)


def make_s3_client(config_path: Path):
    import boto3
    from botocore.config import Config

    cfg = json.loads(config_path.read_text())
    client = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint_url"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        config=Config(signature_version="s3v4"),
    )
    return client, cfg["bucket"]


def iter_keys(client, bucket: str, prefix: str, exts=(".jpg", ".jpeg", ".png", ".webp")):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(exts):
                yield obj["Key"], obj["Size"]


def load_manifest(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(path.read_text(encoding="utf-8").splitlines())


def process_main_with_rotation(reader, image_bgr: np.ndarray, conf_threshold: float = 0.5):
    """Пробует 0/90/180/270; при уверенном результате на 0° не тратит время на остальные."""
    result = reader.process(image_bgr, return_crops=True)
    if any(d.doc_type == "passport" and d.doc_confidence >= conf_threshold for d in result.documents):
        return result

    best, best_score = result, -1.0
    for rot in _ROTATIONS[1:]:
        img = cv2.rotate(image_bgr, rot)
        try:
            candidate = reader.process(img, return_crops=True)
        except Exception:
            continue
        score = max(
            (d.doc_confidence for d in candidate.documents if d.doc_type == "passport"),
            default=-1.0,
        )
        if score > best_score:
            best, best_score = candidate, score
    return best


def run_main(args: argparse.Namespace) -> None:
    from postprocess import MrzOcr, build_reader, refine_passport
    from postprocess.passport import _zones

    out_dir = args.out
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.tsv"
    manifest_path = out_dir / "manifest.txt"
    stats_path = out_dir / "stats.json"

    manifest = load_manifest(manifest_path)
    client, bucket = make_s3_client(args.s3_config)

    reader = build_reader()
    mrz_ocr = MrzOcr(gpu=True)

    deadline = time.time() + args.time_limit_hours * 3600 if args.time_limit_hours else None
    tmp_dir = Path(tempfile.mkdtemp(prefix="s3_passport_"))

    stats = {"seen": 0, "downloaded": 0, "passport_found": 0, "mrz_ok": 0,
             "field_pairs": 0, "errors": 0, "skipped_cached": 0}
    per_field = {f: 0 for f in AUTO_FIELDS}
    t_start = time.time()

    with open(labels_path, "a", encoding="utf-8", newline="") as labels_fh, \
         open(manifest_path, "a", encoding="utf-8", newline="") as manifest_fh:
        for key, size in iter_keys(client, bucket, args.prefix):
            stats["seen"] += 1
            if key in manifest:
                stats["skipped_cached"] += 1
                continue
            if args.limit and stats["downloaded"] >= args.limit:
                break
            if deadline and time.time() > deadline:
                logger.warning("Достигнут лимит времени, останавливаюсь")
                break

            local = tmp_dir / Path(key).name
            try:
                client.download_file(bucket, key, str(local))
                stats["downloaded"] += 1
                image = cv2.imread(str(local))
                if image is None:
                    raise ValueError("cv2.imread вернул None")

                result = process_main_with_rotation(reader, image, args.conf_threshold)
                for doc_idx, doc in enumerate(result.documents):
                    if doc.doc_type != "passport":
                        continue
                    stats["passport_found"] += 1
                    refined = refine_passport(doc, mrz_ocr)
                    mrz_field = refined.get("mrz")
                    if mrz_field and mrz_field.confirmed:
                        stats["mrz_ok"] += 1

                    stem = f"{Path(key).stem}_{doc_idx}"
                    for field in AUTO_FIELDS:
                        fld = refined.get(field)
                        if fld is None or not fld.value or not fld.confirmed:
                            continue
                        for zone_idx, zone in enumerate(_zones(doc, field)):
                            if zone.crop_image is None or zone.crop_image.size == 0:
                                continue
                            name = f"{stem}_{field}_{zone_idx}.jpg"
                            cv2.imwrite(str(images_dir / name), zone.crop_image)
                            labels_fh.write(f"images/{name}\t{fld.value}\n")
                            stats["field_pairs"] += 1
                            per_field[field] += 1
            except Exception as exc:
                stats["errors"] += 1
                logger.warning("Ошибка на %s: %s", key, exc)
            finally:
                local.unlink(missing_ok=True)  # PII не копится на диске
                manifest_fh.write(key + "\n")
                manifest_fh.flush()

            if stats["downloaded"] % 100 == 0:
                elapsed = time.time() - t_start
                rate = stats["downloaded"] / max(elapsed, 1) * 3600
                print(
                    f"[{stats['downloaded']} скачано, {elapsed/60:.0f} мин, "
                    f"{rate:.0f}/ч] passport={stats['passport_found']} "
                    f"mrz_ok={stats['mrz_ok']} pairs={stats['field_pairs']}"
                )
                stats_path.write_text(
                    json.dumps({**stats, "per_field": per_field, "elapsed_min": elapsed / 60}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    stats_path.write_text(
        json.dumps({**stats, "per_field": per_field, "elapsed_min": (time.time() - t_start) / 60},
                    ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n=== ГОТОВО (main) ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"per_field: {per_field}")


def run_registration(args: argparse.Namespace) -> None:
    """Диагностика детектора на реальных страницах — БЕЗ сохранения обучающих
    пар (для адреса нет независимого источника истины типа MRZ-чексуммы)."""
    from postprocess.registration_detector import RegistrationDetectorReader

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.txt"
    report_path = out_dir / "report.jsonl"
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    manifest = load_manifest(manifest_path)
    client, bucket = make_s3_client(args.s3_config)

    weights = str(Path.home() / ".cache" / "docreader" / "models" / "regpage_obb.pt")
    reader = RegistrationDetectorReader(weights, confidence=0.35)

    deadline = time.time() + args.time_limit_hours * 3600 if args.time_limit_hours else None
    tmp_dir = Path(tempfile.mkdtemp(prefix="s3_reg_"))

    stats = {"seen": 0, "downloaded": 0, "with_address": 0, "with_region": 0,
             "with_city": 0, "with_date": 0, "errors": 0}
    overlay_saved = 0
    t_start = time.time()

    with open(report_path, "a", encoding="utf-8", newline="") as report_fh, \
         open(manifest_path, "a", encoding="utf-8", newline="") as manifest_fh:
        for key, size in iter_keys(client, bucket, args.prefix):
            stats["seen"] += 1
            if key in manifest:
                continue
            if args.limit and stats["downloaded"] >= args.limit:
                break
            if deadline and time.time() > deadline:
                logger.warning("Достигнут лимит времени, останавливаюсь")
                break

            local = tmp_dir / Path(key).name
            try:
                client.download_file(bucket, key, str(local))
                stats["downloaded"] += 1
                image = cv2.imread(str(local))
                if image is None:
                    raise ValueError("cv2.imread вернул None")

                det = reader.read(image, auto_rotate=True)
                if det.address:
                    stats["with_address"] += 1
                if det.region:
                    stats["with_region"] += 1
                if det.city:
                    stats["with_city"] += 1
                if det.date:
                    stats["with_date"] += 1
                report_fh.write(json.dumps({"key": key, **det.as_dict()}, ensure_ascii=False) + "\n")

                if overlay_saved < args.overlay_samples and (det.region or det.address):
                    cv2.imwrite(str(overlay_dir / f"{Path(key).stem}.jpg"), image)
                    overlay_saved += 1
            except Exception as exc:
                stats["errors"] += 1
                logger.warning("Ошибка на %s: %s", key, exc)
            finally:
                local.unlink(missing_ok=True)
                manifest_fh.write(key + "\n")
                manifest_fh.flush()

            if stats["downloaded"] % 100 == 0:
                elapsed = time.time() - t_start
                print(
                    f"[{stats['downloaded']} скачано, {elapsed/60:.0f} мин] "
                    f"address={stats['with_address']} region={stats['with_region']} "
                    f"city={stats['with_city']} date={stats['with_date']}"
                )

    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    print(f"\n=== ГОТОВО (registration, диагностика) ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Сбор обучающих данных из S3")
    parser.add_argument("mode", choices=["main", "registration"])
    parser.add_argument("--s3-config", type=Path, required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-limit-hours", type=float, default=0)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--overlay-samples", type=int, default=30)
    args = parser.parse_args()

    if args.prefix is None:
        args.prefix = "passports/main/" if args.mode == "main" else "passports/registration/"

    if args.mode == "main":
        run_main(args)
    else:
        run_registration(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
