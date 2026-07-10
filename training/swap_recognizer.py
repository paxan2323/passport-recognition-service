"""Безопасная подмена продакшн-распознавалки с автооткатом.

Замеряет CER текущей (живой) модели на synth_v1, подменяет её на
дообученную (training/EasyOCR/trainer/saved_models/ru_passport/
best_accuracy.pth), замеряет CER новой. Если новая модель заметно хуже
(CER вырос более чем в 1.5 раза — щедрый порог, т.к. синтетика уже не
целевой домен после подмешивания реальных данных) — откатывает
подмену автоматически и это видно в логе. Иначе оставляет новую.

Предназначен для неконтролируемых (ночных) прогонов, где некому
подтвердить подмену вручную.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.eval_recognizer import evaluate

MODEL_PATH = (
    Path.home() / ".cache" / "docreader" / "models" / "easyocr"
    / "model" / "custom_example.pth"
)
# Бэкап версионируется по времени — иначе повторные запуски (несколько
# раундов дообучения за ночь) затирают backup друг друга одним и тем же
# именем, и откат тихо восстанавливает не то состояние (баг 04.07.2026).
BACKUP_PATH = MODEL_PATH.parent / f"custom_example.backup_{time.strftime('%Y%m%d_%H%M%S')}.pth"
CANDIDATE_PATH = (
    Path(__file__).parent / "EasyOCR" / "trainer" / "saved_models"
    / "ru_passport" / "best_accuracy.pth"
)
BENCHMARK_SYNTH = Path(__file__).parent / "data" / "synth_v1"
# Реальный held-out бенчмарк (build_real_benchmark.py) — ПЕРВИЧНЫЙ критерий:
# после подмешивания реальных данных синтетический CER может слегка
# ухудшаться при реальном улучшении, поэтому решение принимаем по нему.
BENCHMARK_REAL = Path(__file__).parent / "data" / "real_bench"
# Прежний порог 1.5x пропустил реальную регрессию CER 3.3%->4.4% (1.33x)
# как "приемлемую" — модель осталась жить в проде с ухудшением (найдено
# и откачено вручную 04.07.2026). Порог сделан строже.
REGRESSION_FACTOR = 1.15


def _pick_benchmark() -> tuple[Path, str]:
    labels = BENCHMARK_REAL / "labels.tsv"
    if labels.is_file() and sum(1 for _ in open(labels, encoding="utf-8")) >= 100:
        return BENCHMARK_REAL, "real_bench (реальные паспорта)"
    return BENCHMARK_SYNTH, "synth_v1 (синтетика — реальный бенчмарк недоступен)"


def main() -> int:
    if not CANDIDATE_PATH.is_file():
        print(f"Нет кандидата {CANDIDATE_PATH} — подмена отменена", file=sys.stderr)
        return 1

    benchmark, bench_name = _pick_benchmark()
    print(f"Бенчмарк для решения: {bench_name}")

    print("=== Замер ДО подмены (текущая живая модель) ===")
    before = evaluate(benchmark, limit=500, show_worst=0)
    if not isinstance(before, dict):
        print("Не удалось замерить baseline — подмена отменена", file=sys.stderr)
        return 1
    print(f"CER до: {before['cer']:.1%}, acc: {before['accuracy']:.1%}")

    shutil.copyfile(MODEL_PATH, BACKUP_PATH)
    shutil.copyfile(CANDIDATE_PATH, MODEL_PATH)
    print(f"\nПодменено: {CANDIDATE_PATH.name} -> {MODEL_PATH.name} (бэкап: {BACKUP_PATH.name})")

    print("\n=== Замер ПОСЛЕ подмены ===")
    after = evaluate(benchmark, limit=500, show_worst=10)
    if not isinstance(after, dict):
        print("Не удалось замерить новую модель — откатываю", file=sys.stderr)
        shutil.copyfile(BACKUP_PATH, MODEL_PATH)
        return 1
    print(f"CER после: {after['cer']:.1%}, acc: {after['accuracy']:.1%}")

    if after["cer"] > before["cer"] * REGRESSION_FACTOR:
        shutil.copyfile(BACKUP_PATH, MODEL_PATH)
        print(
            f"\n!!! РЕГРЕССИЯ: CER {before['cer']:.1%} -> {after['cer']:.1%} "
            f"(> {REGRESSION_FACTOR}x) — ОТКАЧЕНО на предыдущую модель."
        )
        return 1

    print(f"\nOK: CER {before['cer']:.1%} -> {after['cer']:.1%}. Новая модель оставлена.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
