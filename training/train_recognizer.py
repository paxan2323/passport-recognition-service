"""Раннер дообучения кастомной CRNN docreader на подготовленных данных.

Собирает opt из ru_passport_config.yaml, подставляет ТОЧНЫЙ алфавит
предобученной сети (порядок символов обязан совпасть с головой
custom_example.pth) и запускает трейнер EasyOCR с дообучением от
custom_example.pth.

Порядок работы:
    1. python training\\synth_generator.py --count 100000 --out training\\data\\synth_v1
    2. python training\\prepare_trainer_data.py training\\data\\synth_v1 [training\\data\\real]
    3. python training\\train_recognizer.py                 # полное обучение
       python training\\train_recognizer.py --smoke          # быстрый прогон (CPU)

После обучения новую сеть подключить:
    copy training\\EasyOCR\\trainer\\saved_models\\ru_passport\\best_accuracy.pth
         %USERPROFILE%\\.cache\\docreader\\models\\easyocr\\model\\custom_example.pth
(сначала сохранив backup исходной).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
TRAINER_DIR = HERE / "EasyOCR" / "trainer"
CONFIG_PATH = HERE / "ru_passport_config.yaml"
CHARSET_YAML = (
    Path.home() / ".cache" / "docreader" / "models" / "easyocr"
    / "user_network" / "custom_example.yaml"
)
PRETRAINED = (
    Path.home() / ".cache" / "docreader" / "models" / "easyocr"
    / "model" / "custom_example.pth"
)


class AttrDict(dict):
    """dict с доступом через атрибуты — трейнер обращается к opt.field."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def load_character() -> str:
    with open(CHARSET_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["character_list"]


def build_opt(smoke: bool) -> AttrDict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        opt = AttrDict(yaml.safe_load(fh))

    # Алфавит из предобученной сети — порядок критичен для совместимости головы
    opt.character = load_character()
    opt.num_class = len(opt.character) + 1  # + [blank] для CTC

    opt.train_data = "all_data"
    opt.valid_data = str(Path("all_data") / "ru_val")
    opt.select_data = "ru_train"
    opt.saved_model = str(PRETRAINED)

    if smoke:
        opt.experiment_name = "ru_passport_smoke"
        opt.num_iter = 60
        opt.valInterval = 30

    (TRAINER_DIR / "saved_models" / opt.experiment_name).mkdir(parents=True, exist_ok=True)
    return opt


def main() -> int:
    parser = argparse.ArgumentParser(description="Дообучение распознавалки паспортов")
    parser.add_argument("--smoke", action="store_true", help="Короткий прогон для проверки пайплайна")
    args = parser.parse_args()

    if not PRETRAINED.is_file():
        print(f"Нет предобученной модели {PRETRAINED}", file=sys.stderr)
        return 1
    train_split = TRAINER_DIR / "all_data" / "ru_train" / "labels.csv"
    if not train_split.is_file():
        print("Нет подготовленных данных. Сначала: python training\\prepare_trainer_data.py ...", file=sys.stderr)
        return 1

    opt = build_opt(smoke=args.smoke)

    # Трейнер написан на относительные импорты и пути (./saved_models)
    sys.path.insert(0, str(TRAINER_DIR))
    os.chdir(TRAINER_DIR)

    import train as trainer

    print(f"Старт: experiment={opt.experiment_name}, num_iter={opt.num_iter}, "
          f"num_class={opt.num_class}, device={'cuda' if _cuda() else 'cpu'}")
    trainer.train(opt, amp=False)  # train() завершается через sys.exit()
    return 0


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
