"""Постобработка результатов docreader-ocr: валидация, MRZ-арбитраж, ретраи OCR.

Импорты ленивые: лёгкие модули (mrz, normalize, division_codes) можно
использовать без загрузки easyocr/torch.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from postprocess.factory import build_reader
    from postprocess.mrz import parse_mrz
    from postprocess.mrz_ocr import MrzOcr
    from postprocess.passport import RefinedField, refine_passport

__all__ = ["build_reader", "parse_mrz", "MrzOcr", "RefinedField", "refine_passport"]

_LAZY = {
    "build_reader": "postprocess.factory",
    "parse_mrz": "postprocess.mrz",
    "MrzOcr": "postprocess.mrz_ocr",
    "RefinedField": "postprocess.passport",
    "refine_passport": "postprocess.passport",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'postprocess' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
