"""Image validation: magic bytes, re-encode, size limits."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps

from service.core.config import get_settings
from service.core.errors import (
    FILE_TOO_LARGE,
    INVALID_IMAGE,
    UNSUPPORTED_MEDIA_TYPE,
    ServiceError,
)

_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",  # checked further
}

_ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


@dataclass
class ValidatedImage:
    data: bytes
    width: int
    height: int
    format: str
    scaled: bool = False
    original_width: Optional[int] = None
    original_height: Optional[int] = None


class ValidationError(Exception):
    def __init__(self, error: ServiceError, details: Optional[dict] = None):
        self.error = error
        self.details = details or {}


def downscale_warning(validated: ValidatedImage) -> Optional[dict]:
    if not validated.scaled:
        return None
    ow = validated.original_width or validated.width
    oh = validated.original_height or validated.height
    limit = get_settings().max_image_side_px
    return {
        "code": "IMAGE_DOWNSCALED",
        "field": None,
        "message": (
            f"Изображение уменьшено с {ow}×{oh} до {validated.width}×{validated.height} "
            f"(лимит {limit}px по длинной стороне)"
        ),
    }


def _detect_format(raw: bytes) -> Optional[str]:
    if raw[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return None


def validate_image(raw: bytes) -> ValidatedImage:
    settings = get_settings()
    if len(raw) > settings.max_image_bytes:
        raise ValidationError(
            FILE_TOO_LARGE, {"max_bytes": settings.max_image_bytes, "actual_bytes": len(raw)}
        )

    fmt = _detect_format(raw)
    if fmt is None:
        raise ValidationError(UNSUPPORTED_MEDIA_TYPE, {"detected": "unknown"})

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise ValidationError(INVALID_IMAGE, {"reason": str(exc)}) from exc

    pil_format = img.format
    if pil_format not in _ALLOWED_FORMATS:
        raise ValidationError(UNSUPPORTED_MEDIA_TYPE, {"detected": pil_format or "unknown"})

    img = ImageOps.exif_transpose(img)

    orig_w, orig_h = img.size
    w, h = orig_w, orig_h
    if max(w, h) > settings.max_decode_side_px:
        raise ValidationError(
            INVALID_IMAGE,
            {"reason": "resolution_too_large", "width": w, "height": h},
        )

    scaled = False
    out_fmt = "JPEG" if pil_format == "JPEG" else pil_format
    if out_fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Downscale длинной стороны до max_image_side_px (LANCZOS)
    if max(w, h) > settings.max_image_side_px:
        scale = settings.max_image_side_px / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        scaled = True

    buf = io.BytesIO()
    if out_fmt == "JPEG":
        img.save(buf, format="JPEG", quality=92, optimize=True)
    elif out_fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
    else:
        img.save(buf, format="WEBP", quality=90)

    return ValidatedImage(
        data=buf.getvalue(),
        width=w,
        height=h,
        format=fmt,
        scaled=scaled,
        original_width=orig_w if scaled else None,
        original_height=orig_h if scaled else None,
    )
