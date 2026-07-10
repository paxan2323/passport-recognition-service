"""Error catalog (plan section 10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ServiceError:
    code: str
    message: str
    http_status: int

    def to_body(self, request_id: str, details: Optional[dict[str, Any]] = None) -> dict:
        body: dict[str, Any] = {
            "success": False,
            "error": {"code": self.code, "message": self.message},
            "request_id": request_id,
        }
        if details:
            body["error"]["details"] = details
        return body


# Catalog
INVALID_IMAGE = ServiceError("INVALID_IMAGE", "Файл не является корректным изображением", 400)
FILE_TOO_LARGE = ServiceError("FILE_TOO_LARGE", "Размер файла превышает допустимый лимит", 413)
UNSUPPORTED_MEDIA_TYPE = ServiceError(
    "UNSUPPORTED_MEDIA_TYPE", "Поддерживаются только JPEG, PNG и WebP", 415
)
DOCUMENT_NOT_FOUND = ServiceError(
    "DOCUMENT_NOT_FOUND", "Документ не обнаружен на изображении", 422
)
NOT_A_PASSPORT = ServiceError(
    "NOT_A_PASSPORT", "Документ на изображении не является паспортом РФ", 422
)
LOW_QUALITY = ServiceError(
    "LOW_QUALITY", "Качество изображения недостаточно для распознавания", 422
)
OCR_FAILED = ServiceError(
    "OCR_FAILED", "Не удалось распознать обязательные поля паспорта", 422
)
UNAUTHORIZED = ServiceError("UNAUTHORIZED", "API-ключ отсутствует или недействителен", 401)
FORBIDDEN = ServiceError("FORBIDDEN", "Доступ запрещён", 403)
RATE_LIMITED = ServiceError("RATE_LIMITED", "Превышен лимит запросов", 429)
QUEUE_FULL = ServiceError("QUEUE_FULL", "Очередь обработки переполнена, повторите позже", 503)
PROCESSING_TIMEOUT = ServiceError(
    "PROCESSING_TIMEOUT", "Превышено время ожидания обработки", 504
)
INTERNAL = ServiceError("INTERNAL", "Внутренняя ошибка сервиса", 500)
TASK_NOT_FOUND = ServiceError("TASK_NOT_FOUND", "Задача не найдена или истёк срок хранения", 404)
TASK_PENDING = ServiceError("TASK_PENDING", "Задача ещё обрабатывается", 202)
