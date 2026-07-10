"""Structured JSON logging without PII."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line — no personal data fields."""

    _ALLOWED = frozenset({
        "timestamp", "level", "logger", "message", "request_id", "task_id",
        "api_key_id", "client_ip", "status", "error_code", "duration_ms",
        "queue_depth", "worker_id", "image_bytes", "confidence",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key in self._ALLOWED:
                    payload[key] = value
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, message, (), None
    )
    record.extra_fields = fields  # type: ignore[attr-defined]
    logger.handle(record)
