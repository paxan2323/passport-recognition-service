"""Service configuration from environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Redis
    redis_url: str = "redis://redis:6379/0"
    redis_op_timeout_s: float = 2.0
    img_ttl_s: int = 120
    result_ttl_s: int = 300

    # Queue
    queue_key: str = "passport:queue"
    queue_max_depth: int = 50
    worker_timeout_s: float = 20.0
    api_sync_timeout_s: float = 30.0

    # Auth — comma-separated key:client_id pairs
    passport_api_keys: str = "dev-key-001:default"

    # Rate limit (token bucket per API key)
    rate_limit_rps: float = 5.0
    rate_limit_burst: int = 10
    rate_limit_daily: int = 10000

    # Circuit breaker
    circuit_breaker_window_s: int = 60
    circuit_breaker_failure_ratio: float = 0.5
    circuit_breaker_min_samples: int = 10

    # Image validation
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_side_px: int = 1600
    max_decode_side_px: int = 8000
    min_quality_blur: float = 50.0
    min_quality_glare_ratio: float = 0.35
    min_doc_confidence: float = 0.5

    # Worker
    worker_id: str = "worker-1"
    omp_num_threads: int = 4

    # Logging
    log_level: str = "INFO"
    json_logs: bool = True

    # Webhook
    webhook_timeout_s: float = 5.0

    # Dashboard UI
    dashboard_enabled: bool = True
    worker_metrics_url: str = "http://worker:9091/metrics"

    # Custom models (optional — see docker-compose models volume)
    regpage_weights_path: str = ""

    # Перебор 0/90/180/270° если MRZ/поля не сходятся на исходной ориентации
    auto_rotate_passport: bool = True
    # probe — MRZ-probe + 1 полный прогон; full — полный OCR на каждом повороте
    orientation_mode: str = "probe"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def parse_api_keys(raw: Optional[str] = None) -> dict[str, str]:
    """Parse PASSPORT_API_KEYS into {key: client_id}."""
    settings = get_settings()
    text = raw if raw is not None else settings.passport_api_keys
    result: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, client_id = part.split(":", 1)
            result[key.strip()] = client_id.strip()
        else:
            result[part] = "default"
    return result
