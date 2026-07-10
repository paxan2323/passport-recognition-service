"""Admin API key and session storage in Redis."""

from __future__ import annotations

import logging
import secrets
from typing import Optional

import redis

from service.core.config import get_settings

logger = logging.getLogger(__name__)

ADMIN_API_KEY_KEY = "admin:api_key"
ADMIN_SESSION_PREFIX = "admin:session:"


def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


def get_admin_api_key() -> Optional[str]:
    try:
        value = _client().get(ADMIN_API_KEY_KEY)
        return value if value else None
    except redis.RedisError:
        logger.warning("Redis unavailable for admin API key lookup")
        return None


def set_admin_api_key(key: str) -> None:
    _client().set(ADMIN_API_KEY_KEY, key)


def ensure_admin_api_key() -> str:
    """Return active admin API key, creating one if missing."""
    existing = get_admin_api_key()
    if existing:
        return existing

    settings = get_settings()
    if settings.admin_api_key:
        key = settings.admin_api_key
    else:
        for part in settings.passport_api_keys.split(","):
            part = part.strip()
            if not part:
                continue
            key = part.split(":", 1)[0].strip()
            if key:
                set_admin_api_key(key)
                return key
        key = secrets.token_urlsafe(32)

    set_admin_api_key(key)
    logger.info("Initialized admin API key in Redis")
    return key


def regenerate_admin_api_key() -> str:
    key = secrets.token_urlsafe(32)
    set_admin_api_key(key)
    logger.info("Admin API key regenerated")
    return key


def create_admin_session() -> str:
    settings = get_settings()
    token = secrets.token_urlsafe(32)
    _client().setex(
        f"{ADMIN_SESSION_PREFIX}{token}",
        settings.admin_session_ttl_s,
        "1",
    )
    return token


def verify_admin_session(token: Optional[str]) -> bool:
    if not token:
        return False
    try:
        return bool(_client().exists(f"{ADMIN_SESSION_PREFIX}{token}"))
    except redis.RedisError:
        return False


def revoke_admin_session(token: Optional[str]) -> None:
    if not token:
        return
    try:
        _client().delete(f"{ADMIN_SESSION_PREFIX}{token}")
    except redis.RedisError:
        pass
