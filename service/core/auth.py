"""API Key authentication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, Request

from service.core.config import parse_api_keys
from service.core.errors import UNAUTHORIZED


@dataclass(frozen=True)
class AuthContext:
    api_key: str
    client_id: str
    api_key_id: str  # masked for logs


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def verify_api_key(raw_key: Optional[str]) -> AuthContext:
    if not raw_key:
        raise AuthError(UNAUTHORIZED)
    keys = parse_api_keys()
    client_id = keys.get(raw_key)
    if client_id is None:
        raise AuthError(UNAUTHORIZED)
    return AuthContext(api_key=raw_key, client_id=client_id, api_key_id=_mask_key(raw_key))


class AuthError(Exception):
    def __init__(self, error):
        self.error = error


async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> AuthContext:
    ctx = verify_api_key(x_api_key)
    request.state.auth = ctx
    return ctx
