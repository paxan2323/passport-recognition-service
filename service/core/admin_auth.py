"""Admin dashboard session authentication."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Cookie, HTTPException, Response

from service.core.admin_store import (
    create_admin_session,
    ensure_admin_api_key,
    revoke_admin_session,
    verify_admin_session,
)
from service.core.config import get_settings

SESSION_COOKIE = "admin_session"


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}…{key[-4:]}"


def verify_admin_password(password: str) -> bool:
    expected = get_settings().admin_password
    if not expected:
        return False
    return secrets.compare_digest(password, expected)


def set_session_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=settings.admin_cookie_secure,
        samesite="lax",
        max_age=settings.admin_session_ttl_s,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


async def require_admin_session(
    admin_session: Optional[str] = Cookie(None, alias=SESSION_COOKIE),
) -> str:
    if not verify_admin_session(admin_session):
        raise HTTPException(status_code=401, detail="Требуется вход в панель администратора")
    return admin_session  # type: ignore[return-value]


def admin_session_info() -> dict:
    api_key = ensure_admin_api_key()
    return {
        "authenticated": True,
        "api_key": api_key,
        "api_key_masked": _mask_key(api_key),
    }


def login_admin(response: Response, password: str) -> dict:
    if not verify_admin_password(password):
        raise HTTPException(status_code=401, detail="Неверный пароль")
    token = create_admin_session()
    set_session_cookie(response, token)
    return admin_session_info()


def logout_admin(response: Response, token: Optional[str]) -> dict:
    revoke_admin_session(token)
    clear_session_cookie(response)
    return {"ok": True}
