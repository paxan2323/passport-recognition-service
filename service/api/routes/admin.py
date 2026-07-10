"""Admin dashboard: login, API key management."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from service.core.admin_auth import (
    SESSION_COOKIE,
    admin_session_info,
    login_admin,
    logout_admin,
    require_admin_session,
)
from service.core.admin_store import regenerate_admin_api_key, verify_admin_session

router = APIRouter(prefix="/api/admin", tags=["admin"])


class LoginBody(BaseModel):
    password: str = Field(min_length=1)


@router.post("/login")
async def admin_login(body: LoginBody, response: Response):
    return login_admin(response, body.password)


@router.post("/logout")
async def admin_logout(
    response: Response,
    admin_session: Optional[str] = Cookie(None, alias=SESSION_COOKIE),
):
    return logout_admin(response, admin_session)


@router.get("/me")
async def admin_me(admin_session: Optional[str] = Cookie(None, alias=SESSION_COOKIE)):
    if not verify_admin_session(admin_session):
        raise HTTPException(status_code=401, detail="Не авторизован")
    return admin_session_info()


@router.post("/regenerate-api-key")
async def admin_regenerate_api_key(
    response: Response,
    _session: str = Depends(require_admin_session),
):
    key = regenerate_admin_api_key()
    from service.core.admin_auth import _mask_key

    return {
        "ok": True,
        "api_key": key,
        "api_key_masked": _mask_key(key),
        "message": "Новый ключ активен. Старый ключ больше не работает.",
    }
