"""Web dashboard: manual upload + live stats."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, JSONResponse

from service.core.config import get_settings
from service.core.queue import TaskQueue
from service.core.stats import collect_stats

router = APIRouter(tags=["dashboard"])

_STATIC = Path(__file__).resolve().parent.parent / "static" / "dashboard.html"

_queue: TaskQueue | None = None


def get_queue() -> TaskQueue:
    global _queue
    if _queue is None:
        _queue = TaskQueue()
    return _queue


def _dashboard_enabled() -> bool:
    return get_settings().dashboard_enabled


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    if not _dashboard_enabled():
        return HTMLResponse("<h1>Dashboard disabled</h1>", status_code=404)
    return HTMLResponse(_STATIC.read_text(encoding="utf-8"))


@router.get("/api/dashboard/stats")
async def dashboard_stats(queue: TaskQueue = Depends(get_queue)):
    if not _dashboard_enabled():
        return JSONResponse({"error": "disabled"}, status_code=404)
    return collect_stats(queue)
