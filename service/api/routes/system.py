"""Health, ready, metrics, limits routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from service.core.auth import AuthContext, require_api_key
from service.core.config import get_settings
from service.core.queue import TaskQueue
from service.core.schemas import LimitsResponse

health_router = APIRouter(tags=["health"])
tasks_router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

_queue: TaskQueue | None = None


def get_queue() -> TaskQueue:
    global _queue
    if _queue is None:
        _queue = TaskQueue()
    return _queue


@health_router.get("/health")
async def health():
    return {"status": "ok"}


@health_router.get("/ready")
async def ready(queue: TaskQueue = Depends(get_queue)):
    redis_ok = queue.ping()
    ready = redis_ok
    status = 200 if ready else 503
    return Response(
        content='{"ready":' + ("true" if ready else "false") + "}",
        status_code=status,
        media_type="application/json",
    )


@health_router.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@health_router.get("/v1/limits")
async def limits(auth: AuthContext = Depends(require_api_key)):
    settings = get_settings()
    return LimitsResponse(
        client_id=auth.client_id,
        rate_limit_rps=settings.rate_limit_rps,
        rate_limit_burst=settings.rate_limit_burst,
        rate_limit_daily=settings.rate_limit_daily,
        queue_max_depth=settings.queue_max_depth,
    )


@tasks_router.get("/{task_id}")
async def get_task(
    task_id: str,
    auth: AuthContext = Depends(require_api_key),
    queue: TaskQueue = Depends(get_queue),
):
    from service.core.errors import TASK_NOT_FOUND
    from service.core.schemas import TaskStatus

    meta = queue.get_meta(task_id)
    if meta is None:
        return Response(
            status_code=TASK_NOT_FOUND.http_status,
            content=TASK_NOT_FOUND.to_body(task_id),
            media_type="application/json",
        )

    if meta.client_id != auth.client_id:
        from service.core.errors import FORBIDDEN
        return Response(
            status_code=FORBIDDEN.http_status,
            content=FORBIDDEN.to_body(task_id),
            media_type="application/json",
        )

    result = queue.fetch_result(task_id)
    if result is None:
        body = TaskStatus(
            task_id=task_id,
            request_id=meta.request_id,
            status="pending",
        )
        return Response(status_code=202, content=body.model_dump_json(), media_type="application/json")

    queue.delete_result(task_id)
    status = "completed" if result.get("success") else "failed"
    body = TaskStatus(task_id=task_id, request_id=meta.request_id, status=status, result=result)
    return body.model_dump()
