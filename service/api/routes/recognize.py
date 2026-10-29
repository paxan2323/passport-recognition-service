"""Recognition and task routes."""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from service.core.auth import AuthContext, AuthError, require_api_key
from service.core.circuit_breaker import breaker
from service.core.config import get_settings
from service.core.errors import INTERNAL, PROCESSING_TIMEOUT, QUEUE_FULL, ServiceError
from service.core.ids import new_request_id, new_task_id
from service.core.logging_config import log_event
from service.core.metrics import (
    PROCESSING_DURATION,
    QUEUE_DEPTH,
    RECOGNITION_CONFIDENCE,
    REJECTIONS_TOTAL,
    REQUESTS_TOTAL,
)
from service.core.queue import QueueFullError, TaskQueue
from service.core.rate_limit import RateLimitExceeded, RateLimiter
from service.core.schemas import AsyncTaskAccepted, TaskPayload
from service.core.validation import ValidationError, downscale_warning, validate_image, ValidatedImage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/passport", tags=["passport"])

_queue: Optional[TaskQueue] = None
_rate_limiter: Optional[RateLimiter] = None


def get_queue() -> TaskQueue:
    global _queue
    if _queue is None:
        _queue = TaskQueue()
    return _queue


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(client=get_queue().client)
    return _rate_limiter


def _error_response(error: ServiceError, request_id: str, details: Optional[dict] = None, headers: Optional[dict] = None):
    REQUESTS_TOTAL.labels(status=str(error.http_status), error_code=error.code, api_key_id="unknown").inc()
    REJECTIONS_TOTAL.labels(reason=error.code).inc()
    return JSONResponse(
        status_code=error.http_status,
        content=error.to_body(request_id, details),
        headers=headers or {},
    )


async def _read_and_validate(
    image: UploadFile,
    request_id: str,
) -> ValidatedImage:
    raw = await image.read()
    try:
        return validate_image(raw)
    except ValidationError as exc:
        raise exc


def _inject_downscale_warning(result: dict, validated: ValidatedImage) -> None:
    warning = downscale_warning(validated)
    if not warning or not result.get("success"):
        return
    doc = result.setdefault("document", {})
    warnings = doc.setdefault("warnings", [])
    warnings.append(warning)


def _enqueue_task(
    queue: TaskQueue,
    auth: AuthContext,
    request_id: str,
    validated: ValidatedImage,
    mode: str,
    webhook_url: Optional[str],
    min_quality: str,
    include_raw: bool,
    main_only: bool = False,
) -> str:
    task_id = new_task_id()
    task = TaskPayload(
        task_id=task_id,
        request_id=request_id,
        client_id=auth.client_id,
        api_key_id=auth.api_key_id,
        mode=mode,  # type: ignore[arg-type]
        webhook_url=webhook_url,
        min_quality=min_quality,
        main_only=main_only,
        include_raw=include_raw,
        enqueued_at=time.time(),
        image_scaled=validated.scaled,
        image_original_width=validated.original_width,
        image_original_height=validated.original_height,
        image_width=validated.width,
        image_height=validated.height,
    )
    queue.enqueue(task, validated.data)
    QUEUE_DEPTH.set(queue.queue_depth())
    return task_id


@router.post("/recognize")
async def recognize_sync(
    request: Request,
    image: UploadFile = File(...),
    include_raw: bool = Query(False),
    min_quality: str = Query("normal"),
    main_only: bool = Query(False),
    auth: AuthContext = Depends(require_api_key),
    queue: TaskQueue = Depends(get_queue),
    limiter: RateLimiter = Depends(get_rate_limiter),
):
    request_id = new_request_id()
    t0 = time.perf_counter()

    try:
        limit_result = limiter.check(auth.api_key_id)
        if not limit_result.allowed:
            raise RateLimitExceeded(limit_result)
    except RateLimitExceeded as exc:
        return _error_response(
            exc.error, request_id, headers={"Retry-After": str(exc.result.retry_after_s)}
        )

    if breaker.is_open():
        return _error_response(QUEUE_FULL, request_id)

    try:
        validated = await _read_and_validate(image, request_id)
    except ValidationError as exc:
        REQUESTS_TOTAL.labels(
            status=str(exc.error.http_status), error_code=exc.error.code, api_key_id=auth.api_key_id
        ).inc()
        return _error_response(exc.error, request_id, exc.details)

    try:
        task_id = _enqueue_task(
            queue, auth, request_id, validated, "sync", None, min_quality, include_raw, main_only
        )
    except QueueFullError as exc:
        return _error_response(exc.error, request_id)

    settings = get_settings()
    result = queue.wait_result(task_id, settings.api_sync_timeout_s)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if result is None:
        breaker.record(False)
        PROCESSING_DURATION.labels(phase="total").observe(elapsed_ms / 1000)
        REQUESTS_TOTAL.labels(status="504", error_code="PROCESSING_TIMEOUT", api_key_id=auth.api_key_id).inc()
        return _error_response(PROCESSING_TIMEOUT, request_id)

    breaker.record(True)
    queue.delete_result(task_id)

    status = result.get("success")
    if status is True:
        doc = result.get("document", {})
        conf = doc.get("overall_confidence", 0)
        RECOGNITION_CONFIDENCE.observe(conf)
        REQUESTS_TOTAL.labels(status="200", error_code="", api_key_id=auth.api_key_id).inc()
        PROCESSING_DURATION.labels(phase="total").observe(elapsed_ms / 1000)
        log_event(
            logger, "recognize_ok",
            request_id=request_id, task_id=task_id, api_key_id=auth.api_key_id,
            duration_ms=elapsed_ms, confidence=conf,
        )
        _inject_downscale_warning(result, validated)
        return JSONResponse(content=result)

    err = result.get("error", {})
    code = err.get("code", "INTERNAL")
    http = _http_for_code(code)
    REQUESTS_TOTAL.labels(status=str(http), error_code=code, api_key_id=auth.api_key_id).inc()
    REJECTIONS_TOTAL.labels(reason=code).inc()
    log_event(
        logger, "recognize_fail",
        request_id=request_id, task_id=task_id, api_key_id=auth.api_key_id,
        error_code=code, duration_ms=elapsed_ms,
    )
    return JSONResponse(status_code=http, content=result)


@router.post("/recognize/async", status_code=202)
async def recognize_async(
    request: Request,
    image: UploadFile = File(...),
    webhook_url: Optional[str] = Form(None),
    include_raw: bool = Query(False),
    min_quality: str = Query("normal"),
    main_only: bool = Query(False),
    auth: AuthContext = Depends(require_api_key),
    queue: TaskQueue = Depends(get_queue),
    limiter: RateLimiter = Depends(get_rate_limiter),
):
    request_id = new_request_id()

    try:
        limit_result = limiter.check(auth.api_key_id)
        if not limit_result.allowed:
            raise RateLimitExceeded(limit_result)
    except RateLimitExceeded as exc:
        return _error_response(
            exc.error, request_id, headers={"Retry-After": str(exc.result.retry_after_s)}
        )

    if breaker.is_open():
        return _error_response(QUEUE_FULL, request_id)

    try:
        validated = await _read_and_validate(image, request_id)
    except ValidationError as exc:
        return _error_response(exc.error, request_id, exc.details)

    try:
        task_id = _enqueue_task(
            queue, auth, request_id, validated, "async", webhook_url, min_quality, include_raw, main_only
        )
    except QueueFullError as exc:
        return _error_response(exc.error, request_id)

    body = AsyncTaskAccepted(
        task_id=task_id,
        request_id=request_id,
        poll_url=f"/v1/tasks/{task_id}",
    )
    REQUESTS_TOTAL.labels(status="202", error_code="", api_key_id=auth.api_key_id).inc()
    log_event(logger, "async_enqueued", request_id=request_id, task_id=task_id, api_key_id=auth.api_key_id)
    return JSONResponse(status_code=202, content=body.model_dump())


def _http_for_code(code: str) -> int:
    mapping = {
        "INVALID_IMAGE": 400,
        "FILE_TOO_LARGE": 413,
        "UNSUPPORTED_MEDIA_TYPE": 415,
        "DOCUMENT_NOT_FOUND": 422,
        "NOT_A_PASSPORT": 422,
        "LOW_QUALITY": 422,
        "OCR_FAILED": 422,
        "UNAUTHORIZED": 401,
        "FORBIDDEN": 403,
        "RATE_LIMITED": 429,
        "QUEUE_FULL": 503,
        "PROCESSING_TIMEOUT": 504,
        "INTERNAL": 500,
    }
    return mapping.get(code, 500)
