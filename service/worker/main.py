"""OCR worker — consumes Redis queue, runs RecognitionEngine."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Optional

import httpx
import redis

from service.core.config import get_settings
from service.core.engine import RecognitionEngine, RecognitionOutcome, create_engine
from service.core.errors import INTERNAL, LOW_QUALITY, PROCESSING_TIMEOUT, ServiceError
from service.core.logging_config import log_event, setup_logging
from service.core.metrics import (
    FIELD_CONFIRMED_RATIO,
    PROCESSING_DURATION,
    QUEUE_DEPTH,
    REJECTIONS_TOTAL,
    WORKER_TASKS_TOTAL,
    WORKERS_BUSY,
    WORKERS_TOTAL,
)
from service.core.queue import META_PREFIX, TaskQueue
from service.core.response_mapper import map_error, map_success
from service.core.schemas import TaskPayload

logger = logging.getLogger(__name__)

_running = True


def _handle_signal(signum, _frame):
    global _running
    logger.info("Signal %s received, shutting down", signum)
    _running = False


def _outcome_to_payload(
    outcome: RecognitionOutcome,
    task: TaskPayload,
    processing_ms: int,
) -> dict[str, Any]:
    if outcome.success:
        success = map_success(outcome, task.request_id, processing_ms)
        payload = success.model_dump()
        # Update field confirmed metrics
        for name, fv in success.document.fields.items():
            FIELD_CONFIRMED_RATIO.labels(field=name).set(1.0 if fv.confirmed else 0.0)
        return payload

    err = outcome.error or INTERNAL
    details = outcome.error_details
    body = map_error(err.code, err.message, task.request_id, details)
    REJECTIONS_TOTAL.labels(reason=err.code).inc()
    return body.model_dump()


def _send_webhook(url: str, payload: dict[str, Any], timeout: float) -> None:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            log_event(logger, "webhook_sent", status=resp.status_code)
    except Exception as exc:
        log_event(logger, "webhook_failed", message=str(exc))


def process_task(
    engine: RecognitionEngine,
    queue: TaskQueue,
    task_id: str,
    settings,
) -> None:
    meta_raw = queue.client.get(f"{META_PREFIX}{task_id}")
    if meta_raw is None:
        logger.warning("No meta for task %s", task_id)
        return
    task = TaskPayload.model_validate_json(meta_raw)

    image = queue.fetch_image(task_id)
    if image is None:
        payload = map_error("INTERNAL", "Image blob missing", task.request_id).model_dump()
        queue.store_result(task_id, payload)
        return

    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(engine.recognize, image, task.min_quality)
            try:
                outcome: RecognitionOutcome = future.result(timeout=settings.worker_timeout_s)
            except FuturesTimeout:
                payload = map_error(
                    PROCESSING_TIMEOUT.code,
                    PROCESSING_TIMEOUT.message,
                    task.request_id,
                ).model_dump()
                queue.store_result(task_id, payload)
                WORKER_TASKS_TOTAL.labels(status="timeout").inc()
                return
    except Exception:
        logger.exception("Worker error task_id=%s", task_id)
        payload = map_error(INTERNAL.code, INTERNAL.message, task.request_id).model_dump()
        queue.store_result(task_id, payload)
        WORKER_TASKS_TOTAL.labels(status="error").inc()
        return
    finally:
        del image

    processing_ms = int((time.perf_counter() - t0) * 1000)
    if outcome.timings:
        for phase, ms in outcome.timings.items():
            PROCESSING_DURATION.labels(phase=phase.replace("_ms", "")).observe(ms / 1000)

    payload = _outcome_to_payload(outcome, task, processing_ms)
    if task.image_scaled and payload.get("success"):
        from service.core.validation import ValidatedImage, downscale_warning

        warning = downscale_warning(ValidatedImage(
            data=b"",
            width=task.image_width or 0,
            height=task.image_height or 0,
            format="jpeg",
            scaled=True,
            original_width=task.image_original_width,
            original_height=task.image_original_height,
        ))
        if warning:
            payload.setdefault("document", {}).setdefault("warnings", []).append(warning)
    queue.store_result(task_id, payload)
    WORKER_TASKS_TOTAL.labels(status="ok" if outcome.success else "reject").inc()

    log_event(
        logger, "task_done",
        task_id=task_id, request_id=task.request_id,
        duration_ms=processing_ms,
        status="ok" if outcome.success else payload.get("error", {}).get("code"),
    )

    if task.webhook_url and task.mode == "async":
        _send_webhook(task.webhook_url, payload, settings.webhook_timeout_s)


def run_worker() -> int:
    settings = get_settings()
    setup_logging(settings.log_level, settings.json_logs)
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.omp_num_threads))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Worker %s starting", settings.worker_id)
    WORKERS_TOTAL.set(1)

    engine = create_engine()
    engine.warmup()

    queue = TaskQueue(settings)
    if not queue.ping():
        logger.error("Redis unavailable")
        return 1

    from prometheus_client import start_http_server

    start_http_server(9091)
    logger.info("Metrics on :9091")

    logger.info("Worker ready, listening on %s", settings.queue_key)

    while _running:
        try:
            item = queue.client.blpop(settings.queue_key, timeout=2)
        except redis.RedisError as exc:
            logger.error("Redis error: %s", exc)
            time.sleep(1)
            continue

        if item is None:
            continue

        _, task_id_raw = item
        task_id = task_id_raw.decode() if isinstance(task_id_raw, bytes) else task_id_raw
        WORKERS_BUSY.set(1)
        QUEUE_DEPTH.set(queue.queue_depth())

        try:
            process_task(engine, queue, task_id, settings)
        finally:
            WORKERS_BUSY.set(0)
            QUEUE_DEPTH.set(queue.queue_depth())

    logger.info("Worker stopped")
    return 0


def main() -> int:
    return run_worker()


if __name__ == "__main__":
    sys.exit(main())
