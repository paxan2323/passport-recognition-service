"""Prometheus metrics (plan section 11)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter(
    "passport_requests_total",
    "Total API requests",
    ["status", "error_code", "api_key_id"],
)

PROCESSING_DURATION = Histogram(
    "passport_processing_duration_seconds",
    "End-to-end processing duration",
    ["phase"],
    buckets=(0.1, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30),
)

QUEUE_DEPTH = Gauge("passport_queue_depth", "Tasks waiting in queue")

WORKERS_BUSY = Gauge("passport_workers_busy", "Busy OCR workers")
WORKERS_TOTAL = Gauge("passport_workers_total", "Total OCR workers")

RECOGNITION_CONFIDENCE = Histogram(
    "passport_recognition_confidence",
    "Overall recognition confidence",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
)

FIELD_CONFIRMED_RATIO = Gauge(
    "passport_field_confirmed_ratio",
    "Share of confirmed fields",
    ["field"],
)

REJECTIONS_TOTAL = Counter(
    "passport_rejections_total",
    "Recognition rejections",
    ["reason"],
)

WORKER_TASKS_TOTAL = Counter(
    "passport_worker_tasks_total",
    "Worker processed tasks",
    ["status"],
)
