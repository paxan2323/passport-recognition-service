"""Aggregate service stats for dashboard (Prometheus + Redis + worker scrape)."""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import httpx
from prometheus_client import REGISTRY

from service.core.circuit_breaker import breaker
from service.core.config import Settings, get_settings
from service.core.queue import TaskQueue

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)"
)


def _parse_labels(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def parse_prometheus_text(text: str) -> list[dict[str, Any]]:
    """Parse exposition format into flat samples."""
    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        samples.append(
            {
                "name": m.group("name"),
                "labels": _parse_labels(m.group("labels") or ""),
                "value": float(m.group("value")),
            }
        )
    return samples


def _collect_local_metrics() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            samples.append(
                {
                    "name": sample.name,
                    "labels": dict(sample.labels),
                    "value": float(sample.value),
                }
            )
    return samples


def _sum_counter(samples: list[dict[str, Any]], name: str, **labels: str) -> float:
    total = 0.0
    for s in samples:
        if s["name"] != name:
            continue
        if labels and not all(s["labels"].get(k) == v for k, v in labels.items()):
            continue
        total += s["value"]
    return total


def _get_gauge(samples: list[dict[str, Any]], name: str, default: float = 0.0) -> float:
    for s in samples:
        if s["name"] == name:
            return s["value"]
    return default


def _fetch_worker_metrics(settings: Settings) -> list[dict[str, Any]]:
    url = settings.worker_metrics_url
    if not url:
        return []
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return parse_prometheus_text(resp.text)
    except Exception:
        return []


def collect_stats(queue: Optional[TaskQueue] = None) -> dict[str, Any]:
    settings = get_settings()
    q = queue or TaskQueue(settings)

    local = _collect_local_metrics()
    worker = _fetch_worker_metrics(settings)
    merged = local + worker

    requests_total = _sum_counter(merged, "passport_requests_total")
    requests_ok = _sum_counter(merged, "passport_requests_total", status="200")
    requests_4xx = sum(
        _sum_counter(merged, "passport_requests_total", status=str(c))
        for c in (400, 401, 403, 413, 415, 422, 429)
    )
    requests_5xx = sum(
        _sum_counter(merged, "passport_requests_total", status=str(c))
        for c in (500, 503, 504)
    )

    rejections: dict[str, float] = {}
    for s in merged:
        if s["name"] == "passport_rejections_total":
            reason = s["labels"].get("reason", "unknown")
            rejections[reason] = rejections.get(reason, 0.0) + s["value"]

    worker_tasks: dict[str, float] = {}
    for s in merged:
        if s["name"] == "passport_worker_tasks_total":
            status = s["labels"].get("status", "unknown")
            worker_tasks[status] = worker_tasks.get(status, 0.0) + s["value"]

    queue_depth = q.queue_depth() if q.ping() else -1
    redis_ok = queue_depth >= 0

    return {
        "timestamp": time.time(),
        "system": {
            "redis_ok": redis_ok,
            "circuit_breaker_open": breaker.is_open(),
            "ready": redis_ok and not breaker.is_open(),
        },
        "queue": {
            "depth": max(queue_depth, 0),
            "max_depth": settings.queue_max_depth,
            "utilization_pct": round(
                100 * max(queue_depth, 0) / settings.queue_max_depth, 1
            )
            if redis_ok
            else None,
        },
        "workers": {
            "busy": int(_get_gauge(merged, "passport_workers_busy")),
            "total": int(_get_gauge(merged, "passport_workers_total")),
            "tasks": worker_tasks,
            "metrics_reachable": bool(worker),
        },
        "requests": {
            "total": int(requests_total),
            "success": int(requests_ok),
            "client_errors": int(requests_4xx),
            "server_errors": int(requests_5xx),
            "success_rate_pct": round(100 * requests_ok / requests_total, 1)
            if requests_total
            else None,
        },
        "rejections": rejections,
        "limits": {
            "rate_limit_rps": settings.rate_limit_rps,
            "rate_limit_burst": settings.rate_limit_burst,
            "rate_limit_daily": settings.rate_limit_daily,
        },
    }
