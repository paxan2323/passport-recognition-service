"""Integration test: Redis keys cleaned after processing."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import redis
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _jpeg_bytes() -> bytes:
    img = Image.new("RGB", (800, 600), color=(180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def redis_url():
    url = "redis://localhost:6379/14"
    try:
        r = redis.from_url(url)
        r.ping()
    except redis.RedisError:
        pytest.skip("Redis not available on localhost:6379")
    r.flushdb()
    yield url
    r.flushdb()


def test_no_img_result_keys_after_sync_flow(redis_url, monkeypatch):
    """After API fetches result, img:* and result:* must be gone."""
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("PASSPORT_API_KEYS", "lifecycle-key:client")
    monkeypatch.setenv("QUEUE_MAX_DEPTH", "100")

    from service.core.config import get_settings

    get_settings.cache_clear()

    from service.core.engine import RecognitionOutcome
    from service.core.queue import TaskQueue
    from service.core.schemas import TaskPayload
    from service.core.ids import new_request_id, new_task_id
    from postprocess.passport import RefinedField

    queue = TaskQueue()
    task_id = new_task_id()
    request_id = new_request_id()
    image = _jpeg_bytes()

    task = TaskPayload(
        task_id=task_id,
        request_id=request_id,
        client_id="client",
        api_key_id="****",
        mode="sync",
    )
    queue.enqueue(task, image)

    counts = queue.count_data_keys()
    assert counts["img"] >= 1

    # Simulate worker: fetch image (deletes img key), store result
    fetched = queue.fetch_image(task_id)
    assert fetched is not None
    assert queue.client.get(f"img:{task_id}") is None

    success_payload = {
        "success": True,
        "request_id": request_id,
        "processing_time_ms": 50,
        "document": {"type": "passport_rf_internal", "confidence": 0.9, "fields": {}},
    }
    queue.store_result(task_id, success_payload)

    # Simulate API sync wait + cleanup
    queue.wait_result(task_id, timeout_s=2)
    result = queue.fetch_result(task_id)
    assert result is not None
    queue.delete_result(task_id)

    final = queue.count_data_keys()
    assert final["img"] == 0
    assert final["result"] == 0


def test_worker_deletes_image_immediately(redis_url, monkeypatch):
    monkeypatch.setenv("REDIS_URL", redis_url)
    from service.core.config import get_settings

    get_settings.cache_clear()

    from service.core.queue import TaskQueue
    from service.core.schemas import TaskPayload
    from service.core.ids import new_task_id, new_request_id

    queue = TaskQueue()
    task_id = new_task_id()
    queue.enqueue(
        TaskPayload(
            task_id=task_id,
            request_id=new_request_id(),
            client_id="c",
            api_key_id="k",
        ),
        _jpeg_bytes(),
    )

    queue.fetch_image(task_id)
    assert queue.client.get(f"img:{task_id}") is None
