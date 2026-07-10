"""Redis queue and result backend."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis

from service.core.config import Settings, get_settings
from service.core.schemas import TaskPayload

logger = logging.getLogger(__name__)

IMG_PREFIX = "img:"
RESULT_PREFIX = "result:"
META_PREFIX = "task:meta:"
NOTIFY_PREFIX = "passport:notify:"


class TaskQueue:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._redis = redis.from_url(
            self.settings.redis_url,
            decode_responses=False,
            # BLPOP in wait_result blocks up to api_sync_timeout_s (~30s).
            socket_timeout=None,
            socket_connect_timeout=self.settings.redis_op_timeout_s,
        )

    @property
    def client(self) -> redis.Redis:
        return self._redis

    def ping(self) -> bool:
        try:
            return bool(self._redis.ping())
        except redis.RedisError:
            return False

    def queue_depth(self) -> int:
        return int(self._redis.llen(self.settings.queue_key))

    def enqueue(self, task: TaskPayload, image: bytes) -> None:
        depth = self.queue_depth()
        if depth >= self.settings.queue_max_depth:
            from service.core.errors import QUEUE_FULL

            raise QueueFullError(QUEUE_FULL)

        task_id = task.task_id
        pipe = self._redis.pipeline()
        pipe.set(f"{IMG_PREFIX}{task_id}", image, ex=self.settings.img_ttl_s)
        pipe.set(
            f"{META_PREFIX}{task_id}",
            task.model_dump_json().encode(),
            ex=self.settings.result_ttl_s,
        )
        pipe.rpush(self.settings.queue_key, task_id.encode())
        pipe.execute()
        logger.debug("Enqueued task %s depth=%d", task_id, depth + 1)

    def wait_result(self, task_id: str, timeout_s: float) -> Optional[dict[str, Any]]:
        """BLPOP per-task notify list, then fetch result."""
        notify_key = f"{NOTIFY_PREFIX}{task_id}"
        item = self._redis.blpop(notify_key, timeout=timeout_s)
        if item is None:
            return None
        return self.fetch_result(task_id)

    def fetch_result(self, task_id: str) -> Optional[dict[str, Any]]:
        raw = self._redis.get(f"{RESULT_PREFIX}{task_id}")
        if raw is None:
            return None
        return json.loads(raw.decode())

    def delete_result(self, task_id: str) -> None:
        self._redis.delete(f"{RESULT_PREFIX}{task_id}", f"{META_PREFIX}{task_id}")

    def get_meta(self, task_id: str) -> Optional[TaskPayload]:
        raw = self._redis.get(f"{META_PREFIX}{task_id}")
        if raw is None:
            return None
        return TaskPayload.model_validate_json(raw)

    def store_result(self, task_id: str, payload: dict[str, Any]) -> None:
        pipe = self._redis.pipeline()
        pipe.set(
            f"{RESULT_PREFIX}{task_id}",
            json.dumps(payload, ensure_ascii=False).encode(),
            ex=self.settings.result_ttl_s,
        )
        pipe.lpush(f"{NOTIFY_PREFIX}{task_id}", b"1")
        pipe.expire(f"{NOTIFY_PREFIX}{task_id}", self.settings.result_ttl_s)
        pipe.execute()

    def fetch_image(self, task_id: str) -> Optional[bytes]:
        key = f"{IMG_PREFIX}{task_id}"
        pipe = self._redis.pipeline()
        pipe.get(key)
        pipe.delete(key)
        results = pipe.execute()
        return results[0]

    def count_data_keys(self) -> dict[str, int]:
        """Count img:* and result:* keys (for lifecycle tests)."""
        img = sum(1 for _ in self._redis.scan_iter(f"{IMG_PREFIX}*"))
        res = sum(1 for _ in self._redis.scan_iter(f"{RESULT_PREFIX}*"))
        return {"img": img, "result": res}


class QueueFullError(Exception):
    def __init__(self, error):
        self.error = error
