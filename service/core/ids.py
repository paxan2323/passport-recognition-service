"""Request/task ID generation."""

from __future__ import annotations

import secrets
import time


def new_request_id() -> str:
    ts = int(time.time())
    return f"req_{ts:x}{secrets.token_hex(4)}"


def new_task_id() -> str:
    return f"task_{secrets.token_hex(12)}"
