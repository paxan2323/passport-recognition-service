"""Redis token bucket rate limiter per API key."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import redis

from service.core.config import Settings, get_settings
from service.core.errors import RATE_LIMITED


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_s: int = 0
    remaining_daily: int = 0


class RateLimiter:
    """Token bucket (RPS) + daily counter per API key."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[redis.Redis] = None):
        self.settings = settings or get_settings()
        self._redis = client or redis.from_url(
            self.settings.redis_url,
            decode_responses=True,
            socket_timeout=self.settings.redis_op_timeout_s,
        )

    def check(self, api_key_id: str) -> RateLimitResult:
        now = time.time()
        bucket_key = f"rl:bucket:{api_key_id}"
        daily_key = f"rl:daily:{api_key_id}:{time.strftime('%Y%m%d')}"

        pipe = self._redis.pipeline()
        pipe.get(bucket_key)
        pipe.ttl(bucket_key)
        pipe.get(daily_key)
        bucket_raw, ttl, daily_raw = pipe.execute()

        tokens = float(bucket_raw) if bucket_raw else float(self.settings.rate_limit_burst)
        if ttl is None or ttl < 0:
            ttl = 1

        # Refill
        elapsed = max(0, 1 - ttl / max(1, int(1 / self.settings.rate_limit_rps)))
        tokens = min(
            self.settings.rate_limit_burst,
            tokens + elapsed * self.settings.rate_limit_rps,
        )

        daily_count = int(daily_raw or 0)
        if daily_count >= self.settings.rate_limit_daily:
            return RateLimitResult(
                allowed=False,
                retry_after_s=_seconds_until_midnight(),
                remaining_daily=0,
            )

        if tokens < 1:
            retry = max(1, int(1 / self.settings.rate_limit_rps))
            return RateLimitResult(allowed=False, retry_after_s=retry)

        tokens -= 1
        pipe = self._redis.pipeline()
        pipe.set(bucket_key, str(tokens), ex=max(2, int(self.settings.rate_limit_burst / self.settings.rate_limit_rps)))
        pipe.incr(daily_key)
        pipe.expire(daily_key, _seconds_until_midnight())
        pipe.execute()

        return RateLimitResult(
            allowed=True,
            remaining_daily=self.settings.rate_limit_daily - daily_count - 1,
        )


class RateLimitExceeded(Exception):
    def __init__(self, result: RateLimitResult):
        self.result = result
        self.error = RATE_LIMITED


def _seconds_until_midnight() -> int:
    import datetime as dt

    now = dt.datetime.now()
    midnight = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((midnight - now).total_seconds())
