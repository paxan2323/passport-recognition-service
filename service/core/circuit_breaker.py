"""Circuit breaker for queue timeout storms."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from service.core.config import get_settings


@dataclass
class CircuitBreaker:
    """Opens when >50% outcomes are timeouts in a 60s window."""

    _outcomes: deque[tuple[float, bool]] = field(default_factory=deque)
    _lock: Lock = field(default_factory=Lock)
    _open_until: float = 0.0

    def record(self, success: bool) -> None:
        settings = get_settings()
        now = time.time()
        with self._lock:
            self._outcomes.append((now, success))
            cutoff = now - settings.circuit_breaker_window_s
            while self._outcomes and self._outcomes[0][0] < cutoff:
                self._outcomes.popleft()

            if len(self._outcomes) >= settings.circuit_breaker_min_samples:
                failures = sum(1 for _, ok in self._outcomes if not ok)
                ratio = failures / len(self._outcomes)
                if ratio >= settings.circuit_breaker_failure_ratio:
                    self._open_until = now + settings.circuit_breaker_window_s

    def is_open(self) -> bool:
        return time.time() < self._open_until

    def reset(self) -> None:
        with self._lock:
            self._outcomes.clear()
            self._open_until = 0.0


# Module-level singleton for API process
breaker = CircuitBreaker()
