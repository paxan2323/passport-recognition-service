"""Keep native OCR allocations in a replaceable child, never in the queue worker."""

from __future__ import annotations

import gc
import logging
import multiprocessing
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _create_engine():
    from service.core.engine import create_engine

    return create_engine()


def _child(connection, factory):
    try:
        engine = factory()
        engine.warmup()
        connection.send(("ready", None))
        while True:
            image, quality, main_only = connection.recv()
            outcome = engine.recognize(image, quality, main_only=main_only)
            connection.send(("result", outcome))
            del outcome, image
            gc.collect()
            # PyTorch/OpenCV use native allocations; GC alone cannot return these.
            try:
                import ctypes

                ctypes.CDLL(None).malloc_trim(0)
            except (AttributeError, OSError, TypeError):
                pass
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        # Never send image data, recognized fields or exception text to logs.
        logger.error("OCR child failed (%s)", type(exc).__name__)
    finally:
        connection.close()


def _container_memory_mb():
    for filename in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(Path(filename).read_text()) / (1024 * 1024)
        except (OSError, ValueError):
            continue
    return 0


class ProcessEngine:
    """One sequential child with bounded lifetime, memory and real timeouts."""

    def __init__(self, settings, factory=_create_engine):
        self.settings = settings
        self.factory = factory
        self.process = None
        self.connection = None
        self.completed = 0

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            self.process.close()
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None
        self.completed = 0

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connection.poll(min(0.1, max(0, deadline - time.monotonic()))):
                try:
                    return self.connection.recv()
                except EOFError as exc:
                    raise RuntimeError("OCR child exited") from exc
            if not self.process.is_alive():
                raise RuntimeError("OCR child exited")
            if _container_memory_mb() >= self.settings.worker_memory_limit_mb:
                raise RuntimeError("OCR memory safety limit reached")
        raise TimeoutError("OCR child timed out")

    def warmup(self):
        if self.process is not None and self.process.is_alive():
            if self.completed < self.settings.worker_max_tasks:
                return
        self.close()
        context = multiprocessing.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(target=_child, args=(child_connection, self.factory), daemon=True)
        self.process.start()
        child_connection.close()
        try:
            kind, _ = self._receive(self.settings.worker_startup_timeout_s)
            if kind != "ready":
                raise RuntimeError("OCR child did not become ready")
        except Exception:
            self.close()
            raise
        logger.info("OCR child ready pid=%s", self.process.pid)

    def recognize(self, image, min_quality="normal", *, main_only=False):
        self.warmup()
        try:
            self.connection.send((image, min_quality, main_only))
            kind, outcome = self._receive(self.settings.worker_timeout_s)
            if kind != "result":
                raise RuntimeError("Unexpected OCR child response")
            self.completed += 1
            return outcome
        except Exception:
            self.close()
            raise
