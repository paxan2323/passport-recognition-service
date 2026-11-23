"""Exercise real process death, hangs and recycling without loading OCR models."""

import os
import time
from types import SimpleNamespace

import pytest

from service.worker.process_engine import ProcessEngine


class FakeEngine:
    def warmup(self):
        pass

    def recognize(self, image, quality, *, main_only=False):
        if image == b"hang":
            time.sleep(60)
        if image == b"crash":
            os._exit(9)
        return (os.getpid(), image, quality, main_only)


def fake_engine():
    return FakeEngine()


@pytest.fixture
def engine():
    settings = SimpleNamespace(worker_max_tasks=2, worker_memory_limit_mb=100000,
                               worker_startup_timeout_s=15, worker_timeout_s=2)
    worker = ProcessEngine(settings, factory=fake_engine)
    yield worker
    worker.close()


def test_recycles_only_between_documents(engine):
    first = engine.recognize(b"first", "strict", main_only=True)
    second = engine.recognize(b"second")
    third = engine.recognize(b"third")
    assert first[0] == second[0]
    assert third[0] != second[0]
    assert first[1:] == (b"first", "strict", True)
    assert third[1] == b"third"


def test_hung_document_is_killed_and_next_document_works(engine):
    engine.warmup()
    process = engine.process
    pid = process.pid
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        engine.recognize(b"hang")
    assert time.monotonic() - started < 6
    assert engine.process is None
    assert engine.recognize(b"next")[0] != pid


def test_child_crash_does_not_kill_queue_worker(engine):
    with pytest.raises(RuntimeError, match="exited"):
        engine.recognize(b"crash")
    assert engine.recognize(b"next")[1] == b"next"


def test_memory_guard_kills_child_before_container_limit(engine, monkeypatch):
    engine.warmup()
    monkeypatch.setattr("service.worker.process_engine._container_memory_mb", lambda: 4000)
    engine.settings.worker_memory_limit_mb = 3584
    with pytest.raises(RuntimeError, match="memory safety"):
        engine.recognize(b"hang")
    assert engine.process is None
    monkeypatch.setattr("service.worker.process_engine._container_memory_mb", lambda: 0)
    assert engine.recognize(b"next")[1] == b"next"
