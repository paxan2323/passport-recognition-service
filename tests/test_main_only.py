"""Main-spread checks must not run expensive registration-page fallback."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from service.api.routes.recognize import _enqueue_task
from service.core.auth import AuthContext
from service.core.engine import DocReaderEngine
from service.core.errors import DOCUMENT_NOT_FOUND
from service.core.schemas import TaskPayload
from service.core.validation import ValidatedImage


def test_main_only_rejects_missing_document_without_registration():
    engine = DocReaderEngine()
    engine._ready = True
    engine._reader = Mock()
    engine._try_registration = Mock(side_effect=AssertionError("Must skip registration"))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    page = SimpleNamespace(documents=[])
    with (
        patch("postprocess.orientation.decode_image", return_value=image),
        patch("postprocess.orientation.process_with_rotation", return_value=(page, image, 0)),
    ):
        result = engine.recognize(b"image", main_only=True)
    assert not result.success
    assert result.error == DOCUMENT_NOT_FOUND
    engine._try_registration.assert_not_called()


def test_default_mode_preserves_registration_fallback():
    engine = DocReaderEngine()
    engine._ready = True
    engine._reader = Mock()
    engine._try_registration = Mock(return_value=None)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    page = SimpleNamespace(documents=[])
    with (
        patch("postprocess.orientation.decode_image", return_value=image),
        patch("postprocess.orientation.process_with_rotation", return_value=(page, image, 0)),
    ):
        engine.recognize(b"image")
    engine._try_registration.assert_called_once()


def test_main_only_flag_is_forwarded_to_queue():
    queue = Mock()
    queue.queue_depth.return_value = 1
    auth = AuthContext(api_key="test", client_id="test", api_key_id="***")
    image = ValidatedImage(data=b"image", width=100, height=100, format="JPEG")
    _enqueue_task(queue, auth, "request-test", image, "sync", None, "normal", False, True)
    task, _ = queue.enqueue.call_args.args
    assert task.main_only is True


def test_old_tasks_keep_default_mode():
    task = TaskPayload(task_id="test", request_id="test", client_id="test", api_key_id="test")
    assert task.main_only is False
