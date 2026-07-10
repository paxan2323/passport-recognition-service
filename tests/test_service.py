"""Service unit tests (no OCR models)."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postprocess.passport import RefinedField
from service.core.auth import verify_api_key, AuthError
from service.core.circuit_breaker import CircuitBreaker
from service.core.config import parse_api_keys
from service.core.engine import RecognitionOutcome
from service.core.errors import UNAUTHORIZED
from service.core.ids import new_request_id, new_task_id
from service.core.response_mapper import map_success
from service.core.validation import ValidationError, validate_image


def _make_jpeg(w: int = 1000, h: int = 700) -> bytes:
    img = Image.new("RGB", (w, h), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestAuth:
    def test_valid_key(self, monkeypatch):
        monkeypatch.setenv("PASSPORT_API_KEYS", "dev-key-001:default")
        from service.core.config import get_settings
        get_settings.cache_clear()
        ctx = verify_api_key("dev-key-001")
        assert ctx.client_id == "default"

    def test_invalid_key(self):
        with pytest.raises(AuthError) as exc:
            verify_api_key("bad-key")
        assert exc.value.error == UNAUTHORIZED


class TestValidation:
    def test_valid_jpeg(self):
        data = _make_jpeg()
        result = validate_image(data)
        assert result.format == "jpeg"
        assert len(result.data) > 0

    def test_rejects_svg(self):
        svg = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
        with pytest.raises(ValidationError) as exc:
            validate_image(svg)
        assert exc.value.error.code == "UNSUPPORTED_MEDIA_TYPE"

    def test_downscales_large_image(self):
        from unittest.mock import patch

        large = _make_jpeg(3200, 2400)
        with patch("service.core.validation.get_settings") as mock_settings:
            mock_settings.return_value.max_image_bytes = 10 * 1024 * 1024
            mock_settings.return_value.max_decode_side_px = 8000
            mock_settings.return_value.max_image_side_px = 1600
            result = validate_image(large)
        assert result.scaled
        assert result.original_width == 3200
        assert max(result.width, result.height) == 1600
        assert len(result.data) > 0

    def test_keeps_small_image(self):
        small = _make_jpeg(1200, 900)
        result = validate_image(small)
        assert not result.scaled
        assert result.width == 1200
        assert result.height == 900


class TestResponseMapper:
    def test_series_number_split(self):
        doc = MagicMock()
        doc.doc_confidence = 0.9
        doc.zones = []

        refined = {
            "series_number": RefinedField("03 20 579029", "visual", True),
            "surname": RefinedField("ПИВЕНЬ", "visual", True),
            "name": RefinedField("АНДРЕЙ", "visual", True),
            "patronymic": RefinedField("АНДРЕЕВИЧ", "visual", True),
            "birth_date": RefinedField("25.06.2000", "visual", True),
            "sex": RefinedField("МУЖ.", "visual", True),
            "birth_place": RefinedField("Г. КРАСНОДАР", "visual", False),
            "issue_date": RefinedField("08.07.2020", "mrz", True),
            "issue_code": RefinedField("230-002", "visual", True),
            "issued_by": RefinedField("ГУ МВД", "lookup", True),
        }

        outcome = RecognitionOutcome(success=True, doc=doc, refined=refined, warnings=[])
        result = map_success(outcome, "req_test", 100)

        assert result.document.fields["series"].value == "03 20"
        assert result.document.fields["number"].value == "579029"
        assert result.document.fields["registration_address"].value is None
        assert any(w.code.value == "FIELD_NOT_FOUND" for w in result.document.warnings)
        assert result.document.overall_confidence > 0


class TestEngineDocPick:
    def test_pick_passport_among_false_positives(self):
        from unittest.mock import MagicMock

        from service.core.engine import DocReaderEngine

        docs = [
            MagicMock(doc_type="attestat/diplom", doc_confidence=0.68, zones=[]),
            MagicMock(doc_type="passport", doc_confidence=0.61, zones=[1] * 7),
        ]
        page = MagicMock(documents=docs)
        picked = DocReaderEngine._pick_passport_doc(page)
        assert picked.doc_type == "passport"

    def test_skip_registration_on_main_spread(self):
        from unittest.mock import MagicMock

        from service.core.engine import DocReaderEngine
        from service.core.config import get_settings

        settings = get_settings()
        zone = MagicMock()
        zone.name = "passport_num"
        doc = MagicMock(
            doc_type="passport",
            doc_confidence=0.9,
            zones=[zone],
        )
        assert DocReaderEngine._looks_like_main_passport(doc, settings)

        reg_only = MagicMock(doc_type="passport", doc_confidence=0.9, zones=[])
        assert not DocReaderEngine._looks_like_main_passport(reg_only, settings)


class TestCircuitBreaker:
    def test_opens_on_failures(self):
        cb = CircuitBreaker()
        with patch("service.core.circuit_breaker.get_settings") as mock:
            mock.return_value.circuit_breaker_window_s = 60
            mock.return_value.circuit_breaker_failure_ratio = 0.5
            mock.return_value.circuit_breaker_min_samples = 4
            for _ in range(3):
                cb.record(False)
            assert not cb.is_open()
            cb.record(False)
            assert cb.is_open()


class TestIds:
    def test_unique_ids(self):
        assert new_request_id() != new_request_id()
        assert new_task_id() != new_task_id()


class TestApiEndpoints:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("PASSPORT_API_KEYS", "test-key:client1")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
        monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-pass")

        from service.core.config import get_settings

        get_settings.cache_clear()

        from service.api.main import create_app

        return TestClient(create_app())

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_unauthorized(self, client):
        r = client.post("/v1/passport/recognize")
        assert r.status_code == 401

    def test_metrics(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert b"passport_requests_total" in r.content

    def test_dashboard_page(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "Распознавание паспортов" in r.text

    def test_dashboard_stats(self, client, monkeypatch):
        from service.api.routes import dashboard as dash_mod
        from service.core.config import get_settings

        monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-pass")
        get_settings.cache_clear()

        monkeypatch.setattr(
            dash_mod,
            "collect_stats",
            lambda queue=None: {"system": {"ready": True}, "queue": {"depth": 0}},
        )
        r = client.get("/api/dashboard/stats")
        assert r.status_code == 401

        login = client.post("/api/admin/login", json={"password": "test-admin-pass"})
        assert login.status_code == 200
        r = client.get("/api/dashboard/stats")
        assert r.status_code == 200
        assert r.json()["system"]["ready"] is True

    def test_admin_login_and_regenerate(self, client, monkeypatch):
        from service.core.auth import verify_api_key

        bad = client.post("/api/admin/login", json={"password": "wrong"})
        assert bad.status_code == 401

        ok = client.post("/api/admin/login", json={"password": "test-admin-pass"})
        assert ok.status_code == 200
        assert ok.json()["api_key"]

        regen = client.post("/api/admin/regenerate-api-key")
        assert regen.status_code == 200
        new_key = regen.json()["api_key"]
        assert new_key != ok.json()["api_key"]

        ctx = verify_api_key(new_key)
        assert ctx.client_id == "admin"


class TestStatsParser:
    def test_parse_prometheus(self):
        from service.core.stats import parse_prometheus_text

        text = '# HELP x\npassport_queue_depth 3\npassport_requests_total{status="200"} 42\n'
        samples = parse_prometheus_text(text)
        assert any(s["name"] == "passport_queue_depth" and s["value"] == 3 for s in samples)
        assert any(s["name"] == "passport_requests_total" and s["value"] == 42 for s in samples)


class TestLoggingNoPii:
    def test_json_formatter_blocks_pii_fields(self):
        from service.core.logging_config import JsonFormatter
        import logging

        fmt = JsonFormatter()
        record = logging.LogRecord("t", logging.INFO, "", 0, "ok", (), None)
        record.extra_fields = {
            "request_id": "req_1",
            "surname": "SECRET",
            "duration_ms": 100,
        }
        parsed = json.loads(fmt.format(record))
        assert "surname" not in parsed
        assert parsed["request_id"] == "req_1"
