"""Locust load test for sync recognize endpoint.

Usage:
  locust -f locustfile.py --host=http://localhost:8080
"""

from __future__ import annotations

import io
import os

from locust import HttpUser, between, task
from PIL import Image


def _sample_image() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1200, 800), color=(200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


IMAGE = _sample_image()
API_KEY = os.environ.get("PASSPORT_API_KEY", "dev-key-001")


class PassportUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def recognize(self):
        self.client.post(
            "/v1/passport/recognize",
            headers={"X-API-Key": API_KEY},
            files={"image": ("test.jpg", IMAGE, "image/jpeg")},
            name="/v1/passport/recognize",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
