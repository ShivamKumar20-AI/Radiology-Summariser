"""Runs the real API in-process. Skipped where torch/transformers are not installed (CI)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "model" / "final").exists():
    pytest.skip("model/final not available", allow_module_level=True)

sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import src.api as api  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(api, "create_summary", lambda findings: "No acute abnormality.")
    api.per_ip_limiter = api.PerKeyLimiter(api.MAX_REQUESTS_PER_IP_PER_MIN)
    api.global_limiter = api.SlidingWindowLimiter(api.MAX_REQUESTS_ALL_PER_MIN)


client = TestClient(api.app)


def test_docs_and_schema_are_not_public():
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_oversized_and_empty_input_are_rejected_by_the_server():
    assert client.post("/get_summary", json={"findings": "x" * 5001}).status_code == 422
    assert client.post("/get_summary", json={"findings": ""}).status_code == 422
    assert client.post("/get_summary", json={"findings": "x" * 5000}).status_code == 200


def test_one_address_is_rate_limited_but_another_is_not():
    headers = {"fly-client-ip": "203.0.113.7"}
    codes = [client.post("/get_summary", json={"findings": "Lungs clear."}, headers=headers).status_code
             for _ in range(api.MAX_REQUESTS_PER_IP_PER_MIN + 1)]
    assert codes[:-1] == [200] * api.MAX_REQUESTS_PER_IP_PER_MIN
    assert codes[-1] == 429
    other = client.post("/get_summary", json={"findings": "Lungs clear."}, headers={"fly-client-ip": "198.51.100.9"})
    assert other.status_code == 200


def test_busy_service_says_so_instead_of_queueing(monkeypatch):
    for _ in range(api.MAX_CONCURRENT_SUMMARIES):
        api.work_slots.acquire()
    try:
        assert client.post("/get_summary", json={"findings": "Lungs clear."}).status_code == 503
    finally:
        for _ in range(api.MAX_CONCURRENT_SUMMARIES):
            api.work_slots.release()


def test_health_and_page_still_work():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200
