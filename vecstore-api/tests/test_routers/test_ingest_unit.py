"""Unit tests for /ingest router.

No real Redis or Gemini API -- all services are mocked via dependency_overrides.
Covers validation, response shape, async_mode job dispatch, and error paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder, get_job_store, get_summariser, get_vectorstore
from app.main import app
from app.models.summarise import TraceSummary
from app.services.jobs import JobStore

HEADERS = {"X-API-Key": "dev-secret-change-me"}

DIM = 64
_FAKE_VECTOR = [0.1] * DIM


def _mock_embedder() -> MagicMock:
    e = MagicMock()
    e.embed = AsyncMock(return_value=(_FAKE_VECTOR, 5))
    e.embed_batch = AsyncMock(return_value=([_FAKE_VECTOR], 5))
    return e


def _fake_summary() -> TraceSummary:
    return TraceSummary(
        narrative="Template summary",
        root_cause=None,
        affected_services=["svc"],
        severity="low",
        used_llm=False,
    )


def _mock_summariser() -> MagicMock:
    s = MagicMock()
    s.log_to_text = MagicMock(return_value="svc [INFO] message text")
    s.trace_to_text = MagicMock(return_value="svc trace text")
    s.summarise_log = AsyncMock(return_value=(_fake_summary(), None))
    s.summarise_trace = AsyncMock(return_value=(_fake_summary(), None))
    return s


def _mock_vectorstore() -> MagicMock:
    vs = MagicMock()
    vs.store = MagicMock(return_value=1)
    return vs


@pytest.fixture(autouse=True)
def override_deps():
    app.dependency_overrides[get_embedder] = lambda: _mock_embedder()
    app.dependency_overrides[get_vectorstore] = lambda: _mock_vectorstore()
    app.dependency_overrides[get_summariser] = lambda: _mock_summariser()
    app.dependency_overrides[get_job_store] = lambda: JobStore(ttl_seconds=60)
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ── /ingest/log ───────────────────────────────────────────────────────────────


class TestIngestLogUnit:
    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log",
            json={"entry": {"message": "disk full", "service": "storage", "severity": "ERROR"},
                  "summarise": False},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_element_id_in_result(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log",
            json={"entry": {"message": "started", "service": "api", "severity": "INFO"},
                  "summarise": False},
            headers=HEADERS,
        )
        assert "element_id" in resp.json()["result"]

    def test_summarise_false_does_not_call_llm(self, client: TestClient) -> None:
        mock_s = _mock_summariser()
        app.dependency_overrides[get_summariser] = lambda: mock_s

        client.post(
            "/ingest/log",
            json={"entry": {"message": "ok", "service": "svc", "severity": "INFO"}, "summarise": False},
            headers=HEADERS,
        )
        mock_s.summarise_log.assert_not_awaited()
        mock_s.log_to_text.assert_called_once()

    def test_summarise_true_calls_summarise_log(self, client: TestClient) -> None:
        mock_s = _mock_summariser()
        app.dependency_overrides[get_summariser] = lambda: mock_s

        client.post(
            "/ingest/log",
            json={"entry": {"message": "panic", "service": "svc", "severity": "FATAL"}, "summarise": True},
            headers=HEADERS,
        )
        mock_s.summarise_log.assert_awaited_once()

    def test_missing_message_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log",
            json={"entry": {"service": "svc", "severity": "INFO"}, "summarise": False},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_invalid_severity_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log",
            json={"entry": {"message": "x", "service": "svc", "severity": "VERBOSE"}, "summarise": False},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log",
            json={"entry": {"message": "x", "service": "svc", "severity": "INFO"}, "summarise": False},
        )
        assert resp.status_code == 401

    def test_vset_key_in_result(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log",
            json={"entry": {"message": "ok", "service": "svc", "severity": "INFO"}, "summarise": False},
            headers=HEADERS,
        )
        result = resp.json()["result"]
        assert "vset_key" in result
        assert "logs" in result["vset_key"]


# ── /ingest/log/batch ─────────────────────────────────────────────────────────


class TestIngestLogBatchUnit:
    def _entries(self, n: int = 2) -> list[dict]:
        return [{"message": f"msg-{i}", "service": "svc", "severity": "INFO"} for i in range(n)]

    def test_sync_batch_returns_count(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log/batch",
            json={"entries": self._entries(3), "summarise": False, "async_mode": False},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["count"] == 3
        assert len(data["results"]) == 3

    def test_async_mode_returns_job_id(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log/batch",
            json={"entries": self._entries(2), "summarise": False, "async_mode": True},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert "status_url" in data

    def test_empty_entries_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/log/batch",
            json={"entries": [], "summarise": False},
            headers=HEADERS,
        )
        assert resp.status_code == 422


# ── /ingest/trace ─────────────────────────────────────────────────────────────


class TestIngestTraceUnit:
    def _payload(self, summarise: bool = False) -> dict:
        return {
            "entry": {
                "trace_id": "tid-unit-test",
                "root_operation": "GET /items",
                "root_service": "shop",
                "duration_ms": 180,
                "status": "ok",
                "spans": [],
            },
            "summarise": summarise,
        }

    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.post("/ingest/trace", json=self._payload(), headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_element_id_prefixed_trace(self, client: TestClient) -> None:
        resp = client.post("/ingest/trace", json=self._payload(), headers=HEADERS)
        assert resp.json()["result"]["element_id"].startswith("trace-")

    def test_vset_key_contains_traces(self, client: TestClient) -> None:
        resp = client.post("/ingest/trace", json=self._payload(), headers=HEADERS)
        assert "traces" in resp.json()["result"]["vset_key"]

    def test_summarise_false_uses_trace_to_text(self, client: TestClient) -> None:
        mock_s = _mock_summariser()
        app.dependency_overrides[get_summariser] = lambda: mock_s

        client.post("/ingest/trace", json=self._payload(summarise=False), headers=HEADERS)
        mock_s.trace_to_text.assert_called_once()
        mock_s.summarise_trace.assert_not_awaited()

    def test_summarise_true_calls_summarise_trace(self, client: TestClient) -> None:
        mock_s = _mock_summariser()
        app.dependency_overrides[get_summariser] = lambda: mock_s

        client.post("/ingest/trace", json=self._payload(summarise=True), headers=HEADERS)
        mock_s.summarise_trace.assert_awaited_once()

    def test_missing_trace_id_returns_422(self, client: TestClient) -> None:
        payload = self._payload()
        del payload["entry"]["trace_id"]
        resp = client.post("/ingest/trace", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/ingest/trace", json=self._payload())
        assert resp.status_code == 401


# ── /ingest/trace/batch ───────────────────────────────────────────────────────


class TestIngestTraceBatchUnit:
    def _entry(self, tid: str) -> dict:
        return {
            "trace_id": tid,
            "root_operation": "GET /ping",
            "root_service": "ping-svc",
            "duration_ms": 10,
            "status": "ok",
        }

    def test_sync_batch_returns_results(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/trace/batch",
            json={"entries": [self._entry("t1"), self._entry("t2")], "summarise": False, "async_mode": False},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    def test_async_mode_returns_job_reference(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/trace/batch",
            json={"entries": [self._entry("t1")], "summarise": False, "async_mode": True},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
