"""Extended tests for /status router.

Covers the full status response shape, vset cardinality fields,
job status endpoint (found/not found), and the auth boundary.
Uses app.dependency_overrides so no real Redis or embedder is needed.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder, get_job_store, get_ttl_watcher, get_vectorstore
from app.main import app
from app.services.jobs import Job, JobStore


def _mock_embedder(queue_depth: int = 0) -> MagicMock:
    e = MagicMock()
    e.queue_depth = queue_depth
    return e


def _mock_vectorstore(cardinality: int = 0) -> MagicMock:
    vs = MagicMock()
    vs.cardinality = MagicMock(return_value=cardinality)
    return vs


def _mock_watcher(running: bool = True, pending: int = 0) -> MagicMock:
    w = MagicMock()
    w.is_running = running
    w.pending_count = pending
    return w


HEADERS = {"X-API-Key": "abcdef"}


def _fake_redis() -> MagicMock:
    r = MagicMock()
    r.ping = MagicMock()
    r.close = MagicMock()
    return r


@pytest.fixture(autouse=True)
def override_deps():
    app.dependency_overrides[get_embedder] = lambda: _mock_embedder()
    app.dependency_overrides[get_vectorstore] = lambda: _mock_vectorstore()
    app.dependency_overrides[get_ttl_watcher] = lambda: _mock_watcher()
    app.dependency_overrides[get_job_store] = lambda: JobStore(ttl_seconds=60)
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/status/health").status_code == 200

    def test_no_auth_required(self, client: TestClient) -> None:
        resp = client.get("/status/health")
        assert resp.status_code == 200

    def test_response_has_status_ok(self, client: TestClient) -> None:
        data = client.get("/status/health").json()
        assert data.get("status") == "ok"


class TestStatusEndpoint:
    def test_requires_auth(self, client: TestClient) -> None:
        assert client.get("/status").status_code == 401

    def test_returns_200_with_auth(self, client: TestClient) -> None:
        with patch("redis.Redis.from_url", return_value=_fake_redis()):
            resp = client.get("/status", headers=HEADERS)
        assert resp.status_code == 200

    def test_response_shape(self, client: TestClient) -> None:
        with patch("redis.Redis.from_url", return_value=_fake_redis()):
            data = client.get("/status", headers=HEADERS).json()
        assert data["service"] == "vecstore-api"
        assert "version" in data
        assert isinstance(data["redis_connected"], bool)
        assert isinstance(data["embed_queue_depth"], int)
        assert isinstance(data["ttl_watcher_running"], bool)
        assert isinstance(data["vsets"], list)
        assert isinstance(data["active_jobs"], int)

    def test_vsets_has_three_entries(self, client: TestClient) -> None:
        with patch("redis.Redis.from_url", return_value=_fake_redis()):
            data = client.get("/status", headers=HEADERS).json()
        assert len(data["vsets"]) == 3
        for vset in data["vsets"]:
            assert "key" in vset
            assert "cardinality" in vset

    def test_redis_connected_false_when_ping_fails(self, client: TestClient) -> None:
        import redis as redis_lib
        r = _fake_redis()
        r.ping = MagicMock(side_effect=redis_lib.RedisError("refused"))
        with patch("redis.Redis.from_url", return_value=r):
            data = client.get("/status", headers=HEADERS).json()
        assert data["redis_connected"] is False

    def test_embed_queue_depth_reflected(self, client: TestClient) -> None:
        app.dependency_overrides[get_embedder] = lambda: _mock_embedder(queue_depth=3)
        with patch("redis.Redis.from_url", return_value=_fake_redis()):
            data = client.get("/status", headers=HEADERS).json()
        assert data["embed_queue_depth"] == 3

    def test_ttl_watcher_not_running_reflected(self, client: TestClient) -> None:
        app.dependency_overrides[get_ttl_watcher] = lambda: _mock_watcher(running=False)
        with patch("redis.Redis.from_url", return_value=_fake_redis()):
            data = client.get("/status", headers=HEADERS).json()
        assert data["ttl_watcher_running"] is False


class TestJobStatusEndpoint:
    def test_unknown_job_returns_404(self, client: TestClient) -> None:
        resp = client.get("/status/job/no-such-job", headers=HEADERS)
        assert resp.status_code == 404

    def test_known_job_returns_200(self, client: TestClient) -> None:
        store = JobStore(ttl_seconds=60)
        job = store.create(total=5)
        app.dependency_overrides[get_job_store] = lambda: store

        resp = client.get(f"/status/job/{job.job_id}", headers=HEADERS)
        assert resp.status_code == 200

    def test_job_response_shape(self, client: TestClient) -> None:
        store = JobStore(ttl_seconds=60)
        job = store.create(total=10)
        app.dependency_overrides[get_job_store] = lambda: store

        data = client.get(f"/status/job/{job.job_id}", headers=HEADERS).json()
        assert data["job_id"] == job.job_id
        assert data["status"] == "pending"
        assert data["total"] == 10
        assert data["processed"] == 0
        assert "submitted_at" in data

    def test_job_status_requires_auth(self, client: TestClient) -> None:
        store = JobStore(ttl_seconds=60)
        job = store.create(total=1)
        app.dependency_overrides[get_job_store] = lambda: store

        resp = client.get(f"/status/job/{job.job_id}")
        assert resp.status_code == 401
