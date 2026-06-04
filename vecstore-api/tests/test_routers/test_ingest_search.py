"""Integration tests for /ingest and /search routes.

Uses a real Redis at REDIS_URL but replaces the Embedder with a fake that
returns deterministic vectors -- no Gemini API key needed.

Run with: pytest -m integration
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import redis as redis_lib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.config import Settings
from app.middleware.auth import APIKeyMiddleware
from app.middleware.correlation import CorrelationMiddleware
from app.routers import cache, ingest, search, status
from app.services.cache import CacheService
from app.services.jobs import JobStore
from app.services.summariser import Summariser
from app.services.vectorstore import VectorStore

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
DIM = 64
API_KEY = "test-key"

VSET_LOGS = "test:ingest:logs"
VSET_TRACES = "test:ingest:traces"
VSET_CACHE = "test:ingest:cache"


def _unit_vec(index: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[index % dim] = 1.0
    return v


class FakeEmbedder:
    """Returns a deterministic unit vector; no external API call."""

    queue_depth: int = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def embed(self, text: str, dim: int = DIM) -> tuple[list[float], int]:
        bucket = abs(hash(text)) % dim
        return _unit_vec(bucket, dim), 1

    async def embed_batch(self, texts: list[str], dim: int = DIM) -> list[tuple[list[float], int]]:
        return [await self.embed(t, dim) for t in texts]


def _noop_ttl_watcher() -> Any:
    watcher = MagicMock()
    watcher.is_running = True
    watcher.pending_count = 0
    return watcher


@pytest.fixture(scope="module")
def redis_client() -> redis_lib.Redis:
    r = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        r.ping()
    except redis_lib.RedisError as exc:
        pytest.skip(f"Redis not reachable at {REDIS_URL}: {exc}")
    yield r
    r.close()


@pytest.fixture(scope="module")
def test_settings() -> Settings:
    return Settings(
        api_key=API_KEY,
        redis_url=REDIS_URL,
        gemini_api_key="fake",
        embed_dim=DIM,
        vset_logs=VSET_LOGS,
        vset_traces=VSET_TRACES,
        vset_cache=VSET_CACHE,
        otel_enabled=False,
    )


@pytest.fixture(scope="module")
def integration_client(redis_client: redis_lib.Redis, test_settings: Settings) -> TestClient:
    """TestClient wired to real Redis + fake embedder, using isolated vset keys."""
    fake_embedder = FakeEmbedder()
    vs = VectorStore(redis_client)
    cache_svc = CacheService(test_settings, fake_embedder, vs, redis_client)  # type: ignore[arg-type]
    summariser = Summariser(test_settings)
    job_store = JobStore(ttl_seconds=60)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.redis = redis_client
        app.state.vectorstore = vs
        app.state.embedder = fake_embedder
        app.state.cache_service = cache_svc
        app.state.summariser = summariser
        app.state.ttl_watcher = _noop_ttl_watcher()
        app.state.job_store = job_store
        yield

    app = FastAPI(lifespan=_lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(APIKeyMiddleware, api_key=API_KEY)
    app.add_middleware(CorrelationMiddleware)
    app.include_router(ingest.router)
    app.include_router(search.router)
    app.include_router(cache.router)
    app.include_router(status.router)

    # Patch get_settings so routers pick up test vset keys / config
    with patch("app.config.get_settings", return_value=test_settings), patch(
        "app.routers.ingest.get_settings", return_value=test_settings
    ), patch("app.routers.search.get_settings", return_value=test_settings), patch(
        "app.routers.cache.get_settings", return_value=test_settings
    ), patch(
        "app.routers.status.get_settings", return_value=test_settings
    ):
        with TestClient(app) as c:
            yield c

    for key in [VSET_LOGS, VSET_TRACES, VSET_CACHE]:
        redis_client.delete(key)


HEADERS = {"X-API-Key": API_KEY}


@pytest.mark.integration
class TestIngestLog:
    def test_returns_200(self, integration_client: TestClient) -> None:
        resp = integration_client.post(
            "/ingest/log",
            json={"entry": {"message": "Connection pool exhausted", "service": "gateway", "severity": "WARN"},
                  "summarise": False},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_response_shape(self, integration_client: TestClient) -> None:
        resp = integration_client.post(
            "/ingest/log",
            json={"entry": {"message": "Auth token expired", "service": "auth", "severity": "ERROR"},
                  "summarise": False},
            headers=HEADERS,
        )
        data = resp.json()
        assert data["ok"] is True
        result = data["result"]
        assert "element_id" in result
        assert result["vset_key"] == VSET_LOGS
        assert result["summarised"] is False
        assert isinstance(result["embed_ms"], int)

    def test_persists_to_redis(self, integration_client: TestClient, redis_client: redis_lib.Redis) -> None:
        resp = integration_client.post(
            "/ingest/log",
            json={"entry": {"message": "Disk usage above 90%", "service": "monitor", "severity": "WARN"},
                  "summarise": False},
            headers=HEADERS,
        )
        element_id = resp.json()["result"]["element_id"]
        attrs = VectorStore(redis_client).get_attrs(VSET_LOGS, element_id)
        assert attrs is not None
        assert attrs["service"] == "monitor"
        assert attrs["severity"] == "WARN"

    def test_requires_auth(self, integration_client: TestClient) -> None:
        resp = integration_client.post(
            "/ingest/log",
            json={"entry": {"message": "test", "service": "svc", "severity": "INFO"}, "summarise": False},
        )
        assert resp.status_code == 401


@pytest.mark.integration
class TestIngestTrace:
    def _payload(self, trace_id: str | None = None) -> dict[str, Any]:
        tid = trace_id or f"trace-{int(time.time() * 1000)}"
        return {
            "entry": {
                "trace_id": tid,
                "root_operation": "POST /checkout",
                "root_service": "shop-api",
                "duration_ms": 320,
                "status": "ok",
                "spans": [
                    {
                        "name": "db.query",
                        "span_id": "span-001",
                        "service": "postgres",
                        "start_time_ms": 0,
                        "duration_ms": 120,
                        "status": "ok",
                    }
                ],
            },
            "summarise": False,
        }

    def test_returns_200(self, integration_client: TestClient) -> None:
        resp = integration_client.post("/ingest/trace", json=self._payload(), headers=HEADERS)
        assert resp.status_code == 200

    def test_response_shape(self, integration_client: TestClient) -> None:
        resp = integration_client.post("/ingest/trace", json=self._payload("tid-shape"), headers=HEADERS)
        data = resp.json()
        assert data["ok"] is True
        result = data["result"]
        assert result["vset_key"] == VSET_TRACES
        assert result["element_id"].startswith("trace-")
        assert result["summarised"] is False

    def test_persists_metadata(self, integration_client: TestClient, redis_client: redis_lib.Redis) -> None:
        tid = "tid-persist-check"
        resp = integration_client.post("/ingest/trace", json=self._payload(tid), headers=HEADERS)
        element_id = resp.json()["result"]["element_id"]
        attrs = VectorStore(redis_client).get_attrs(VSET_TRACES, element_id)
        assert attrs is not None
        assert attrs["trace_id"] == tid
        assert attrs["root_service"] == "shop-api"
        assert attrs["type"] == "trace"


@pytest.mark.integration
class TestSearch:
    def test_returns_results_after_ingest(self, integration_client: TestClient) -> None:
        msg = "OOM kill on worker node"
        integration_client.post(
            "/ingest/log",
            json={"entry": {"message": msg, "service": "k8s", "severity": "FATAL"}, "summarise": False},
            headers=HEADERS,
        )
        resp = integration_client.post(
            "/search",
            json={"query": msg, "keys": [VSET_LOGS], "top_k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_hits"] >= 1
        assert data["results"][0]["score"] > 0.0

    def test_response_shape(self, integration_client: TestClient) -> None:
        resp = integration_client.post(
            "/search",
            json={"query": "anything", "keys": [VSET_LOGS], "top_k": 1},
            headers=HEADERS,
        )
        data = resp.json()
        assert "query" in data
        assert "keys_searched" in data
        assert "total_hits" in data
        assert isinstance(data["results"], list)

    def test_empty_vset_returns_zero_hits(self, integration_client: TestClient) -> None:
        resp = integration_client.post(
            "/search",
            json={"query": "ghost query", "keys": ["test:nonexistent:vset"], "top_k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["total_hits"] == 0

    def test_requires_auth(self, integration_client: TestClient) -> None:
        resp = integration_client.post("/search", json={"query": "test", "top_k": 1})
        assert resp.status_code == 401


@pytest.mark.integration
class TestCacheRoutes:
    def test_store_and_lookup_hit(self, integration_client: TestClient) -> None:
        question = "What is the retry policy for failed LLM calls?"
        answer = "Exponential backoff with jitter, up to 5 retries."

        store_resp = integration_client.post(
            "/cache",
            json={"question": question, "answer": answer, "model": "gemini-2.5-flash"},
            headers=HEADERS,
        )
        assert store_resp.status_code == 200
        store_data = store_resp.json()
        assert "element_id" in store_data
        assert store_data["cardinality"] >= 1

        # Identical question produces the same vector, so score will be 1.0
        lookup_resp = integration_client.post(
            "/cache/query",
            json={"question": question, "threshold": 0.99},
            headers=HEADERS,
        )
        assert lookup_resp.status_code == 200
        lookup_data = lookup_resp.json()
        assert lookup_data["hit"] is True
        assert lookup_data["result"]["answer"] == answer

    def test_status_returns_cardinality(self, integration_client: TestClient) -> None:
        resp = integration_client.get("/cache/status", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "cardinality" in data
        assert isinstance(data["cardinality"], int)

    def test_delete_removes_entry(self, integration_client: TestClient) -> None:
        store_resp = integration_client.post(
            "/cache",
            json={"question": "Delete me please", "answer": "OK"},
            headers=HEADERS,
        )
        element_id = store_resp.json()["element_id"]
        del_resp = integration_client.delete(f"/cache/{element_id}", headers=HEADERS)
        assert del_resp.status_code == 200
