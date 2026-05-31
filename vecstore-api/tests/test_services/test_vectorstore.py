"""Integration tests for VectorStore against a real Redis instance.

Requires Redis running at REDIS_URL (default: redis://localhost:6379).
Run with: pytest -m integration
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest
import redis as redis_lib

from app.services.vectorstore import VectorStore

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
VSET_KEY = "test:vecstore:integration"
DIM = 64


def _unit_vec(index: int, dim: int = DIM) -> list[float]:
    """Deterministic unit vector with energy concentrated at `index`."""
    v = [0.0] * dim
    v[index % dim] = 1.0
    return v


def _sample_metadata() -> list[dict[str, Any]]:
    """Return 5 realistic log-event metadata dicts for use in tests."""
    return [
        {"service": "auth", "severity": "ERROR", "latency_ms": 342, "region": "eu-west-1", "retries": 2},
        {"service": "gateway", "severity": "WARN", "latency_ms": 87, "region": "us-east-1", "retries": 0},
        {"service": "embedder", "severity": "INFO", "latency_ms": 1204, "region": "eu-west-1", "retries": 1},
        {"service": "ingest", "severity": "DEBUG", "latency_ms": 15, "region": "ap-southeast-1", "retries": 0},
        {"service": "vecstore", "severity": "ERROR", "latency_ms": 9800, "region": "us-east-1", "retries": 3},
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


@pytest.fixture(scope="module")
def redis_client() -> redis_lib.Redis:
    r = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        r.ping()
        r.delete(VSET_KEY)
        r.ping()
        
    except redis_lib.RedisError as exc:
        pytest.skip(f"Redis not reachable at {REDIS_URL}: {exc}")
    yield r
    r.close()


# @pytest.fixture(autouse=True)
# def clean_vsets(redis_client: redis_lib.Redis) -> None:
#     redis_client.delete(VSET_KEY)
#     yield
#     redis_client.delete(VSET_KEY)


@pytest.fixture(autouse=True)
def clean_vset(redis_client: redis_lib.Redis, request: pytest.FixtureRequest) -> None:
    #redis_client.delete(VSET_KEY)
    yield
    marker = request.node.get_closest_marker("keep_vset")
    if marker:
        import time
        time.sleep(marker.args[0])
    
    redis_client.delete(VSET_KEY)

@pytest.fixture()
def vs(redis_client: redis_lib.Redis) -> VectorStore:
    return VectorStore(redis_client)


@pytest.mark.integration
class TestVectorStoreBasics:
    def test_store_returns_cardinality(self, vs: VectorStore) -> None:
        card = vs.store(VSET_KEY, "elem-0", _unit_vec(0), {"label": "zero"})
        assert card == 1
        card = vs.store(VSET_KEY, "elem-1", _unit_vec(1), {"label": "one"})
        assert card == 2

    def test_cardinality_empty_key(self, vs: VectorStore) -> None:
        assert vs.cardinality(VSET_KEY) == 0

    def test_cardinality_after_store(self, vs: VectorStore) -> None:
        vs.store(VSET_KEY, "elem-0", _unit_vec(0), {})
        vs.store(VSET_KEY, "elem-1", _unit_vec(1), {})
        assert vs.cardinality(VSET_KEY) == 2

    def test_get_attrs_round_trips_metadata(self, vs: VectorStore) -> None:
        meta = {"service": "auth", "severity": "ERROR", "count": 3}
        vs.store(VSET_KEY, "elem-0", _unit_vec(0), meta)
        result = vs.get_attrs(VSET_KEY, "elem-0")
        assert result is not None
        assert result["service"] == "auth"
        assert result["severity"] == "ERROR"
        assert result["count"] == 3

    def test_get_attrs_missing_element_returns_none(self, vs: VectorStore) -> None:
        vs.store(VSET_KEY, "elem-0", _unit_vec(0), {})
        assert vs.get_attrs(VSET_KEY, "no-such-element") is None

    def test_remove_existing_element(self, vs: VectorStore) -> None:
        vs.store(VSET_KEY, "elem-0", _unit_vec(0), {})
        assert vs.cardinality(VSET_KEY) == 1
        removed = vs.remove(VSET_KEY, "elem-0")
        assert removed is True
        assert vs.cardinality(VSET_KEY) == 0

    def test_remove_nonexistent_element_returns_false(self, vs: VectorStore) -> None:
        vs.store(VSET_KEY, "elem-0", _unit_vec(0), {})
        result = vs.remove(VSET_KEY, "no-such")
        assert result is False

    def test_flush_key(self, vs: VectorStore) -> None:
        vs.store(VSET_KEY, "elem-0", _unit_vec(0), {})
        vs.store(VSET_KEY, "elem-1", _unit_vec(1), {})
        flushed = vs.flush_key(VSET_KEY)
        assert flushed is True
        assert vs.cardinality(VSET_KEY) == 0


@pytest.mark.integration
class TestVectorStoreSearch:
    def test_search_empty_key_returns_empty(self, vs: VectorStore) -> None:
        results = vs.search(VSET_KEY, _unit_vec(0))
        assert results == []

    def test_search_finds_nearest_neighbour(self, vs: VectorStore) -> None:
        for i in range(4):
            vs.store(VSET_KEY, f"elem-{i}", _unit_vec(i), {"idx": i})
        results = vs.search(VSET_KEY, _unit_vec(2), top_k=1)
        assert len(results) == 1
        assert results[0]["element"] == "elem-2"
        assert results[0]["score"] > 0.99

    def test_search_top_k_limits_results(self, vs: VectorStore) -> None:
        for i in range(6):
            vs.store(VSET_KEY, f"elem-{i}", _unit_vec(i), {"idx": i})
        results = vs.search(VSET_KEY, _unit_vec(0), top_k=3)
        assert len(results) <= 3

    def test_search_includes_attrs_in_results(self, vs: VectorStore) -> None:
        vs.store(VSET_KEY, "elem-0", _unit_vec(0), {"service": "gateway", "severity": "WARN"})
        results = vs.search(VSET_KEY, _unit_vec(0), top_k=1)
        assert len(results) == 1
        attrs = results[0]["attrs"]
        assert attrs.get("service") == "gateway"
        assert attrs.get("severity") == "WARN"

    def test_search_scores_are_floats_between_0_and_1(self, vs: VectorStore) -> None:
        for i in range(3):
            vs.store(VSET_KEY, f"elem-{i}", _unit_vec(i), {})
        results = vs.search(VSET_KEY, _unit_vec(0), top_k=3)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


@pytest.mark.integration
class TestVectorStoreTTLShadowKeys:
    @pytest.mark.keep_vset(10)
    def test_set_ttl_creates_shadow_key(self, vs: VectorStore, redis_client: redis_lib.Redis) -> None:
        _sm = _sample_metadata()
        vs.store(VSET_KEY, "elem-creates_shadow_key", _unit_vec(0), _sm[0])
        vs.set_ttl(VSET_KEY, "elem-creates_shadow_key", ttl_seconds=5)
        shadow = f"ttl:{VSET_KEY}:elem-creates_shadow_key"
        ttl = redis_client.ttl(shadow)
        assert ttl > 0

    @pytest.mark.keep_vset(10)
    def test_set_ttl_shadow_key_expires(self, vs: VectorStore, redis_client: redis_lib.Redis) -> None:
        _sm = _sample_metadata()
        vs.store(VSET_KEY, "elem-shadow_key_expires", _unit_vec(0), _sm[0])
        vs.set_ttl(VSET_KEY, "elem-shadow_key_expires", ttl_seconds=5)
        shadow = f"ttl:{VSET_KEY}:elem-shadow_key_expires"
        assert redis_client.exists(shadow)
        ttl = redis_client.ttl(shadow)
        assert 4 < ttl <= 5
