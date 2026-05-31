"""Redis vector set abstraction.

Wraps vadd / vsim / vrem / vgetattr / vsetattr behind typed methods.
All methods are synchronous (called from executor) or use redis-py async client.
We use the sync client here and call from run_in_executor — simpler than
async redis for now and matches the existing test_vecstore.py patterns.
"""

import json
import math
from typing import Any

import redis as redis_lib
from redis.commands.vectorset.commands import QuantizationOptions

from app.logging_config import get_logger

log = get_logger(__name__)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


class VectorStore:
    def __init__(self, redis_client: redis_lib.Redis) -> None:
        self._r = redis_client

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        vset_key: str,
        element_id: str,
        vector: list[float],
        metadata: dict[str, Any],
        quantization: QuantizationOptions = QuantizationOptions.Q8,
    ) -> int:
        """Store vector + metadata. Returns new cardinality."""
        self._r.vset().vadd(vset_key, vector, element_id, quantization=quantization)
        self._r.vset().vsetattr(vset_key, element_id, json.dumps(metadata, ensure_ascii=False))
        cardinality: int = self._r.execute_command("VCARD", vset_key)
        log.debug("vset_store", key=vset_key, element=element_id, cardinality=cardinality)
        return cardinality

    def remove(self, vset_key: str, element_id: str) -> bool:
        """Remove a single element. Returns True if it existed."""
        existed: bool = bool(self._r.vset().vrem(vset_key, element_id))
        log.debug("vset_remove", key=vset_key, element=element_id, existed=existed)
        return existed

    def flush_key(self, vset_key: str) -> bool:
        deleted = self._r.delete(vset_key)
        log.info("vset_flush", key=vset_key, deleted=bool(deleted))
        return bool(deleted)

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        vset_key: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked results [{element, score, attrs}]."""
        try:
            hits: dict[str, float] = self._r.vset().vsim(
                vset_key,
                query_vector,
                count=top_k,
                with_scores=True,
                filter=filter_expr,
            )
        except redis_lib.ResponseError as exc:
            if "no such key" in str(exc).lower():
                return []
            raise
        except AttributeError:
            # redis-py parses an empty/missing vset response as a list instead of
            # a dict when with_scores=True; treat as an empty result set
            return []

        results = []
        for element, score in hits.items():
            raw = self._r.vset().vgetattr(vset_key, element)
            attrs = raw if isinstance(raw, dict) else (json.loads(raw) if raw else {})
            results.append({"element": element, "score": round(float(score), 6), "attrs": attrs})
        return results

    def get_attrs(self, vset_key: str, element_id: str) -> dict[str, Any] | None:
        try:
            raw = self._r.vset().vgetattr(vset_key, element_id)
            return raw if isinstance(raw, dict) else (json.loads(raw) if raw else None)
        except redis_lib.ResponseError:
            return None

    def cardinality(self, vset_key: str) -> int:
        try:
            return int(self._r.execute_command("VCARD", vset_key))
        except redis_lib.ResponseError:
            return 0

    def list_members(self, vset_key: str, limit: int = 1000) -> list[dict[str, Any]]:
        try:
            members = self._r.execute_command("VRANDMEMBER", vset_key, limit) or []
        except redis_lib.ResponseError:
            return []
        results = []
        for m in members:
            attrs = self.get_attrs(vset_key, m) or {}
            results.append({"element": m, "attrs": attrs})
        return results

    # ── TTL shadow keys ───────────────────────────────────────────────────────

    def set_ttl(self, vset_key: str, element_id: str, ttl_seconds: int) -> None:
        shadow_key = f"ttl:{vset_key}:{element_id}"
        self._r.set(shadow_key, "1", ex=ttl_seconds)
        log.debug("vset_ttl_set", key=vset_key, element=element_id, ttl=ttl_seconds)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def norm(vector: list[float]) -> float:
        return _norm(vector)
