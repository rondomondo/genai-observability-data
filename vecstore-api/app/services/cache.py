"""Semantic cache service.

Wraps vector store + embedder into a cache-oriented API.
Handles hit/miss logic, TTL shadow keys, and metadata normalisation.
"""

import re
import time
from typing import Any

import redis as redis_lib

from app.config import Settings
from app.logging_config import get_logger
from app.models.cache import CacheHit, CacheLookupResponse, CacheStoreResponse
from app.services.embedder import Embedder
from app.services.vectorstore import VectorStore

log = get_logger(__name__)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())[:48].strip("-")
    return slug or "doc"


class CacheService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vectorstore: VectorStore,
        redis_client: redis_lib.Redis,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vs = vectorstore
        self._r = redis_client

    async def lookup(
        self,
        question: str,
        threshold: float | None = None,
        top_k: int = 1,
    ) -> CacheLookupResponse:
        effective_threshold = threshold if threshold is not None else self._settings.cache_hit_threshold
        vector, embed_ms = await self._embedder.embed(question)

        import asyncio

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._vs.search(self._settings.vset_cache, vector, top_k=top_k),
        )

        hit = False
        result: CacheHit | None = None
        score: float | None = None

        if results:
            top = results[0]
            score = top["score"]
            if score >= effective_threshold:
                hit = True
                attrs = top["attrs"]
                result = CacheHit(
                    element_id=top["element"],
                    score=score,
                    question=attrs.get("question", ""),
                    answer=attrs.get("answer", ""),
                    sources=attrs.get("sources", []),
                    model=attrs.get("model"),
                    cached_at=attrs.get("cached_at", 0),
                    metadata=attrs.get("metadata", {}),
                )
                log.info("cache_hit", element=top["element"], score=score)
            else:
                log.debug("cache_miss", top_score=score, threshold=effective_threshold)

        return CacheLookupResponse(
            hit=hit,
            score=score,
            threshold=effective_threshold,
            result=result,
            embed_ms=embed_ms,
        )

    async def store(
        self,
        question: str,
        answer: str,
        sources: list[str] | None = None,
        model: str | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheStoreResponse:
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._settings.cache_ttl_seconds
        vector, embed_ms = await self._embedder.embed(question)

        element_id = _slug(question)
        attrs: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "model": model,
            "cached_at": int(time.time()),
            "metadata": metadata or {},
        }

        import asyncio

        loop = asyncio.get_running_loop()

        def _write() -> int:
            cardinality = self._vs.store(self._settings.vset_cache, element_id, vector, attrs)
            self._vs.set_ttl(self._settings.vset_cache, element_id, effective_ttl)
            return cardinality

        cardinality = await loop.run_in_executor(None, _write)
        log.info("cache_stored", element=element_id, ttl=effective_ttl, cardinality=cardinality)

        return CacheStoreResponse(
            element_id=element_id,
            key=self._settings.vset_cache,
            cardinality=cardinality,
            ttl_seconds=effective_ttl,
            embed_ms=embed_ms,
        )

    async def delete(self, element_id: str) -> bool:
        import asyncio

        loop = asyncio.get_running_loop()
        removed = await loop.run_in_executor(
            None,
            lambda: self._vs.remove(self._settings.vset_cache, element_id),
        )
        if removed:
            # Also clean up shadow TTL key
            shadow = f"ttl:{self._settings.vset_cache}:{element_id}"
            self._r.delete(shadow)
        return removed
