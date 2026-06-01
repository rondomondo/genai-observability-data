from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import get_settings
from app.dependencies import get_cache_service, get_ttl_watcher, get_vectorstore
from app.rate_limits import limit_for
from app.models.cache import (
    CacheLookupRequest,
    CacheLookupResponse,
    CacheStatusResponse,
    CacheStoreRequest,
    CacheStoreResponse,
)
from app.services.cache import CacheService
from app.services.ttl_watcher import TTLWatcher
from app.services.vectorstore import VectorStore

router = APIRouter(prefix="/cache", tags=["cache"])


@router.post("/query", response_model=CacheLookupResponse, summary="Semantic cache lookup")
@limit_for("rate_limit_cache")
async def cache_query(
    request: Request,
    body: CacheLookupRequest,
    cache: CacheService = Depends(get_cache_service),
) -> CacheLookupResponse:
    return await cache.lookup(body.question, body.threshold, body.top_k)


@router.post("", response_model=CacheStoreResponse, summary="Store a Q&A pair in the cache")
@limit_for("rate_limit_cache")
async def cache_store(
    request: Request,
    body: CacheStoreRequest,
    cache: CacheService = Depends(get_cache_service),
) -> CacheStoreResponse:
    return await cache.store(
        question=body.question,
        answer=body.answer,
        sources=body.sources,
        model=body.model,
        ttl_seconds=body.ttl_seconds,
        metadata=body.metadata,
    )


@router.delete("/{element_id}", summary="Remove a cache entry by element ID")
async def cache_delete(
    element_id: str,
    cache: CacheService = Depends(get_cache_service),
) -> dict:
    removed = await cache.delete(element_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Element '{element_id}' not found in cache")
    return {"ok": True, "element_id": element_id}


@router.get("/status", response_model=CacheStatusResponse, summary="Cache status")
async def cache_status(
    vs: VectorStore = Depends(get_vectorstore),
    watcher: TTLWatcher = Depends(get_ttl_watcher),
) -> CacheStatusResponse:
    settings = get_settings()
    return CacheStatusResponse(
        key=settings.vset_cache,
        cardinality=vs.cardinality(settings.vset_cache),
        ttl_watcher_running=watcher.is_running,
        pending_expirations=watcher.pending_count,
    )
