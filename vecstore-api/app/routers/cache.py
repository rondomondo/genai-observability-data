from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import get_settings
from app.dependencies import get_cache_service, get_ttl_watcher, get_vectorstore
from app.logging_config import get_logger
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

log = get_logger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


@router.post("/query", response_model=CacheLookupResponse, summary="Semantic cache lookup")
@limit_for("rate_limit_cache")
async def cache_query(
    request: Request,
    body: CacheLookupRequest,
    cache: CacheService = Depends(get_cache_service),
) -> CacheLookupResponse:
    log.debug("cache_query_request", question_len=len(body.question), threshold=body.threshold, top_k=body.top_k)
    result = await cache.lookup(body.question, body.threshold, body.top_k)
    log.info("cache_query_complete", hit=result.hit, hits_returned=len(result.hits))
    return result


@router.post("", response_model=CacheStoreResponse, summary="Store a Q&A pair in the cache")
@limit_for("rate_limit_cache")
async def cache_store(
    request: Request,
    body: CacheStoreRequest,
    cache: CacheService = Depends(get_cache_service),
) -> CacheStoreResponse:
    log.debug(
        "cache_store_request",
        question_len=len(body.question),
        answer_len=len(body.answer),
        model=body.model,
        ttl_seconds=body.ttl_seconds,
        sources_count=len(body.sources) if body.sources else 0,
    )
    result = await cache.store(
        question=body.question,
        answer=body.answer,
        sources=body.sources,
        model=body.model,
        ttl_seconds=body.ttl_seconds,
        metadata=body.metadata,
    )
    log.info("cache_store_complete", element_id=result.element_id)
    return result


@router.delete("/{element_id}", summary="Remove a cache entry by element ID")
async def cache_delete(
    element_id: str,
    cache: CacheService = Depends(get_cache_service),
) -> dict:
    log.debug("cache_delete_request", element_id=element_id)
    removed = await cache.delete(element_id)
    if not removed:
        log.warning("cache_delete_not_found", element_id=element_id)
        raise HTTPException(status_code=404, detail=f"Element '{element_id}' not found in cache")
    log.info("cache_delete_complete", element_id=element_id)
    return {"ok": True, "element_id": element_id}


@router.get("/status", response_model=CacheStatusResponse, summary="Cache status")
async def cache_status(
    vs: VectorStore = Depends(get_vectorstore),
    watcher: TTLWatcher = Depends(get_ttl_watcher),
) -> CacheStatusResponse:
    settings = get_settings()
    cardinality = vs.cardinality(settings.vset_cache)
    log.debug(
        "cache_status_request",
        key=settings.vset_cache,
        cardinality=cardinality,
        ttl_watcher_running=watcher.is_running,
        pending_expirations=watcher.pending_count,
    )
    return CacheStatusResponse(
        key=settings.vset_cache,
        cardinality=cardinality,
        ttl_watcher_running=watcher.is_running,
        pending_expirations=watcher.pending_count,
    )
