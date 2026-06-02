from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import get_embedder, get_job_store, get_ttl_watcher, get_vectorstore
from app.models.status import JobStatus, StatusResponse, VSetInfo
from app.services.embedder import Embedder
from app.services.jobs import JobStore
from app.services.ttl_watcher import TTLWatcher
from app.services.vectorstore import VectorStore

router = APIRouter(prefix="/status", tags=["status"])

VERSION = "0.1.0"


@router.get("/health", summary="Liveness probe", include_in_schema=False)
async def health() -> dict[str, str]:
    """Return a simple liveness response, no auth required."""
    return {"status": "ok"}


@router.get("", response_model=StatusResponse, summary="Overall service status")
async def status(
    embedder: Embedder = Depends(get_embedder),
    vs: VectorStore = Depends(get_vectorstore),
    watcher: TTLWatcher = Depends(get_ttl_watcher),
    job_store: JobStore = Depends(get_job_store),
) -> StatusResponse:
    settings = get_settings()

    import redis as redis_lib

    r = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        r.ping()
        redis_ok = True
    except redis_lib.RedisError:
        redis_ok = False
    finally:
        r.close()

    vsets = [
        VSetInfo(key=k, cardinality=vs.cardinality(k))
        for k in [settings.vset_cache, settings.vset_logs, settings.vset_traces]
    ]

    return StatusResponse(
        service="vecstore-api",
        version=VERSION,
        redis_connected=redis_ok,
        embed_workers=settings.embed_workers,
        embed_queue_depth=embedder.queue_depth,
        embed_backend=settings.embed_backend,
        embed_model=settings.embed_model,
        ttl_watcher_running=watcher.is_running,
        vsets=vsets,
        active_jobs=job_store.active_count(),
    )


@router.get("/job/{job_id}", response_model=JobStatus, summary="Async batch job status")
async def job_status(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> JobStatus:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        completed_at=job.completed_at,
        total=job.total,
        processed=job.processed,
        errors=job.errors,
        results=job.results,
        error_detail=job.error_detail,
    )
