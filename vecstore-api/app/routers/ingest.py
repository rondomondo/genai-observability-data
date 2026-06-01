"""Ingest router — logs and traces, sync and async batch."""

import asyncio
import re
import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.config import get_settings
from app.dependencies import get_embedder, get_job_store, get_summariser, get_vectorstore
from app.logging_config import get_logger
from app.rate_limits import limit_for
from app.models.common import JobReference
from app.models.ingest import (
    BatchIngestResponse,
    IngestResponse,
    IngestResult,
    LogBatchIngestRequest,
    LogIngestRequest,
    TraceBatchIngestRequest,
    TraceIngestRequest,
)
from app.services.embedder import Embedder
from app.services.jobs import JobStore
from app.services.summariser import Summariser
from app.services.vectorstore import VectorStore

log = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())[:48].strip("-")
    return slug or "doc"


# ── Log endpoints ─────────────────────────────────────────────────────────────


@router.post("/log", response_model=IngestResponse, summary="Ingest a single log entry")
@limit_for("rate_limit_ingest")
async def ingest_log(
    request: Request,
    body: LogIngestRequest,
    embedder: Embedder = Depends(get_embedder),
    vs: VectorStore = Depends(get_vectorstore),
    summariser: Summariser = Depends(get_summariser),
) -> IngestResponse:
    settings = get_settings()
    entry = body.entry

    if body.summarise:
        summary, _ = await summariser.summarise_log(entry)
        text = summary.narrative
        summarised = True
    else:
        text = summariser.log_to_text(entry)
        summarised = False

    vector, embed_ms = await embedder.embed(text)
    element_id = _slug(f"{entry.service}-{entry.message[:32]}-{int(time.time())}")

    metadata: dict[str, Any] = {
        "type": "log",
        "service": entry.service,
        "severity": entry.severity,
        "message": entry.message,
        "text": text,
        "trace_id": entry.trace_id,
        "span_id": entry.span_id,
        "timestamp": entry.timestamp or int(time.time() * 1000),
        **entry.metadata,
    }

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: vs.store(settings.vset_logs, element_id, vector, metadata))

    return IngestResponse(
        result=IngestResult(
            element_id=element_id,
            vset_key=settings.vset_logs,
            summarised=summarised,
            summary=text if summarised else None,
            embed_ms=embed_ms,
        )
    )


@router.post("/log/batch", summary="Ingest a batch of log entries (sync or async)")
@limit_for("rate_limit_ingest")
async def ingest_log_batch(
    request: Request,
    body: LogBatchIngestRequest,
    embedder: Embedder = Depends(get_embedder),
    vs: VectorStore = Depends(get_vectorstore),
    summariser: Summariser = Depends(get_summariser),
    job_store: JobStore = Depends(get_job_store),
):
    settings = get_settings()

    async def _process() -> list[IngestResult]:
        results = []
        for entry in body.entries:
            if body.summarise:
                s, _ = await summariser.summarise_log(entry)
                text = s.narrative
                summarised = True
            else:
                text = summariser.log_to_text(entry)
                summarised = False

            vector, embed_ms = await embedder.embed(text)
            element_id = _slug(f"{entry.service}-{entry.message[:32]}-{int(time.time() * 1000)}")
            metadata: dict[str, Any] = {
                "type": "log",
                "service": entry.service,
                "severity": entry.severity,
                "message": entry.message,
                "text": text,
                "trace_id": entry.trace_id,
                "span_id": entry.span_id,
                "timestamp": entry.timestamp or int(time.time() * 1000),
                **entry.metadata,
            }
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda e=element_id, v=vector, m=metadata: vs.store(settings.vset_logs, e, v, m)
            )
            results.append(
                IngestResult(
                    element_id=element_id, vset_key=settings.vset_logs, summarised=summarised, embed_ms=embed_ms
                )
            )
        return results

    if body.async_mode:
        job = job_store.create(total=len(body.entries))

        async def _run_job():
            results = await _process()
            job.results = [r.model_dump() for r in results]
            job.processed = len(results)

        job_store.submit(job, _run_job)
        return JobReference(job_id=job.job_id, status_url=f"/status/job/{job.job_id}")

    results = await _process()
    return BatchIngestResponse(count=len(results), results=results)


# ── Trace endpoints ───────────────────────────────────────────────────────────


@router.post("/trace", response_model=IngestResponse, summary="Ingest a single trace")
@limit_for("rate_limit_ingest")
async def ingest_trace(
    request: Request,
    body: TraceIngestRequest,
    embedder: Embedder = Depends(get_embedder),
    vs: VectorStore = Depends(get_vectorstore),
    summariser: Summariser = Depends(get_summariser),
) -> IngestResponse:
    settings = get_settings()
    entry = body.entry

    if body.summarise:
        summary, _ = await summariser.summarise_trace(entry)
        text = summary.narrative
        summarised = True
        extra_meta: dict[str, Any] = {
            "summary_severity": summary.severity,
            "affected_services": summary.affected_services,
            "root_cause": summary.root_cause,
            "used_llm": summary.used_llm,
        }
    else:
        text = summariser.trace_to_text(entry)
        summarised = False
        extra_meta = {}

    vector, embed_ms = await embedder.embed(text)
    element_id = _slug(f"trace-{entry.trace_id}")

    metadata: dict[str, Any] = {
        "type": "trace",
        "trace_id": entry.trace_id,
        "root_operation": entry.root_operation,
        "root_service": entry.root_service,
        "duration_ms": entry.duration_ms,
        "status": entry.status,
        "text": text,
        **extra_meta,
        **entry.metadata,
    }

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: vs.store(settings.vset_traces, element_id, vector, metadata))

    return IngestResponse(
        result=IngestResult(
            element_id=element_id,
            vset_key=settings.vset_traces,
            summarised=summarised,
            summary=text if summarised else None,
            embed_ms=embed_ms,
        )
    )


@router.post("/trace/batch", summary="Ingest a batch of traces (sync or async)")
@limit_for("rate_limit_ingest")
async def ingest_trace_batch(
    request: Request,
    body: TraceBatchIngestRequest,
    embedder: Embedder = Depends(get_embedder),
    vs: VectorStore = Depends(get_vectorstore),
    summariser: Summariser = Depends(get_summariser),
    job_store: JobStore = Depends(get_job_store),
):
    settings = get_settings()

    async def _process() -> list[IngestResult]:
        results = []
        for entry in body.entries:
            if body.summarise:
                s, _ = await summariser.summarise_trace(entry)
                text = s.narrative
                summarised = True
            else:
                text = summariser.trace_to_text(entry)
                summarised = False

            vector, embed_ms = await embedder.embed(text)
            element_id = _slug(f"trace-{entry.trace_id}")
            metadata: dict[str, Any] = {
                "type": "trace",
                "trace_id": entry.trace_id,
                "root_operation": entry.root_operation,
                "root_service": entry.root_service,
                "duration_ms": entry.duration_ms,
                "status": entry.status,
                "text": text,
                **entry.metadata,
            }
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda e=element_id, v=vector, m=metadata: vs.store(settings.vset_traces, e, v, m)
            )
            results.append(
                IngestResult(
                    element_id=element_id, vset_key=settings.vset_traces, summarised=summarised, embed_ms=embed_ms
                )
            )
        return results

    if body.async_mode:
        job = job_store.create(total=len(body.entries))

        async def _run_job():
            results = await _process()
            job.results = [r.model_dump() for r in results]
            job.processed = len(results)

        job_store.submit(job, _run_job)
        return JobReference(job_id=job.job_id, status_url=f"/status/job/{job.job_id}")

    results = await _process()
    return BatchIngestResponse(count=len(results), results=results)
