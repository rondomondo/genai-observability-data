from fastapi import APIRouter, Depends, Request

from app.dependencies import get_summariser
from app.rate_limits import limit_for
from app.models.ingest import LogEntry, TraceEntry
from app.models.summarise import (
    SummariseLogRequest,
    SummariseResponse,
    SummariseTraceRequest,
)
from app.services.summariser import Summariser

router = APIRouter(prefix="/summarise", tags=["summarise"])


@router.post("/trace", response_model=SummariseResponse, summary="Summarise a trace (explicit endpoint)")
@limit_for("rate_limit_summarise")
async def summarise_trace(
    request: Request,
    body: SummariseTraceRequest,
    summariser: Summariser = Depends(get_summariser),
) -> SummariseResponse:
    trace = TraceEntry(
        trace_id=body.trace_id,
        root_operation=body.root_operation,
        root_service=body.root_service,
        duration_ms=body.duration_ms,
        status=body.status,  # type: ignore[arg-type]
        spans=[],
    )
    summary, model_ms = await summariser.summarise_trace(trace, force_llm=body.force_llm)
    return SummariseResponse(summary=summary, model_ms=model_ms)


@router.post("/log", response_model=SummariseResponse, summary="Summarise a log entry (explicit endpoint)")
@limit_for("rate_limit_summarise")
async def summarise_log(
    request: Request,
    body: SummariseLogRequest,
    summariser: Summariser = Depends(get_summariser),
) -> SummariseResponse:
    entry = LogEntry(
        message=body.message,
        service=body.service,
        severity=body.severity,  # type: ignore[arg-type]
        metadata=body.context,
    )
    summary, model_ms = await summariser.summarise_log(entry)
    return SummariseResponse(summary=summary, model_ms=model_ms)
