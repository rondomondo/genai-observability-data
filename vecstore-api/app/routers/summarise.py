from fastapi import APIRouter, Depends

from app.dependencies import get_summariser
from app.models.ingest import LogEntry, TraceEntry
from app.models.summarise import (
    SummariseLogRequest,
    SummariseResponse,
    SummariseTraceRequest,
)
from app.services.summariser import Summariser

router = APIRouter(prefix="/summarise", tags=["summarise"])


@router.post("/trace", response_model=SummariseResponse, summary="Summarise a trace (explicit endpoint)")
async def summarise_trace(
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
async def summarise_log(
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
