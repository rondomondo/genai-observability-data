from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Log ingestion ─────────────────────────────────────────────────────────────


class LogEntry(BaseModel):
    message: str = Field(..., min_length=1)
    service: str = Field(..., min_length=1)
    severity: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"] = "INFO"
    timestamp: int | None = None  # unix epoch ms; defaults to now
    trace_id: str | None = None
    span_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogIngestRequest(BaseModel):
    entry: LogEntry
    summarise: bool = False  # request LLM summarisation


class LogBatchIngestRequest(BaseModel):
    entries: list[LogEntry] = Field(..., min_length=1, max_length=256)
    summarise: bool = False
    async_mode: bool = Field(default=False, description="Return job ID immediately")


# ── Trace ingestion ───────────────────────────────────────────────────────────


class Span(BaseModel):
    name: str
    span_id: str
    parent_span_id: str | None = None
    service: str
    start_time_ms: int
    duration_ms: int
    status: Literal["ok", "error", "unset"] = "unset"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)


class TraceEntry(BaseModel):
    trace_id: str
    root_operation: str
    root_service: str
    duration_ms: int
    status: Literal["ok", "error", "unset"] = "unset"
    spans: list[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceIngestRequest(BaseModel):
    entry: TraceEntry
    summarise: bool = True  # default true — traces benefit more from summarisation


class TraceBatchIngestRequest(BaseModel):
    entries: list[TraceEntry] = Field(..., min_length=1, max_length=64)
    summarise: bool = True
    async_mode: bool = False


# ── Shared response ───────────────────────────────────────────────────────────


class IngestResult(BaseModel):
    element_id: str
    vset_key: str
    summarised: bool
    summary: str | None = None
    embed_ms: int


class IngestResponse(BaseModel):
    ok: bool = True
    result: IngestResult


class BatchIngestResponse(BaseModel):
    ok: bool = True
    count: int
    results: list[IngestResult]
