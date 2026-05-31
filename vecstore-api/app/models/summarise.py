from typing import Any, Literal

from pydantic import BaseModel, Field


class SummariseLogRequest(BaseModel):
    message: str = Field(..., min_length=1)
    service: str
    severity: str
    context: dict[str, Any] = Field(default_factory=dict)


class SummariseTraceRequest(BaseModel):
    trace_id: str
    root_operation: str
    root_service: str
    duration_ms: int
    status: str
    spans: list[dict[str, Any]] = Field(default_factory=list)
    force_llm: bool = Field(default=False, description="Skip template heuristic, always use LLM")


class TraceSummary(BaseModel):
    """Structured output from the pydantic-ai summariser agent."""

    narrative: str = Field(..., description="2-3 sentence plain English summary")
    root_cause: str | None = Field(default=None, description="Inferred root cause if status=error")
    affected_services: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high", "critical"] = "low"
    recommended_action: str | None = None
    used_llm: bool = False


class SummariseResponse(BaseModel):
    summary: TraceSummary
    model_ms: int | None = None
