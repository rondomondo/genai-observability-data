"""
Core data models for synthetic log and trace generation.
All models are dataclasses with OTLP-compatible field names.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


class SpanKind(StrEnum):
    CLIENT = "CLIENT"
    SERVER = "SERVER"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"
    INTERNAL = "INTERNAL"


_SPAN_KIND_INT = {
    SpanKind.INTERNAL: 1,
    SpanKind.SERVER: 2,
    SpanKind.CLIENT: 3,
    SpanKind.PRODUCER: 4,
    SpanKind.CONSUMER: 5,
}


class SpanStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


_SPAN_STATUS_INT = {
    SpanStatus.UNSET: 0,
    SpanStatus.OK: 1,
    SpanStatus.ERROR: 2,
}


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"  # token budget exhausted
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class GenAIOperation(StrEnum):
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    RETRIEVE = "retrieve"
    EXECUTE_TOOL = "execute_tool"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"


class GenAIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI = "azure.ai_inference"


# ---------------------------------------------------------------------------
# Trace / Span models
# ---------------------------------------------------------------------------


def new_id(n: int = 16) -> str:
    """Generate a random hex ID of n hex chars."""
    return uuid.uuid4().hex[:n]


@dataclass
class Span:
    """
    OTLP-compatible span. Attributes follow OTel semantic conventions.
    GenAI attributes follow the experimental GenAI semconv 1.41.0.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    service_name: str
    start_time_unix_nano: int  # epoch nanoseconds
    end_time_unix_nano: int
    status: SpanStatus = SpanStatus.OK
    status_message: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    resource_attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time_unix_nano - self.start_time_unix_nano) / 1_000_000

    def to_otlp_dict(self) -> dict:
        """Serialise to OTLP JSON span format."""
        d: dict = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": _SPAN_KIND_INT[self.kind],
            "startTimeUnixNano": str(self.start_time_unix_nano),
            "endTimeUnixNano": str(self.end_time_unix_nano),
            "status": {
                "code": _SPAN_STATUS_INT[self.status],
                "message": self.status_message,
            },
            "attributes": _attrs_to_otlp(self.attributes),
            "events": [e.to_otlp_dict() for e in self.events],
        }
        if self.parent_span_id:
            d["parentSpanId"] = self.parent_span_id
        return d


@dataclass
class SpanEvent:
    name: str
    time_unix_nano: int
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_otlp_dict(self) -> dict:
        return {
            "name": self.name,
            "timeUnixNano": str(self.time_unix_nano),
            "attributes": _attrs_to_otlp(self.attributes),
        }


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)

    def root_span(self) -> Span | None:
        return next((s for s in self.spans if s.parent_span_id is None), None)

    def to_otlp_dict(self) -> dict:
        """
        Group spans by service for OTLP resourceSpans format.
        """
        by_service: dict[str, list[Span]] = {}
        for span in self.spans:
            by_service.setdefault(span.service_name, []).append(span)

        resource_spans = []
        for svc, spans in by_service.items():
            resource_spans.append(
                {
                    "resource": {
                        "attributes": _attrs_to_otlp(spans[0].resource_attributes),
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": f"synthetic.{svc}", "version": "1.0.0"},
                            "spans": [s.to_otlp_dict() for s in spans],
                        }
                    ],
                }
            )
        return {"resourceSpans": resource_spans}


# ---------------------------------------------------------------------------
# Log model
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    """
    Structured log entry. Maps to OTLP LogRecord format.
    trace_id / span_id allow correlation with trace data.
    """

    timestamp_unix_nano: int
    service_name: str
    level: LogLevel
    body: str
    trace_id: str | None = None
    span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    resource_attributes: dict[str, Any] = field(default_factory=dict)

    def to_otlp_dict(self) -> dict:
        return {
            "timeUnixNano": str(self.timestamp_unix_nano),
            "severityText": self.level.value,
            "severityNumber": _severity_number(self.level),
            "body": {"stringValue": self.body},
            "traceId": self.trace_id or "",
            "spanId": self.span_id or "",
            "attributes": _attrs_to_otlp(self.attributes),
        }

    def to_loki_dict(self) -> dict:
        """Loki push API format."""
        return {
            "stream": {
                "service": self.service_name,
                "level": self.level.value,
                **{
                    k: str(v) for k, v in self.resource_attributes.items() if k in ("environment", "region", "version")
                },
            },
            "values": [[str(self.timestamp_unix_nano), self.body]],
        }

    def to_logfmt(self) -> str:
        """Human-readable logfmt line."""
        ts_ms = self.timestamp_unix_nano // 1_000_000
        parts = [
            f"ts={ts_ms}",
            f"level={self.level.value}",
            f"service={self.service_name}",
            f'msg="{self.body}"',
        ]
        if self.trace_id:
            parts.append(f"trace_id={self.trace_id}")
        if self.span_id:
            parts.append(f"span_id={self.span_id}")
        for k, v in self.attributes.items():
            parts.append(f"{k}={v}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attrs_to_otlp(attrs: dict[str, Any]) -> list[dict]:
    """Convert a flat dict to OTLP key-value attribute list."""
    result = []
    for k, v in attrs.items():
        if isinstance(v, bool):
            result.append({"key": k, "value": {"boolValue": v}})
        elif isinstance(v, int):
            result.append({"key": k, "value": {"intValue": str(v)}})
        elif isinstance(v, float):
            result.append({"key": k, "value": {"doubleValue": v}})
        elif isinstance(v, list):
            result.append({"key": k, "value": {"arrayValue": {"values": [{"stringValue": str(i)} for i in v]}}})
        else:
            result.append({"key": k, "value": {"stringValue": str(v)}})
    return result


def _severity_number(level: LogLevel) -> int:
    return {
        LogLevel.DEBUG: 5,
        LogLevel.INFO: 9,
        LogLevel.WARN: 13,
        LogLevel.ERROR: 17,
        LogLevel.FATAL: 21,
    }[level]
