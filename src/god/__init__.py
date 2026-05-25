"""god — GenAI Observability Data generator.

Generates realistic distributed traces and logs with correlated failures
and OTel GenAI semantic conventions (semconv 1.41.0, experimental).
"""

__version__ = "0.1.0"

from god.models import (
    FinishReason,
    GenAIOperation,
    GenAIProvider,
    LogEntry,
    LogLevel,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    Trace,
    new_id,
)
from god.orchestrator import generate_dataset, generate_trace
from god.scenarios import SCENARIO_PRESETS, Scenario, ScenarioEngine
from god.services import DEFAULT_SERVICES, LatencyProfile, ServiceDef, ServiceRegistry, ServiceType
from god.sinks import FileSink, MultiSink, OTLPHttpSink, Sink, StdoutSink

__all__ = [
    "__version__",
    "FinishReason",
    "GenAIOperation",
    "GenAIProvider",
    "LogEntry",
    "LogLevel",
    "Span",
    "SpanEvent",
    "SpanKind",
    "SpanStatus",
    "Trace",
    "new_id",
    "generate_dataset",
    "generate_trace",
    "SCENARIO_PRESETS",
    "Scenario",
    "ScenarioEngine",
    "DEFAULT_SERVICES",
    "LatencyProfile",
    "ServiceDef",
    "ServiceRegistry",
    "ServiceType",
    "FileSink",
    "MultiSink",
    "OTLPHttpSink",
    "Sink",
    "StdoutSink",
]
