"""Tests for span generators — genai and infra."""

from __future__ import annotations

import numpy as np
import pytest

from god.generators.genai import (
    build_embedding_span,
    build_inference_span,
    build_mcp_tool_span,
    build_rag_trace,
    build_retrieval_span,
    build_tool_span,
    generate_genai_logs,
)
from god.generators.infra import (
    build_db_span,
    build_grpc_span,
    build_http_span,
    build_queue_span,
    generate_logs_for_span,
)
from god.models import GenAIOperation, SpanKind, SpanStatus
from god.services import LatencyProfile, ServiceDef, ServiceRegistry, ServiceType


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _llm_svc() -> ServiceDef:
    return ServiceDef(
        name="llm-anthropic",
        type=ServiceType.GENAI_LLM,
        latency=LatencyProfile(p50_ms=1000, p99_ms=5000, timeout_ms=30000),
        error_rate=0.0,
        provider="anthropic",
        model="claude-sonnet-4-20250514",
    )


def _embed_svc() -> ServiceDef:
    return ServiceDef(
        name="embed-service",
        type=ServiceType.GENAI_EMBED,
        latency=LatencyProfile(p50_ms=100, p99_ms=500),
        error_rate=0.0,
        provider="anthropic",
        model="voyage-3",
    )


def _vector_svc() -> ServiceDef:
    return ServiceDef(
        name="vector-db",
        type=ServiceType.GENAI_VECTOR_DB,
        latency=LatencyProfile(p50_ms=20, p99_ms=150, timeout_ms=2000),
        error_rate=0.0,
    )


def _http_svc() -> ServiceDef:
    return ServiceDef(
        name="api-gateway",
        type=ServiceType.HTTP_API,
        latency=LatencyProfile(p50_ms=15, p99_ms=120),
        error_rate=0.0,
    )


def _db_svc() -> ServiceDef:
    return ServiceDef(
        name="postgres-auth",
        type=ServiceType.DATABASE,
        latency=LatencyProfile(p50_ms=3, p99_ms=40, timeout_ms=3000),
        error_rate=0.0,
    )


def _queue_svc() -> ServiceDef:
    return ServiceDef(
        name="kafka",
        type=ServiceType.QUEUE,
        latency=LatencyProfile(p50_ms=10, p99_ms=50),
        error_rate=0.0,
    )


def _grpc_svc() -> ServiceDef:
    return ServiceDef(
        name="auth-service",
        type=ServiceType.GRPC_SERVICE,
        latency=LatencyProfile(p50_ms=8, p99_ms=60),
        error_rate=0.0,
    )


# Inference span


def test_inference_span_basic_structure() -> None:
    svc = _llm_svc()
    span, end_ms = build_inference_span(svc, "trace1", None, 1000.0, _rng())
    assert span.name.startswith("chat ")
    assert span.kind == SpanKind.CLIENT
    assert span.service_name == "llm-anthropic"
    assert span.attributes["gen_ai.operation.name"] == GenAIOperation.CHAT.value
    assert end_ms > 1000.0


def test_inference_span_has_ttft_event_on_success() -> None:
    svc = _llm_svc()
    svc.error_rate = 0.0
    span, _ = build_inference_span(svc, "t", None, 0.0, _rng(42))
    ttft_events = [e for e in span.events if e.name == "gen_ai.content.first_token"]
    assert len(ttft_events) == 1


def test_inference_span_error_has_exception_event() -> None:
    svc = _llm_svc()
    svc._error_rate_override = 1.0
    span, _ = build_inference_span(svc, "t", None, 0.0, _rng(0))
    assert span.status == SpanStatus.ERROR
    exc_events = [e for e in span.events if e.name == "exception"]
    assert len(exc_events) == 1


def test_inference_span_deterministic_with_seed() -> None:
    svc = _llm_svc()
    span1, _ = build_inference_span(svc, "t", None, 0.0, _rng(99))
    svc2 = _llm_svc()
    span2, _ = build_inference_span(svc2, "t", None, 0.0, _rng(99))
    assert span1.duration_ms == pytest.approx(span2.duration_ms, rel=1e-6)


def test_inference_span_capture_content() -> None:
    svc = _llm_svc()
    span, _ = build_inference_span(svc, "t", None, 0.0, _rng(1), capture_content=True)
    assert "gen_ai.system_instructions" in span.attributes
    assert "gen_ai.input.messages" in span.attributes


def test_inference_span_ttft_multiplier() -> None:
    svc = _llm_svc()
    span_base, _ = build_inference_span(svc, "t", None, 0.0, _rng(7))
    svc2 = _llm_svc()
    span_mult, _ = build_inference_span(svc2, "t", None, 0.0, _rng(7), ttft_multiplier=5.0)
    assert span_mult.duration_ms > span_base.duration_ms


# Embedding span


def test_embedding_span_structure() -> None:
    svc = _embed_svc()
    span, end_ms = build_embedding_span(svc, "t", None, 0.0, _rng())
    assert span.attributes["gen_ai.operation.name"] == GenAIOperation.EMBEDDINGS.value
    assert "gen_ai.embedding.batch_size" in span.attributes
    assert end_ms > 0.0


# Retrieval span


def test_retrieval_span_structure() -> None:
    svc = _vector_svc()
    span, end_ms = build_retrieval_span(svc, "t", None, 0.0, _rng())
    assert span.attributes["gen_ai.operation.name"] == GenAIOperation.RETRIEVE.value
    assert "db.vector.query.top_k" in span.attributes
    assert "db.vector.query.returned" in span.attributes


def test_retrieval_span_latency_multiplier() -> None:
    svc = _vector_svc()
    span_base, _ = build_retrieval_span(svc, "t", None, 0.0, _rng(3))
    svc2 = _vector_svc()
    span_slow, _ = build_retrieval_span(svc2, "t", None, 0.0, _rng(3), latency_multiplier=6.0)
    assert span_slow.duration_ms > span_base.duration_ms


# Tool spans


def test_tool_span_structure() -> None:
    svc = _llm_svc()
    span, end_ms = build_tool_span(svc, "t", None, 0.0, _rng())
    assert span.attributes["gen_ai.operation.name"] == GenAIOperation.EXECUTE_TOOL.value
    assert "gen_ai.tool.name" in span.attributes


def test_mcp_tool_span_has_mcp_attrs() -> None:
    svc = _llm_svc()
    span, _ = build_mcp_tool_span(svc, "t", None, 0.0, _rng())
    assert "mcp.method.name" in span.attributes
    assert "mcp.session.id" in span.attributes
    assert span.attributes["mcp.protocol.version"] == "2025-03-26"


# Full RAG trace


def test_rag_trace_returns_multiple_spans() -> None:
    reg = ServiceRegistry()
    spans, end_ms = build_rag_trace(reg, "trace1", None, 0.0, _rng())
    assert len(spans) >= 3  # embed + retrieve + infer
    ops = {s.attributes.get("gen_ai.operation.name") for s in spans}
    assert GenAIOperation.EMBEDDINGS.value in ops
    assert GenAIOperation.RETRIEVE.value in ops
    assert GenAIOperation.CHAT.value in ops


def test_rag_trace_end_time_after_start() -> None:
    reg = ServiceRegistry()
    _, end_ms = build_rag_trace(reg, "t", None, 100.0, _rng())
    assert end_ms > 100.0


# GenAI log generators


def test_genai_logs_chat_success() -> None:
    svc = _llm_svc()
    span, _ = build_inference_span(svc, "t", None, 0.0, _rng(5))
    logs = generate_genai_logs(span, svc, _rng())
    assert len(logs) >= 1
    from god.models import LogLevel

    assert any(l.level in (LogLevel.INFO, LogLevel.WARN) for l in logs)


def test_genai_logs_embedding() -> None:
    svc = _embed_svc()
    span, _ = build_embedding_span(svc, "t", None, 0.0, _rng())
    logs = generate_genai_logs(span, svc, _rng())
    assert len(logs) >= 1


# Infra span builders


def test_http_span_structure() -> None:
    svc = _http_svc()
    span, end_ms = build_http_span(svc, "t", None, 0.0, _rng())
    assert "http.request.method" in span.attributes
    assert "http.response.status_code" in span.attributes
    assert end_ms > 0.0


def test_grpc_span_structure() -> None:
    svc = _grpc_svc()
    span, end_ms = build_grpc_span(svc, "t", None, 0.0, _rng())
    assert span.attributes["rpc.system"] == "grpc"
    assert "rpc.grpc.status_code" in span.attributes


def test_db_span_structure() -> None:
    svc = _db_svc()
    span, end_ms = build_db_span(svc, "t", None, 0.0, _rng())
    assert span.attributes["db.system.name"] == "postgresql"
    assert "db.operation.name" in span.attributes


def test_queue_span_producer() -> None:
    svc = _queue_svc()
    span, _ = build_queue_span(svc, "t", None, 0.0, _rng(), is_producer=True)
    assert span.kind == SpanKind.PRODUCER
    assert span.attributes["messaging.operation.name"] == "publish"


def test_queue_span_consumer() -> None:
    svc = _queue_svc()
    span, _ = build_queue_span(svc, "t", None, 0.0, _rng(), is_producer=False)
    assert span.kind == SpanKind.CONSUMER


def test_infra_logs_for_http_span() -> None:
    svc = _http_svc()
    span, _ = build_http_span(svc, "t", None, 0.0, _rng())
    logs = generate_logs_for_span(span, svc, _rng())
    assert len(logs) >= 1
    for log in logs:
        assert log.service_name == svc.name
        assert log.trace_id == span.trace_id


def test_infra_logs_correlated_trace_id() -> None:
    svc = _db_svc()
    span, _ = build_db_span(svc, "t_abc", None, 0.0, _rng())
    logs = generate_logs_for_span(span, svc, _rng())
    for log in logs:
        assert log.trace_id == "t_abc"
