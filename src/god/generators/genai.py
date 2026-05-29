"""
GenAI span generators implementing OTel GenAI Semantic Conventions.
Spec reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
Status: experimental (semconv 1.41.0, May 2026)

Covers:
  - Inference (chat) spans    gen_ai.operation.name = "chat"
  - Embedding spans           gen_ai.operation.name = "embeddings"
  - Retrieval spans           gen_ai.operation.name = "retrieve"
  - Tool execution spans      gen_ai.operation.name = "execute_tool"
  - MCP tool spans            mcp.* attributes
  - Create agent spans        gen_ai.operation.name = "create_agent"
  - Invoke agent spans        gen_ai.operation.name = "invoke_agent"
  - Memory operation spans    gen_ai.operation.name = "search_memory" / "create_memory" / etc.
  - Workflow spans            gen_ai.operation.name = "invoke_workflow"
  - Plan spans                gen_ai.operation.name = "plan"
  - GenAI metrics (TTFT, inter-token latency, token counts)
  - Content capture events    gen_ai.content.prompt / gen_ai.content.completion
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

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
    new_id,
)
from god.services import ServiceDef

# ---------------------------------------------------------------------------
# Realistic synthetic content pools
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = [
    "You are a helpful assistant for financial services queries.",
    "You are an expert at analysing log data and identifying anomalies.",
    "You are a code review assistant. Be concise and cite line numbers.",
    "You are a RAG assistant. Always cite retrieved sources.",
]

_USER_QUERIES = [
    "Why is the payments service returning 503 errors?",
    "Summarise the last 24 hours of anomalies in the auth service.",
    "Explain the root cause of the database connection pool exhaustion.",
    "What changed in the deployment at 14:32 UTC?",
    "Are there any patterns in the slow query logs?",
    "Compare error rates across services for the past hour.",
]

_TOOL_NAMES = [
    "search_logs",
    "get_trace",
    "query_metrics",
    "fetch_deployment_history",
    "get_service_config",
]

_MCP_TOOL_NAMES = [
    "vector_search",
    "get_context_chunks",
    "log_retrieval",
    "trace_lookup",
]

_EMBEDDING_INPUT_TEXTS = [
    "connection pool exhausted max connections reached",
    "upstream timeout payments service 503",
    "slow query detected duration exceeded threshold",
    "authentication token validation failed",
    "kafka consumer lag elevated topic payments.events",
]

_VECTOR_DB_COLLECTIONS = [
    "logs_embeddings",
    "trace_summaries",
    "runbooks",
    "incident_history",
]

_MODELS: dict[str, dict] = {
    # Anthropic
    "claude-sonnet-4-20250514": {
        "provider": GenAIProvider.ANTHROPIC,
        "input_token_rate": 1.0,  # relative cost multiplier
        "context_window": 200000,
        "typical_input_tokens": (800, 4000),
        "typical_output_tokens": (150, 800),
        "ttft_p50_ms": 400,
        "ttft_p99_ms": 2000,
        "inter_token_p50_ms": 18,
        "inter_token_p99_ms": 60,
    },
    "claude-haiku-4-5-20251001": {
        "provider": GenAIProvider.ANTHROPIC,
        "input_token_rate": 0.1,
        "context_window": 200000,
        "typical_input_tokens": (200, 1500),
        "typical_output_tokens": (50, 400),
        "ttft_p50_ms": 150,
        "ttft_p99_ms": 800,
        "inter_token_p50_ms": 8,
        "inter_token_p99_ms": 25,
    },
    # OpenAI
    "gpt-4o": {
        "provider": GenAIProvider.OPENAI,
        "input_token_rate": 0.8,
        "context_window": 128000,
        "typical_input_tokens": (600, 3000),
        "typical_output_tokens": (100, 600),
        "ttft_p50_ms": 350,
        "ttft_p99_ms": 1800,
        "inter_token_p50_ms": 20,
        "inter_token_p99_ms": 70,
    },
    "text-embedding-3-small": {
        "provider": GenAIProvider.OPENAI,
        "input_token_rate": 0.02,
        "context_window": 8191,
        "typical_input_tokens": (50, 512),
        "typical_output_tokens": (0, 0),
        "ttft_p50_ms": 80,
        "ttft_p99_ms": 400,
        "inter_token_p50_ms": 0,
        "inter_token_p99_ms": 0,
    },
    "voyage-3": {
        "provider": GenAIProvider.ANTHROPIC,
        "input_token_rate": 0.02,
        "context_window": 32000,
        "typical_input_tokens": (30, 400),
        "typical_output_tokens": (0, 0),
        "ttft_p50_ms": 90,
        "ttft_p99_ms": 450,
        "inter_token_p50_ms": 0,
        "inter_token_p99_ms": 0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(ms: float) -> int:
    return int(ms * 1_000_000)


def _lognormal_sample(rng: np.random.Generator, p50: float, p99: float) -> float:
    if p50 <= 0:
        return 0.0
    mu = math.log(p50)
    sigma = (math.log(max(p99, p50 + 1)) - mu) / 2.326
    return float(rng.lognormal(mu, max(sigma, 0.01)))


def _token_count(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Sample token count from a roughly realistic distribution."""
    return int(rng.integers(lo, hi + 1))


def _model_info(model: str) -> dict:
    return _MODELS.get(model, _MODELS["claude-sonnet-4-20250514"])


# ---------------------------------------------------------------------------
# Inference (chat) span
# ---------------------------------------------------------------------------


def build_inference_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    model: str | None = None,
    capture_content: bool = False,
    force_finish_reason: FinishReason | None = None,
    ttft_multiplier: float = 1.0,
) -> tuple[Span, float]:
    """
    Build a GenAI inference (chat) span following semconv 1.41.0.

    Span name: "{gen_ai.operation.name} {gen_ai.request.model}"
    Span kind: CLIENT

    Key attributes (all gen_ai.*):
      gen_ai.operation.name       = "chat"
      gen_ai.request.model        = model name
      gen_ai.provider.name        = provider discriminator
      gen_ai.usage.input_tokens
      gen_ai.usage.output_tokens
      gen_ai.response.finish_reasons  (array)
      gen_ai.request.max_tokens
      gen_ai.request.temperature
      gen_ai.request.top_p

    Content capture (opt-in, large attributes):
      gen_ai.system_instructions
      gen_ai.input.messages
      gen_ai.output.messages

    Metrics captured as span events:
      gen_ai.content.first_token   (time-to-first-token)
    """
    model = model or svc.model or "claude-sonnet-4-20250514"
    info = _model_info(model)
    span_id = new_id(16)

    # Token counts
    in_lo, in_hi = info["typical_input_tokens"]
    out_lo, out_hi = info["typical_output_tokens"]
    input_tokens = _token_count(rng, in_lo, in_hi)
    output_tokens = _token_count(rng, out_lo, out_hi)

    is_err = rng.random() < svc.effective_error_rate
    is_length_trunc = (
        not is_err and rng.random() < 0.05  # 5% chance of token budget exhaustion
    )

    # Finish reason
    if force_finish_reason:
        finish_reason = force_finish_reason
    elif is_err:
        finish_reason = FinishReason.ERROR
        output_tokens = 0
    elif is_length_trunc:
        finish_reason = FinishReason.LENGTH
    else:
        finish_reason = FinishReason.STOP

    # TTFT — log-normal with optional multiplier (for GPU saturation scenario)
    ttft_ms = _lognormal_sample(
        rng,
        info["ttft_p50_ms"] * ttft_multiplier,
        info["ttft_p99_ms"] * ttft_multiplier,
    )

    # Inter-token latency and total generation time
    inter_p50 = info["inter_token_p50_ms"]
    inter_p99 = info["inter_token_p99_ms"]
    if output_tokens > 0 and inter_p50 > 0:
        per_token_ms = _lognormal_sample(rng, inter_p50, inter_p99)
        generation_ms = ttft_ms + per_token_ms * output_tokens
    else:
        generation_ms = ttft_ms + 50

    end_ms = start_ms + generation_ms

    # Attributes
    attrs: dict[str, Any] = {
        # Required GenAI attributes
        "gen_ai.operation.name": GenAIOperation.CHAT.value,
        "gen_ai.request.model": model,
        "gen_ai.provider.name": info["provider"].value,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "gen_ai.response.finish_reasons": [finish_reason.value],
        # Request parameters
        "gen_ai.request.max_tokens": random.choice([1024, 2048, 4096, 8192]),
        "gen_ai.request.temperature": round(float(rng.uniform(0.0, 1.0)), 2),
        "gen_ai.request.top_p": round(float(rng.uniform(0.8, 1.0)), 2),
        # Latency breakdown
        "gen_ai.server.time_to_first_token": round(ttft_ms / 1000, 4),  # seconds per spec
        "gen_ai.server.ttft_ms": round(ttft_ms, 2),  # convenience
        # Server info
        "server.address": f"{info['provider'].value}.api",
        "server.port": 443,
        "network.transport": "tcp",
    }

    if is_err:
        attrs["error.type"] = random.choice(
            [
                "rate_limit_error",
                "overloaded_error",
                "api_timeout",
                "server_error",
            ]
        )

    # Optional content capture (disabled by default — large + privacy-sensitive)
    if capture_content:
        attrs["gen_ai.system_instructions"] = random.choice(_SYSTEM_PROMPTS)
        attrs["gen_ai.input.messages"] = str([{"role": "user", "content": random.choice(_USER_QUERIES)}])
        if output_tokens > 0:
            attrs["gen_ai.output.messages"] = str(
                [{"role": "assistant", "content": "Based on the retrieved context, I can see..."}]
            )

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"chat {model}",
        kind=SpanKind.CLIENT,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        status_message=attrs.get("error.type", ""),
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )

    # TTFT event — spec-recommended for streaming observability
    if not is_err:
        span.events.append(
            SpanEvent(
                name="gen_ai.content.first_token",
                time_unix_nano=_ns(start_ms + ttft_ms),
                attributes={
                    "gen_ai.server.time_to_first_token": round(ttft_ms / 1000, 4),
                },
            )
        )

    # Exception event on error
    if is_err:
        span.events.append(
            SpanEvent(
                name="exception",
                time_unix_nano=_ns(start_ms + ttft_ms * 0.5),
                attributes={
                    "exception.type": attrs.get("error.type", "APIError"),
                    "exception.message": f"GenAI API error: {attrs.get('error.type')}",
                },
            )
        )

    return span, end_ms


# ---------------------------------------------------------------------------
# Embedding span
# ---------------------------------------------------------------------------


def build_embedding_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    model: str | None = None,
) -> tuple[Span, float]:
    """
    gen_ai.operation.name = "embeddings"
    Returns a span and end_time_ms.
    """
    model = model or svc.model or "voyage-3"
    info = _model_info(model)
    span_id = new_id(16)

    in_lo, in_hi = info["typical_input_tokens"]
    input_tokens = _token_count(rng, in_lo, in_hi)
    duration_ms = _lognormal_sample(rng, info["ttft_p50_ms"], info["ttft_p99_ms"])
    end_ms = start_ms + duration_ms
    is_err = rng.random() < svc.effective_error_rate

    # Batch size for embedding requests (often batch > 1)
    batch_size = int(rng.integers(1, 16))
    embedding_dim = 1536 if "openai" in model else 1024

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": GenAIOperation.EMBEDDINGS.value,
        "gen_ai.request.model": model,
        "gen_ai.provider.name": info["provider"].value,
        "gen_ai.usage.input_tokens": input_tokens * batch_size,
        "gen_ai.request.encoding_formats": ["float"],
        "gen_ai.embedding.batch_size": batch_size,
        "gen_ai.embedding.dimensions": embedding_dim,
        "server.address": f"{info['provider'].value}.api",
        "server.port": 443,
    }
    if is_err:
        attrs["error.type"] = "embedding_error"

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"embeddings {model}",
        kind=SpanKind.CLIENT,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Retrieval (vector search) span
# ---------------------------------------------------------------------------


def build_retrieval_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    latency_multiplier: float = 1.0,
) -> tuple[Span, float]:
    """
    gen_ai.operation.name = "retrieve"
    Used for vector DB similarity search.
    """
    span_id = new_id(16)
    base_p50 = svc.effective_latency.p50_ms * latency_multiplier
    base_p99 = svc.effective_latency.p99_ms * latency_multiplier
    duration_ms = _lognormal_sample(rng, base_p50, base_p99)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < svc.effective_error_rate

    collection = random.choice(_VECTOR_DB_COLLECTIONS)
    top_k = random.choice([3, 5, 10, 20])
    returned = int(rng.integers(0, top_k + 1)) if not is_err else 0

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": GenAIOperation.RETRIEVE.value,
        "db.system.name": "qdrant",  # or "weaviate", "pgvector"
        "db.collection.name": collection,
        "db.vector.query.top_k": top_k,
        "db.vector.query.returned": returned,
        "db.vector.dimension": 1024,
        "db.vector.similarity_metric": "cosine",
        "server.address": svc.name,
        "server.port": 6333,
        "db.query.duration_ms": round(duration_ms, 2),
    }
    if is_err:
        attrs["error.type"] = "vector_db_error"

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"retrieve {collection}",
        kind=SpanKind.CLIENT,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Tool execution span (LLM-initiated)
# ---------------------------------------------------------------------------


def build_tool_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    tool_name: str | None = None,
) -> tuple[Span, float]:
    """
    gen_ai.operation.name = "execute_tool"
    Represents a tool call made by the LLM during inference.
    """
    span_id = new_id(16)
    duration_ms = _lognormal_sample(rng, 50, 400)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < 0.05

    tool = tool_name or random.choice(_TOOL_NAMES)

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": GenAIOperation.EXECUTE_TOOL.value,
        "gen_ai.tool.name": tool,
        "gen_ai.tool.call.id": new_id(12),
        "gen_ai.tool.description": f"Tool for {tool.replace('_', ' ')}",
    }
    if is_err:
        attrs["error.type"] = "tool_execution_error"
        attrs["gen_ai.tool.error"] = "tool returned non-zero exit code"

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"execute_tool {tool}",
        kind=SpanKind.INTERNAL,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# MCP tool span
# ---------------------------------------------------------------------------


def build_mcp_tool_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    tool_name: str | None = None,
) -> tuple[Span, float]:
    """
    MCP (Model Context Protocol) tool call span.

    Attributes follow OTel MCP semconv:
      gen_ai.operation.name = "execute_tool"
      mcp.method.name       = "tools/call"
      mcp.session.id
      mcp.protocol.version  = "2025-03-26"
      gen_ai.tool.name
      network.transport     = "tcp" (HTTP transport)
      network.protocol.name = "http"

    Per spec: if outer GenAI span already tracks tool execution,
    MCP instrumentation enriches that span rather than creating a duplicate.
    Here we model it as a child CLIENT span from the LLM service.
    """
    span_id = new_id(16)
    duration_ms = _lognormal_sample(rng, 30, 300)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < 0.04

    tool = tool_name or random.choice(_MCP_TOOL_NAMES)

    attrs: dict[str, Any] = {
        # GenAI execute_tool base
        "gen_ai.operation.name": GenAIOperation.EXECUTE_TOOL.value,
        "gen_ai.tool.name": tool,
        "gen_ai.tool.call.id": new_id(12),
        # MCP-specific
        "mcp.method.name": "tools/call",
        "mcp.session.id": f"session-{new_id(8)}",
        "mcp.protocol.version": "2025-03-26",
        "mcp.server.name": f"mcp-{tool.replace('_', '-')}",
        # Transport
        "network.transport": "tcp",
        "network.protocol.name": "http",
        "server.address": "mcp.internal",
        "server.port": 8765,
        # JSON-RPC
        "jsonrpc.request.id": str(int(rng.integers(1, 9999))),
        "jsonrpc.version": "2.0",
    }
    if is_err:
        attrs["error.type"] = "mcp_tool_error"

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"execute_tool {tool}",
        kind=SpanKind.CLIENT,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Full RAG pipeline trace builder
# ---------------------------------------------------------------------------


def build_rag_trace(
    registry,  # ServiceRegistry
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    capture_content: bool = False,
    ttft_multiplier: float = 1.0,
    retrieval_latency_multiplier: float = 1.0,
) -> tuple[list[Span], float]:
    """
    Build a full RAG pipeline subtrace:
      1. embed query            (embeddings span)
      2. vector search          (retrieve span)
      3. LLM inference          (chat span)
         └─ optional tool call  (execute_tool or MCP span)

    Returns (spans, end_time_ms).
    """
    spans: list[Span] = []
    cursor = start_ms

    embed_svc = registry.get("embed-service")
    vector_svc = registry.get("vector-db")
    llm_svc = registry.get("llm-anthropic")

    # 1. Embedding
    embed_span, cursor = build_embedding_span(
        embed_svc,
        trace_id,
        parent_span_id,
        cursor,
        rng,
    )
    spans.append(embed_span)
    cursor += float(rng.uniform(1, 5))  # network gap

    # 2. Retrieval
    retrieve_span, cursor = build_retrieval_span(
        vector_svc,
        trace_id,
        parent_span_id,
        cursor,
        rng,
        latency_multiplier=retrieval_latency_multiplier,
    )
    spans.append(retrieve_span)
    cursor += float(rng.uniform(1, 5))

    # 3. LLM inference
    llm_span, cursor = build_inference_span(
        llm_svc,
        trace_id,
        parent_span_id,
        cursor,
        rng,
        capture_content=capture_content,
        ttft_multiplier=ttft_multiplier,
    )
    spans.append(llm_span)

    # 3a. Optional tool call (25% of requests)
    lo = (llm_span.start_time_unix_nano / 1e6) + 100
    hi = (llm_span.end_time_unix_nano / 1e6) - 200
    if rng.random() < 0.25 and hi > lo:
        tool_start = float(rng.uniform(lo, hi))
        if rng.random() < 0.5:
            tool_span, _ = build_mcp_tool_span(
                llm_svc,
                trace_id,
                llm_span.span_id,
                tool_start,
                rng,
            )
        else:
            tool_span, _ = build_tool_span(
                llm_svc,
                trace_id,
                llm_span.span_id,
                tool_start,
                rng,
            )
        spans.append(tool_span)

    return spans, cursor


# ---------------------------------------------------------------------------
# Agent / memory / workflow content pools
# ---------------------------------------------------------------------------

_AGENT_NAMES = [
    "ResearchAgent",
    "SummaryAgent",
    "CodeReviewAgent",
    "IncidentAnalyser",
    "AnomalyDetector",
    "DataExtractionAgent",
]

_AGENT_DESCRIPTIONS = [
    "Analyses logs and traces to identify root causes",
    "Summarises incident timelines from trace data",
    "Reviews code changes for regressions and security issues",
    "Extracts structured data from unstructured log output",
    "Detects anomalies in metric streams using LLM reasoning",
]

_WORKFLOW_NAMES = [
    "incident_triage_pipeline",
    "multi_agent_rag",
    "customer_support_pipeline",
    "code_review_workflow",
    "anomaly_resolution_pipeline",
]

_MEMORY_QUERIES = [
    "user preferences for alert thresholds",
    "past incident resolutions for service degradation",
    "prior tool call results for payments-service",
    "historical anomaly patterns in auth-service",
    "previously resolved root causes for connection pool exhaustion",
]

_TOOL_TYPES = ["function", "extension", "datastore"]

_EXT_TOOL_NAMES = [
    "fetch_pagerduty_incidents",
    "query_grafana_dashboard",
    "get_github_pr_diff",
    "call_jira_api",
    "invoke_aws_lambda",
]

_FUNCTION_TOOL_NAMES = [
    "parse_log_line",
    "compute_error_rate",
    "format_trace_summary",
    "extract_service_name",
    "rank_candidates",
]

_DATASTORE_TOOL_NAMES = [
    "search_runbook_db",
    "query_incident_history",
    "lookup_service_config",
    "fetch_deployment_manifest",
    "retrieve_alert_rules",
]


# ---------------------------------------------------------------------------
# Create agent span
# ---------------------------------------------------------------------------


def build_create_agent_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    agent_name: str | None = None,
) -> tuple[Span, float]:
    """
    gen_ai.operation.name = "create_agent"
    Represents provisioning a remote agent (e.g. OpenAI Assistants, Bedrock Agents).
    Span name: "create_agent {agent_name}"
    Span kind: CLIENT
    """
    span_id = new_id(16)
    duration_ms = _lognormal_sample(rng, 200, 1200)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < svc.effective_error_rate

    model = svc.model or "gpt-4o"
    info = _model_info(model)
    name = agent_name or random.choice(_AGENT_NAMES)
    agent_id = f"asst_{new_id(24)}"

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": GenAIOperation.CREATE_AGENT.value,
        "gen_ai.provider.name": info["provider"].value,
        "gen_ai.request.model": model,
        "gen_ai.agent.id": agent_id,
        "gen_ai.agent.name": name,
        "gen_ai.agent.description": random.choice(_AGENT_DESCRIPTIONS),
        "gen_ai.agent.version": f"1.{int(rng.integers(0, 5))}.0",
        "server.address": f"{info['provider'].value}.api",
        "server.port": 443,
    }
    if is_err:
        attrs["error.type"] = random.choice(["agent_creation_failed", "quota_exceeded", "invalid_model"])

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"create_agent {name}",
        kind=SpanKind.CLIENT,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Invoke agent span
# ---------------------------------------------------------------------------


def build_invoke_agent_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    agent_name: str | None = None,
    is_remote: bool = True,
    conversation_id: str | None = None,
) -> tuple[Span, float]:
    """
    gen_ai.operation.name = "invoke_agent"
    Remote invocation (CLIENT kind) or in-process (INTERNAL kind).
    Span name: "invoke_agent {agent_name}"
    """
    span_id = new_id(16)
    # Agent invocations are longer — they encompass multiple LLM round-trips
    duration_ms = _lognormal_sample(rng, svc.effective_latency.p50_ms, svc.effective_latency.p99_ms)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < svc.effective_error_rate

    model = svc.model or "gpt-4o"
    info = _model_info(model)
    name = agent_name or random.choice(_AGENT_NAMES)

    # Aggregate token usage across the full agent invocation
    in_lo, in_hi = info["typical_input_tokens"]
    out_lo, out_hi = info["typical_output_tokens"]
    turns = int(rng.integers(2, 6))
    input_tokens = sum(_token_count(rng, in_lo, in_hi) for _ in range(turns))
    output_tokens = sum(_token_count(rng, out_lo, out_hi) for _ in range(turns))

    finish_reasons = [FinishReason.STOP.value] if not is_err else [FinishReason.ERROR.value]

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": GenAIOperation.INVOKE_AGENT.value,
        "gen_ai.provider.name": info["provider"].value,
        "gen_ai.request.model": model,
        "gen_ai.agent.id": f"asst_{new_id(24)}",
        "gen_ai.agent.name": name,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "gen_ai.response.finish_reasons": finish_reasons,
        "gen_ai.request.temperature": round(float(rng.uniform(0.0, 0.8)), 2),
        "gen_ai.request.max_tokens": random.choice([2048, 4096, 8192]),
        "gen_ai.conversation.id": conversation_id or f"conv_{new_id(20)}",
    }
    if is_remote:
        attrs["server.address"] = f"{info['provider'].value}.api"
        attrs["server.port"] = 443

    if is_err:
        attrs["error.type"] = random.choice(["agent_timeout", "agent_loop_limit", "context_length_exceeded"])

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"invoke_agent {name}",
        kind=SpanKind.CLIENT if is_remote else SpanKind.INTERNAL,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Rich tool call span (function / extension / datastore)
# ---------------------------------------------------------------------------


def build_rich_tool_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    tool_type: str | None = None,
) -> tuple[Span, float]:
    """
    gen_ai.operation.name = "execute_tool"
    Covers all three tool types from the spec: function, extension, datastore.
    Populates gen_ai.tool.call.arguments and gen_ai.tool.call.result.
    """
    span_id = new_id(16)
    t_type = tool_type or random.choice(_TOOL_TYPES)

    if t_type == "extension":
        tool_name = random.choice(_EXT_TOOL_NAMES)
        base_p50, base_p99 = 300, 2000
    elif t_type == "datastore":
        tool_name = random.choice(_DATASTORE_TOOL_NAMES)
        base_p50, base_p99 = 60, 400
    else:
        tool_name = random.choice(_FUNCTION_TOOL_NAMES)
        base_p50, base_p99 = 5, 50

    duration_ms = _lognormal_sample(rng, base_p50 * svc._latency_multiplier, base_p99 * svc._latency_multiplier)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < svc.effective_error_rate

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": GenAIOperation.EXECUTE_TOOL.value,
        "gen_ai.tool.name": tool_name,
        "gen_ai.tool.type": t_type,
        "gen_ai.tool.call.id": f"call_{new_id(24)}",
        "gen_ai.tool.description": f"{t_type.capitalize()} tool: {tool_name.replace('_', ' ')}",
        "gen_ai.tool.call.arguments": f'{{"service": "payments-service", "window": "1h"}}',
    }
    if not is_err:
        attrs["gen_ai.tool.call.result"] = '{"status": "ok", "records": 42}'
    if is_err:
        attrs["error.type"] = random.choice(["tool_timeout", "tool_execution_error", "invalid_arguments"])

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"execute_tool {tool_name}",
        kind=SpanKind.INTERNAL,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Memory operation span
# ---------------------------------------------------------------------------

_MEMORY_OPS = [
    GenAIOperation.SEARCH_MEMORY,
    GenAIOperation.UPSERT_MEMORY,
    GenAIOperation.CREATE_MEMORY,
    GenAIOperation.DELETE_MEMORY,
]

_MEMORY_OP_WEIGHTS = [0.55, 0.25, 0.15, 0.05]


def build_memory_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    operation: GenAIOperation | None = None,
) -> tuple[Span, float]:
    """
    Covers all memory CRUD operations from the spec.
    gen_ai.operation.name = one of: search_memory, create_memory, upsert_memory, delete_memory
    Span name: "{operation}"
    Span kind: CLIENT
    """
    span_id = new_id(16)
    op = operation or GenAIOperation(
        rng.choice(  # type: ignore[call-arg]
            [o.value for o in _MEMORY_OPS],
            p=_MEMORY_OP_WEIGHTS,
        )
    )
    duration_ms = _lognormal_sample(rng, svc.effective_latency.p50_ms, svc.effective_latency.p99_ms)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < svc.effective_error_rate

    store_id = f"ms_{new_id(8)}"
    record_count = int(rng.integers(1, 10)) if op != GenAIOperation.SEARCH_MEMORY else int(rng.integers(0, 6))

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": op.value,
        "gen_ai.memory.store.id": store_id,
        "gen_ai.memory.record.count": record_count,
        "server.address": svc.name,
        "server.port": 8080,
    }
    if op == GenAIOperation.SEARCH_MEMORY:
        attrs["gen_ai.memory.query.text"] = random.choice(_MEMORY_QUERIES)
    if op in (GenAIOperation.CREATE_MEMORY, GenAIOperation.UPSERT_MEMORY):
        attrs["gen_ai.memory.record.id"] = f"mem_{new_id(20)}"
    if is_err:
        attrs["error.type"] = random.choice(["memory_store_unavailable", "memory_write_failed", "store_quota_exceeded"])

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=op.value,
        kind=SpanKind.CLIENT,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Plan span
# ---------------------------------------------------------------------------


def build_plan_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    agent_name: str | None = None,
) -> tuple[Span, float]:
    """
    gen_ai.operation.name = "plan"
    Represents the task decomposition phase before tool/agent execution.
    Span kind: INTERNAL
    """
    span_id = new_id(16)
    duration_ms = _lognormal_sample(rng, 600, 3000)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < svc.effective_error_rate

    name = agent_name or random.choice(_AGENT_NAMES)

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": GenAIOperation.PLAN.value,
        "gen_ai.agent.name": name,
        "gen_ai.agent.id": f"asst_{new_id(24)}",
    }
    if is_err:
        attrs["error.type"] = "planning_failed"

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"plan {name}",
        kind=SpanKind.INTERNAL,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Workflow span (invoke_workflow)
# ---------------------------------------------------------------------------


def build_workflow_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    workflow_name: str | None = None,
) -> tuple[Span, float]:
    """
    gen_ai.operation.name = "invoke_workflow"
    Top-level span covering a coordinated multi-agent pipeline.
    Span kind: INTERNAL
    """
    span_id = new_id(16)
    duration_ms = _lognormal_sample(rng, svc.effective_latency.p50_ms, svc.effective_latency.p99_ms)
    end_ms = start_ms + duration_ms
    is_err = rng.random() < svc.effective_error_rate

    wf_name = workflow_name or random.choice(_WORKFLOW_NAMES)

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": GenAIOperation.INVOKE_WORKFLOW.value,
        "gen_ai.workflow.name": wf_name,
    }
    if is_err:
        attrs["error.type"] = random.choice(["workflow_timeout", "agent_step_failed", "orchestration_error"])

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"invoke_workflow {wf_name}",
        kind=SpanKind.INTERNAL,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Full agentic pipeline trace builder
# ---------------------------------------------------------------------------


def build_agentic_trace(
    registry,  # ServiceRegistry
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
) -> tuple[list[Span], float]:
    """
    Build a full agentic pipeline subtrace:
      1. invoke_workflow         (workflow span — root of agent pipeline)
         ├─ plan {agent}        (planning phase)
         ├─ search_memory       (retrieve prior context)
         ├─ invoke_agent        (agent does LLM + tool calls)
         │    ├─ chat {model}   (underlying LLM call)
         │    └─ execute_tool   (one or more tool calls)
         └─ upsert_memory       (persist new memories)

    Returns (spans, end_time_ms).
    """
    spans: list[Span] = []
    agent_name = random.choice(_AGENT_NAMES)
    conversation_id = f"conv_{new_id(20)}"

    workflow_svc = registry.get("workflow-orchestrator")
    agent_svc = registry.get("agent-service")
    memory_svc = registry.get("memory-service")
    tool_svc = registry.get("tool-executor")
    llm_svc = registry.get("llm-anthropic")

    # 1. Workflow root span
    wf_span, wf_end = build_workflow_span(workflow_svc, trace_id, parent_span_id, start_ms, rng)
    spans.append(wf_span)
    cursor = start_ms + float(rng.uniform(5, 20))

    # 2. Plan phase (child of workflow)
    plan_span, cursor = build_plan_span(agent_svc, trace_id, wf_span.span_id, cursor, rng, agent_name)
    spans.append(plan_span)
    cursor += float(rng.uniform(2, 8))

    # 3. Search memory before acting
    mem_search_span, cursor = build_memory_span(
        memory_svc, trace_id, wf_span.span_id, cursor, rng, GenAIOperation.SEARCH_MEMORY
    )
    spans.append(mem_search_span)
    cursor += float(rng.uniform(2, 8))

    # 4. Invoke agent (encompasses the LLM + tools)
    invoke_span, invoke_end = build_invoke_agent_span(
        agent_svc, trace_id, wf_span.span_id, cursor, rng, agent_name=agent_name, conversation_id=conversation_id
    )
    spans.append(invoke_span)

    # 4a. Underlying LLM chat span (child of invoke_agent)
    llm_span, llm_end = build_inference_span(
        llm_svc, trace_id, invoke_span.span_id, cursor + float(rng.uniform(5, 15)), rng
    )
    spans.append(llm_span)

    # 4b. Tool calls (1-3 tools, children of invoke_agent)
    n_tools = int(rng.integers(1, 4))
    tool_cursor = llm_end + float(rng.uniform(2, 10))
    for _ in range(n_tools):
        tool_span, tool_cursor = build_rich_tool_span(
            tool_svc, trace_id, invoke_span.span_id, tool_cursor, rng
        )
        spans.append(tool_span)
        tool_cursor += float(rng.uniform(1, 5))

    cursor = max(invoke_end, tool_cursor)

    # 5. Persist new memories after agent run
    mem_write_span, cursor = build_memory_span(
        memory_svc, trace_id, wf_span.span_id, cursor, rng, GenAIOperation.UPSERT_MEMORY
    )
    spans.append(mem_write_span)

    # Align workflow end to last child
    wf_end = max(wf_end, cursor)
    return spans, wf_end


# ---------------------------------------------------------------------------
# GenAI log generators
# ---------------------------------------------------------------------------


def generate_genai_logs(
    span: Span,
    svc: ServiceDef,
    rng: np.random.Generator,
) -> list[LogEntry]:
    """Generate realistic GenAI-specific log entries correlated to a span."""
    logs: list[LogEntry] = []
    op = span.attributes.get("gen_ai.operation.name", "")
    model = span.attributes.get("gen_ai.request.model", "unknown")
    is_err = span.status == SpanStatus.ERROR
    in_tok = span.attributes.get("gen_ai.usage.input_tokens", 0)
    out_tok = span.attributes.get("gen_ai.usage.output_tokens", 0)
    finish = span.attributes.get("gen_ai.response.finish_reasons", ["stop"])
    if isinstance(finish, list):
        finish_str = finish[0]
    else:
        finish_str = str(finish)

    mid_ns = span.start_time_unix_nano + (span.end_time_unix_nano - span.start_time_unix_nano) // 2

    if op == GenAIOperation.CHAT.value:
        if is_err:
            body = (
                f"LLM inference failed model={model} "
                f"error={span.attributes.get('error.type', 'unknown')} "
                f"input_tokens={in_tok}"
            )
            level = LogLevel.ERROR
        elif finish_str == FinishReason.LENGTH.value:
            body = (
                f"LLM response truncated at token limit model={model} "
                f"input_tokens={in_tok} output_tokens={out_tok} "
                f"finish_reason=length — consider increasing max_tokens"
            )
            level = LogLevel.WARN
        else:
            ttft = span.attributes.get("gen_ai.server.ttft_ms", 0)
            body = (
                f"LLM inference complete model={model} "
                f"input_tokens={in_tok} output_tokens={out_tok} "
                f"ttft_ms={ttft:.1f} finish_reason={finish_str}"
            )
            level = LogLevel.INFO

    elif op == GenAIOperation.EMBEDDINGS.value:
        batch = span.attributes.get("gen_ai.embedding.batch_size", 1)
        body = f"Embeddings generated model={model} batch_size={batch} input_tokens={in_tok}"
        level = LogLevel.ERROR if is_err else LogLevel.DEBUG

    elif op == GenAIOperation.RETRIEVE.value:
        returned = span.attributes.get("db.vector.query.returned", 0)
        top_k = span.attributes.get("db.vector.query.top_k", 0)
        body = (
            f"Vector search complete collection={span.attributes.get('db.collection.name', '?')} "
            f"top_k={top_k} returned={returned} "
            f"duration_ms={span.duration_ms:.1f}"
        )
        level = LogLevel.ERROR if is_err else LogLevel.DEBUG

    elif op == GenAIOperation.EXECUTE_TOOL.value:
        tool = span.attributes.get("gen_ai.tool.name", "unknown")
        tool_type = span.attributes.get("gen_ai.tool.type", "")
        mcp = "mcp.session.id" in span.attributes
        prefix = "MCP tool" if mcp else f"Tool[{tool_type}]" if tool_type else "Tool"
        body = f"{prefix} execution tool={tool} duration_ms={span.duration_ms:.1f} " + (
            "error=" + str(span.attributes.get("error.type", "")) if is_err else "ok=true"
        )
        level = LogLevel.ERROR if is_err else LogLevel.INFO

    elif op == GenAIOperation.CREATE_AGENT.value:
        agent_name = span.attributes.get("gen_ai.agent.name", "unknown")
        agent_id = span.attributes.get("gen_ai.agent.id", "")
        if is_err:
            body = f"Agent creation failed agent={agent_name} error={span.attributes.get('error.type', 'unknown')}"
            level = LogLevel.ERROR
        else:
            body = f"Agent provisioned agent={agent_name} agent_id={agent_id} duration_ms={span.duration_ms:.1f}"
            level = LogLevel.INFO

    elif op == GenAIOperation.INVOKE_AGENT.value:
        agent_name = span.attributes.get("gen_ai.agent.name", "unknown")
        in_tok = span.attributes.get("gen_ai.usage.input_tokens", 0)
        out_tok = span.attributes.get("gen_ai.usage.output_tokens", 0)
        conv = span.attributes.get("gen_ai.conversation.id", "")
        if is_err:
            body = (
                f"Agent invocation failed agent={agent_name} "
                f"error={span.attributes.get('error.type', 'unknown')} "
                f"conversation_id={conv}"
            )
            level = LogLevel.ERROR
        else:
            body = (
                f"Agent invocation complete agent={agent_name} "
                f"input_tokens={in_tok} output_tokens={out_tok} "
                f"duration_ms={span.duration_ms:.1f} conversation_id={conv}"
            )
            level = LogLevel.INFO

    elif op in (
        GenAIOperation.SEARCH_MEMORY.value,
        GenAIOperation.CREATE_MEMORY.value,
        GenAIOperation.UPSERT_MEMORY.value,
        GenAIOperation.DELETE_MEMORY.value,
    ):
        store = span.attributes.get("gen_ai.memory.store.id", "unknown")
        count = span.attributes.get("gen_ai.memory.record.count", 0)
        if is_err:
            body = (
                f"Memory operation failed op={op} store={store} "
                f"error={span.attributes.get('error.type', 'unknown')}"
            )
            level = LogLevel.ERROR
        elif op == GenAIOperation.SEARCH_MEMORY.value:
            query = span.attributes.get("gen_ai.memory.query.text", "")
            body = (
                f"Memory search complete store={store} records_returned={count} "
                f"duration_ms={span.duration_ms:.1f} query=\"{query}\""
            )
            level = LogLevel.DEBUG
        else:
            body = (
                f"Memory {op} complete store={store} records_affected={count} "
                f"duration_ms={span.duration_ms:.1f}"
            )
            level = LogLevel.DEBUG

    elif op == GenAIOperation.INVOKE_WORKFLOW.value:
        wf_name = span.attributes.get("gen_ai.workflow.name", "unknown")
        if is_err:
            body = (
                f"Workflow failed workflow={wf_name} "
                f"error={span.attributes.get('error.type', 'unknown')} "
                f"duration_ms={span.duration_ms:.1f}"
            )
            level = LogLevel.ERROR
        else:
            body = f"Workflow complete workflow={wf_name} duration_ms={span.duration_ms:.1f}"
            level = LogLevel.INFO

    elif op == GenAIOperation.PLAN.value:
        agent_name = span.attributes.get("gen_ai.agent.name", "unknown")
        if is_err:
            body = f"Agent planning failed agent={agent_name} error={span.attributes.get('error.type', 'unknown')}"
            level = LogLevel.ERROR
        else:
            body = f"Agent planning complete agent={agent_name} duration_ms={span.duration_ms:.1f}"
            level = LogLevel.DEBUG

    else:
        return logs

    logs.append(
        LogEntry(
            timestamp_unix_nano=mid_ns,
            service_name=svc.name,
            level=level,
            body=body,
            trace_id=span.trace_id,
            span_id=span.span_id,
            attributes={
                "gen_ai.operation.name": op,
                "gen_ai.request.model": model,
            },
            resource_attributes=svc.resource_attributes(),
        )
    )
    return logs
