"""
Trace orchestrator.

Assembles full distributed traces from the service topology by walking
the dependency graph and generating correlated spans and logs.
Also generates standalone log bursts (background noise logs not tied
to a specific user-initiated request).
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

from god.generators.genai import (
    build_embedding_span,
    build_inference_span,
    build_rag_trace,
    build_retrieval_span,
    generate_genai_logs,
)
from god.generators.infra import (
    build_db_span,
    build_grpc_span,
    build_http_span,
    build_queue_span,
    generate_logs_for_span,
)
from god.models import LogEntry, Span, Trace, new_id
from god.scenarios import ScenarioEngine
from god.services import ServiceRegistry, ServiceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(ms: float) -> int:
    return int(ms * 1_000_000)


def _scenario_ttft_multiplier(engine: ScenarioEngine | None) -> float:
    if engine and "ttft" in engine.active_tags():
        return 5.0
    return 1.0


def _scenario_retrieval_multiplier(engine: ScenarioEngine | None) -> float:
    if engine and "vector_db" in engine.active_tags():
        return 6.0
    return 1.0


# ---------------------------------------------------------------------------
# Single span builder dispatch
# ---------------------------------------------------------------------------


def _build_span_for_service(
    svc,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
) -> tuple[list[Span], list[LogEntry], float]:
    """
    Build 1..N spans and associated logs for a single service call.
    Returns (spans, logs, end_time_ms).
    """
    spans: list[Span] = []
    logs: list[LogEntry] = []

    if svc.type == ServiceType.HTTP_API:
        span, end_ms = build_http_span(svc, trace_id, parent_span_id, start_ms, rng)
        spans.append(span)
        logs.extend(generate_logs_for_span(span, svc, rng))

    elif svc.type == ServiceType.GRPC_SERVICE:
        span, end_ms = build_grpc_span(svc, trace_id, parent_span_id, start_ms, rng)
        spans.append(span)
        logs.extend(generate_logs_for_span(span, svc, rng))

    elif svc.type == ServiceType.DATABASE:
        span, end_ms = build_db_span(svc, trace_id, parent_span_id, start_ms, rng)
        spans.append(span)
        logs.extend(generate_logs_for_span(span, svc, rng))

    elif svc.type == ServiceType.QUEUE:
        span, end_ms = build_queue_span(svc, trace_id, parent_span_id, start_ms, rng)
        spans.append(span)
        logs.extend(generate_logs_for_span(span, svc, rng))

    elif svc.type == ServiceType.GENAI_LLM:
        span, end_ms = build_inference_span(svc, trace_id, parent_span_id, start_ms, rng)
        spans.append(span)
        logs.extend(generate_genai_logs(span, svc, rng))

    elif svc.type == ServiceType.GENAI_EMBED:
        span, end_ms = build_embedding_span(svc, trace_id, parent_span_id, start_ms, rng)
        spans.append(span)
        logs.extend(generate_genai_logs(span, svc, rng))

    elif svc.type == ServiceType.GENAI_VECTOR_DB:
        span, end_ms = build_retrieval_span(svc, trace_id, parent_span_id, start_ms, rng)
        spans.append(span)
        logs.extend(generate_genai_logs(span, svc, rng))

    elif svc.type in (ServiceType.GENAI_RAG, ServiceType.GENAI_AGENT):
        # RAG builds its own multi-span subtrace
        rag_spans, end_ms = build_rag_trace(
            None,  # registry injected below, handled in generate_trace
            trace_id,
            parent_span_id,
            start_ms,
            rng,
        )
        spans.extend(rag_spans)
        for s in rag_spans:
            svc_obj = _mock_svc_for_span(s)
            logs.extend(generate_genai_logs(s, svc_obj, rng))
    else:
        end_ms = start_ms + 10

    return spans, logs, end_ms


def _mock_svc_for_span(span: Span):
    """Create a minimal ServiceDef-like object for log generation."""
    from god.services import LatencyProfile, ServiceDef, ServiceType

    return ServiceDef(
        name=span.service_name,
        type=ServiceType.GENAI_LLM,
        latency=LatencyProfile(p50_ms=100, p99_ms=1000),
    )


# ---------------------------------------------------------------------------
# Full trace generator
# ---------------------------------------------------------------------------


def generate_trace(
    registry: ServiceRegistry,
    start_ms: float,
    rng: np.random.Generator,
    engine: ScenarioEngine | None = None,
    capture_content: bool = False,
) -> tuple[Trace, list[LogEntry]]:
    """
    Generate a single end-to-end trace starting at api-gateway.
    The trace walks the dependency graph, with child spans starting
    after a small propagation delay from their parent.

    Returns (Trace, logs).
    """
    trace_id = new_id(32)
    all_spans: list[Span] = []
    all_logs: list[LogEntry] = []

    ttft_mult = _scenario_ttft_multiplier(engine)
    retrieval_mult = _scenario_retrieval_multiplier(engine)

    # ── api-gateway (root) ────────────────────────────────────────────────
    gw_svc = registry.get("api-gateway")
    gw_span, gw_end = build_http_span(gw_svc, trace_id, None, start_ms, rng)
    all_spans.append(gw_span)
    all_logs.extend(generate_logs_for_span(gw_span, gw_svc, rng))

    # Small propagation delay before child spans
    child_start = start_ms + float(rng.uniform(1, 5))

    # ── auth-service ──────────────────────────────────────────────────────
    auth_svc = registry.get("auth-service")
    auth_span, auth_end = build_grpc_span(auth_svc, trace_id, gw_span.span_id, child_start, rng)
    all_spans.append(auth_span)
    all_logs.extend(generate_logs_for_span(auth_span, auth_svc, rng))

    # postgres-auth child of auth
    pg_auth_svc = registry.get("postgres-auth")
    pg_auth_span, _ = build_db_span(
        pg_auth_svc, trace_id, auth_span.span_id, auth_end - float(rng.uniform(10, 30)), rng
    )
    all_spans.append(pg_auth_span)
    all_logs.extend(generate_logs_for_span(pg_auth_span, pg_auth_svc, rng))

    # ── Decide which downstream path to take ─────────────────────────────
    # 50% RAG, 35% payments, 15% both
    path_roll = rng.random()

    if path_roll < 0.50 or path_roll >= 0.85:
        # RAG pipeline
        rag_start = auth_end + float(rng.uniform(2, 8))
        rag_spans, rag_end = build_rag_trace(
            registry,
            trace_id,
            gw_span.span_id,
            rag_start,
            rng,
            capture_content=capture_content,
            ttft_multiplier=ttft_mult,
            retrieval_latency_multiplier=retrieval_mult,
        )
        all_spans.extend(rag_spans)
        for s in rag_spans:
            svc_name = s.service_name
            try:
                svc = registry.get(svc_name)
            except KeyError:
                svc = _mock_svc_for_span(s)
            all_logs.extend(generate_genai_logs(s, svc, rng))

    if path_roll >= 0.35:
        # Payments path
        pay_svc = registry.get("payments-service")
        pay_start = auth_end + float(rng.uniform(2, 8))
        pay_span, pay_end = build_http_span(pay_svc, trace_id, gw_span.span_id, pay_start, rng)
        all_spans.append(pay_span)
        all_logs.extend(generate_logs_for_span(pay_span, pay_svc, rng))

        # postgres-payments
        pg_pay = registry.get("postgres-payments")
        pg_pay_span, _ = build_db_span(pg_pay, trace_id, pay_span.span_id, pay_end - float(rng.uniform(20, 80)), rng)
        all_spans.append(pg_pay_span)
        all_logs.extend(generate_logs_for_span(pg_pay_span, pg_pay, rng))

        # kafka publish
        kafka_svc = registry.get("kafka")
        kafka_span, _ = build_queue_span(
            kafka_svc, trace_id, pay_span.span_id, pay_end - float(rng.uniform(5, 20)), rng, is_producer=True
        )
        all_spans.append(kafka_span)
        all_logs.extend(generate_logs_for_span(kafka_span, kafka_svc, rng))

    trace = Trace(trace_id=trace_id, spans=all_spans)
    return trace, all_logs


# ---------------------------------------------------------------------------
# Bulk generator
# ---------------------------------------------------------------------------


def generate_dataset(
    registry: ServiceRegistry,
    n_traces: int = 100,
    base_epoch_ms: float = 0.0,
    traces_per_second: float = 10.0,
    rng_seed: int | None = None,
    engine: ScenarioEngine | None = None,
    capture_content: bool = False,
) -> Generator[tuple[Trace, list[LogEntry]], None, None]:
    """
    Yield (trace, logs) pairs at the requested rate.

    Virtual time advances at 1s intervals per (1 / traces_per_second) * n steps.
    ScenarioEngine.tick() is called for each virtual second.

    Args:
        registry:          ServiceRegistry with service definitions
        n_traces:          Total number of traces to generate
        base_epoch_ms:     Starting epoch millisecond timestamp
        traces_per_second: Simulated request arrival rate
        rng_seed:          Set for deterministic output (test fixtures)
        engine:            ScenarioEngine for fault injection (optional)
        capture_content:   Include gen_ai.input/output.messages (large, opt-in)
    """
    rng = np.random.default_rng(rng_seed)
    interval_ms = 1000.0 / traces_per_second
    last_tick_s = -1.0

    for i in range(n_traces):
        elapsed_ms = i * interval_ms
        elapsed_s = elapsed_ms / 1000.0

        # Tick scenario engine once per virtual second
        if engine and int(elapsed_s) > last_tick_s:
            last_tick_s = int(elapsed_s)
            activated = engine.tick(elapsed_s)
            if activated:
                # Emit a synthetic log for the scenario activation
                pass  # handled in main.py

        # Jitter arrival time (Poisson-ish)
        jitter_ms = float(rng.exponential(interval_ms))
        start_ms = base_epoch_ms + elapsed_ms + jitter_ms

        trace, logs = generate_trace(
            registry,
            start_ms,
            rng,
            engine=engine,
            capture_content=capture_content,
        )
        yield trace, logs
