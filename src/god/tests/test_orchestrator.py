"""Tests for the orchestrator — full trace generation and dataset iteration."""

from __future__ import annotations

import numpy as np

from god.models import SpanStatus, Trace
from god.orchestrator import generate_dataset, generate_trace
from god.scenarios import ScenarioState
from god.services import ServiceRegistry


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_generate_trace_returns_trace_and_logs() -> None:
    reg = ServiceRegistry()
    trace, logs = generate_trace(reg, start_ms=1000.0, rng=_rng())
    assert isinstance(trace, Trace)
    assert len(trace.spans) >= 3
    assert isinstance(logs, list)
    assert len(logs) > 0


def test_generate_trace_has_root_span() -> None:
    reg = ServiceRegistry()
    trace, _ = generate_trace(reg, start_ms=0.0, rng=_rng())
    root = trace.root_span()
    assert root is not None
    assert root.parent_span_id is None


def test_generate_trace_spans_ordered_in_time() -> None:
    reg = ServiceRegistry()
    trace, _ = generate_trace(reg, start_ms=0.0, rng=_rng())
    for span in trace.spans:
        assert span.end_time_unix_nano >= span.start_time_unix_nano


def test_generate_trace_all_spans_share_trace_id() -> None:
    reg = ServiceRegistry()
    trace, _ = generate_trace(reg, start_ms=0.0, rng=_rng())
    assert all(s.trace_id == trace.trace_id for s in trace.spans)


def test_generate_trace_produces_valid_structure() -> None:
    reg = ServiceRegistry()
    trace, logs = generate_trace(reg, start_ms=0.0, rng=_rng(99))
    assert trace.trace_id is not None
    assert len(trace.spans) >= 3
    root = trace.root_span()
    assert root is not None
    assert root.service_name == "api-gateway"
    for span in trace.spans:
        assert span.end_time_unix_nano >= span.start_time_unix_nano


def test_generate_dataset_yields_n_traces() -> None:
    reg = ServiceRegistry()
    results = list(generate_dataset(reg, n_traces=10, rng_seed=0))
    assert len(results) == 10


def test_generate_dataset_all_traces_have_spans() -> None:
    reg = ServiceRegistry()
    for trace, logs in generate_dataset(reg, n_traces=5, rng_seed=1):
        assert len(trace.spans) > 0


def test_generate_dataset_timestamps_increase() -> None:
    reg = ServiceRegistry()
    base = 1_700_000_000_000.0
    starts = []
    for trace, _ in generate_dataset(reg, n_traces=20, base_epoch_ms=base, rng_seed=2):
        root = trace.root_span()
        if root:
            starts.append(root.start_time_unix_nano)
    # Timestamps are approximately increasing — Poisson jitter can occasionally invert
    # adjacent pairs, so we check the overall trend rather than strict monotonicity
    assert starts[0] < starts[-1], "First trace should start before last trace"
    assert min(starts) >= int(base * 1_000_000), "All starts should be at or after base epoch"


def test_generate_dataset_with_scenario_engine() -> None:
    from god.scenarios import ScenarioEngine, llm_latency_spike

    reg = ServiceRegistry()
    engine = ScenarioEngine(reg, [llm_latency_spike(start_s=0, duration_s=100)])
    results = list(generate_dataset(reg, n_traces=20, rng_seed=3, engine=engine))
    assert len(results) == 20


def test_generate_trace_with_scenario_tags() -> None:
    from god.scenarios import ScenarioEngine, vector_db_slowdown

    reg = ServiceRegistry()
    engine = ScenarioEngine(reg, [vector_db_slowdown(start_s=0, duration_s=100)])
    engine.tick(0.0)
    trace, _ = generate_trace(reg, start_ms=0.0, rng=_rng(), engine=engine)
    assert len(trace.spans) >= 3


def test_generate_trace_some_are_errors() -> None:
    """With high error rates, expect some error spans across many traces."""
    reg = ServiceRegistry()
    for svc in reg.all():
        svc._error_rate_override = 0.5

    error_count = 0
    for trace, _ in generate_dataset(reg, n_traces=30, rng_seed=4):
        error_count += sum(1 for s in trace.spans if s.status == SpanStatus.ERROR)
    assert error_count > 0


def test_generate_dataset_logs_correlated_to_traces() -> None:
    reg = ServiceRegistry()
    trace_ids = set()
    log_trace_ids = set()
    for trace, logs in generate_dataset(reg, n_traces=5, rng_seed=5):
        trace_ids.add(trace.trace_id)
        for log in logs:
            if log.trace_id:
                log_trace_ids.add(log.trace_id)
    assert trace_ids.issubset(log_trace_ids | {None})


def test_generate_trace_capture_content_enabled() -> None:
    """capture_content=True should not raise and still produce valid spans."""
    reg = ServiceRegistry()
    trace, logs = generate_trace(reg, start_ms=0.0, rng=_rng(7), capture_content=True)
    assert len(trace.spans) >= 3
    assert isinstance(logs, list)


def test_generate_dataset_capture_content() -> None:
    reg = ServiceRegistry()
    results = list(generate_dataset(reg, n_traces=5, rng_seed=8, capture_content=True))
    assert len(results) == 5
    for trace, logs in results:
        assert len(trace.spans) > 0


def test_generate_trace_rag_path() -> None:
    """Seed chosen to hit the RAG branch (path_roll < 0.35)."""
    reg = ServiceRegistry()
    # Run enough traces to hit RAG; seed 42 reliably hits it
    service_names = set()
    for _, _ in zip(range(50), (generate_trace(reg, start_ms=float(i * 100), rng=np.random.default_rng(i)) for i in range(50))):
        pass
    # Generate a batch and confirm rag-related services appear across them
    for i in range(30):
        trace, _ = generate_trace(reg, start_ms=float(i * 100), rng=np.random.default_rng(i))
        for span in trace.spans:
            service_names.add(span.service_name)
    assert any("llm" in n or "vector" in n or "embed" in n for n in service_names)


def test_generate_trace_payments_path() -> None:
    """Confirm payments-service spans appear across a batch of traces."""
    reg = ServiceRegistry()
    service_names: set[str] = set()
    for i in range(30):
        trace, _ = generate_trace(reg, start_ms=float(i * 100), rng=np.random.default_rng(i + 100))
        for span in trace.spans:
            service_names.add(span.service_name)
    assert "payments-service" in service_names


def test_generate_trace_agentic_path() -> None:
    """Confirm agentic workflow spans appear across a batch of traces."""
    reg = ServiceRegistry()
    service_names: set[str] = set()
    for i in range(30):
        trace, _ = generate_trace(reg, start_ms=float(i * 100), rng=np.random.default_rng(i + 200))
        for span in trace.spans:
            service_names.add(span.service_name)
    # Agentic path produces agent-service or workflow-related spans
    assert any("agent" in n or "workflow" in n or "tool" in n or "memory" in n for n in service_names)


def test_generate_dataset_scenario_tick_fires() -> None:
    """ScenarioEngine should activate mid-dataset, changing error rates for later traces."""
    from god.scenarios import ScenarioEngine, auth_outage

    reg = ServiceRegistry()
    # auth_outage fires at 5s; at 10 tps that's trace 50
    engine = ScenarioEngine(reg, [auth_outage(start_s=5, duration_s=100)])
    results = list(generate_dataset(reg, n_traces=200, traces_per_second=10.0, rng_seed=9, engine=engine))
    assert len(results) == 200
    # Engine should have activated the scenario (state is no longer all PENDING)
    assert engine.scenarios[0].state != ScenarioState.PENDING


def test_generate_dataset_zero_traces() -> None:
    reg = ServiceRegistry()
    results = list(generate_dataset(reg, n_traces=0, rng_seed=0))
    assert results == []


def test_generate_trace_auth_service_in_every_trace() -> None:
    """auth-service is always called — it is on the critical path."""
    reg = ServiceRegistry()
    for i in range(10):
        trace, _ = generate_trace(reg, start_ms=float(i * 100), rng=_rng(i))
        names = {s.service_name for s in trace.spans}
        assert "auth-service" in names
