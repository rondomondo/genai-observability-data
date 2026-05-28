"""Tests for the pydantic-ai scenario analysis agent in mock mode."""

from __future__ import annotations

import pytest

from god.mock.activate import activate_mock_mode, deactivate_mock_mode, is_active


@pytest.fixture(autouse=True)
def _mock_mode():
    """Activate mock mode for all agent tests; clean up after."""
    activate_mock_mode()
    yield
    deactivate_mock_mode()


def test_mock_mode_flag_set() -> None:
    assert is_active()


@pytest.mark.asyncio
async def test_analyse_batch_returns_analysis() -> None:
    from god.agent import analyse_batch
    from god.deps import GodDeps
    from god.orchestrator import generate_dataset
    from god.services import ServiceRegistry

    reg = ServiceRegistry()
    deps = GodDeps(registry=reg)
    traces_and_logs = list(generate_dataset(reg, n_traces=10, rng_seed=0))
    traces = [t for t, _ in traces_and_logs]
    logs = [log for _, ls in traces_and_logs for log in ls]

    result = await analyse_batch(traces, logs, deps)

    assert isinstance(result.error_rate, float)
    assert 0.0 <= result.error_rate <= 1.0
    assert result.p99_latency_ms >= 0.0
    assert isinstance(result.anomaly_signals, list)
    assert isinstance(result.recommended_action, str)
    assert len(result.recommended_action) > 0
    assert isinstance(result.summary, str)


@pytest.mark.asyncio
async def test_analyse_batch_with_active_scenario() -> None:
    from god.agent import analyse_batch
    from god.deps import GodDeps
    from god.orchestrator import generate_dataset
    from god.scenarios import ScenarioEngine, llm_latency_spike
    from god.services import ServiceRegistry

    reg = ServiceRegistry()
    engine = ScenarioEngine(reg, [llm_latency_spike(start_s=0, duration_s=1000)])
    engine.tick(0.0)

    deps = GodDeps(registry=reg, engine=engine)
    traces_and_logs = list(generate_dataset(reg, n_traces=10, rng_seed=1, engine=engine))
    traces = [t for t, _ in traces_and_logs]
    logs = [log for _, ls in traces_and_logs for log in ls]

    result = await analyse_batch(traces, logs, deps)

    assert "llm_latency_spike" in result.active_scenarios


@pytest.mark.asyncio
async def test_analyse_batch_high_error_rate_signals() -> None:
    from god.agent import analyse_batch
    from god.deps import GodDeps
    from god.orchestrator import generate_dataset
    from god.services import ServiceRegistry

    reg = ServiceRegistry()
    for svc in reg.all():
        svc._error_rate_override = 0.9

    deps = GodDeps(registry=reg)
    traces_and_logs = list(generate_dataset(reg, n_traces=20, rng_seed=2))
    traces = [t for t, _ in traces_and_logs]
    logs = [log for _, ls in traces_and_logs for log in ls]

    result = await analyse_batch(traces, logs, deps)

    assert result.error_rate > 0.3
    assert any(s.severity in ("high", "critical") for s in result.anomaly_signals)


@pytest.mark.asyncio
async def test_analyse_batch_nominal_no_signals() -> None:
    from god.agent import analyse_batch
    from god.deps import GodDeps
    from god.orchestrator import generate_dataset
    from god.services import ServiceRegistry

    reg = ServiceRegistry()
    for svc in reg.all():
        svc._error_rate_override = 0.0

    deps = GodDeps(registry=reg)
    traces_and_logs = list(generate_dataset(reg, n_traces=10, rng_seed=3))
    traces = [t for t, _ in traces_and_logs]
    logs = [log for _, ls in traces_and_logs for log in ls]

    result = await analyse_batch(traces, logs, deps)

    assert result.error_rate == pytest.approx(0.0)
