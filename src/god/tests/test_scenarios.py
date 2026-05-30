"""Tests for the scenario engine and fault injection presets."""

from __future__ import annotations

import pytest

from god.scenarios import (
    SCENARIO_PRESETS,
    ScenarioEngine,
    ScenarioState,
    agent_bootstrap_failure,
    auth_outage,
    db_degradation,
    kafka_lag,
    llm_latency_spike,
    memory_store_degradation,
    payments_error_burst,
    provider_failover,
    retrieval_quality_drop,
    token_budget_breach,
    tool_call_timeout,
    vector_db_slowdown,
)
from god.services import ServiceRegistry


def _registry() -> ServiceRegistry:
    return ServiceRegistry()


def test_scenario_starts_pending() -> None:
    s = db_degradation(start_s=60, duration_s=30)
    assert s.state == ScenarioState.PENDING


def test_scenario_activates_at_start_offset() -> None:
    reg = _registry()
    s = db_degradation(start_s=10, duration_s=30)
    s.tick(5.0, reg)
    assert s.state == ScenarioState.PENDING
    s.tick(10.0, reg)
    assert s.state == ScenarioState.ACTIVE


def test_scenario_recovers_at_end_offset() -> None:
    reg = _registry()
    s = auth_outage(start_s=0, duration_s=10)
    s.tick(0.0, reg)
    assert s.state == ScenarioState.ACTIVE
    s.tick(10.0, reg)
    assert s.state == ScenarioState.DONE


def test_scenario_done_does_not_reactivate() -> None:
    reg = _registry()
    s = auth_outage(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    assert s.state == ScenarioState.DONE
    s.tick(100.0, reg)
    assert s.state == ScenarioState.DONE


def test_db_degradation_applies_fault() -> None:
    reg = _registry()
    s = db_degradation("postgres-payments", start_s=0, duration_s=30, error_rate=0.5)
    s.tick(0.0, reg)
    pg = reg.get("postgres-payments")
    assert pg.effective_error_rate == pytest.approx(0.5)


def test_db_degradation_clears_on_recover() -> None:
    reg = _registry()
    s = db_degradation("postgres-payments", start_s=0, duration_s=5, error_rate=0.9)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    pg = reg.get("postgres-payments")
    assert pg._error_rate_override is None


def test_llm_latency_spike_applies_multiplier() -> None:
    reg = _registry()
    s = llm_latency_spike(start_s=0, duration_s=10, ttft_multiplier=4.0, error_rate=0.2)
    s.tick(0.0, reg)
    llm = reg.get("llm-gemini")
    assert llm._latency_multiplier == pytest.approx(4.0)
    assert llm.effective_error_rate == pytest.approx(0.2)


def test_auth_outage_100pct_errors() -> None:
    reg = _registry()
    s = auth_outage(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    assert reg.get("auth-service").effective_error_rate == pytest.approx(1.0)


def test_scenario_engine_tick_activates() -> None:
    reg = _registry()
    engine = ScenarioEngine(reg, [db_degradation(start_s=10, duration_s=30)])
    activated = engine.tick(5.0)
    assert activated == []
    activated = engine.tick(10.0)
    assert "db_degradation" in activated


def test_scenario_engine_active_names() -> None:
    reg = _registry()
    engine = ScenarioEngine(
        reg,
        [
            db_degradation(start_s=0, duration_s=30),
            auth_outage(start_s=0, duration_s=30),
        ],
    )
    engine.tick(0.0)
    names = engine.active_scenario_names()
    assert "db_degradation" in names
    assert "auth_outage" in names


def test_scenario_engine_active_tags() -> None:
    reg = _registry()
    engine = ScenarioEngine(reg, [vector_db_slowdown(start_s=0, duration_s=60)])
    engine.tick(0.0)
    tags = engine.active_tags()
    assert "vector_db" in tags
    assert "latency" in tags


def test_scenario_engine_from_preset_cascading_failure() -> None:
    reg = _registry()
    engine = ScenarioEngine.from_preset("cascading_failure", reg)
    assert len(engine.scenarios) == 4


def test_scenario_engine_from_preset_quiet() -> None:
    reg = _registry()
    engine = ScenarioEngine.from_preset("quiet", reg)
    assert engine.scenarios == []


def test_scenario_engine_from_preset_unknown_raises() -> None:
    reg = _registry()
    with pytest.raises(ValueError, match="Unknown scenario preset"):
        ScenarioEngine.from_preset("does_not_exist", reg)


def test_all_presets_return_list() -> None:
    for name, factory in SCENARIO_PRESETS.items():
        scenarios = factory()
        assert isinstance(scenarios, list), f"Preset '{name}' did not return a list"


def test_scenario_duration_property() -> None:
    s = db_degradation(start_s=10, duration_s=90)
    assert s.duration_s == pytest.approx(90)


def test_llm_latency_spike_recover_restores_multiplier() -> None:
    reg = _registry()
    llm = reg.get("llm-gemini")
    original_multiplier = llm._latency_multiplier
    s = llm_latency_spike(start_s=0, duration_s=5, ttft_multiplier=4.0, error_rate=0.1)
    s.tick(0.0, reg)
    assert llm._latency_multiplier == pytest.approx(4.0)
    s.tick(5.0, reg)
    assert llm._latency_multiplier == pytest.approx(original_multiplier)
    assert llm._error_rate_override is None


def test_token_budget_breach_sets_zero_error_rate() -> None:
    reg = _registry()
    s = token_budget_breach(start_s=0, duration_s=10)
    s.tick(0.0, reg)
    llm = reg.get("llm-gemini")
    assert llm._error_rate_override == pytest.approx(0.0)


def test_token_budget_breach_clears_on_recover() -> None:
    reg = _registry()
    s = token_budget_breach(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    assert reg.get("llm-gemini")._error_rate_override is None


def test_provider_failover_applies_high_error_rate() -> None:
    reg = _registry()
    s = provider_failover(start_s=0, duration_s=10)
    s.tick(0.0, reg)
    assert reg.get("llm-gemini").effective_error_rate == pytest.approx(0.95)


def test_provider_failover_recovers() -> None:
    reg = _registry()
    s = provider_failover(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    llm = reg.get("llm-gemini")
    assert llm._error_rate_override is None
    assert llm._latency_multiplier == pytest.approx(1.0)


def test_payments_error_burst_applies_fault() -> None:
    reg = _registry()
    s = payments_error_burst(start_s=0, duration_s=10)
    s.tick(0.0, reg)
    assert reg.get("payments-service").effective_error_rate == pytest.approx(0.6)


def test_payments_error_burst_recovers() -> None:
    reg = _registry()
    s = payments_error_burst(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    assert reg.get("payments-service")._error_rate_override is None
    assert reg.get("postgres-payments")._error_rate_override is None


def test_kafka_lag_applies_latency() -> None:
    reg = _registry()
    s = kafka_lag(start_s=0, duration_s=10)
    s.tick(0.0, reg)
    assert reg.get("kafka")._latency_multiplier == pytest.approx(10.0)


def test_kafka_lag_recovers() -> None:
    reg = _registry()
    s = kafka_lag(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    assert reg.get("kafka")._error_rate_override is None
    assert reg.get("kafka")._latency_multiplier == pytest.approx(1.0)


def test_tool_call_timeout_applies_fault() -> None:
    reg = _registry()
    s = tool_call_timeout(start_s=0, duration_s=10, error_rate=0.55, latency_multiplier=12.0)
    s.tick(0.0, reg)
    svc = reg.get("tool-executor")
    assert svc.effective_error_rate == pytest.approx(0.55)
    assert svc._latency_multiplier == pytest.approx(12.0)


def test_tool_call_timeout_recovers() -> None:
    reg = _registry()
    s = tool_call_timeout(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    svc = reg.get("tool-executor")
    assert svc._error_rate_override is None
    assert svc._latency_multiplier == pytest.approx(1.0)


def test_memory_store_degradation_applies_fault() -> None:
    reg = _registry()
    s = memory_store_degradation(start_s=0, duration_s=10, latency_multiplier=8.0, error_rate=0.25)
    s.tick(0.0, reg)
    svc = reg.get("memory-service")
    assert svc.effective_error_rate == pytest.approx(0.25)
    assert svc._latency_multiplier == pytest.approx(8.0)


def test_memory_store_degradation_recovers() -> None:
    reg = _registry()
    s = memory_store_degradation(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    svc = reg.get("memory-service")
    assert svc._error_rate_override is None
    assert svc._latency_multiplier == pytest.approx(1.0)


def test_agent_bootstrap_failure_applies_fault() -> None:
    reg = _registry()
    s = agent_bootstrap_failure(start_s=0, duration_s=10)
    s.tick(0.0, reg)
    assert reg.get("agent-service").effective_error_rate == pytest.approx(0.9)


def test_agent_bootstrap_failure_recovers() -> None:
    reg = _registry()
    s = agent_bootstrap_failure(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    svc = reg.get("agent-service")
    assert svc._error_rate_override is None
    assert svc._latency_multiplier == pytest.approx(1.0)


def test_retrieval_quality_drop_applies_fault() -> None:
    reg = _registry()
    s = retrieval_quality_drop(start_s=0, duration_s=10, latency_multiplier=3.0, error_rate=0.08)
    s.tick(0.0, reg)
    svc = reg.get("vector-db")
    assert svc.effective_error_rate == pytest.approx(0.08)
    assert svc._latency_multiplier == pytest.approx(3.0)


def test_retrieval_quality_drop_recovers() -> None:
    reg = _registry()
    s = retrieval_quality_drop(start_s=0, duration_s=5)
    s.tick(0.0, reg)
    s.tick(5.0, reg)
    svc = reg.get("vector-db")
    assert svc._error_rate_override is None
    assert svc._latency_multiplier == pytest.approx(1.0)


def test_engine_activated_scenario_names_includes_done() -> None:
    reg = _registry()
    engine = ScenarioEngine(reg, [auth_outage(start_s=0, duration_s=5)])
    engine.tick(0.0)
    engine.tick(5.0)
    assert engine.active_scenario_names() == []
    assert "auth_outage" in engine.activated_scenario_names()


def test_engine_activated_names_includes_active_and_done() -> None:
    reg = _registry()
    engine = ScenarioEngine(
        reg,
        [
            auth_outage(start_s=0, duration_s=5),
            db_degradation(start_s=0, duration_s=100),
        ],
    )
    engine.tick(0.0)
    engine.tick(5.0)
    activated = engine.activated_scenario_names()
    assert "auth_outage" in activated
    assert "db_degradation" in activated


def test_engine_active_tags_empty_when_no_active_scenarios() -> None:
    reg = _registry()
    engine = ScenarioEngine(reg, [auth_outage(start_s=100, duration_s=10)])
    assert engine.active_tags() == set()


@pytest.mark.parametrize(
    "preset,expected_count",
    [
        ("event_degradation", 1),
        ("db_incident", 1),
        ("llm_degradation", 2),
        ("rag_meltdown", 3),
        ("payment_outage", 2),
        ("cascading_failure", 4),
        ("tool_meltdown", 2),
        ("memory_incident", 2),
        ("agent_failure", 1),
        ("rag_quality_degradation", 2),
        ("agentic_cascading_failure", 4),
        ("quiet", 0),
    ],
)
def test_preset_scenario_counts(preset: str, expected_count: int) -> None:
    scenarios = SCENARIO_PRESETS[preset]()
    assert len(scenarios) == expected_count
