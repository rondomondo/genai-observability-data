"""Tests for the scenario engine and fault injection presets."""
from __future__ import annotations

import pytest

from god.scenarios import (
    SCENARIO_PRESETS,
    Scenario,
    ScenarioEngine,
    ScenarioState,
    auth_outage,
    db_degradation,
    kafka_lag,
    llm_latency_spike,
    payments_error_burst,
    provider_failover,
    token_budget_breach,
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
    llm = reg.get("llm-anthropic")
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
    engine = ScenarioEngine(reg, [
        db_degradation(start_s=0, duration_s=30),
        auth_outage(start_s=0, duration_s=30),
    ])
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
    reg = _registry()
    for name, factory in SCENARIO_PRESETS.items():
        scenarios = factory()
        assert isinstance(scenarios, list), f"Preset '{name}' did not return a list"


def test_scenario_duration_property() -> None:
    s = db_degradation(start_s=10, duration_s=90)
    assert s.duration_s == pytest.approx(90)
