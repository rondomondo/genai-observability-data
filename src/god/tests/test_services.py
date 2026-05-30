"""Tests for service registry and dependency graph."""

from __future__ import annotations

import pytest

from god.services import LatencyProfile, ServiceDef, ServiceRegistry, ServiceType


def test_latency_profile_mu_sigma_positive() -> None:
    lp = LatencyProfile(p50_ms=50, p99_ms=500)
    mu, sigma = lp.mu_sigma()
    assert mu > 0
    assert sigma > 0


def test_latency_profile_timeout_default() -> None:
    lp = LatencyProfile(p50_ms=10, p99_ms=100)
    assert lp.timeout_ms == 5000.0


def test_service_def_effective_error_rate_no_override() -> None:
    svc = ServiceDef(name="x", type=ServiceType.HTTP_API, error_rate=0.05)
    assert svc.effective_error_rate == pytest.approx(0.05)


def test_service_def_effective_error_rate_override() -> None:
    svc = ServiceDef(name="x", type=ServiceType.HTTP_API, error_rate=0.05)
    svc._error_rate_override = 0.9
    assert svc.effective_error_rate == pytest.approx(0.9)


def test_service_def_effective_latency_no_multiplier() -> None:
    svc = ServiceDef(name="x", type=ServiceType.HTTP_API, latency=LatencyProfile(10, 100))
    assert svc.effective_latency.p50_ms == pytest.approx(10)
    assert svc.effective_latency.p99_ms == pytest.approx(100)


def test_service_def_effective_latency_with_multiplier() -> None:
    svc = ServiceDef(name="x", type=ServiceType.HTTP_API, latency=LatencyProfile(10, 100))
    svc._latency_multiplier = 3.0
    el = svc.effective_latency
    assert el.p50_ms == pytest.approx(30)
    assert el.p99_ms == pytest.approx(300)


def test_service_def_resource_attributes() -> None:
    svc = ServiceDef(
        name="my-svc",
        type=ServiceType.GENAI_LLM,
        version="2.0.0",
        environment="staging",
        provider="google",
    )
    attrs = svc.resource_attributes()
    assert attrs["service.name"] == "my-svc"
    assert attrs["service.version"] == "2.0.0"
    assert attrs["deployment.environment"] == "staging"
    assert attrs["gen_ai.provider.name"] == "google"


def test_registry_get_known_service() -> None:
    reg = ServiceRegistry()
    svc = reg.get("api-gateway")
    assert svc.name == "api-gateway"
    assert svc.type == ServiceType.HTTP_API


def test_registry_get_unknown_raises() -> None:
    reg = ServiceRegistry()
    with pytest.raises(KeyError):
        reg.get("does-not-exist")


def test_registry_all_returns_all_defaults() -> None:
    reg = ServiceRegistry()
    names = {s.name for s in reg.all()}
    assert "api-gateway" in names
    assert "llm-gemini" in names
    assert "vector-db" in names


def test_registry_dependencies_of() -> None:
    reg = ServiceRegistry()
    deps = reg.dependencies_of("rag-service")
    dep_names = {d.name for d in deps}
    assert "embed-service" in dep_names
    assert "vector-db" in dep_names
    assert "llm-gemini" in dep_names


def test_registry_dependents_of() -> None:
    reg = ServiceRegistry()
    callers = reg.dependents_of("postgres-auth")
    assert any(c.name == "auth-service" for c in callers)


def test_registry_propagate_fault_and_clear() -> None:
    reg = ServiceRegistry()
    reg.propagate_fault("postgres-payments", error_rate=0.8, latency_multiplier=5.0)

    pg = reg.get("postgres-payments")
    assert pg.effective_error_rate == pytest.approx(0.8)
    assert pg.effective_latency.p50_ms == pytest.approx(pg.latency.p50_ms * 5.0)

    # Upstream bleed (payments-service depends on postgres-payments)
    pay = reg.get("payments-service")
    assert pay._error_rate_override is not None
    assert pay._error_rate_override > 0

    reg.clear_faults()
    assert pg._error_rate_override is None
    assert pg._latency_multiplier == pytest.approx(1.0)
    assert pay._error_rate_override is None


def test_registry_genai_services() -> None:
    reg = ServiceRegistry()
    genai = reg.genai_services()
    types = {s.type for s in genai}
    assert ServiceType.GENAI_LLM in types
    assert ServiceType.GENAI_EMBED in types
    assert ServiceType.HTTP_API not in types


def test_registry_custom_services() -> None:
    custom = [ServiceDef(name="my-llm", type=ServiceType.GENAI_LLM)]
    reg = ServiceRegistry(custom)
    assert reg.get("my-llm").name == "my-llm"
    with pytest.raises(KeyError):
        reg.get("api-gateway")
