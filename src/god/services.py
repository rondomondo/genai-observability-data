"""
Service registry and dependency graph.

Services are defined as dataclasses with latency profiles, error rates,
and upstream/downstream dependencies. The graph drives correlated failure
propagation in the scenario engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ServiceType(str, Enum):
    HTTP_API = "http_api"
    GRPC_SERVICE = "grpc_service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    GENAI_LLM = "genai_llm"
    GENAI_EMBED = "genai_embed"
    GENAI_VECTOR_DB = "genai_vector_db"
    GENAI_RAG = "genai_rag"
    GENAI_AGENT = "genai_agent"


@dataclass
class LatencyProfile:
    """
    Log-normal latency distribution parameters.
    p50_ms / p99_ms are used to derive mu/sigma for numpy.random.lognormal.
    """
    p50_ms: float
    p99_ms: float
    timeout_ms: float = 5000.0

    def mu_sigma(self) -> tuple[float, float]:
        """Derive lognormal mu/sigma from p50 and p99."""
        import math
        mu = math.log(self.p50_ms)
        # p99 is approximately mu + 2.326 * sigma for lognormal
        sigma = (math.log(self.p99_ms) - mu) / 2.326
        return mu, max(sigma, 0.01)


@dataclass
class ServiceDef:
    """Definition of a service in the topology."""
    name: str
    type: ServiceType
    version: str = "1.0.0"
    environment: str = "production"
    region: str = "eu-west-1"
    latency: LatencyProfile = field(default_factory=lambda: LatencyProfile(50, 500))
    error_rate: float = 0.01          # baseline probability of error
    dependencies: list[str] = field(default_factory=list)  # downstream service names
    # GenAI-specific
    provider: str | None = None       # gen_ai.provider.name
    model: str | None = None          # gen_ai.request.model
    # Runtime-injected overrides (set by scenario engine)
    _error_rate_override: float | None = field(default=None, repr=False)
    _latency_multiplier: float = field(default=1.0, repr=False)

    @property
    def effective_error_rate(self) -> float:
        return self._error_rate_override if self._error_rate_override is not None \
            else self.error_rate

    @property
    def effective_latency(self) -> LatencyProfile:
        if self._latency_multiplier == 1.0:
            return self.latency
        return LatencyProfile(
            p50_ms=self.latency.p50_ms * self._latency_multiplier,
            p99_ms=self.latency.p99_ms * self._latency_multiplier,
            timeout_ms=self.latency.timeout_ms,
        )

    def resource_attributes(self) -> dict:
        attrs = {
            "service.name": self.name,
            "service.version": self.version,
            "deployment.environment": self.environment,
            "cloud.region": self.region,
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        }
        if self.provider:
            attrs["gen_ai.provider.name"] = self.provider
        return attrs


# ---------------------------------------------------------------------------
# Default topology
# ---------------------------------------------------------------------------
#
#   api-gateway
#     ├── auth-service         → postgres-auth
#     ├── rag-service          → embed-service, vector-db, llm-anthropic
#     │     llm-anthropic      → tool: retrieval (vector-db)
#     └── payments-service     → postgres-payments, kafka
#
#   notifications-service (kafka consumer) → (no outbound deps in traces)

DEFAULT_SERVICES: list[ServiceDef] = [
    # ── Edge ──────────────────────────────────────────────────────────────
    ServiceDef(
        name="api-gateway",
        type=ServiceType.HTTP_API,
        latency=LatencyProfile(p50_ms=15, p99_ms=120),
        error_rate=0.005,
        dependencies=["auth-service", "rag-service", "payments-service"],
    ),

    # ── Auth ──────────────────────────────────────────────────────────────
    ServiceDef(
        name="auth-service",
        type=ServiceType.GRPC_SERVICE,
        latency=LatencyProfile(p50_ms=8, p99_ms=60),
        error_rate=0.003,
        dependencies=["postgres-auth"],
    ),
    ServiceDef(
        name="postgres-auth",
        type=ServiceType.DATABASE,
        latency=LatencyProfile(p50_ms=3, p99_ms=40, timeout_ms=3000),
        error_rate=0.002,
    ),

    # ── RAG pipeline ──────────────────────────────────────────────────────
    ServiceDef(
        name="rag-service",
        type=ServiceType.GENAI_RAG,
        latency=LatencyProfile(p50_ms=800, p99_ms=4000),
        error_rate=0.02,
        dependencies=["embed-service", "vector-db", "llm-anthropic"],
    ),
    ServiceDef(
        name="embed-service",
        type=ServiceType.GENAI_EMBED,
        latency=LatencyProfile(p50_ms=120, p99_ms=600),
        error_rate=0.01,
        provider="anthropic",
        model="voyage-3",
    ),
    ServiceDef(
        name="vector-db",
        type=ServiceType.GENAI_VECTOR_DB,
        latency=LatencyProfile(p50_ms=20, p99_ms=150, timeout_ms=2000),
        error_rate=0.005,
    ),
    ServiceDef(
        name="llm-anthropic",
        type=ServiceType.GENAI_LLM,
        latency=LatencyProfile(p50_ms=1200, p99_ms=8000, timeout_ms=30000),
        error_rate=0.015,
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        dependencies=["vector-db"],   # tool call dependency
    ),

    # ── Payments ──────────────────────────────────────────────────────────
    ServiceDef(
        name="payments-service",
        type=ServiceType.HTTP_API,
        latency=LatencyProfile(p50_ms=200, p99_ms=1500),
        error_rate=0.03,
        dependencies=["postgres-payments", "kafka"],
    ),
    ServiceDef(
        name="postgres-payments",
        type=ServiceType.DATABASE,
        latency=LatencyProfile(p50_ms=5, p99_ms=80, timeout_ms=5000),
        error_rate=0.005,
    ),
    ServiceDef(
        name="kafka",
        type=ServiceType.QUEUE,
        latency=LatencyProfile(p50_ms=10, p99_ms=50),
        error_rate=0.001,
    ),

    # ── Notifications (async consumer) ────────────────────────────────────
    ServiceDef(
        name="notifications-service",
        type=ServiceType.GRPC_SERVICE,
        latency=LatencyProfile(p50_ms=50, p99_ms=300),
        error_rate=0.01,
        dependencies=["kafka"],
    ),
]


class ServiceRegistry:
    """Lookup and dependency graph over ServiceDef instances."""

    def __init__(self, services: list[ServiceDef] | None = None):
        self._services: dict[str, ServiceDef] = {}
        for svc in (services or DEFAULT_SERVICES):
            self._services[svc.name] = svc

    def get(self, name: str) -> ServiceDef:
        return self._services[name]

    def all(self) -> list[ServiceDef]:
        return list(self._services.values())

    def dependencies_of(self, name: str) -> list[ServiceDef]:
        return [self._services[d] for d in self.get(name).dependencies
                if d in self._services]

    def dependents_of(self, name: str) -> list[ServiceDef]:
        """Services that depend ON this service (upstream callers)."""
        return [s for s in self._services.values() if name in s.dependencies]

    def propagate_fault(
        self,
        service_name: str,
        error_rate: float | None = None,
        latency_multiplier: float = 1.0,
    ) -> None:
        """
        Apply a fault to a service and propagate effects upstream.
        Upstream callers see a reduced but non-zero bleed-through effect
        (cascading degradation, not full outage).
        """
        svc = self._services[service_name]
        svc._error_rate_override = error_rate
        svc._latency_multiplier = latency_multiplier

        bleed_error = (error_rate or 0) * 0.3
        bleed_latency = 1 + (latency_multiplier - 1) * 0.5

        for caller in self.dependents_of(service_name):
            if bleed_error > 0:
                existing = caller._error_rate_override or caller.error_rate
                caller._error_rate_override = min(existing + bleed_error, 1.0)
            if bleed_latency > 1.0:
                caller._latency_multiplier = max(
                    caller._latency_multiplier, bleed_latency
                )

    def clear_faults(self) -> None:
        for svc in self._services.values():
            svc._error_rate_override = None
            svc._latency_multiplier = 1.0

    def genai_services(self) -> list[ServiceDef]:
        return [s for s in self._services.values()
                if s.type in (
                    ServiceType.GENAI_LLM, ServiceType.GENAI_EMBED,
                    ServiceType.GENAI_VECTOR_DB, ServiceType.GENAI_RAG,
                    ServiceType.GENAI_AGENT,
                )]
