"""
Scenario engine for fault injection and anomaly simulation.

Scenarios are composable, time-bounded perturbations applied to the
ServiceRegistry. They drive correlated failures across the dependency graph,
producing realistic anomaly patterns in the generated data.

Built-in scenarios:
  - db_degradation           DB connection pool exhaustion → upstream timeouts
  - llm_latency_spike        GPU saturation → high TTFT, timeouts
  - token_budget_breach      LLM hitting max_tokens → LENGTH finish reason
  - provider_failover        Primary LLM provider times out → retry on fallback
  - vector_db_slowdown       Vector index contention → elevated retrieval latency
  - payments_error_burst     Payment processor errors → cascading to gateway
  - auth_outage              Auth service down → 401s across all services
  - kafka_lag                Consumer lag spike → notification delivery delays
  - tool_call_timeout        External tool (extension type) starts timing out
  - memory_store_degradation Memory service latency spikes → agent context loss
  - agent_bootstrap_failure  Agent creation fails → invoke_agent errors cascade
  - retrieval_quality_drop   Vector DB returns stale index → low-score documents
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from god.services import ServiceRegistry


class ScenarioState(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    RECOVERING = "recovering"
    DONE = "done"


@dataclass
class Scenario:
    """
    A time-bounded fault scenario.

    start_offset_s / end_offset_s are relative to a reference epoch.
    apply() is called once when the scenario becomes active.
    recover() is called once when it expires.
    """

    name: str
    description: str
    start_offset_s: float  # seconds after epoch
    end_offset_s: float  # seconds after epoch
    apply: Callable[[ServiceRegistry], None]
    recover: Callable[[ServiceRegistry], None]
    state: ScenarioState = ScenarioState.PENDING
    tags: list[str] = field(default_factory=list)

    def tick(self, elapsed_s: float, registry: ServiceRegistry) -> None:
        if self.state == ScenarioState.DONE:
            return
        if elapsed_s >= self.start_offset_s and self.state == ScenarioState.PENDING:
            self.state = ScenarioState.ACTIVE
            self.apply(registry)
        if elapsed_s >= self.end_offset_s and self.state == ScenarioState.ACTIVE:
            self.state = ScenarioState.RECOVERING
            self.recover(registry)
            self.state = ScenarioState.DONE

    @property
    def duration_s(self) -> float:
        return self.end_offset_s - self.start_offset_s


# ---------------------------------------------------------------------------
# Scenario factory functions
# ---------------------------------------------------------------------------


def db_degradation(
    target: str = "postgres-payments",
    start_s: float = 60,
    duration_s: float = 120,
    error_rate: float = 0.4,
    latency_multiplier: float = 8.0,
) -> Scenario:
    """
    Simulates connection pool exhaustion on a database.
    Upstream callers (e.g. payments-service) see cascading errors.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault(target, error_rate, latency_multiplier)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get(target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0
        # Upstream bleed recovery — gradual
        for caller in reg.dependents_of(target):
            caller._error_rate_override = None
            caller._latency_multiplier = max(caller._latency_multiplier * 0.5, 1.0)

    return Scenario(
        name="db_degradation",
        description=f"Connection pool exhaustion {target}: error_rate={error_rate:.0%}, latency x{latency_multiplier}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["database", "latency", "error_burst"],
    )


def llm_latency_spike(
    target: str = "llm-anthropic",
    start_s: float = 30,
    duration_s: float = 90,
    ttft_multiplier: float = 5.0,
    error_rate: float = 0.15,
) -> Scenario:
    """
    Simulates GPU saturation or cold model loading.
    TTFT spikes dramatically; some requests timeout entirely.
    """
    _original_multiplier: list[float] = []

    def apply(reg: ServiceRegistry) -> None:
        svc = reg.get(target)
        _original_multiplier.append(svc._latency_multiplier)
        svc._latency_multiplier = ttft_multiplier
        svc._error_rate_override = error_rate

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get(target)
        svc._latency_multiplier = _original_multiplier[0] if _original_multiplier else 1.0
        svc._error_rate_override = None

    return Scenario(
        name="llm_latency_spike",
        description=f"GPU saturation on {target}: TTFT x{ttft_multiplier}, error_rate={error_rate:.0%}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["genai", "latency", "ttft"],
    )


def token_budget_breach(
    target: str = "llm-anthropic",
    start_s: float = 0,
    duration_s: float = 300,
) -> Scenario:
    """
    Simulates a misconfigured max_tokens causing frequent LENGTH truncations.
    Not an error per se, but a quality degradation signal.
    """

    def apply(reg: ServiceRegistry) -> None:
        # Mark with a custom attribute rather than error rate
        svc = reg.get(target)
        svc._error_rate_override = 0.0  # no hard errors, just truncations

    def recover(reg: ServiceRegistry) -> None:
        reg.get(target)._error_rate_override = None

    return Scenario(
        name="token_budget_breach",
        description="Frequent LLM response truncations due to low max_tokens config",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["genai", "quality", "token_limit"],
    )


def provider_failover(
    primary: str = "llm-anthropic",
    start_s: float = 45,
    duration_s: float = 60,
) -> Scenario:
    """
    Primary LLM provider becomes unavailable.
    Models a retry/failover pattern — primary errors spike, then recover.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault(primary, error_rate=0.95, latency_multiplier=3.0)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get(primary)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0

    return Scenario(
        name="provider_failover",
        description=f"Primary provider {primary} unavailable — failover triggered",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["genai", "provider", "failover"],
    )


def vector_db_slowdown(
    target: str = "vector-db",
    start_s: float = 20,
    duration_s: float = 180,
    latency_multiplier: float = 6.0,
) -> Scenario:
    """
    Vector index rebuild or high concurrency causing retrieval degradation.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault(target, error_rate=0.05, latency_multiplier=latency_multiplier)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get(target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0

    return Scenario(
        name="vector_db_slowdown",
        description=f"Vector index contention on {target}: latency x{latency_multiplier}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["genai", "vector_db", "latency"],
    )


def payments_error_burst(
    start_s: float = 0,
    duration_s: float = 45,
) -> Scenario:
    """
    Payment processor returns errors — cascades up to gateway.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault("payments-service", error_rate=0.6, latency_multiplier=4.0)

    def recover(reg: ServiceRegistry) -> None:
        for name in ("payments-service", "postgres-payments"):
            svc = reg.get(name)
            svc._error_rate_override = None
            svc._latency_multiplier = 1.0

    return Scenario(
        name="payments_error_burst",
        description="Payment processor error burst: 60% error rate",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["payments", "error_burst"],
    )


def auth_outage(
    start_s: float = 10,
    duration_s: float = 30,
) -> Scenario:
    """
    Auth service hard down — all authenticated endpoints return 401/503.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault("auth-service", error_rate=1.0, latency_multiplier=1.0)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get("auth-service")
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0

    return Scenario(
        name="auth_outage",
        description="Auth service hard down: 100% error rate for 30s",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["auth", "outage"],
    )


def kafka_lag(
    start_s: float = 0,
    duration_s: float = 600,
) -> Scenario:
    """
    Kafka consumer lag spike — notification delays, no hard errors.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault("kafka", error_rate=0.02, latency_multiplier=10.0)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get("kafka")
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0

    return Scenario(
        name="kafka_lag",
        description="Kafka consumer lag: 10x latency, notifications delayed",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["kafka", "lag", "async"],
    )


def tool_call_timeout(
    target: str = "tool-executor",
    start_s: float = 30,
    duration_s: float = 120,
    error_rate: float = 0.55,
    latency_multiplier: float = 12.0,
) -> Scenario:
    """
    External API tool calls (extension type) begin timing out — e.g. a
    downstream REST API is overloaded. The tool-executor error rate spikes
    and latency balloons, causing invoke_agent spans to see cascading failures
    as the agent retries or gives up.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault(target, error_rate, latency_multiplier)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get(target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0
        for caller in reg.dependents_of(target):
            caller._error_rate_override = None
            caller._latency_multiplier = max(caller._latency_multiplier * 0.5, 1.0)

    return Scenario(
        name="tool_call_timeout",
        description=f"Extension tool timeouts on {target}: error_rate={error_rate:.0%}, latency x{latency_multiplier}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["genai", "tool", "latency", "error_burst"],
    )


def memory_store_degradation(
    target: str = "memory-service",
    start_s: float = 20,
    duration_s: float = 180,
    latency_multiplier: float = 8.0,
    error_rate: float = 0.25,
) -> Scenario:
    """
    Memory service latency spikes and starts failing writes — simulates a
    saturated or network-partitioned memory backend (mem0, Zep, etc.).
    Agents lose the ability to retrieve prior context, causing degraded
    response quality and write failures on upsert_memory spans.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault(target, error_rate, latency_multiplier)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get(target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0

    return Scenario(
        name="memory_store_degradation",
        description=f"Memory store degraded on {target}: latency x{latency_multiplier}, error_rate={error_rate:.0%}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["genai", "memory", "latency", "agent"],
    )


def agent_bootstrap_failure(
    agent_target: str = "agent-service",
    start_s: float = 10,
    duration_s: float = 60,
) -> Scenario:
    """
    The remote agent provisioning endpoint becomes unavailable — create_agent
    calls fail at 90% rate. Downstream invoke_agent spans see cascading errors
    because agents cannot be initialised. Models an OpenAI Assistants or
    Bedrock Agents service disruption.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault(agent_target, error_rate=0.9, latency_multiplier=2.0)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get(agent_target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0
        for caller in reg.dependents_of(agent_target):
            caller._error_rate_override = None
            caller._latency_multiplier = 1.0

    return Scenario(
        name="agent_bootstrap_failure",
        description=f"Agent provisioning unavailable on {agent_target}: 90% create_agent failures",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["genai", "agent", "error_burst", "provider"],
    )


def retrieval_quality_drop(
    target: str = "vector-db",
    start_s: float = 0,
    duration_s: float = 240,
    latency_multiplier: float = 3.0,
    error_rate: float = 0.08,
) -> Scenario:
    """
    Vector DB is serving from a stale or partially rebuilt index — low latency
    impact but elevated error rate and degraded retrieval quality. Documents
    returned have lower relevance scores. Manifests in retrieval spans with
    db.vector.query.returned dropping (fewer results above threshold) and
    gen_ai.evaluation.result events showing low scores on downstream LLM calls.
    """

    def apply(reg: ServiceRegistry) -> None:
        reg.propagate_fault(target, error_rate, latency_multiplier)

    def recover(reg: ServiceRegistry) -> None:
        svc = reg.get(target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0

    return Scenario(
        name="retrieval_quality_drop",
        description=f"Stale vector index on {target}: degraded retrieval quality, error_rate={error_rate:.0%}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        apply=apply,
        recover=recover,
        tags=["genai", "vector_db", "quality", "retrieval"],
    )


# ---------------------------------------------------------------------------
# Named preset scenario sets
# ---------------------------------------------------------------------------

SCENARIO_PRESETS: dict[str, Callable[[], list[Scenario]]] = {
    "event_degradation": lambda: [
        kafka_lag(start_s=0, duration_s=600),
    ],
    "db_incident": lambda: [
        db_degradation("postgres-payments", start_s=60, duration_s=120),
    ],
    "llm_degradation": lambda: [
        llm_latency_spike(start_s=30, duration_s=90),
        token_budget_breach(start_s=0, duration_s=300),
    ],
    "rag_meltdown": lambda: [
        vector_db_slowdown(start_s=0, duration_s=180),
        llm_latency_spike(start_s=30, duration_s=60),
        provider_failover(start_s=90, duration_s=60),
    ],
    "payment_outage": lambda: [
        payments_error_burst(start_s=0, duration_s=45),
        db_degradation("postgres-payments", start_s=10, duration_s=90),
    ],
    "cascading_failure": lambda: [
        db_degradation("postgres-auth", start_s=0, duration_s=30),
        auth_outage(start_s=5, duration_s=25),
        payments_error_burst(start_s=15, duration_s=60),
        llm_latency_spike(start_s=40, duration_s=90),
    ],
    # Agentic pipeline scenarios
    "tool_meltdown": lambda: [
        tool_call_timeout(start_s=30, duration_s=120),
        agent_bootstrap_failure(start_s=60, duration_s=60),
    ],
    "memory_incident": lambda: [
        memory_store_degradation(start_s=20, duration_s=180),
        tool_call_timeout(start_s=90, duration_s=60),
    ],
    "agent_failure": lambda: [
        agent_bootstrap_failure(start_s=10, duration_s=60),
    ],
    "rag_quality_degradation": lambda: [
        retrieval_quality_drop(start_s=0, duration_s=240),
        llm_latency_spike(start_s=60, duration_s=90),
    ],
    "agentic_cascading_failure": lambda: [
        agent_bootstrap_failure(start_s=0, duration_s=30),
        tool_call_timeout(start_s=15, duration_s=120),
        memory_store_degradation(start_s=30, duration_s=180),
        llm_latency_spike(start_s=60, duration_s=90),
    ],
    "quiet": lambda: [],
}

SCENARIO_PRESETS["all"] = lambda: [
    s for name, factory in SCENARIO_PRESETS.items() if name not in ("all", "quiet") for s in factory()
]


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


class ScenarioEngine:
    """
    Applies and expires scenarios as virtual time advances.
    Call tick(elapsed_s) before each trace/log generation batch.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        scenarios: list[Scenario] | None = None,
    ):
        self.registry = registry
        self.scenarios = scenarios or []

    def tick(self, elapsed_s: float) -> list[str]:
        """
        Advance virtual time. Returns names of newly activated scenarios.
        """
        activated = []
        for scenario in self.scenarios:
            prev = scenario.state
            scenario.tick(elapsed_s, self.registry)
            if prev == ScenarioState.PENDING and scenario.state == ScenarioState.ACTIVE:
                activated.append(scenario.name)
        return activated

    def active_scenario_names(self) -> list[str]:
        return [s.name for s in self.scenarios if s.state == ScenarioState.ACTIVE]

    def activated_scenario_names(self) -> list[str]:
        """Return names of all scenarios that were activated, including those already completed."""
        return [s.name for s in self.scenarios if s.state in (ScenarioState.ACTIVE, ScenarioState.DONE)]

    def active_tags(self) -> set[str]:
        tags: set[str] = set()
        for s in self.scenarios:
            if s.state == ScenarioState.ACTIVE:
                tags.update(s.tags)
        return tags

    @classmethod
    def from_preset(cls, name: str, registry: ServiceRegistry) -> ScenarioEngine:
        factory = SCENARIO_PRESETS.get(name)
        if factory is None:
            raise ValueError(f"Unknown scenario preset '{name}'. Available: {list(SCENARIO_PRESETS.keys())}")
        return cls(registry, factory())
