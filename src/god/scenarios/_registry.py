"""Scenario factory functions and named preset sets.

Factory functions are the public authoring API: they accept typed, validated
parameters and return a ready-to-run Scenario. Each factory builds a ScenarioSpec
and pairs it with the appropriate handler via build_handler().

Adding a new scenario:
  1. Write a ScenarioHandler subclass in _handlers.py and register it.
  2. Add a factory function here following the pattern below.
  3. Optionally add a preset entry to SCENARIO_PRESETS.
"""

from collections.abc import Callable

from god.scenarios._base import Scenario, ScenarioSpec
from god.scenarios._handlers import build_handler


def _make(spec: ScenarioSpec) -> Scenario:
    return Scenario(spec, build_handler(spec.handler_type))


def db_degradation(
    target: str = "postgres-payments",
    start_s: float = 60,
    duration_s: float = 120,
    error_rate: float = 0.4,
    latency_multiplier: float = 8.0,
) -> Scenario:
    """Simulate connection pool exhaustion on a database.

    Args:
        target: service name of the database to degrade.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
        error_rate: fraction of requests that error (0-1).
        latency_multiplier: factor applied to baseline latency.
    """
    return _make(ScenarioSpec(
        name="db_degradation",
        handler_type="db_degradation",
        description=f"Connection pool exhaustion {target}: error_rate={error_rate:.0%}, latency x{latency_multiplier}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=target,
        error_rate=error_rate,
        latency_multiplier=latency_multiplier,
        tags=["database", "latency", "error_burst"],
    ))


def llm_latency_spike(
    target: str = "llm-gemini",
    start_s: float = 30,
    duration_s: float = 90,
    ttft_multiplier: float = 5.0,
    error_rate: float = 0.15,
) -> Scenario:
    """Simulate GPU saturation or cold model loading.

    Args:
        target: LLM service name.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
        ttft_multiplier: latency multiplier applied to the LLM service.
        error_rate: fraction of requests that timeout entirely.
    """
    return _make(ScenarioSpec(
        name="llm_latency_spike",
        handler_type="llm_latency_spike",
        description=f"GPU saturation on {target}: TTFT x{ttft_multiplier}, error_rate={error_rate:.0%}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=target,
        error_rate=error_rate,
        latency_multiplier=ttft_multiplier,
        tags=["genai", "latency", "ttft"],
    ))


def token_budget_breach(
    target: str = "llm-gemini",
    start_s: float = 0,
    duration_s: float = 300,
) -> Scenario:
    """Simulate a misconfigured max_tokens causing frequent LENGTH truncations.

    Args:
        target: LLM service name.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
    """
    return _make(ScenarioSpec(
        name="token_budget_breach",
        handler_type="token_budget_breach",
        description="Frequent LLM response truncations due to low max_tokens config",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=target,
        tags=["genai", "quality", "token_limit"],
    ))


def provider_failover(
    primary: str = "llm-gemini",
    start_s: float = 45,
    duration_s: float = 60,
) -> Scenario:
    """Simulate primary LLM provider becoming unavailable, triggering failover.

    Args:
        primary: service name of the primary LLM provider.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
    """
    return _make(ScenarioSpec(
        name="provider_failover",
        handler_type="simple_propagate",
        description=f"Primary provider {primary} unavailable -- failover triggered",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=primary,
        error_rate=0.95,
        latency_multiplier=3.0,
        tags=["genai", "provider", "failover"],
    ))


def vector_db_slowdown(
    target: str = "vector-db",
    start_s: float = 20,
    duration_s: float = 180,
    latency_multiplier: float = 6.0,
) -> Scenario:
    """Simulate vector index rebuild or high concurrency causing retrieval degradation.

    Args:
        target: vector database service name.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
        latency_multiplier: factor applied to baseline retrieval latency.
    """
    return _make(ScenarioSpec(
        name="vector_db_slowdown",
        handler_type="simple_propagate",
        description=f"Vector index contention on {target}: latency x{latency_multiplier}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=target,
        error_rate=0.05,
        latency_multiplier=latency_multiplier,
        tags=["genai", "vector_db", "latency"],
    ))


def payments_error_burst(
    start_s: float = 0,
    duration_s: float = 45,
) -> Scenario:
    """Simulate payment processor errors cascading up to the gateway.

    Args:
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
    """
    return _make(ScenarioSpec(
        name="payments_error_burst",
        handler_type="payments_error_burst",
        description="Payment processor error burst: 60% error rate",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target="payments-service",
        error_rate=0.6,
        latency_multiplier=4.0,
        extra={"db_target": "postgres-payments"},
        tags=["payments", "error_burst"],
    ))


def auth_outage(
    start_s: float = 10,
    duration_s: float = 30,
) -> Scenario:
    """Simulate auth service going hard down, causing 401/503s on all authenticated endpoints.

    Args:
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
    """
    return _make(ScenarioSpec(
        name="auth_outage",
        handler_type="simple_propagate",
        description="Auth service hard down: 100% error rate for 30s",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target="auth-service",
        error_rate=1.0,
        latency_multiplier=1.0,
        tags=["auth", "outage"],
    ))


def kafka_lag(
    start_s: float = 0,
    duration_s: float = 600,
) -> Scenario:
    """Simulate Kafka consumer lag spike causing notification delivery delays.

    Args:
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
    """
    return _make(ScenarioSpec(
        name="kafka_lag",
        handler_type="simple_propagate",
        description="Kafka consumer lag: 10x latency, notifications delayed",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target="kafka",
        error_rate=0.02,
        latency_multiplier=10.0,
        tags=["kafka", "lag", "async"],
    ))


def tool_call_timeout(
    target: str = "tool-executor",
    start_s: float = 30,
    duration_s: float = 120,
    error_rate: float = 0.55,
    latency_multiplier: float = 12.0,
) -> Scenario:
    """Simulate external API tool calls timing out due to downstream overload.

    Args:
        target: tool executor service name.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
        error_rate: fraction of tool calls that error.
        latency_multiplier: factor applied to baseline tool call latency.
    """
    return _make(ScenarioSpec(
        name="tool_call_timeout",
        handler_type="tool_call_timeout",
        description=f"Extension tool timeouts on {target}: error_rate={error_rate:.0%}, latency x{latency_multiplier}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=target,
        error_rate=error_rate,
        latency_multiplier=latency_multiplier,
        tags=["genai", "tool", "latency", "error_burst"],
    ))


def memory_store_degradation(
    target: str = "memory-service",
    start_s: float = 20,
    duration_s: float = 180,
    latency_multiplier: float = 8.0,
    error_rate: float = 0.25,
) -> Scenario:
    """Simulate a saturated or network-partitioned memory backend.

    Args:
        target: memory service name.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
        latency_multiplier: factor applied to baseline memory operation latency.
        error_rate: fraction of memory writes that fail.
    """
    return _make(ScenarioSpec(
        name="memory_store_degradation",
        handler_type="simple_propagate",
        description=f"Memory store degraded on {target}: latency x{latency_multiplier}, error_rate={error_rate:.0%}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=target,
        error_rate=error_rate,
        latency_multiplier=latency_multiplier,
        tags=["genai", "memory", "latency", "agent"],
    ))


def agent_bootstrap_failure(
    agent_target: str = "agent-service",
    start_s: float = 10,
    duration_s: float = 60,
) -> Scenario:
    """Simulate remote agent provisioning endpoint becoming unavailable.

    Args:
        agent_target: agent service name.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
    """
    return _make(ScenarioSpec(
        name="agent_bootstrap_failure",
        handler_type="agent_bootstrap_failure",
        description=f"Agent provisioning unavailable on {agent_target}: 90% create_agent failures",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=agent_target,
        error_rate=0.9,
        latency_multiplier=2.0,
        tags=["genai", "agent", "error_burst", "provider"],
    ))


def retrieval_quality_drop(
    target: str = "vector-db",
    start_s: float = 0,
    duration_s: float = 240,
    latency_multiplier: float = 3.0,
    error_rate: float = 0.08,
) -> Scenario:
    """Simulate a stale or partially rebuilt vector index degrading retrieval quality.

    Args:
        target: vector database service name.
        start_s: seconds after epoch to activate.
        duration_s: how long the fault lasts.
        latency_multiplier: factor applied to baseline retrieval latency.
        error_rate: fraction of retrievals that error or return below-threshold results.
    """
    return _make(ScenarioSpec(
        name="retrieval_quality_drop",
        handler_type="simple_propagate",
        description=f"Stale vector index on {target}: degraded retrieval quality, error_rate={error_rate:.0%}",
        start_offset_s=start_s,
        end_offset_s=start_s + duration_s,
        target=target,
        error_rate=error_rate,
        latency_multiplier=latency_multiplier,
        tags=["genai", "vector_db", "quality", "retrieval"],
    ))


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
