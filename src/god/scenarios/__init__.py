"""Scenario engine for fault injection and anomaly simulation.

Scenarios are composable, time-bounded perturbations applied to the
ServiceRegistry. They drive correlated failures across the dependency graph,
producing realistic anomaly patterns in the generated data.

Scenario presets (SCENARIO_PRESETS):
  - event_degradation          Kafka consumer lag spike -> notification delivery delays
  - db_incident                DB connection pool exhaustion -> upstream timeouts
  - llm_degradation            GPU saturation + token budget breach
  - rag_meltdown               Vector DB slowdown + LLM spike + provider failover
  - payment_outage             Payment processor errors + DB degradation
  - cascading_failure          Auth DB -> auth outage -> payments -> LLM spike
  - tool_meltdown              Tool call timeouts + agent bootstrap failures
  - memory_incident            Memory store degradation + tool call timeouts
  - agent_failure              Agent creation fails -> invoke_agent errors cascade
  - rag_quality_degradation    Stale vector index + LLM latency spike
  - agentic_cascading_failure  Agent bootstrap -> tool timeouts -> memory -> LLM

Extension points:
  - ScenarioSpec: serialisable config - load from YAML/JSON for dynamic discovery
  - ScenarioHandler / register_handler: plug in custom fault logic
  - SCENARIO_PRESETS: add named preset groups
"""

from god.scenarios._base import Scenario, ScenarioEngine, ScenarioSpec, ScenarioState
from god.scenarios._handlers import (
    AgentBootstrapFailureHandler,
    DbDegradationHandler,
    LlmLatencySpikeHandler,
    PaymentsErrorBurstHandler,
    ScenarioHandler,
    SimplePropagateHandler,
    TokenBudgetBreachHandler,
    ToolCallTimeoutHandler,
    build_handler,
    register_handler,
)
from god.scenarios._registry import (
    SCENARIO_PRESETS,
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

__all__ = [
    # Core primitives
    "Scenario",
    "ScenarioEngine",
    "ScenarioSpec",
    "ScenarioState",
    # Handler base and extension API
    "ScenarioHandler",
    "SimplePropagateHandler",
    "DbDegradationHandler",
    "LlmLatencySpikeHandler",
    "TokenBudgetBreachHandler",
    "PaymentsErrorBurstHandler",
    "ToolCallTimeoutHandler",
    "AgentBootstrapFailureHandler",
    "build_handler",
    "register_handler",
    # Factory functions
    "db_degradation",
    "llm_latency_spike",
    "token_budget_breach",
    "provider_failover",
    "vector_db_slowdown",
    "payments_error_burst",
    "auth_outage",
    "kafka_lag",
    "tool_call_timeout",
    "memory_store_degradation",
    "agent_bootstrap_failure",
    "retrieval_quality_drop",
    # Presets
    "SCENARIO_PRESETS",
]
