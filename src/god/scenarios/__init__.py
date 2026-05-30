"""Scenario engine for fault injection and anomaly simulation.

Scenarios are composable, time-bounded perturbations applied to the
ServiceRegistry. They drive correlated failures across the dependency graph,
producing realistic anomaly patterns in the generated data.

Built-in scenarios:
  - db_degradation           DB connection pool exhaustion -> upstream timeouts
  - llm_latency_spike        GPU saturation -> high TTFT, timeouts
  - token_budget_breach      LLM hitting max_tokens -> LENGTH finish reason
  - provider_failover        Primary LLM provider times out -> retry on fallback
  - vector_db_slowdown       Vector index contention -> elevated retrieval latency
  - payments_error_burst     Payment processor errors -> cascading to gateway
  - auth_outage              Auth service down -> 401s across all services
  - kafka_lag                Consumer lag spike -> notification delivery delays
  - tool_call_timeout        External tool (extension type) starts timing out
  - memory_store_degradation Memory service latency spikes -> agent context loss
  - agent_bootstrap_failure  Agent creation fails -> invoke_agent errors cascade
  - retrieval_quality_drop   Vector DB returns stale index -> low-score documents

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
