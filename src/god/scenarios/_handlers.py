"""Scenario handlers: abstract base and all concrete fault implementations.

A handler is stateless - all config lives in ScenarioSpec. This makes handlers
safe to share across scenario instances and easy to register for dynamic discovery.
"""

from abc import ABC, abstractmethod

from god.scenarios._base import ScenarioSpec
from god.services import ServiceRegistry


class ScenarioHandler(ABC):
    """Stateless fault injector paired with a ScenarioSpec at runtime."""

    @abstractmethod
    def apply(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        """Inject the fault described by spec into registry."""

    @abstractmethod
    def recover(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        """Undo the fault and restore affected services to baseline."""


class SimplePropagateHandler(ScenarioHandler):
    """Generic handler for scenarios that call propagate_fault on a single target.

    Covers the majority of scenarios: apply propagates the fault with bleed-through
    to upstreams; recover clears the override on the target only.
    """

    def apply(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        registry.propagate_fault(spec.target, spec.error_rate, spec.latency_multiplier)

    def recover(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        svc = registry.get(spec.target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0


class DbDegradationHandler(ScenarioHandler):
    """Connection pool exhaustion - gradual upstream bleed recovery."""

    def apply(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        registry.propagate_fault(spec.target, spec.error_rate, spec.latency_multiplier)

    def recover(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        svc = registry.get(spec.target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0
        for caller in registry.dependents_of(spec.target):
            caller._error_rate_override = None
            caller._latency_multiplier = max(caller._latency_multiplier * 0.5, 1.0)


class LlmLatencySpikeHandler(ScenarioHandler):
    """GPU saturation - captures original multiplier so recovery is exact.

    Uses a per-instance list as a single-element cell to capture state across
    apply/recover without making the handler stateful at the class level.
    This is safe because Scenario constructs one handler instance per scenario.
    """

    def __init__(self) -> None:
        self._original_multiplier: list[float] = []

    def apply(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        svc = registry.get(spec.target)
        self._original_multiplier.clear()
        self._original_multiplier.append(svc._latency_multiplier)
        svc._latency_multiplier = spec.latency_multiplier
        svc._error_rate_override = spec.error_rate

    def recover(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        svc = registry.get(spec.target)
        svc._latency_multiplier = self._original_multiplier[0] if self._original_multiplier else 1.0
        svc._error_rate_override = None


class TokenBudgetBreachHandler(ScenarioHandler):
    """Misconfigured max_tokens - sets zero error rate to signal truncations without hard errors."""

    def apply(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        registry.get(spec.target)._error_rate_override = 0.0

    def recover(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        registry.get(spec.target)._error_rate_override = None


class PaymentsErrorBurstHandler(ScenarioHandler):
    """Payment processor errors - recovers both payments-service and its DB."""

    def apply(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        registry.propagate_fault(spec.target, spec.error_rate, spec.latency_multiplier)

    def recover(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        for name in (spec.target, spec.extra.get("db_target", "postgres-payments")):
            svc = registry.get(name)
            svc._error_rate_override = None
            svc._latency_multiplier = 1.0


class ToolCallTimeoutHandler(ScenarioHandler):
    """External tool timeouts - gradual upstream bleed recovery, same as DB degradation."""

    def apply(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        registry.propagate_fault(spec.target, spec.error_rate, spec.latency_multiplier)

    def recover(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        svc = registry.get(spec.target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0
        for caller in registry.dependents_of(spec.target):
            caller._error_rate_override = None
            caller._latency_multiplier = max(caller._latency_multiplier * 0.5, 1.0)


class AgentBootstrapFailureHandler(ScenarioHandler):
    """Remote agent provisioning unavailable - clears all upstream callers fully."""

    def apply(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        registry.propagate_fault(spec.target, spec.error_rate, spec.latency_multiplier)

    def recover(self, registry: ServiceRegistry, spec: ScenarioSpec) -> None:
        svc = registry.get(spec.target)
        svc._error_rate_override = None
        svc._latency_multiplier = 1.0
        for caller in registry.dependents_of(spec.target):
            caller._error_rate_override = None
            caller._latency_multiplier = 1.0


# Handler registry - maps handler_type string to handler class.
# Register new handlers here (or via register_handler) to enable dynamic discovery.
_HANDLER_REGISTRY: dict[str, type[ScenarioHandler]] = {
    "simple_propagate": SimplePropagateHandler,
    "db_degradation": DbDegradationHandler,
    "llm_latency_spike": LlmLatencySpikeHandler,
    "token_budget_breach": TokenBudgetBreachHandler,
    "payments_error_burst": PaymentsErrorBurstHandler,
    "tool_call_timeout": ToolCallTimeoutHandler,
    "agent_bootstrap_failure": AgentBootstrapFailureHandler,
}


def register_handler(handler_type: str, cls: type[ScenarioHandler]) -> None:
    """Register a custom handler for dynamic scenario discovery.

    Args:
        handler_type: the string key used in ScenarioSpec.handler_type.
        cls: the handler class to instantiate when this type is encountered.
    """
    _HANDLER_REGISTRY[handler_type] = cls


def build_handler(handler_type: str) -> ScenarioHandler:
    """Instantiate a handler by its registered type string.

    Args:
        handler_type: key in the handler registry.

    Raises:
        ValueError: if handler_type is not registered.
    """
    cls = _HANDLER_REGISTRY.get(handler_type)
    if cls is None:
        raise ValueError(
            f"Unknown handler type '{handler_type}'. Registered: {list(_HANDLER_REGISTRY.keys())}"
        )
    return cls()
