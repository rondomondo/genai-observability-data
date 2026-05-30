"""Core scenario primitives: spec, state machine, and engine."""

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from god.scenarios._handlers import ScenarioHandler

from god.services import ServiceRegistry


class ScenarioState(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    RECOVERING = "recovering"
    DONE = "done"


class ScenarioSpec(BaseModel):
    """Validated, serialisable configuration for a single fault scenario.

    Separating config from behaviour means specs can be loaded from YAML/JSON
    or generated dynamically without coupling to handler code.
    """

    name: str
    handler_type: str
    description: str
    start_offset_s: float = Field(ge=0.0)
    end_offset_s: float = Field(ge=0.0)
    tags: list[str] = Field(default_factory=list)
    target: str = ""
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_multiplier: float = Field(default=1.0, ge=0.0)
    extra: dict[str, str] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_offset_s - self.start_offset_s

    def model_post_init(self, __context: object) -> None:
        if self.end_offset_s < self.start_offset_s:
            raise ValueError("end_offset_s must be >= start_offset_s")


class Scenario:
    """Runtime instance: pairs a validated spec with its handler and tracks state.

    The handler drives apply/recover; the spec holds all config. This keeps
    the state machine and the fault logic fully decoupled.
    """

    def __init__(self, spec: ScenarioSpec, handler: "ScenarioHandler") -> None:
        self.spec = spec
        self.handler = handler
        self.state: ScenarioState = ScenarioState.PENDING

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def tags(self) -> list[str]:
        return self.spec.tags

    @property
    def duration_s(self) -> float:
        return self.spec.duration_s

    @property
    def start_offset_s(self) -> float:
        return self.spec.start_offset_s

    @property
    def end_offset_s(self) -> float:
        return self.spec.end_offset_s

    def tick(self, elapsed_s: float, registry: ServiceRegistry) -> None:
        if self.state == ScenarioState.DONE:
            return
        if elapsed_s >= self.spec.start_offset_s and self.state == ScenarioState.PENDING:
            self.state = ScenarioState.ACTIVE
            self.handler.apply(registry, self.spec)
        if elapsed_s >= self.spec.end_offset_s and self.state == ScenarioState.ACTIVE:
            self.state = ScenarioState.RECOVERING
            self.handler.recover(registry, self.spec)
            self.state = ScenarioState.DONE


class ScenarioEngine:
    """Applies and expires scenarios as virtual time advances.

    Call tick(elapsed_s) before each trace/log generation batch.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        scenarios: list[Scenario] | None = None,
    ) -> None:
        self.registry = registry
        self.scenarios: list[Scenario] = scenarios or []

    def tick(self, elapsed_s: float) -> list[str]:
        """Advance virtual time.

        Args:
            elapsed_s: seconds since the reference epoch.

        Returns:
            Names of newly activated scenarios in this tick.
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
        """Return names of all scenarios that were activated, including completed ones."""
        return [s.name for s in self.scenarios if s.state in (ScenarioState.ACTIVE, ScenarioState.DONE)]

    def active_tags(self) -> set[str]:
        tags: set[str] = set()
        for s in self.scenarios:
            if s.state == ScenarioState.ACTIVE:
                tags.update(s.tags)
        return tags

    @classmethod
    def from_preset(cls, name: str, registry: ServiceRegistry) -> "ScenarioEngine":
        """Construct an engine from a named preset.

        Args:
            name: preset name from SCENARIO_PRESETS.
            registry: the service registry to apply faults to.

        Raises:
            ValueError: if the preset name is not found.
        """
        from god.scenarios._registry import SCENARIO_PRESETS

        factory = SCENARIO_PRESETS.get(name)
        if factory is None:
            raise ValueError(f"Unknown scenario preset '{name}'. Available: {list(SCENARIO_PRESETS.keys())}")
        return cls(registry, factory())
