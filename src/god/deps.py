"""Pydantic-AI dependency container for the scenario analysis agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from god.config import Settings, get_settings
from god.scenarios import ScenarioEngine
from god.services import ServiceRegistry


@dataclass
class GodDeps:
    """Runtime dependencies injected into the god agent via RunContext."""

    settings: Settings = field(default_factory=get_settings)
    registry: ServiceRegistry = field(default_factory=ServiceRegistry)
    engine: ScenarioEngine | None = None
    scenario_preset: str | None = None

    @classmethod
    def from_settings(cls, s: Settings) -> GodDeps:
        return cls(settings=s)
