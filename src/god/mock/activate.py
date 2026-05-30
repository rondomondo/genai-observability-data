"""Mock mode activation for the god package.

Calling activate_mock_mode() replaces the pydantic-ai scenario_agent
with a FunctionModel — no network calls, no API key required.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

_active = False
_saved_model: Any = None


def is_active() -> bool:
    return _active


def activate_mock_mode() -> None:
    """Patch the scenario_agent with a FunctionModel mock."""
    global _active, _saved_model
    if _active:
        return

    from god.agent import scenario_agent
    from god.mock.models import make_analysis_model

    _saved_model = scenario_agent._model
    scenario_agent._model = make_analysis_model()
    _active = True
    _log.info("god mock mode activated — no API calls will be made")


def deactivate_mock_mode() -> None:
    """Reverse mock patches (used in tests)."""
    global _active, _saved_model

    from god.agent import scenario_agent

    scenario_agent._model = _saved_model
    _saved_model = None
    _active = False
