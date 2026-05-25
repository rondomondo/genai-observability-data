"""Mock mode activation for the god package.

Calling activate_mock_mode() replaces the pydantic-ai scenario_agent
with a FunctionModel — no network calls, no API key required.
The override is permanent for the process lifetime (using agent.override()).
"""
from __future__ import annotations

import contextlib
import logging

_log = logging.getLogger(__name__)

_active = False
_overrides: list[contextlib.AbstractContextManager] = []


def is_active() -> bool:
    return _active


def activate_mock_mode() -> None:
    """Patch the scenario_agent with a FunctionModel mock."""
    global _active
    if _active:
        return

    from god.agent import scenario_agent
    from god.mock.models import make_analysis_model

    ctx = scenario_agent.override(model=make_analysis_model())
    _overrides.append(ctx)
    ctx.__enter__()
    _active = True
    _log.info("god mock mode activated — no API calls will be made")


def deactivate_mock_mode() -> None:
    """Reverse mock patches (used in tests)."""
    global _active
    for ctx in reversed(_overrides):
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass
    _overrides.clear()
    _active = False
