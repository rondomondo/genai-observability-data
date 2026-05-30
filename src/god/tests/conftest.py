"""Shared pytest fixtures for all god tests."""

from __future__ import annotations

import pytest

from god.mock.activate import activate_mock_mode, deactivate_mock_mode


@pytest.fixture(autouse=True, scope="session")
def _session_mock_mode():
    """Activate mock mode for the entire test session so no real API calls are made."""
    activate_mock_mode()
    yield
    deactivate_mock_mode()
