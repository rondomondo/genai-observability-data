"""Tests for the embedder service."""

from __future__ import annotations

import pytest


@pytest.mark.integration
async def test_embed_returns_vector() -> None:
    """Requires a real GEMINI_API_KEY and network access."""
    pytest.skip("integration stub — implement with a real embedder fixture")
