"""Shared pytest fixtures for vecstore-api tests."""

from __future__ import annotations

import os

os.environ.setdefault("OTEL_ENABLED", "false")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Return a synchronous TestClient for the FastAPI app."""
    return TestClient(app)
