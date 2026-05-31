"""Tests for the /status router."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/status/health")
    assert response.status_code == 200


def test_health_response_shape(client: TestClient) -> None:
    response = client.get("/status/health")
    data = response.json()
    assert "status" in data
