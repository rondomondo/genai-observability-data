"""Tests for the embedder-sidecar FastAPI app.

sentence_transformers is patched via sys.modules so no model download or
torch install is needed. Tests cover the /embed and /health endpoints plus
the LocalEmbedder wrapper logic.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


DIM = 64


def _mock_st_module(model_name: str = "test-model", dim: int = DIM) -> MagicMock:
    """Return a sys.modules patch dict for sentence_transformers."""
    mock_array = MagicMock()
    mock_array.tolist.return_value = [0.5] * dim

    mock_model = MagicMock()
    mock_model.prompts = {}
    mock_model.encode.return_value = [mock_array]
    mock_model.get_sentence_embedding_dimension.return_value = dim

    mock_st_cls = MagicMock(return_value=mock_model)
    mock_module = MagicMock()
    mock_module.SentenceTransformer = mock_st_cls
    return mock_module


@pytest.fixture()
def st_patch():
    mock_module = _mock_st_module()
    with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
        yield mock_module


@pytest.fixture()
def mock_embedder(st_patch) -> MagicMock:
    mock_array = MagicMock()
    mock_array.tolist.return_value = [0.5] * DIM

    e = MagicMock()
    e.model_name = "test-model"
    e.dim = DIM
    e.embed.return_value = [[0.5] * DIM]
    return e


@pytest.fixture()
def client(mock_embedder) -> TestClient:
    from concurrent.futures import ThreadPoolExecutor

    import app.main as main_mod

    main_mod._embedder = mock_embedder
    main_mod._executor = ThreadPoolExecutor(max_workers=1)

    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    main_mod._executor.shutdown(wait=False)
    main_mod._embedder = None
    main_mod._executor = None


# LocalEmbedder unit tests (no HTTP layer)

class TestLocalEmbedder:
    def test_embed_returns_list_of_vectors(self, st_patch) -> None:
        from app.embedder import LocalEmbedder
        from app.config import Settings

        embedder = LocalEmbedder(Settings(embed_model="test-model", embed_dim=DIM))
        result = embedder.embed(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 1  # mock returns one item regardless

    def test_no_prompt_when_model_lacks_query_key(self, st_patch) -> None:
        from app.embedder import LocalEmbedder
        from app.config import Settings

        embedder = LocalEmbedder(Settings(embed_model="test-model", embed_dim=DIM))
        embedder._model.prompts = {}
        embedder.embed(["text"], is_query=True)
        _, kwargs = embedder._model.encode.call_args
        assert "prompt_name" not in kwargs

    def test_query_prompt_applied_when_supported(self, st_patch) -> None:
        from app.embedder import LocalEmbedder
        from app.config import Settings

        embedder = LocalEmbedder(Settings(embed_model="test-model", embed_dim=DIM))
        embedder._model.prompts = {"query": "Instruct: ..."}
        embedder.embed(["what is gravity?"], is_query=True)
        _, kwargs = embedder._model.encode.call_args
        assert kwargs.get("prompt_name") == "query"

    def test_dim_property(self, st_patch) -> None:
        from app.embedder import LocalEmbedder
        from app.config import Settings

        embedder = LocalEmbedder(Settings(embed_model="test-model", embed_dim=DIM))
        assert embedder.dim == DIM


# /health endpoint

class TestHealthEndpoint:
    def test_returns_ok_when_ready(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model" in body
        assert "dim" in body


# /embed endpoint

class TestEmbedEndpoint:
    def test_returns_vectors_for_single_text(self, client: TestClient) -> None:
        resp = client.post("/embed", json={"texts": ["hello world"]})
        assert resp.status_code == 200
        body = resp.json()
        assert "vectors" in body
        assert len(body["vectors"]) == 1
        assert isinstance(body["vectors"][0], list)

    def test_returns_vectors_for_batch(self, client: TestClient) -> None:
        resp = client.post("/embed", json={"texts": ["alpha", "beta", "gamma"]})
        assert resp.status_code == 200

    def test_response_includes_model_and_dim(self, client: TestClient) -> None:
        resp = client.post("/embed", json={"texts": ["test"]})
        body = resp.json()
        assert "model" in body
        assert "dim" in body
        assert isinstance(body["dim"], int)

    def test_is_query_false_by_default(self, client: TestClient) -> None:
        resp = client.post("/embed", json={"texts": ["doc text"]})
        assert resp.status_code == 200

    def test_is_query_true_accepted(self, client: TestClient) -> None:
        resp = client.post("/embed", json={"texts": ["query?"], "is_query": True})
        assert resp.status_code == 200

    def test_empty_texts_rejected(self, client: TestClient) -> None:
        resp = client.post("/embed", json={"texts": []})
        assert resp.status_code == 422

    def test_missing_texts_field_rejected(self, client: TestClient) -> None:
        resp = client.post("/embed", json={"is_query": False})
        assert resp.status_code == 422
