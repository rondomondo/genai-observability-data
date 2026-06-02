"""Unit tests for _SidecarEmbeddingModel and Embedder sidecar dispatch.

No real HTTP server required. httpx.AsyncClient.post is patched via AsyncMock
so we test request shape and response parsing precisely.

Tests cover:
- _SidecarEmbeddingModel.embed() posts to the correct endpoint
- is_query mapping: True -> "query", False -> "document"
- Batch requests are sent as a single POST
- HTTP errors propagate
- close() shuts down the async client
- Embedder.embed_batch() uses a single HTTP request for the sidecar backend
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic_ai.embeddings import EmbeddingResult

from app.config import Settings
from app.services.embedder import Embedder, _SidecarEmbeddingModel


DIM = 64
SIDECAR_URL = "http://sidecar-test:8001"


def _sidecar_settings(**overrides) -> Settings:
    base: dict = {
        "embed_backend": "sidecar",
        "embed_model": "test-model",
        "embed_dim": DIM,
        "embed_workers": 2,
        "embed_concurrency": 2,
        "embed_batch_size": 32,
        "embed_sidecar_url": SIDECAR_URL,
        "otel_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)


def _embed_url() -> str:
    return f"{SIDECAR_URL}/embed"


def _mock_response(vectors: list[list[float]]) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"vectors": vectors, "model": "test-model", "dim": DIM}
    return resp


def _fake_result(vectors: list[list[float]], input_type: str = "document") -> EmbeddingResult:
    return EmbeddingResult(
        embeddings=vectors,
        inputs=[""] * len(vectors),
        input_type=input_type,  # type: ignore[arg-type]
        model_name="test-model",
        provider_name="test",
    )


def _make_sidecar_model() -> tuple[_SidecarEmbeddingModel, AsyncMock]:
    """Return a _SidecarEmbeddingModel with a mocked async httpx client."""
    model = _SidecarEmbeddingModel(_sidecar_settings())
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    model._client = mock_client
    return model, mock_client


def _make_embedder() -> tuple[Embedder, AsyncMock]:
    """Return an Embedder (sidecar backend) with a mocked async httpx client."""
    e = Embedder(_sidecar_settings())
    assert isinstance(e._model, _SidecarEmbeddingModel)
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    e._model._client = mock_client
    return e, mock_client


# _SidecarEmbeddingModel

class TestSidecarEmbeddingModel:
    async def test_posts_to_embed_endpoint(self) -> None:
        model, client = _make_sidecar_model()
        client.post.return_value = _mock_response([[0.1] * DIM])
        await model.embed("hello", input_type="document")
        call_url = client.post.call_args[0][0]
        assert call_url == _embed_url()

    async def test_sends_text_as_single_item_list(self) -> None:
        model, client = _make_sidecar_model()
        client.post.return_value = _mock_response([[0.1] * DIM])
        await model.embed("my text", input_type="document")
        payload = client.post.call_args[1]["json"]
        assert payload["texts"] == ["my text"]

    async def test_document_input_type_maps_to_is_query_false(self) -> None:
        model, client = _make_sidecar_model()
        client.post.return_value = _mock_response([[0.1] * DIM])
        await model.embed("doc", input_type="document")
        payload = client.post.call_args[1]["json"]
        assert payload["is_query"] is False

    async def test_query_input_type_maps_to_is_query_true(self) -> None:
        model, client = _make_sidecar_model()
        client.post.return_value = _mock_response([[0.1] * DIM])
        await model.embed("query text", input_type="query")
        payload = client.post.call_args[1]["json"]
        assert payload["is_query"] is True

    async def test_returns_embedding_result_with_vectors(self) -> None:
        model, client = _make_sidecar_model()
        expected = [float(i) for i in range(DIM)]
        client.post.return_value = _mock_response([expected])
        result = await model.embed("text", input_type="document")
        assert list(result.embeddings[0]) == expected

    async def test_batch_sent_as_single_post(self) -> None:
        texts = ["a", "b", "c"]
        model, client = _make_sidecar_model()
        client.post.return_value = _mock_response([[0.1] * DIM] * 3)
        result = await model.embed(texts, input_type="document")
        assert client.post.call_count == 1
        assert len(result.embeddings) == 3

    async def test_http_error_propagates(self) -> None:
        model, client = _make_sidecar_model()
        err_resp = MagicMock(spec=httpx.Response)
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=MagicMock()
        )
        client.post.return_value = err_resp
        with pytest.raises(httpx.HTTPStatusError):
            await model.embed("text", input_type="document")

    async def test_close_shuts_down_async_client(self) -> None:
        model, client = _make_sidecar_model()
        await model.close()
        client.aclose.assert_called_once()


# Embedder with sidecar backend

class TestEmbedderSidecarDispatch:
    async def test_embed_returns_vector_and_elapsed(self) -> None:
        e, client = _make_embedder()
        client.post.return_value = _mock_response([[0.1] * DIM])
        await e.start()
        vector, elapsed_ms = await e.embed("hello")
        await e.stop()
        assert len(vector) == DIM
        assert elapsed_ms >= 0

    async def test_embed_batch_sends_one_request(self) -> None:
        texts = ["a", "b", "c"]
        e, client = _make_embedder()
        client.post.return_value = _mock_response([[0.1] * DIM] * 3)
        await e.start()
        vectors, _ = await e.embed_batch(texts)
        await e.stop()
        assert len(vectors) == 3
        assert client.post.call_count == 1
        payload = client.post.call_args[1]["json"]
        assert payload["texts"] == texts

    async def test_empty_batch_skips_request(self) -> None:
        e, client = _make_embedder()
        await e.start()
        vectors, elapsed_ms = await e.embed_batch([])
        await e.stop()
        assert vectors == []
        assert elapsed_ms == 0
        client.post.assert_not_called()

    async def test_is_query_propagated_in_batch(self) -> None:
        e, client = _make_embedder()
        client.post.return_value = _mock_response([[0.1] * DIM] * 2)
        await e.start()
        await e.embed_batch(["q1", "q2"], is_query=True)
        await e.stop()
        payload = client.post.call_args[1]["json"]
        assert payload["is_query"] is True

    async def test_stop_closes_sidecar_client(self) -> None:
        e, client = _make_embedder()
        await e.start()
        await e.stop()
        client.aclose.assert_called_once()

    async def test_http_error_propagates(self) -> None:
        e, client = _make_embedder()
        err_resp = MagicMock(spec=httpx.Response)
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=MagicMock()
        )
        client.post.return_value = err_resp
        await e.start()
        with pytest.raises(httpx.HTTPStatusError):
            await e.embed("will fail")
        await e.stop()
