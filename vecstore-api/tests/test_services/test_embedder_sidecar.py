"""Unit tests for the _SidecarBackend and Embedder sidecar dispatch.

No real HTTP server or torch/sentence-transformers required. httpx.Client is
patched via respx so we test request shape and response parsing precisely.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.services.embedder import Embedder, _SidecarBackend


DIM = 64
SIDECAR_URL = "http://sidecar-test:8001"


def _sidecar_settings(**overrides) -> Settings:
    return Settings(
        embed_backend="sidecar",
        embed_model="test-model",
        embed_dim=DIM,
        embed_workers=2,
        embed_concurrency=2,
        embed_batch_size=32,
        embed_sidecar_url=SIDECAR_URL,
        otel_enabled=False,
        **overrides,
    )


def _fake_response(vectors: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "vectors": vectors,
        "model": "test-model",
        "dim": DIM,
    }
    return resp


def _make_sidecar_backend(settings: Settings | None = None) -> tuple[_SidecarBackend, MagicMock]:
    s = settings or _sidecar_settings()
    mock_client = MagicMock()
    with patch("app.services.embedder.httpx.Client", return_value=mock_client):
        backend = _SidecarBackend(s)
    return backend, mock_client


# _SidecarBackend unit tests

class TestSidecarBackendEmbed:
    def test_posts_to_embed_endpoint(self) -> None:
        backend, client = _make_sidecar_backend()
        client.post.return_value = _fake_response([[0.1] * DIM])
        backend.embed("hello", DIM)
        call_url = client.post.call_args[0][0]
        assert call_url == f"{SIDECAR_URL}/embed"

    def test_sends_text_as_single_item_list(self) -> None:
        backend, client = _make_sidecar_backend()
        client.post.return_value = _fake_response([[0.1] * DIM])
        backend.embed("my text", DIM)
        payload = client.post.call_args[1]["json"]
        assert payload["texts"] == ["my text"]

    def test_is_query_false_by_default(self) -> None:
        backend, client = _make_sidecar_backend()
        client.post.return_value = _fake_response([[0.1] * DIM])
        backend.embed("doc", DIM)
        payload = client.post.call_args[1]["json"]
        assert payload["is_query"] is False

    def test_is_query_true_forwarded(self) -> None:
        backend, client = _make_sidecar_backend()
        client.post.return_value = _fake_response([[0.1] * DIM])
        backend.embed("query text", DIM, is_query=True)
        payload = client.post.call_args[1]["json"]
        assert payload["is_query"] is True

    def test_returns_first_vector(self) -> None:
        backend, client = _make_sidecar_backend()
        expected = [float(i) for i in range(DIM)]
        client.post.return_value = _fake_response([expected])
        result = backend.embed("text", DIM)
        assert result == expected

    def test_raises_on_http_error(self) -> None:
        import httpx

        backend, client = _make_sidecar_backend()
        client.post.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock()))
        )
        with pytest.raises(httpx.HTTPStatusError):
            backend.embed("text", DIM)

    def test_close_shuts_down_client(self) -> None:
        backend, client = _make_sidecar_backend()
        backend.close()
        client.close.assert_called_once()


class TestSidecarBackendBulk:
    def test_sends_full_batch_in_one_request(self) -> None:
        backend, client = _make_sidecar_backend()
        texts = ["a", "b", "c"]
        client.post.return_value = _fake_response([[0.1] * DIM] * 3)
        backend.embed_batch_bulk(texts)
        assert client.post.call_count == 1
        payload = client.post.call_args[1]["json"]
        assert payload["texts"] == texts

    def test_returns_all_vectors(self) -> None:
        backend, client = _make_sidecar_backend()
        texts = ["x", "y"]
        vecs = [[float(i)] * DIM for i in range(2)]
        client.post.return_value = _fake_response(vecs)
        result = backend.embed_batch_bulk(texts)
        assert result == vecs

    def test_is_query_forwarded_in_bulk(self) -> None:
        backend, client = _make_sidecar_backend()
        client.post.return_value = _fake_response([[0.1] * DIM] * 2)
        backend.embed_batch_bulk(["q1", "q2"], is_query=True)
        payload = client.post.call_args[1]["json"]
        assert payload["is_query"] is True


# Embedder integration with sidecar backend

def _make_embedder(fake_vectors: list[list[float]] | None = None) -> tuple[Embedder, MagicMock]:
    vecs = fake_vectors or [[0.1] * DIM]
    mock_client = MagicMock()
    mock_client.post.return_value = _fake_response(vecs)
    with patch("app.services.embedder.httpx.Client", return_value=mock_client):
        e = Embedder(_sidecar_settings())
    return e, mock_client


class TestEmbedderSidecarDispatch:
    async def test_embed_uses_sidecar_backend(self) -> None:
        e, client = _make_embedder()
        await e.start()
        vector, elapsed_ms = await e.embed("hello")
        await e.stop()
        assert len(vector) == DIM
        assert elapsed_ms >= 0
        client.post.assert_called_once()

    async def test_embed_batch_sends_one_request(self) -> None:
        texts = ["a", "b", "c"]
        e, client = _make_embedder([[0.1] * DIM] * 3)
        await e.start()
        vectors, _ = await e.embed_batch(texts)
        await e.stop()
        assert len(vectors) == 3
        # Bulk path: one POST for the whole batch
        assert client.post.call_count == 1
        payload = client.post.call_args[1]["json"]
        assert payload["texts"] == texts

    async def test_empty_batch_skips_request(self) -> None:
        e, client = _make_embedder()
        await e.start()
        vectors, _ = await e.embed_batch([])
        await e.stop()
        assert vectors == []
        client.post.assert_not_called()

    async def test_is_query_propagated_in_batch(self) -> None:
        e, client = _make_embedder([[0.1] * DIM] * 2)
        await e.start()
        await e.embed_batch(["q1", "q2"], is_query=True)
        await e.stop()
        payload = client.post.call_args[1]["json"]
        assert payload["is_query"] is True

    async def test_stop_closes_sidecar_client(self) -> None:
        e, client = _make_embedder()
        await e.start()
        await e.stop()
        client.close.assert_called_once()

    async def test_http_error_propagates(self) -> None:
        import httpx

        e, client = _make_embedder()
        client.post.return_value = MagicMock(
            raise_for_status=MagicMock(
                side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock())
            )
        )
        await e.start()
        with pytest.raises(httpx.HTTPStatusError):
            await e.embed("will fail")
        await e.stop()
