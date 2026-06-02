"""Unit tests for the Embedder service.

No external APIs or network access required. All backend models are replaced
with pydantic-ai's TestEmbeddingModel or lightweight mocks so no real API
calls are made.

Tests cover:
- Backend factory (_make_model) selects the right model per embed_backend
- Lifecycle: start/stop log correctly, sidecar close is called on stop
- embed() public API: returns vector + elapsed_ms, dim forwarded, is_query mapping
- embed_batch() public API: returns vectors + elapsed_ms, empty batch, is_query mapping
- Error propagation from the underlying model
- queue_depth compatibility shim always returns 0
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.embeddings import EmbeddingResult, TestEmbeddingModel

from app.config import Settings
from app.services.embedder import Embedder, _SidecarEmbeddingModel, _make_model


DIM = 64


def _settings(**overrides) -> Settings:
    base: dict = {
        "embed_backend": "mock",
        "embed_model": "test-model",
        "embed_dim": DIM,
        "embed_workers": 2,
        "embed_concurrency": 2,
        "embed_batch_size": 32,
        "otel_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)


def _fake_result(vectors: list[list[float]], input_type: str = "document") -> EmbeddingResult:
    return EmbeddingResult(
        embeddings=vectors,
        inputs=[""] * len(vectors),
        input_type=input_type,  # type: ignore[arg-type]
        model_name="test-model",
        provider_name="test",
    )


def _make_embedder(settings: Settings | None = None) -> Embedder:
    """Return an Embedder backed by TestEmbeddingModel."""
    s = settings or _settings()
    return Embedder(s)


def _make_embedder_with_mock_model(
    settings: Settings | None = None,
    vectors: list[list[float]] | None = None,
) -> tuple[Embedder, AsyncMock]:
    """Return an Embedder with a fully mocked EmbeddingModel."""
    s = settings or _settings()
    vecs = vectors or [[0.1] * DIM]
    mock_model = AsyncMock()
    mock_model.embed = AsyncMock(return_value=_fake_result(vecs))
    with patch("app.services.embedder._make_model", return_value=mock_model):
        e = Embedder(s)
    return e, mock_model


# Backend factory

class TestMakeModel:
    def test_mock_backend_returns_test_model(self) -> None:
        s = _settings(embed_backend="mock")
        model = _make_model(s)
        assert isinstance(model, TestEmbeddingModel)

    def test_sidecar_backend_returns_sidecar_model(self) -> None:
        s = _settings(embed_backend="sidecar", embed_sidecar_url="http://sidecar:8001")
        model = _make_model(s)
        assert isinstance(model, _SidecarEmbeddingModel)

    def test_gemini_backend_returns_google_model(self) -> None:
        from pydantic_ai.embeddings.google import GoogleEmbeddingModel
        s = _settings(embed_backend="gemini", gemini_api_key="fake-key")
        model = _make_model(s)
        assert isinstance(model, GoogleEmbeddingModel)

    def test_st_backend_import_error_raises_clear_message(self) -> None:
        s = _settings(embed_backend="sentence-transformers")
        with patch.dict("sys.modules", {"pydantic_ai.models.sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                _make_model(s)


# Lifecycle

class TestEmbedderLifecycle:
    async def test_start_does_not_raise(self) -> None:
        e = _make_embedder()
        await e.start()
        await e.stop()

    async def test_stop_closes_sidecar_model(self) -> None:
        s = _settings(embed_backend="sidecar", embed_sidecar_url="http://sidecar:8001")
        mock_model = AsyncMock(spec=_SidecarEmbeddingModel)
        mock_model.embed = AsyncMock(return_value=_fake_result([[0.1] * DIM]))
        with patch("app.services.embedder._make_model", return_value=mock_model):
            e = Embedder(s)
        await e.start()
        await e.stop()
        mock_model.close.assert_called_once()

    async def test_stop_non_sidecar_does_not_call_close(self) -> None:
        e, mock_model = _make_embedder_with_mock_model()
        await e.start()
        await e.stop()
        assert not hasattr(mock_model, "close") or not mock_model.close.called

    def test_queue_depth_always_zero(self) -> None:
        e = _make_embedder()
        assert e.queue_depth == 0


# embed() public API

class TestEmbed:
    async def test_returns_vector_and_elapsed_ms(self) -> None:
        e, _ = _make_embedder_with_mock_model()
        await e.start()
        vector, elapsed_ms = await e.embed("hello world")
        await e.stop()
        assert isinstance(vector, list)
        assert len(vector) == DIM
        assert isinstance(elapsed_ms, int)
        assert elapsed_ms >= 0

    async def test_uses_settings_dim_by_default(self) -> None:
        e, mock_model = _make_embedder_with_mock_model()
        await e.start()
        await e.embed("text")
        await e.stop()
        call_kwargs = mock_model.embed.call_args[1]
        assert call_kwargs["settings"]["dimensions"] == DIM

    async def test_dim_override_forwarded(self) -> None:
        e, mock_model = _make_embedder_with_mock_model(
            vectors=[[0.1] * 128],
        )
        await e.start()
        await e.embed("text", dim=128)
        await e.stop()
        call_kwargs = mock_model.embed.call_args[1]
        assert call_kwargs["settings"]["dimensions"] == 128

    async def test_is_query_false_maps_to_document_input_type(self) -> None:
        e, mock_model = _make_embedder_with_mock_model()
        await e.start()
        await e.embed("some doc")
        await e.stop()
        call_kwargs = mock_model.embed.call_args[1]
        assert call_kwargs["input_type"] == "document"

    async def test_is_query_true_maps_to_query_input_type(self) -> None:
        e, mock_model = _make_embedder_with_mock_model()
        await e.start()
        await e.embed("what is the capital?", is_query=True)
        await e.stop()
        call_kwargs = mock_model.embed.call_args[1]
        assert call_kwargs["input_type"] == "query"

    async def test_api_error_propagates(self) -> None:
        e, mock_model = _make_embedder_with_mock_model()
        mock_model.embed.side_effect = RuntimeError("API unavailable")
        await e.start()
        with pytest.raises(RuntimeError, match="API unavailable"):
            await e.embed("will fail")
        await e.stop()

    async def test_mock_backend_returns_deterministic_vector(self) -> None:
        e = _make_embedder(_settings(embed_backend="mock", embed_dim=64))
        await e.start()
        vector, _ = await e.embed("hello")
        await e.stop()
        assert vector == [1.0] * 64


# embed_batch() public API

class TestEmbedBatch:
    async def test_returns_list_of_vectors_and_elapsed(self) -> None:
        texts = ["alpha", "beta", "gamma"]
        e, mock_model = _make_embedder_with_mock_model(vectors=[[0.1] * DIM] * 3)
        mock_model.embed = AsyncMock(side_effect=[
            _fake_result([[0.1] * DIM]),
            _fake_result([[0.2] * DIM]),
            _fake_result([[0.3] * DIM]),
        ])
        await e.start()
        vectors, elapsed_ms = await e.embed_batch(texts)
        await e.stop()
        assert len(vectors) == len(texts)
        for v in vectors:
            assert len(v) == DIM
        assert isinstance(elapsed_ms, int)

    async def test_empty_batch_returns_empty_list(self) -> None:
        e, _ = _make_embedder_with_mock_model()
        await e.start()
        vectors, elapsed_ms = await e.embed_batch([])
        await e.stop()
        assert vectors == []
        assert elapsed_ms == 0

    async def test_is_query_false_maps_to_document(self) -> None:
        e, mock_model = _make_embedder_with_mock_model(vectors=[[0.1] * DIM] * 2)
        mock_model.embed = AsyncMock(side_effect=[
            _fake_result([[0.1] * DIM]),
            _fake_result([[0.1] * DIM]),
        ])
        await e.start()
        await e.embed_batch(["d1", "d2"])
        await e.stop()
        for call in mock_model.embed.call_args_list:
            assert call[1]["input_type"] == "document"

    async def test_is_query_true_maps_to_query(self) -> None:
        e, mock_model = _make_embedder_with_mock_model(vectors=[[0.1] * DIM] * 2)
        mock_model.embed = AsyncMock(side_effect=[
            _fake_result([[0.1] * DIM]),
            _fake_result([[0.1] * DIM]),
        ])
        await e.start()
        await e.embed_batch(["q1", "q2"], is_query=True)
        await e.stop()
        for call in mock_model.embed.call_args_list:
            assert call[1]["input_type"] == "query"
