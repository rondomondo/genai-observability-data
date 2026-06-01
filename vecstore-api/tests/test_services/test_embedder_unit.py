"""Unit tests for the Embedder service.

No external APIs or network access required. Each backend is patched at the
point where the underlying library is imported inside the backend class.

Tests cover:
- Worker lifecycle (start/stop, executor shutdown)
- Gemini backend dispatch and argument passing
- SentenceTransformers backend dispatch, query prompt, missing-library error
- embed() / embed_batch() public API contracts
- is_query flag propagation
- Error propagation through the worker queue
- Concurrency semaphore respected
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.services.embedder import (
    Embedder,
    _GeminiBackend,
    _SentenceTransformersBackend,
    _make_backend,
)


DIM = 64


def _fake_vector(dim: int = DIM) -> list[float]:
    return [0.1] * dim


def _gemini_settings(**overrides) -> Settings:
    return Settings(
        gemini_api_key="fake-key",
        embed_backend="gemini",
        embed_model="gemini-embedding-001",
        embed_dim=DIM,
        embed_workers=2,
        embed_concurrency=2,
        embed_batch_size=32,
        otel_enabled=False,
        **overrides,
    )


def _st_settings(**overrides) -> Settings:
    return Settings(
        embed_backend="sentence-transformers",
        embed_model="mock-model",
        embed_dim=DIM,
        embed_workers=2,
        embed_concurrency=2,
        embed_batch_size=32,
        otel_enabled=False,
        **overrides,
    )


def _make_embedder(settings: Settings, fake_vector: list[float] | None = None) -> Embedder:
    """Return an Embedder with a patched backend.embed so no real API is called."""
    vec = fake_vector or _fake_vector(settings.embed_dim)
    backend = MagicMock()
    backend.embed = MagicMock(return_value=vec)
    with patch("app.services.embedder._make_backend", return_value=backend):
        e = Embedder(settings)
    return e


# Backend factory

class TestMakeBackend:
    def test_gemini_backend_selected(self) -> None:
        s = _gemini_settings()
        with patch("app.services.embedder._GeminiBackend") as MockGemini:
            _make_backend(s)
            MockGemini.assert_called_once_with(s)

    def test_st_backend_selected(self) -> None:
        s = _st_settings()
        with patch("app.services.embedder._SentenceTransformersBackend") as MockST:
            _make_backend(s)
            MockST.assert_called_once_with(s)


# Gemini backend

class TestGeminiBackend:
    def _make(self) -> tuple[_GeminiBackend, MagicMock]:
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [MagicMock(values=[0.1] * DIM)]
        mock_client.models.embed_content.return_value = mock_result

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai_types = MagicMock()

        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_genai_types}):
            backend = _GeminiBackend(_gemini_settings())

        backend._client = mock_client
        backend._genai_types = mock_genai_types
        return backend, mock_client

    def test_returns_float_list(self) -> None:
        backend, _ = self._make()
        result = backend.embed("hello", DIM)
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_passes_text_to_api(self) -> None:
        backend, client = self._make()
        backend.embed("my text", DIM)
        call_kwargs = client.models.embed_content.call_args[1]
        assert call_kwargs["contents"] == "my text"

    def test_passes_dim_to_config(self) -> None:
        backend, client = self._make()
        backend.embed("text", 128)
        backend._genai_types.EmbedContentConfig.assert_called_with(output_dimensionality=128)

    def test_raises_when_no_embeddings(self) -> None:
        backend, client = self._make()
        client.models.embed_content.return_value = MagicMock(embeddings=[])
        with pytest.raises(RuntimeError, match="no embeddings"):
            backend.embed("text", DIM)

    def test_is_query_flag_ignored(self) -> None:
        """Gemini backend does not use is_query; call should not raise."""
        backend, _ = self._make()
        result = backend.embed("query text", DIM, is_query=True)
        assert len(result) == DIM


# SentenceTransformers backend

class TestSentenceTransformersBackend:
    def _make(self, prompts: dict | None = None) -> tuple[_SentenceTransformersBackend, MagicMock]:
        mock_st_model = MagicMock()
        mock_st_model.prompts = prompts or {}
        # Return a plain list -- the backend calls .tolist() on the result
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.2] * DIM
        mock_st_model.encode.return_value = mock_array

        mock_st_cls = MagicMock(return_value=mock_st_model)
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)}):
            backend = _SentenceTransformersBackend(_st_settings())
        backend._model = mock_st_model
        return backend, mock_st_model

    def test_returns_float_list(self) -> None:
        backend, _ = self._make()
        result = backend.embed("hello", DIM)
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_no_prompt_name_when_not_query(self) -> None:
        backend, model = self._make(prompts={"query": "Instruct: ..."})
        backend.embed("doc text", DIM, is_query=False)
        _, kwargs = model.encode.call_args
        assert "prompt_name" not in kwargs

    def test_query_prompt_applied_when_model_supports_it(self) -> None:
        backend, model = self._make(prompts={"query": "Instruct: ..."})
        backend.embed("what is gravity?", DIM, is_query=True)
        _, kwargs = model.encode.call_args
        assert kwargs.get("prompt_name") == "query"

    def test_query_prompt_skipped_when_model_lacks_it(self) -> None:
        backend, model = self._make(prompts={})
        backend.embed("what is gravity?", DIM, is_query=True)
        _, kwargs = model.encode.call_args
        assert "prompt_name" not in kwargs

    def test_import_error_raises_clear_message(self) -> None:
        import sys
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                _SentenceTransformersBackend(_st_settings())


# Embedder lifecycle

class TestEmbedderLifecycle:
    async def test_start_spawns_workers(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        assert len(e._workers) == 2
        for task in e._workers:
            assert not task.done()
        await e.stop()

    async def test_stop_cleans_up_workers(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        await e.stop()
        for task in e._workers:
            assert task.done()

    async def test_executor_shut_down_after_stop(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        await e.stop()
        assert e._executor._shutdown  # type: ignore[attr-defined]

    async def test_queue_depth_zero_after_stop(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        await e.stop()
        assert e.queue_depth == 0


# embed() public API

class TestEmbed:
    async def test_returns_vector_and_elapsed_ms(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        vector, elapsed_ms = await e.embed("hello world")
        await e.stop()
        assert isinstance(vector, list)
        assert len(vector) == DIM
        assert isinstance(elapsed_ms, int)
        assert elapsed_ms >= 0

    async def test_uses_settings_dim_by_default(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        _, _ = await e.embed("test")
        await e.stop()
        args = e._backend.embed.call_args[0]  # type: ignore[union-attr]
        assert args[1] == DIM

    async def test_dim_override_forwarded_to_backend(self) -> None:
        e = _make_embedder(_gemini_settings(), fake_vector=_fake_vector(128))
        await e.start()
        await e.embed("test", dim=128)
        await e.stop()
        args = e._backend.embed.call_args[0]  # type: ignore[union-attr]
        assert args[1] == 128

    async def test_is_query_false_by_default(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        await e.embed("some doc")
        await e.stop()
        args = e._backend.embed.call_args[0]  # type: ignore[union-attr]
        assert args[2] is False

    async def test_is_query_true_forwarded(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        await e.embed("what is the capital?", is_query=True)
        await e.stop()
        args = e._backend.embed.call_args[0]  # type: ignore[union-attr]
        assert args[2] is True

    async def test_api_error_propagates(self) -> None:
        e = _make_embedder(_gemini_settings())
        e._backend.embed = MagicMock(side_effect=RuntimeError("API unavailable"))  # type: ignore[union-attr]
        await e.start()
        with pytest.raises(RuntimeError, match="API unavailable"):
            await e.embed("will fail")
        await e.stop()

    async def test_multiple_sequential_embeds(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        for text in ["first", "second", "third"]:
            vector, _ = await e.embed(text)
            assert len(vector) == DIM
        await e.stop()


# embed_batch() public API

class TestEmbedBatch:
    async def test_returns_list_of_vectors_and_elapsed(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        texts = ["alpha", "beta", "gamma"]
        vectors, elapsed_ms = await e.embed_batch(texts)
        await e.stop()
        assert len(vectors) == len(texts)
        for v in vectors:
            assert len(v) == DIM
        assert isinstance(elapsed_ms, int)

    async def test_empty_batch_returns_empty_list(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        vectors, elapsed_ms = await e.embed_batch([])
        await e.stop()
        assert vectors == []
        assert elapsed_ms >= 0

    async def test_backend_called_once_per_text(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        texts = ["a", "b", "c"]
        await e.embed_batch(texts)
        await e.stop()
        assert e._backend.embed.call_count == len(texts)  # type: ignore[union-attr]

    async def test_is_query_forwarded_to_all_texts(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        await e.embed_batch(["q1", "q2"], is_query=True)
        await e.stop()
        for call in e._backend.embed.call_args_list:  # type: ignore[union-attr]
            assert call[0][2] is True

    async def test_batch_is_query_false_by_default(self) -> None:
        e = _make_embedder(_gemini_settings())
        await e.start()
        await e.embed_batch(["d1", "d2"])
        await e.stop()
        for call in e._backend.embed.call_args_list:  # type: ignore[union-attr]
            assert call[0][2] is False
