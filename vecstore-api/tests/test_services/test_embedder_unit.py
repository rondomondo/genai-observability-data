"""Unit tests for the Embedder service.

The Gemini API client is patched throughout -- no API key or network needed.
Tests cover the worker lifecycle, embed/embed_batch public API, queue behaviour,
and error propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.services.embedder import Embedder


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        gemini_api_key="fake",
        embed_model="gemini-embedding-001",
        embed_dim=64,
        embed_workers=2,
        embed_batch_size=32,
        otel_enabled=False,
    )


def _fake_vector(dim: int = 64) -> list[float]:
    return [0.1] * dim


def _make_embedder(settings: Settings, fake_vector: list[float] | None = None) -> Embedder:
    """Return an Embedder whose _call_api is patched to return a canned vector."""
    vec = fake_vector or _fake_vector(settings.embed_dim)
    with patch("app.services.embedder.genai.Client"):
        e = Embedder(settings)
    e._call_api = MagicMock(return_value=vec)  # type: ignore[method-assign]
    return e


class TestEmbedderLifecycle:
    async def test_start_spawns_workers(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        assert len(e._workers) == settings.embed_workers
        for task in e._workers:
            assert not task.done()
        await e.stop()

    async def test_stop_cleans_up_workers(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        await e.stop()
        for task in e._workers:
            assert task.done()

    async def test_executor_shut_down_after_stop(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        await e.stop()
        assert e._executor._shutdown  # type: ignore[attr-defined]

    async def test_queue_depth_zero_after_stop(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        await e.stop()
        assert e.queue_depth == 0


class TestQueueDepth:
    def test_queue_depth_property_reflects_qsize(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        assert e.queue_depth == 0


class TestEmbed:
    async def test_returns_vector_and_elapsed_ms(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        vector, elapsed_ms = await e.embed("hello world")
        await e.stop()
        assert isinstance(vector, list)
        assert len(vector) == settings.embed_dim
        assert isinstance(elapsed_ms, int)
        assert elapsed_ms >= 0

    async def test_uses_settings_dim_by_default(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        vector, _ = await e.embed("test")
        await e.stop()
        assert len(vector) == settings.embed_dim

    async def test_dim_override_passed_to_call_api(self, settings: Settings) -> None:
        custom_dim = 128
        e = _make_embedder(settings, fake_vector=_fake_vector(custom_dim))
        await e.start()
        vector, _ = await e.embed("test", dim=custom_dim)
        await e.stop()
        _, call_kwargs = e._call_api.call_args  # type: ignore[union-attr]
        # _call_api is called with positional args (text, dim)
        call_args = e._call_api.call_args[0]  # type: ignore[union-attr]
        assert call_args[1] == custom_dim

    async def test_api_error_propagates_as_exception(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        e._call_api = MagicMock(side_effect=RuntimeError("API unavailable"))  # type: ignore[method-assign]
        await e.start()
        with pytest.raises(RuntimeError, match="API unavailable"):
            await e.embed("will fail")
        await e.stop()

    async def test_multiple_sequential_embeds(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        for text in ["first", "second", "third"]:
            vector, _ = await e.embed(text)
            assert len(vector) == settings.embed_dim
        await e.stop()


class TestEmbedBatch:
    async def test_returns_list_of_vectors_and_elapsed(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        texts = ["alpha", "beta", "gamma"]
        vectors, elapsed_ms = await e.embed_batch(texts)
        await e.stop()
        assert len(vectors) == len(texts)
        for v in vectors:
            assert len(v) == settings.embed_dim
        assert isinstance(elapsed_ms, int)

    async def test_empty_batch_returns_empty_list(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        vectors, elapsed_ms = await e.embed_batch([])
        await e.stop()
        assert vectors == []
        assert elapsed_ms >= 0

    async def test_call_api_called_once_per_text(self, settings: Settings) -> None:
        e = _make_embedder(settings)
        await e.start()
        texts = ["a", "b", "c"]
        await e.embed_batch(texts)
        await e.stop()
        assert e._call_api.call_count == len(texts)  # type: ignore[union-attr]
