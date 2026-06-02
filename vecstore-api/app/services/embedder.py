"""Embedding service.

Supports four backends, selected by ``Settings.embed_backend``:

- ``"gemini"``               -- Google GenAI via pydantic-ai GoogleEmbeddingModel
- ``"sentence-transformers"``-- local model via pydantic-ai SentenceTransformerEmbeddingModel
- ``"sidecar"``              -- HTTP call to the embedder-sidecar container
- ``"mock"``                 -- deterministic test vectors via pydantic-ai TestEmbeddingModel

Usage:
    embedder = Embedder(settings)
    await embedder.start()
    vector, ms = await embedder.embed("some text")
    vectors, ms = await embedder.embed_batch(["text1", "text2"])
    await embedder.stop()
"""

import asyncio
import time
from collections.abc import Sequence
from datetime import datetime, timezone

import httpx
from pydantic_ai.embeddings import EmbeddingModel, EmbeddingResult, EmbedInputType, EmbeddingSettings
from pydantic_ai.usage import RequestUsage

from app.config import Settings
from app.logging_config import get_logger

log = get_logger(__name__)


class _SidecarEmbeddingModel(EmbeddingModel):
    """pydantic-ai EmbeddingModel that delegates to the embedder-sidecar HTTP service."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._url = settings.embed_sidecar_url.rstrip("/") + "/embed"
        self._model = settings.embed_model
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def system(self) -> str:
        return "sidecar"

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        texts, _ = self.prepare_embed(inputs, settings)
        is_query = input_type == "query"
        response = await self._client.post(self._url, json={"texts": texts, "is_query": is_query})
        response.raise_for_status()
        vectors: list[list[float]] = response.json()["vectors"]
        return EmbeddingResult(
            embeddings=vectors,
            inputs=texts,
            input_type=input_type,
            model_name=self._model,
            provider_name="sidecar",
        )

    async def close(self) -> None:
        await self._client.aclose()


def _make_model(settings: Settings) -> EmbeddingModel:
    if settings.embed_backend == "mock":
        from pydantic_ai.embeddings import TestEmbeddingModel
        return TestEmbeddingModel(dimensions=settings.embed_dim)

    if settings.embed_backend == "sentence-transformers":
        try:
            from pydantic_ai.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai sentence-transformers support is required for embed_backend='sentence-transformers'. "
                "Install it with: uv add pydantic-ai-slim[sentence-transformers]"
            ) from exc
        return SentenceTransformerEmbeddingModel(settings.embed_model)

    if settings.embed_backend == "sidecar":
        return _SidecarEmbeddingModel(settings)

    # gemini (default)
    try:
        from pydantic_ai.embeddings.google import GoogleEmbeddingModel
    except ImportError as exc:
        raise ImportError(
            "pydantic-ai google support is required for embed_backend='gemini'. "
            "Install it with: uv add pydantic-ai-slim[google]"
        ) from exc
    from pydantic_ai.providers.google import GoogleProvider
    provider = GoogleProvider(api_key=settings.gemini_api_key)
    return GoogleEmbeddingModel(settings.embed_model, provider=provider)


class Embedder:
    """Async embedding facade over a pydantic-ai EmbeddingModel.

    Maintains the same public API as the previous thread-pool implementation
    so callers require no changes.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = _make_model(settings)
        self._sem = asyncio.Semaphore(settings.embed_concurrency)

    # Lifecycle

    async def start(self) -> None:
        log.info(
            "embedder_started",
            backend=self._settings.embed_backend,
            model=self._settings.embed_model,
        )

    async def stop(self) -> None:
        if isinstance(self._model, _SidecarEmbeddingModel):
            await self._model.close()
        log.info("embedder_stopped")

    # Public API

    async def embed(
        self,
        text: str,
        dim: int | None = None,
        is_query: bool = False,
    ) -> tuple[list[float], int]:
        """Embed a single text.

        Args:
            text: text to embed.
            dim: output dimensionality; falls back to ``settings.embed_dim``.
            is_query: pass True for query texts so backends that support
                      query-specific prompts can apply them.

        Returns:
            A tuple of (vector, elapsed_ms).
        """
        effective_dim = dim or self._settings.embed_dim
        embed_settings = EmbeddingSettings(dimensions=effective_dim)
        input_type: EmbedInputType = "query" if is_query else "document"
        t0 = time.perf_counter()
        async with self._sem:
            result = await self._model.embed(text, input_type=input_type, settings=embed_settings)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return list(result.embeddings[0]), elapsed_ms

    async def embed_batch(
        self,
        texts: list[str],
        dim: int | None = None,
        is_query: bool = False,
    ) -> tuple[list[list[float]], int]:
        """Embed multiple texts.

        For the sidecar backend the full batch is sent in one HTTP request.
        All other backends embed concurrently with individual semaphore slots.

        Args:
            texts: texts to embed.
            dim: output dimensionality; falls back to ``settings.embed_dim``.
            is_query: applied uniformly to all texts in the batch.

        Returns:
            A tuple of (vectors, elapsed_ms).
        """
        if not texts:
            return [], 0

        effective_dim = dim or self._settings.embed_dim
        embed_settings = EmbeddingSettings(dimensions=effective_dim)
        input_type: EmbedInputType = "query" if is_query else "document"
        t0 = time.perf_counter()

        if isinstance(self._model, _SidecarEmbeddingModel):
            async with self._sem:
                result = await self._model.embed(texts, input_type=input_type, settings=embed_settings)
            vectors = [list(v) for v in result.embeddings]
        else:
            async def _one(text: str) -> list[float]:
                async with self._sem:
                    r = await self._model.embed(text, input_type=input_type, settings=embed_settings)
                return list(r.embeddings[0])

            vectors = list(await asyncio.gather(*[_one(t) for t in texts]))

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return vectors, elapsed_ms

    @property
    def queue_depth(self) -> int:
        """Always 0 -- retained for API compatibility; no queue in this implementation."""
        return 0
