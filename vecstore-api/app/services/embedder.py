"""Embedding service.

Supports three backends, selected by ``Settings.embed_backend``:

- ``"gemini"``               -- Google GenAI (synchronous client, network I/O)
- ``"sentence-transformers"``-- local SentenceTransformer model (CPU/GPU, no API key)
- ``"sidecar"``              -- HTTP call to the embedder-sidecar container

The Gemini and sentence-transformers backends are synchronous and run inside a
shared ``ThreadPoolExecutor``. The sidecar backend uses ``httpx`` for async HTTP
and bypasses the executor entirely.

Usage:
    embedder = Embedder(settings)
    await embedder.start()
    vector = await embedder.embed("some text")
    vectors = await embedder.embed_batch(["text1", "text2"])
    await embedder.stop()
"""

import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.config import Settings
from app.logging_config import get_logger

log = get_logger(__name__)


class EmbedRequest:
    __slots__ = ("text", "dim", "future", "is_query")

    def __init__(
        self,
        text: str,
        dim: int,
        loop: asyncio.AbstractEventLoop,
        is_query: bool = False,
    ) -> None:
        self.text = text
        self.dim = dim
        self.future: asyncio.Future[list[float]] = loop.create_future()
        self.is_query = is_query


# Backend protocol

class EmbedBackend(ABC):
    """Synchronous embedding backend, called from a thread-pool worker."""

    @abstractmethod
    def embed(self, text: str, dim: int, is_query: bool = False) -> list[float]:
        """Embed a single text and return a float vector.

        Args:
            text: the text to embed.
            dim: requested output dimensionality (advisory for backends that
                 do not support Matryoshka truncation).
            is_query: when True the backend may apply a query-specific prompt
                      (e.g. Qwen3 ``prompt_name="query"``).
        """


class _GeminiBackend(EmbedBackend):
    def __init__(self, settings: Settings) -> None:
        from google import genai
        from google.genai import types as genai_types

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.embed_model
        self._genai_types = genai_types

    def embed(self, text: str, dim: int, is_query: bool = False) -> list[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._genai_types.EmbedContentConfig(output_dimensionality=dim),
        )
        if not result.embeddings:
            raise RuntimeError("Gemini API returned no embeddings")
        return list(result.embeddings[0].values)


class _SentenceTransformersBackend(EmbedBackend):
    def __init__(self, settings: Settings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embed_backend='sentence-transformers'. "
                "Install it with: uv add sentence-transformers"
            ) from exc

        self._model: "SentenceTransformer" = SentenceTransformer(settings.embed_model)
        log.info("sentence_transformers_model_loaded", model=settings.embed_model)

    def embed(self, text: str, dim: int, is_query: bool = False) -> list[float]:
        prompt_name = "query" if is_query and "query" in (self._model.prompts or {}) else None
        kwargs: dict = {"prompt_name": prompt_name} if prompt_name else {}
        vector = self._model.encode(text, **kwargs)
        return vector.tolist()


class _SidecarBackend(EmbedBackend):
    """Delegates embedding to the embedder-sidecar HTTP service.

    Uses a synchronous httpx client so it fits the existing thread-pool
    worker model without requiring changes to the Embedder dispatch loop.
    Batch calls are collapsed into a single HTTP request to avoid per-text
    round-trip overhead; single embed() calls send a one-item batch.
    """

    def __init__(self, settings: Settings) -> None:
        self._url = settings.embed_sidecar_url.rstrip("/") + "/embed"
        self._client = httpx.Client(timeout=30.0)

    def embed(self, text: str, dim: int, is_query: bool = False) -> list[float]:
        response = self._client.post(self._url, json={"texts": [text], "is_query": is_query})
        response.raise_for_status()
        return response.json()["vectors"][0]

    def embed_batch_bulk(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Send a full batch in one HTTP request.

        Args:
            texts: texts to embed.
            is_query: applied to all texts in the batch.

        Returns:
            List of float vectors, one per input text.
        """
        response = self._client.post(self._url, json={"texts": texts, "is_query": is_query})
        response.raise_for_status()
        return response.json()["vectors"]

    def close(self) -> None:
        self._client.close()


def _make_backend(settings: Settings) -> EmbedBackend:
    if settings.embed_backend == "sentence-transformers":
        return _SentenceTransformersBackend(settings)
    if settings.embed_backend == "sidecar":
        return _SidecarBackend(settings)
    return _GeminiBackend(settings)


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backend = _make_backend(settings)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.embed_workers,
            thread_name_prefix="embedder",
        )
        self._queue: asyncio.Queue[EmbedRequest | None] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._sem = asyncio.Semaphore(settings.embed_concurrency)

    # Lifecycle

    async def start(self) -> None:
        self._running = True
        for i in range(self._settings.embed_workers):
            task = asyncio.create_task(self._worker(i), name=f"embedder-worker-{i}")
            self._workers.append(task)
        log.info(
            "embedder_started",
            backend=self._settings.embed_backend,
            model=self._settings.embed_model,
            workers=self._settings.embed_workers,
        )

    async def stop(self) -> None:
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._executor.shutdown(wait=True)
        if isinstance(self._backend, _SidecarBackend):
            self._backend.close()
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
                      query-specific prompts (e.g. Qwen3) can apply them.

        Returns:
            A tuple of (vector, elapsed_ms).
        """
        effective_dim = dim or self._settings.embed_dim
        loop = asyncio.get_running_loop()
        req = EmbedRequest(text, effective_dim, loop, is_query=is_query)
        t0 = time.perf_counter()
        await self._queue.put(req)
        vector = await req.future
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return vector, elapsed_ms

    async def embed_batch(
        self,
        texts: list[str],
        dim: int | None = None,
        is_query: bool = False,
    ) -> tuple[list[list[float]], int]:
        """Embed multiple texts.

        For the sidecar backend the full batch is sent in one HTTP request.
        All other backends embed concurrently via individual worker tasks.

        Args:
            texts: texts to embed.
            dim: output dimensionality; falls back to ``settings.embed_dim``.
            is_query: applied uniformly to all texts in the batch.

        Returns:
            A tuple of (vectors, elapsed_ms).
        """
        t0 = time.perf_counter()
        if isinstance(self._backend, _SidecarBackend) and texts:
            loop = asyncio.get_running_loop()
            vectors = await loop.run_in_executor(
                self._executor,
                lambda: self._backend.embed_batch_bulk(texts, is_query),  # type: ignore[union-attr]
            )
        else:
            tasks = [self.embed(t, dim, is_query=is_query) for t in texts]
            results = await asyncio.gather(*tasks)
            vectors = [r[0] for r in results]
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return vectors, elapsed_ms

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # Internal

    async def _worker(self, worker_id: int) -> None:
        log.debug("embed_worker_ready", worker_id=worker_id)
        loop = asyncio.get_running_loop()
        while True:
            req = await self._queue.get()
            if req is None:
                break
            try:
                async with self._sem:
                    vector = await loop.run_in_executor(
                        self._executor,
                        lambda r=req: self._backend.embed(r.text, r.dim, r.is_query),
                    )
                req.future.set_result(vector)
            except Exception as exc:
                log.error("embed_worker_error", worker_id=worker_id, error=str(exc))
                req.future.set_exception(exc)
            finally:
                self._queue.task_done()
