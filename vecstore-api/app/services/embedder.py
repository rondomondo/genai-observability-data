"""Embedding service.

Google GenAI client is synchronous. We run it in a thread pool executor
via an asyncio queue so concurrent embed requests are handled by a fixed
pool of workers rather than spawning unlimited threads.

Usage:
    embedder = Embedder(settings)
    await embedder.start()           # call once at startup
    vector = await embedder.embed("some text")
    vectors = await embedder.embed_batch(["text1", "text2"])
    await embedder.stop()            # call once at shutdown
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types as genai_types

from app.config import Settings
from app.logging_config import get_logger

log = get_logger(__name__)


class EmbedRequest:
    __slots__ = ("text", "dim", "future")

    def __init__(self, text: str, dim: int, loop: asyncio.AbstractEventLoop) -> None:
        self.text = text
        self.dim = dim
        self.future: asyncio.Future[list[float]] = loop.create_future()


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.embed_workers,
            thread_name_prefix="embedder",
        )
        self._queue: asyncio.Queue[EmbedRequest | None] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._sem = asyncio.Semaphore(settings.embed_concurrency)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        for i in range(self._settings.embed_workers):
            task = asyncio.create_task(self._worker(i), name=f"embedder-worker-{i}")
            self._workers.append(task)
        log.info("embedder_started", workers=self._settings.embed_workers)

    async def stop(self) -> None:
        self._running = False
        # Poison-pill each worker
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._executor.shutdown(wait=True)
        log.info("embedder_stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed(self, text: str, dim: int | None = None) -> tuple[list[float], int]:
        """Embed a single text. Returns (vector, elapsed_ms)."""
        effective_dim = dim or self._settings.embed_dim
        loop = asyncio.get_running_loop()
        req = EmbedRequest(text, effective_dim, loop)
        t0 = time.perf_counter()
        await self._queue.put(req)
        vector = await req.future
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return vector, elapsed_ms

    async def embed_batch(self, texts: list[str], dim: int | None = None) -> tuple[list[list[float]], int]:
        """Embed multiple texts concurrently. Returns (vectors, elapsed_ms)."""
        t0 = time.perf_counter()
        tasks = [self.embed(t, dim) for t in texts]
        results = await asyncio.gather(*tasks)
        vectors = [r[0] for r in results]
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return vectors, elapsed_ms

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ── Internal ──────────────────────────────────────────────────────────────

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
                        lambda r=req: self._call_api(r.text, r.dim),
                    )
                req.future.set_result(vector)
            except Exception as exc:
                log.error("embed_worker_error", worker_id=worker_id, error=str(exc))
                req.future.set_exception(exc)
            finally:
                self._queue.task_done()

    def _call_api(self, text: str, dim: int) -> list[float]:
        result = self._client.models.embed_content(
            model=self._settings.embed_model,
            contents=text,
            config=genai_types.EmbedContentConfig(output_dimensionality=dim),
        )
        if not result.embeddings:
            raise RuntimeError("Gemini API returned no embeddings")
        return list(result.embeddings[0].values)
