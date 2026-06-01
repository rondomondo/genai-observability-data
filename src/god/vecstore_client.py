"""Async httpx client for the vecstore-api service."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from god.config import HttpClientSettings, get_settings
from god.http_client import HttpClient
from vecstore_models import (
    BatchIngestResponse,
    CacheLookupResponse,
    CacheStatusResponse,
    CacheStoreResponse,
    IngestResponse,
    JobReference,
    JobStatus,
    LogEntry,
    SearchResponse,
    StatusResponse,
    TraceEntry,
)
from vecstore_models.cache import CacheHit

log = logging.getLogger(__name__)


class VecstoreError(Exception):
    """Raised when a vecstore-api call fails after all retries."""

    def __init__(self, method: str, url: str, status: int | None, detail: str) -> None:
        self.method = method
        self.url = url
        self.status = status
        self.detail = detail
        super().__init__(f"{method} {url} -> {status}: {detail}")


class AsyncVecstoreClient:
    """Async client for vecstore-api.

    Usage (preferred - automatic lifecycle):

        async with AsyncVecstoreClient.create("http://localhost:8000") as client:
            result = await client.ingest_log(entry)

    Usage (manual lifecycle, e.g. long-lived sinks):

        client = AsyncVecstoreClient("http://localhost:8000", api_key="...")
        await client.open()
        ...
        await client.close()

    Args:
        base_url: Root URL of vecstore-api, no trailing slash.
        api_key: Value sent as X-API-Key header. Defaults to GOD_VECSTORE_API_KEY from settings.
        settings: HttpClientSettings for timeouts and retry policy. Defaults to env-driven singleton.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        settings: HttpClientSettings | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else get_settings().vecstore_api_key
        self._http = HttpClient(
            settings=settings,
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": resolved_key, "Content-Type": "application/json"},
        )

    async def open(self) -> None:
        """Initialise the underlying httpx connection pool."""
        await self._http.start()

    async def close(self) -> None:
        """Close the underlying httpx connection pool."""
        await self._http.close()

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        base_url: str,
        api_key: str | None = None,
        settings: HttpClientSettings | None = None,
    ) -> AsyncIterator["AsyncVecstoreClient"]:
        """Async context manager that opens and closes the client automatically.

        Args:
            base_url: Root URL of vecstore-api.
            api_key: API key for authentication. Defaults to GOD_VECSTORE_API_KEY from settings.
            settings: HttpClientSettings for timeouts and retry policy.

        Yields:
            An open AsyncVecstoreClient.
        """
        inst = cls(base_url=base_url, api_key=api_key, settings=settings)
        async with inst._http:
            yield inst

    async def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        """Execute a request, mapping httpx errors to VecstoreError.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: Path relative to base_url.
            **kwargs: Forwarded to HttpClient.request.

        Returns:
            The successful httpx.Response.

        Raises:
            VecstoreError: On HTTP error or persistent network failure.
        """
        try:
            return await self._http.request(method, path, **kwargs)
        except httpx.HTTPStatusError as exc:
            raise VecstoreError(method, path, exc.response.status_code, exc.response.text) from exc
        except httpx.TransportError as exc:
            raise VecstoreError(method, path, None, str(exc)) from exc

    # Health

    async def health(self) -> dict[str, object]:
        """Check service liveness (no auth required).

        Returns:
            Parsed JSON response body.
        """
        resp = await self._request("GET", "/health")
        return resp.json()  # type: ignore[no-any-return]

    async def status(self) -> StatusResponse:
        """Return full service status including vector set cardinalities.

        Returns:
            Parsed StatusResponse.
        """
        resp = await self._request("GET", "/status")
        return StatusResponse.model_validate(resp.json())

    # Ingest: logs

    async def ingest_log(self, entry: LogEntry, *, summarise: bool = False) -> IngestResponse:
        """Ingest a single log entry.

        Args:
            entry: LogEntry to store.
            summarise: When True, the API runs LLM summarisation before embedding.

        Returns:
            Parsed IngestResponse.
        """
        resp = await self._request("POST", "/ingest/log", json={"entry": entry.model_dump(), "summarise": summarise})
        return IngestResponse.model_validate(resp.json())

    async def ingest_log_batch(
        self,
        entries: list[LogEntry],
        *,
        summarise: bool = False,
        async_mode: bool = True,
    ) -> BatchIngestResponse | JobReference:
        """Ingest a batch of log entries (max 256 per call).

        Args:
            entries: List of LogEntry objects.
            summarise: Run LLM summarisation for each entry.
            async_mode: When True, returns a JobReference immediately.

        Returns:
            BatchIngestResponse when async_mode=False, JobReference when async_mode=True.

        Raises:
            ValueError: If entries exceeds 256.
        """
        if len(entries) > 256:
            raise ValueError(f"log batch size {len(entries)} exceeds vecstore-api limit of 256")
        payload = {
            "entries": [e.model_dump() for e in entries],
            "summarise": summarise,
            "async_mode": async_mode,
        }
        resp = await self._request("POST", "/ingest/log/batch", json=payload)
        data = resp.json()
        if async_mode:
            return JobReference.model_validate(data)
        return BatchIngestResponse.model_validate(data)

    # Ingest: traces

    async def ingest_trace(self, entry: TraceEntry, *, summarise: bool = True) -> IngestResponse:
        """Ingest a single trace.

        Args:
            entry: TraceEntry to store.
            summarise: When True, the API generates a narrative summary before embedding.

        Returns:
            Parsed IngestResponse.
        """
        resp = await self._request(
            "POST", "/ingest/trace", json={"entry": entry.model_dump(), "summarise": summarise}
        )
        return IngestResponse.model_validate(resp.json())

    async def ingest_trace_batch(
        self,
        entries: list[TraceEntry],
        *,
        summarise: bool = True,
        async_mode: bool = True,
    ) -> BatchIngestResponse | JobReference:
        """Ingest a batch of traces (max 64 per call).

        Args:
            entries: List of TraceEntry objects.
            summarise: Run LLM summarisation for each trace.
            async_mode: When True, returns a JobReference immediately.

        Returns:
            BatchIngestResponse when async_mode=False, JobReference when async_mode=True.

        Raises:
            ValueError: If entries exceeds 64.
        """
        if len(entries) > 64:
            raise ValueError(f"trace batch size {len(entries)} exceeds vecstore-api limit of 64")
        payload = {
            "entries": [e.model_dump() for e in entries],
            "summarise": summarise,
            "async_mode": async_mode,
        }
        resp = await self._request("POST", "/ingest/trace/batch", json=payload)
        data = resp.json()
        if async_mode:
            return JobReference.model_validate(data)
        return BatchIngestResponse.model_validate(data)

    # Search

    async def search(
        self,
        query: str,
        *,
        keys: list[str] | None = None,
        top_k: int = 5,
        filter_expr: str | None = None,
        include_vectors: bool = False,
    ) -> SearchResponse:
        """Semantic search across vector sets.

        Args:
            query: Natural language query text.
            keys: Vector set keys to search. Defaults to all (cache, logs, traces).
            top_k: Max results to return (1-50).
            filter_expr: Redis filter expression, e.g. '.service == "payments"'.
            include_vectors: Include raw embedding vectors in results.

        Returns:
            Parsed SearchResponse.
        """
        payload: dict[str, object] = {"query": query, "top_k": top_k, "include_vectors": include_vectors}
        if keys is not None:
            payload["keys"] = keys
        if filter_expr is not None:
            payload["filter_expr"] = filter_expr
        resp = await self._request("POST", "/search", json=payload)
        return SearchResponse.model_validate(resp.json())

    # Cache

    async def cache_query(
        self,
        question: str,
        *,
        threshold: float | None = None,
        top_k: int = 1,
    ) -> CacheLookupResponse:
        """Semantic cache lookup.

        Args:
            question: The query to look up in the RAG cache.
            threshold: Similarity threshold override (0.0-1.0).
            top_k: Max cache hits to return (1-10).

        Returns:
            Parsed CacheLookupResponse with hit/score/result fields.
        """
        payload: dict[str, object] = {"question": question, "top_k": top_k}
        if threshold is not None:
            payload["threshold"] = threshold
        resp = await self._request("POST", "/cache/query", json=payload)
        return CacheLookupResponse.model_validate(resp.json())

    async def cache_store(
        self,
        question: str,
        answer: str,
        *,
        sources: list[str] | None = None,
        model: str | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CacheStoreResponse:
        """Store a Q&A pair in the semantic cache.

        Args:
            question: The question/prompt text.
            answer: The answer/response text.
            sources: Source document references.
            model: Model name that produced the answer.
            ttl_seconds: Cache entry TTL (min 60).
            metadata: Arbitrary additional metadata.

        Returns:
            Parsed CacheStoreResponse.
        """
        payload: dict[str, object] = {"question": question, "answer": answer, "sources": sources or []}
        if model is not None:
            payload["model"] = model
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds
        if metadata:
            payload["metadata"] = metadata
        resp = await self._request("POST", "/cache", json=payload)
        return CacheStoreResponse.model_validate(resp.json())

    async def cache_delete(self, element_id: str) -> dict[str, object]:
        """Remove a cache entry by element ID.

        Args:
            element_id: The element ID returned by cache_store.

        Returns:
            Parsed response JSON.
        """
        resp = await self._request("DELETE", f"/cache/{element_id}")
        return resp.json()  # type: ignore[no-any-return]

    async def cache_status(self) -> CacheStatusResponse:
        """Return cache status (cardinality, TTL watcher state).

        Returns:
            Parsed CacheStatusResponse.
        """
        resp = await self._request("GET", "/cache/status")
        return CacheStatusResponse.model_validate(resp.json())

    # Job polling

    async def job_status(self, job_id: str) -> JobStatus:
        """Poll async batch job status.

        Args:
            job_id: Job ID returned by an async batch ingest call.

        Returns:
            Parsed JobStatus.
        """
        resp = await self._request("GET", f"/status/job/{job_id}")
        return JobStatus.model_validate(resp.json())

    async def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval_s: float = 1.0,
        timeout_s: float = 120.0,
    ) -> JobStatus:
        """Poll a job until it reaches done/failed or timeout.

        Args:
            job_id: Job ID to poll.
            poll_interval_s: Seconds between polls.
            timeout_s: Maximum total wait time.

        Returns:
            Final JobStatus.

        Raises:
            TimeoutError: If the job does not complete within timeout_s.
            VecstoreError: If the job reaches failed status.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            job = await self.job_status(job_id)
            if job.status == "done":
                return job
            if job.status == "failed":
                raise VecstoreError("JOB", f"/status/job/{job_id}", None, job.error_detail or "job failed")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} did not complete within {timeout_s}s (last status={job.status!r})")
            await asyncio.sleep(poll_interval_s)


# Re-export CacheHit so callers can type-hint cache_query results without a separate import.
__all__ = [
    "AsyncVecstoreClient",
    "CacheHit",
    "VecstoreError",
]
