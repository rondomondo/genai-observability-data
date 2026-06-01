"""Shared async HTTP client with connection pooling, tenacity-based retry, and per-operation concurrency limits."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from god.config import HttpClientSettings, get_http_settings

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError))


class HttpClient:
    """Async HTTP client wrapping httpx.AsyncClient with retry, connection pooling, and per-operation concurrency limits.

    Args:
        settings: HttpClientSettings instance. Defaults to the process singleton.
        base_url: Optional base URL; when set, all request paths are relative to it.
        headers: Default headers merged into every request.
        rates: Optional mapping of operation name to max concurrent requests. Operations not listed
            fall back to settings.http_default_concurrency. Pass an empty dict to disable limiting.
    """

    def __init__(
        self,
        settings: HttpClientSettings | None = None,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        rates: dict[str, int] | None = None,
    ) -> None:
        self._settings = settings or get_http_settings()
        self._base_url = base_url
        self._headers = headers or {}
        self._client: httpx.AsyncClient | None = None
        self._rates = rates
        self._sems: dict[str, asyncio.Semaphore] = {}

    def _sem_for(self, rate_name: str) -> asyncio.Semaphore:
        if rate_name not in self._sems:
            if self._rates is not None:
                limit = self._rates.get(rate_name, self._settings.http_default_concurrency)
            else:
                limit = self._settings.http_default_concurrency
            self._sems[rate_name] = asyncio.Semaphore(limit)
        return self._sems[rate_name]

    async def start(self) -> None:
        """Initialise the underlying connection pool."""
        if self._client is None or self._client.is_closed:
            s = self._settings
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=s.http_connect_timeout,
                    read=s.http_read_timeout,
                    write=s.http_write_timeout,
                    pool=s.http_pool_timeout,
                ),
                limits=httpx.Limits(
                    max_connections=s.http_max_connections,
                    max_keepalive_connections=s.http_max_keepalive,
                ),
                headers={"User-Agent": s.http_user_agent, **self._headers},
            )
            log.debug("HttpClient started base_url=%r", self._base_url)

    async def close(self) -> None:
        """Close the underlying connection pool."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            log.debug("HttpClient closed")

    async def __aenter__(self) -> "HttpClient":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _raw(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            raise RuntimeError("HttpClient not started -- call start() or use as async context manager")
        return self._client

    async def request(self, method: str, path: str, rate_name: str = "default", **kwargs: Any) -> httpx.Response:
        """Execute method with exponential-backoff retry on transient errors.

        Concurrency is gated per rate_name before each attempt. When rates were
        supplied at construction, the named limit applies; unknown names fall back to
        http_default_concurrency. When rates=None (default), all requests share the
        default semaphore.

        Args:
            method: HTTP method string (GET, POST, DELETE, etc.).
            path: Path (relative to base_url when set, otherwise an absolute URL).
            rate_name: Operation name used to select the concurrency limit.
            **kwargs: Forwarded verbatim to httpx.AsyncClient.request.

        Returns:
            The successful httpx.Response.

        Raises:
            httpx.HTTPStatusError: On non-retryable 4xx/5xx after all attempts.
            httpx.TransportError: On persistent network errors after all attempts.
        """
        s = self._settings
        sem = self._sem_for(rate_name)

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(s.http_retry_attempts),
            wait=wait_exponential(multiplier=1, min=s.http_retry_backoff_min, max=s.http_retry_backoff_max),
            reraise=True,
        )
        async def _attempt() -> httpx.Response:
            async with sem:
                resp = await self._raw().request(method, path, **kwargs)
                resp.raise_for_status()
                return resp

        return await _attempt()
