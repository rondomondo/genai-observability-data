"""Shared async HTTP client with connection pooling, tenacity retry, and per-operation rate limiting.

Rate limiting is two-layered per named operation:
  1. Token-bucket (pyrate-limiter) -- proactive throughput cap (req/s), avoids 429s before they happen.
  2. asyncio.Semaphore -- caps simultaneous in-flight requests regardless of throughput.

When a 429 is received the retry handler drains one extra token from the bucket so that
the burst allowance shrinks in proportion to upstream pressure, making the client
self-tuning under sustained quota stress.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from pyrate_limiter import Duration, Limiter, Rate
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from god.config import HttpClientSettings, get_http_settings

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError))


def _is_429(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def _make_limiter(rps: float) -> Limiter | None:
    """Build a Limiter for the given rate, or return None when rps <= 0."""
    if rps <= 0:
        return None
    return Limiter(Rate(int(rps), Duration.SECOND))


class HttpClient:
    """Async HTTP client wrapping httpx.AsyncClient with retry, connection pooling, and per-operation rate limits.

    Rate limiting is two-layered per named operation:
    a token-bucket caps throughput (req/s) and a semaphore caps in-flight concurrency.
    When a 429 is received the bucket for that operation absorbs an extra token so the
    client backs off proactively on subsequent requests.

    Args:
        settings: HttpClientSettings instance. Defaults to the process singleton.
        base_url: Optional base URL; when set, all request paths are relative to it.
        headers: Default headers merged into every request.
        rates: Optional mapping of operation name to max concurrent requests. Operations not listed
            fall back to settings.http_default_concurrency. Pass an empty dict to disable limiting.
        rps: Optional mapping of operation name to max requests-per-second. Operations not listed
            fall back to the settings http_rps_* fields. Pass an empty dict to disable rps limiting.
    """

    def __init__(
        self,
        settings: HttpClientSettings | None = None,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        rates: dict[str, int] | None = None,
        rps: dict[str, float] | None = None,
    ) -> None:
        self._settings = settings or get_http_settings()
        self._base_url = base_url
        self._headers = headers or {}
        self._client: httpx.AsyncClient | None = None
        self._rates = rates
        self._rps = rps
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._limiters: dict[str, Limiter | None] = {}

    # Concurrency gate

    def _sem_for(self, rate_name: str) -> asyncio.Semaphore:
        if rate_name not in self._sems:
            if self._rates is not None:
                limit = self._rates.get(rate_name, self._settings.http_default_concurrency)
            else:
                limit = self._settings.http_default_concurrency
            self._sems[rate_name] = asyncio.Semaphore(limit)
        return self._sems[rate_name]

    # Token-bucket gate

    def _limiter_for(self, rate_name: str) -> Limiter | None:
        if rate_name not in self._limiters:
            rps = self._resolve_rps(rate_name)
            self._limiters[rate_name] = _make_limiter(rps)
        return self._limiters[rate_name]

    def _resolve_rps(self, rate_name: str) -> float:
        if self._rps is not None:
            return self._rps.get(rate_name, 0.0)
        s = self._settings
        mapping = {
            "ingest": s.http_rps_ingest,
            "search": s.http_rps_search,
            "cache": s.http_rps_cache,
            "admin": s.http_rps_admin,
        }
        return mapping.get(rate_name, s.http_rps_default)

    async def _acquire_token(self, rate_name: str) -> None:
        """Wait until the token bucket allows a request for rate_name."""
        limiter = self._limiter_for(rate_name)
        if limiter is None:
            return
        loop = asyncio.get_running_loop()
        # try_acquire is synchronous; poll with exponential sleep to avoid blocking the event loop.
        delay = 0.01
        while True:
            acquired = await loop.run_in_executor(None, lambda: limiter.try_acquire(rate_name))
            if acquired:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 1.0)

    def _drain_token_on_429(self, rate_name: str) -> None:
        """Consume an extra token to shrink the burst allowance after receiving a 429."""
        limiter = self._limiter_for(rate_name)
        if limiter is None:
            return
        try:
            limiter.try_acquire(rate_name)
        except Exception:
            pass

    # Lifecycle

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
        """Execute method with token-bucket gating and exponential-backoff retry on transient errors.

        Each attempt first waits for a token-bucket slot (proactive throughput cap), then acquires
        the concurrency semaphore. On a 429 response an extra token is drained from the bucket so
        the client slows itself down automatically under sustained upstream pressure.

        Args:
            method: HTTP method string (GET, POST, DELETE, etc.).
            path: Path (relative to base_url when set, otherwise an absolute URL).
            rate_name: Operation name used to select both the concurrency limit and the RPS bucket.
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
            await self._acquire_token(rate_name)
            async with sem:
                try:
                    resp = await self._raw().request(method, path, **kwargs)
                    resp.raise_for_status()
                    return resp
                except httpx.HTTPStatusError as exc:
                    if _is_429(exc):
                        self._drain_token_on_429(rate_name)
                        log.warning(
                            "http_429_received rate_name=%r path=%r -- drained bucket token",
                            rate_name,
                            path,
                        )
                    raise

        return await _attempt()
