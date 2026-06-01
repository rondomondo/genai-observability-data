"""Tests for the shared async HTTP client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from god.config import HttpClientSettings
from god.http_client import HttpClient, _is_429, _is_retryable, _make_limiter


# Settings with fast retry values so tests don't sleep
_FAST_SETTINGS = HttpClientSettings(
    HTTP_RETRY_ATTEMPTS=2,
    HTTP_RETRY_BACKOFF_MIN=0.0,
    HTTP_RETRY_BACKOFF_MAX=0.0,
)


def _make_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://example.com")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


# _is_retryable

@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_is_retryable_for_retryable_status_codes(status_code: int) -> None:
    assert _is_retryable(_make_status_error(status_code))


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_is_not_retryable_for_client_errors(status_code: int) -> None:
    assert not _is_retryable(_make_status_error(status_code))


@pytest.mark.parametrize("exc_class", [httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError])
def test_is_retryable_for_transport_errors(exc_class: type) -> None:
    assert _is_retryable(exc_class("network failure"))


def test_is_not_retryable_for_arbitrary_exceptions() -> None:
    assert not _is_retryable(ValueError("not an httpx error"))


# Lifecycle

@pytest.mark.asyncio
async def test_start_creates_client() -> None:
    client = HttpClient(settings=_FAST_SETTINGS)
    assert client._client is None
    await client.start()
    assert client._client is not None
    assert not client._client.is_closed
    await client.close()


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    client = HttpClient(settings=_FAST_SETTINGS)
    await client.start()
    inner = client._client
    await client.start()
    assert client._client is inner
    await client.close()


@pytest.mark.asyncio
async def test_close_shuts_down_pool() -> None:
    client = HttpClient(settings=_FAST_SETTINGS)
    await client.start()
    await client.close()
    assert client._client is None


@pytest.mark.asyncio
async def test_async_context_manager() -> None:
    async with HttpClient(settings=_FAST_SETTINGS) as client:
        assert client._client is not None
        assert not client._client.is_closed
    assert client._client is None


def test_raw_raises_when_not_started() -> None:
    client = HttpClient(settings=_FAST_SETTINGS)
    with pytest.raises(RuntimeError, match="not started"):
        client._raw()


# request

@pytest.mark.asyncio
async def test_request_returns_response_on_success() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()

    async with HttpClient(settings=_FAST_SETTINGS) as client:
        with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
            resp = await client.request("GET", "/ping")

    assert resp is mock_response


@pytest.mark.asyncio
async def test_request_retries_on_retryable_status() -> None:
    ok_response = MagicMock(spec=httpx.Response)
    ok_response.raise_for_status = MagicMock()

    call_count = 0

    async def _flaky(*_args, **_kwargs) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise _make_status_error(503)
        return ok_response

    async with HttpClient(settings=_FAST_SETTINGS) as client:
        with patch.object(client._client, "request", side_effect=_flaky):
            resp = await client.request("GET", "/flaky")

    assert resp is ok_response
    assert call_count == 2


@pytest.mark.asyncio
async def test_request_raises_after_all_attempts_exhausted() -> None:
    async with HttpClient(settings=_FAST_SETTINGS) as client:
        with patch.object(client._client, "request", new_callable=AsyncMock, side_effect=_make_status_error(503)):
            with pytest.raises(httpx.HTTPStatusError):
                await client.request("GET", "/down")


@pytest.mark.asyncio
async def test_request_does_not_retry_on_404() -> None:
    call_count = 0

    async def _not_found(*_args, **_kwargs) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise _make_status_error(404)

    async with HttpClient(settings=_FAST_SETTINGS) as client:
        with patch.object(client._client, "request", side_effect=_not_found):
            with pytest.raises(httpx.HTTPStatusError):
                await client.request("GET", "/missing")

    assert call_count == 1


# Constructor options

@pytest.mark.asyncio
async def test_base_url_is_passed_to_inner_client() -> None:
    async with HttpClient(settings=_FAST_SETTINGS, base_url="http://api.example.com") as client:
        assert str(client._client.base_url) == "http://api.example.com"


@pytest.mark.asyncio
async def test_default_headers_merged_into_client() -> None:
    headers = {"X-Custom": "value"}
    async with HttpClient(settings=_FAST_SETTINGS, headers=headers) as client:
        assert client._client.headers["x-custom"] == "value"


# Concurrency / rate limiting

@pytest.mark.asyncio
async def test_sem_for_named_rate_uses_configured_limit() -> None:
    client = HttpClient(settings=_FAST_SETTINGS, rates={"chat": 3, "embeddings": 1})
    sem = client._sem_for("chat")
    assert sem._value == 3
    assert client._sem_for("embeddings")._value == 1


@pytest.mark.asyncio
async def test_sem_for_unknown_name_falls_back_to_default_concurrency() -> None:
    settings = HttpClientSettings(
        HTTP_RETRY_ATTEMPTS=2,
        HTTP_RETRY_BACKOFF_MIN=0.0,
        HTTP_RETRY_BACKOFF_MAX=0.0,
        HTTP_DEFAULT_CONCURRENCY=7,
    )
    client = HttpClient(settings=settings, rates={"chat": 2})
    sem = client._sem_for("unknown_op")
    assert sem._value == 7


@pytest.mark.asyncio
async def test_sem_for_no_rates_uses_default_concurrency() -> None:
    settings = HttpClientSettings(
        HTTP_RETRY_ATTEMPTS=2,
        HTTP_RETRY_BACKOFF_MIN=0.0,
        HTTP_RETRY_BACKOFF_MAX=0.0,
        HTTP_DEFAULT_CONCURRENCY=5,
    )
    client = HttpClient(settings=settings)
    assert client._sem_for("anything")._value == 5


@pytest.mark.asyncio
async def test_sem_for_returns_same_instance_on_repeated_calls() -> None:
    client = HttpClient(settings=_FAST_SETTINGS, rates={"chat": 2})
    assert client._sem_for("chat") is client._sem_for("chat")


@pytest.mark.asyncio
async def test_request_uses_named_semaphore() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()

    async with HttpClient(settings=_FAST_SETTINGS, rates={"chat": 2}) as client:
        with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
            await client.request("POST", "/chat", rate_name="chat")

    assert "chat" in client._sems


@pytest.mark.asyncio
async def test_request_default_rate_name_when_omitted() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()

    async with HttpClient(settings=_FAST_SETTINGS) as client:
        with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
            await client.request("GET", "/ping")

    assert "default" in client._sems


@pytest.mark.asyncio
async def test_concurrency_limit_is_enforced() -> None:
    """At most N requests proceed simultaneously when the named limit is N."""
    import asyncio

    concurrent_peak = 0
    active = 0

    async def _slow_request(*_args, **_kwargs) -> httpx.Response:
        nonlocal concurrent_peak, active
        active += 1
        concurrent_peak = max(concurrent_peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        resp = MagicMock(spec=httpx.Response)
        resp.raise_for_status = MagicMock()
        return resp

    async with HttpClient(settings=_FAST_SETTINGS, rates={"op": 2}) as client:
        with patch.object(client._client, "request", side_effect=_slow_request):
            await asyncio.gather(*[client.request("GET", "/x", rate_name="op") for _ in range(6)])

    assert concurrent_peak <= 2


# Token-bucket / RPS limiting

def test_make_limiter_returns_none_for_zero_rps() -> None:
    assert _make_limiter(0.0) is None
    assert _make_limiter(-1.0) is None


def test_make_limiter_returns_limiter_for_positive_rps() -> None:
    from pyrate_limiter import Limiter

    limiter = _make_limiter(10.0)
    assert isinstance(limiter, Limiter)


def test_limiter_for_uses_rps_constructor_map() -> None:
    client = HttpClient(settings=_FAST_SETTINGS, rps={"search": 5.0, "ingest": 0.0})
    assert client._limiter_for("search") is not None
    assert client._limiter_for("ingest") is None


def test_limiter_for_returns_none_when_rps_map_omits_name() -> None:
    client = HttpClient(settings=_FAST_SETTINGS, rps={"search": 5.0})
    # "embed" not in the map and default is 0 -- no limiter
    assert client._limiter_for("embed") is None


def test_limiter_for_reads_settings_fields_by_name() -> None:
    settings = HttpClientSettings(
        HTTP_RETRY_ATTEMPTS=2,
        HTTP_RETRY_BACKOFF_MIN=0.0,
        HTTP_RETRY_BACKOFF_MAX=0.0,
        HTTP_RPS_INGEST=8.0,
        HTTP_RPS_SEARCH=4.0,
    )
    client = HttpClient(settings=settings)
    assert client._limiter_for("ingest") is not None
    assert client._limiter_for("search") is not None
    assert client._limiter_for("cache") is None  # default 0.0


def test_limiter_for_returns_same_instance_on_repeated_calls() -> None:
    client = HttpClient(settings=_FAST_SETTINGS, rps={"op": 5.0})
    assert client._limiter_for("op") is client._limiter_for("op")


@pytest.mark.asyncio
async def test_request_succeeds_with_rps_limiter_configured() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()

    async with HttpClient(settings=_FAST_SETTINGS, rps={"default": 100.0}) as client:
        with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
            resp = await client.request("GET", "/ping")

    assert resp is mock_response


# 429 bucket feedback

def test_is_429_true_for_429_status() -> None:
    assert _is_429(_make_status_error(429))


def test_is_429_false_for_other_statuses() -> None:
    for code in (400, 403, 500, 503):
        assert not _is_429(_make_status_error(code))


@pytest.mark.asyncio
async def test_429_drains_bucket_and_is_retried() -> None:
    ok_response = MagicMock(spec=httpx.Response)
    ok_response.raise_for_status = MagicMock()

    call_count = 0

    async def _flaky(*_args, **_kwargs) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise _make_status_error(429)
        return ok_response

    async with HttpClient(settings=_FAST_SETTINGS, rps={"default": 1000.0}) as client:
        with patch.object(client._client, "request", side_effect=_flaky):
            resp = await client.request("GET", "/quota")

    assert resp is ok_response
    assert call_count == 2


@pytest.mark.asyncio
async def test_drain_token_on_429_does_not_raise_when_no_limiter() -> None:
    client = HttpClient(settings=_FAST_SETTINGS)  # rps=None, no limiter
    # Should be a no-op without raising
    client._drain_token_on_429("default")
