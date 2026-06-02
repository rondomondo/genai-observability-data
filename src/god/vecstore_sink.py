"""VecstoreHttpSink -- flushes generated traces and logs to vecstore-api.

The sink adapts god's internal dataclasses (Trace, LogEntry) into vecstore_models
types and sends them via AsyncVecstoreClient.

The underlying HTTP client is opened on the first flush and reused across
subsequent calls to preserve connection pooling. Call close() when done.

Two flush strategies:
  - default: logs errors and continues so generation is never interrupted
  - raise_on_error=True: propagates VecstoreError to the caller
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import vecstore_models as vm

from god.config import HttpClientSettings, get_settings
from god.models import LogEntry as GodLogEntry
from god.models import Span as GodSpan
from god.models import SpanStatus, Trace
from god.sinks import Sink
from god.vecstore_client import AsyncVecstoreClient, VecstoreError

log = logging.getLogger(__name__)

_TRACE_BATCH_LIMIT = 64
_LOG_BATCH_LIMIT = 256


class VecstoreHttpSink(Sink):
    """Flush generated traces and logs to a running vecstore-api instance.

    Args:
        base_url: Root URL of vecstore-api, e.g. "http://localhost:8000".
        api_key: Value sent as X-API-Key. Defaults to GOD_VECSTORE_API_KEY from settings.
        summarise_traces: When True, vecstore-api runs LLM summarisation on each trace.
        summarise_logs: When True, vecstore-api runs LLM summarisation on each log.
        async_mode: When True, batches are submitted asynchronously (job IDs returned, not polled).
        raise_on_error: When False (default), errors are logged and swallowed so generation continues.
        settings: HttpClientSettings for timeouts and retry policy. Defaults to env-driven singleton.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        *,
        summarise_traces: bool = True,
        summarise_logs: bool = False,
        async_mode: bool = True,
        raise_on_error: bool = False,
        settings: HttpClientSettings | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else get_settings().vecstore_api_key
        self._client = AsyncVecstoreClient(base_url=base_url, api_key=resolved_key, settings=settings)
        self._summarise_traces = summarise_traces
        self._summarise_logs = summarise_logs
        self._async_mode = async_mode
        self._raise_on_error = raise_on_error
        self._opened = False

    async def _ensure_open(self) -> None:
        if not self._opened:
            await self._client.open()
            self._opened = True

    # Sink protocol

    def flush(self, traces: list[Trace], logs: list[GodLogEntry]) -> None:
        """Synchronous flush for use in non-async callers (e.g. the CLI).

        Opens a fresh httpx transport, sends, then closes it within the same event loop.
        Each asyncio.run call gets its own loop; the transport must not outlive it.

        Args:
            traces: Generated Trace objects.
            logs: Generated LogEntry objects.
        """
        asyncio.run(self._flush_and_close(traces, logs))

    def close(self) -> None:
        """No-op for the synchronous Sink protocol when flush() manages its own lifecycle."""

    async def _close_async(self) -> None:
        await self._client.close()
        self._opened = False

    async def _flush_and_close(self, traces: list[Trace], logs: list[GodLogEntry]) -> None:
        try:
            await self.flush_async(traces, logs)
        finally:
            await self._close_async()

    async def flush_async(self, traces: list[Trace], logs: list[GodLogEntry]) -> None:
        """Async flush for use in async callers.

        Args:
            traces: Generated Trace objects.
            logs: Generated LogEntry objects.
        """
        await self._ensure_open()
        if traces:
            await self._send_traces(traces)
        if logs:
            await self._send_logs(logs)

    async def _send_traces(self, traces: list[Trace]) -> None:
        entries = [_trace_to_dto(t) for t in traces]
        for chunk in _chunks(entries, _TRACE_BATCH_LIMIT):
            try:
                result = await self._client.ingest_trace_batch(
                    chunk,
                    summarise=self._summarise_traces,
                    async_mode=self._async_mode,
                )
                if isinstance(result, vm.JobReference):
                    log.info("vecstore trace batch submitted job_id=%s count=%d", result.job_id, len(chunk))
                else:
                    log.debug("vecstore trace batch ingested count=%d", len(chunk))
            except VecstoreError as exc:
                log.error("vecstore trace batch failed count=%d: %s", len(chunk), exc)
                if self._raise_on_error:
                    raise

    async def _send_logs(self, logs: list[GodLogEntry]) -> None:
        entries = [_log_to_dto(entry) for entry in logs]
        for chunk in _chunks(entries, _LOG_BATCH_LIMIT):
            try:
                result = await self._client.ingest_log_batch(
                    chunk,
                    summarise=self._summarise_logs,
                    async_mode=self._async_mode,
                )
                if isinstance(result, vm.JobReference):
                    log.info("vecstore log batch submitted job_id=%s count=%d", result.job_id, len(chunk))
                else:
                    log.debug("vecstore log batch ingested count=%d", len(chunk))
            except VecstoreError as exc:
                log.error("vecstore log batch failed count=%d: %s", len(chunk), exc)
                if self._raise_on_error:
                    raise


# Model adapters


def _trace_to_dto(trace: Trace) -> vm.TraceEntry:
    """Convert a god Trace dataclass to a vecstore_models TraceEntry.

    Args:
        trace: A fully generated Trace with one or more Spans.

    Returns:
        TraceEntry ready to pass to AsyncVecstoreClient.
    """
    root = trace.root_span()
    root_operation = root.name if root else "unknown"
    root_service = root.service_name if root else "unknown"

    if trace.spans:
        start_ns = min(s.start_time_unix_nano for s in trace.spans)
        end_ns = max(s.end_time_unix_nano for s in trace.spans)
        duration_ms = int((end_ns - start_ns) / 1_000_000)
    else:
        duration_ms = 0

    has_error = any(s.status == SpanStatus.ERROR for s in trace.spans)
    status: Literal["ok", "error", "unset"] = "error" if has_error else "ok"

    metadata: dict[str, object] = {}
    if root:
        for k in ("gen_ai.operation.name", "gen_ai.system", "gen_ai.request.model"):
            if k in root.attributes:
                metadata[k] = root.attributes[k]

    return vm.TraceEntry(
        trace_id=trace.trace_id,
        root_operation=root_operation,
        root_service=root_service,
        duration_ms=duration_ms,
        status=status,
        spans=[_span_to_dto(s) for s in trace.spans],
        metadata=metadata,
    )


def _span_to_dto(span: GodSpan) -> vm.Span:
    """Convert a god Span dataclass to a vecstore_models Span.

    Args:
        span: A generated Span with OTLP attributes.

    Returns:
        Span ready to embed in a TraceEntry.
    """
    return vm.Span(
        name=span.name,
        span_id=span.span_id,
        parent_span_id=span.parent_span_id or None,
        service=span.service_name,
        start_time_ms=span.start_time_unix_nano // 1_000_000,
        duration_ms=int(span.duration_ms),
        status=span.status.value.lower(),  # type: ignore[arg-type]
        attributes=dict(span.attributes),
        events=[{"name": e.name, "time_unix_nano": e.time_unix_nano, **e.attributes} for e in span.events],
    )


def _log_to_dto(log_entry: GodLogEntry) -> vm.LogEntry:
    """Convert a god LogEntry dataclass to a vecstore_models LogEntry.

    Args:
        log_entry: A generated LogEntry.

    Returns:
        LogEntry ready to pass to AsyncVecstoreClient.
    """
    return vm.LogEntry(
        message=log_entry.body,
        service=log_entry.service_name,
        severity=log_entry.level.value,  # type: ignore[arg-type]
        timestamp=log_entry.timestamp_unix_nano // 1_000_000,
        trace_id=log_entry.trace_id or None,
        span_id=log_entry.span_id or None,
        metadata=dict(log_entry.attributes),
    )


# Utility


def _chunks(items: list, size: int) -> list[list]:  # type: ignore[type-arg]
    """Split a list into chunks of at most size elements.

    Args:
        items: The list to split.
        size: Maximum chunk length.

    Returns:
        List of sub-lists, each of length <= size.
    """
    return [items[i : i + size] for i in range(0, len(items), size)]
