"""Summarisation service.

Uses a pydantic-ai agent for error/anomaly traces (structured output).
Falls back to a fast template summariser for normal traces and all log lines.

The agent returns a TraceSummary with typed fields so callers can route
on severity, affected_services etc. without parsing free text.
"""

import asyncio
import time

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pyrate_limiter import Duration, Limiter, Rate

from app.config import Settings
from app.logging_config import get_logger
from app.models.ingest import LogEntry, TraceEntry
from app.models.summarise import TraceSummary

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are an SRE observability assistant analysing distributed traces and log events.
Be precise and terse. Use actual service names, operation names, and error messages from the data.
Never speculate beyond what the trace or log data supports.
Only populate root_cause if status is error or timeout — leave it null for ok traces.
Return only the structured JSON — no preamble, no markdown fences."""


class Summariser:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mock = settings.summariser_backend == "mock" or settings.summariser_mock
        self._agent: Agent[None, TraceSummary] | None = None
        self._sem = asyncio.Semaphore(settings.summariser_concurrency)
        self._limiter = Limiter(Rate(settings.summariser_rate_limit_rpm, Duration.MINUTE))
        log.info(
            "summariser_started",
            backend=settings.summariser_backend,
            mock=self._mock,
            model=settings.summariser_model,
            llm_threshold_ms=settings.summariser_llm_threshold_duration_ms,
            concurrency=settings.summariser_concurrency,
            rate_limit_rpm=settings.summariser_rate_limit_rpm,
        )

    def _get_agent(self) -> "Agent[None, TraceSummary]":
        if self._agent is None:
            backend = self._settings.summariser_backend
            if backend == "google-cloud":
                try:
                    import google.auth
                    import google.oauth2.credentials
                    from pydantic_ai.providers.google_cloud import GoogleCloudProvider
                except ImportError as exc:
                    raise ImportError(
                        "google-auth and pydantic-ai google support are required for "
                        "summariser_backend='google-cloud'. Install with: uv add google-auth pydantic-ai-slim[google]"
                    ) from exc
                if self._settings.google_access_token:
                    credentials = google.oauth2.credentials.Credentials(token=self._settings.google_access_token)
                    project = self._settings.google_cloud_project or None
                else:
                    credentials, project = google.auth.default()
                    project = self._settings.google_cloud_project or project
                provider = GoogleCloudProvider(
                    credentials=credentials,
                    project=project,
                    location=self._settings.google_cloud_location or None,
                )
                model = GoogleModel(self._settings.summariser_model, provider=provider)
            elif backend == "gemini":
                model = GoogleModel(self._settings.summariser_model)
            else:
                model = AnthropicModel(self._settings.summariser_model)
            self._agent = Agent(
                model=model,
                output_type=TraceSummary,
                system_prompt=_SYSTEM_PROMPT,
            )
        return self._agent

    async def _acquire_token(self) -> None:
        loop = asyncio.get_running_loop()
        delay = 0.05
        waited = 0.0
        while True:
            acquired = await loop.run_in_executor(None, lambda: self._limiter.try_acquire("summariser"))
            if acquired:
                if waited > 0:
                    log.warning("summariser_rate_limit_wait", waited_ms=round(waited * 1000))
                return
            await asyncio.sleep(delay)
            waited += delay
            delay = min(delay * 2, 2.0)

    # ── Public API ────────────────────────────────────────────────────────────

    async def summarise_trace(self, trace: TraceEntry, force_llm: bool = False) -> tuple[TraceSummary, int | None]:
        """Returns (TraceSummary, llm_ms | None)."""
        if not self._settings.summariser_mock and (force_llm or self._should_use_llm(trace)):
            return await self._llm_summarise_trace(trace)
        path = "mock" if self._settings.summariser_mock else "template"
        summary = self._template_summarise_trace(trace)
        log.debug(f"summarise_trace {summary}", path=path, trace_id=trace.trace_id, severity=summary.severity)
        return summary, None

    async def summarise_log(self, entry: LogEntry) -> tuple[TraceSummary, int | None]:
        """Logs use template by default — LLM only for ERROR/FATAL."""
        if not self._settings.summariser_mock and entry.severity in ("ERROR", "FATAL"):
            return await self._llm_summarise_log(entry)
        path = "mock" if self._settings.summariser_mock else "template"
        summary = self._template_summarise_log(entry)
        log.debug(f"summarise_log {summary}", path=path, service=entry.service, severity=entry.severity)
        return summary, None

    def trace_to_text(self, trace: TraceEntry) -> str:
        """Convert a trace to embeddable text (used when not summarising)."""
        errors = [s for s in trace.spans if s.status == "error"]
        slow = [s for s in trace.spans if s.duration_ms > self._settings.summariser_llm_threshold_duration_ms]
        parts = [
            f"{trace.root_service} {trace.root_operation}",
            f"duration={trace.duration_ms}ms status={trace.status}",
        ]
        if errors:
            parts.append("errors: " + ", ".join(s.name for s in errors))
        if slow:
            parts.append("slow_spans: " + ", ".join(s.name for s in slow))
        return " | ".join(parts)

    def log_to_text(self, entry: LogEntry) -> str:
        return f"{entry.service} [{entry.severity}] {entry.message}"

    # ── Heuristic ─────────────────────────────────────────────────────────────

    def _should_use_llm(self, trace: TraceEntry) -> bool:
        if self._settings.summariser_mock:
            return False
        if trace.status == "error":
            return True
        if trace.duration_ms > self._settings.summariser_llm_threshold_duration_ms:
            return True
        if any(s.status == "error" for s in trace.spans):
            return True
        return False

    # ── Template summarisers (fast, no API call) ──────────────────────────────

    def _template_summarise_trace(self, trace: TraceEntry) -> TraceSummary:
        errors = [s for s in trace.spans if s.status == "error"]
        services = list({s.service for s in trace.spans} | {trace.root_service})
        severity = "low"
        if trace.status == "error" or errors:
            severity = "high"
        elif trace.duration_ms > self._settings.summariser_llm_threshold_duration_ms:
            severity = "medium"

        narrative = (
            f"{trace.root_service} executed {trace.root_operation} "
            f"in {trace.duration_ms}ms with status {trace.status}."
        )
        if errors:
            narrative += f" Errors in: {', '.join(s.name for s in errors)}."

        return TraceSummary(
            narrative=narrative,
            root_cause=None,
            affected_services=services,
            severity=severity,  # type: ignore[arg-type]
            recommended_action=None,
            used_llm=False,
        )

    def _template_summarise_log(self, entry: LogEntry) -> TraceSummary:
        severity_map = {"DEBUG": "low", "INFO": "low", "WARN": "medium", "ERROR": "high", "FATAL": "critical"}
        return TraceSummary(
            narrative=f"[{entry.severity}] {entry.service}: {entry.message}",
            root_cause=None,
            affected_services=[entry.service],
            severity=severity_map.get(entry.severity, "low"),  # type: ignore[arg-type]
            recommended_action=None,
            used_llm=False,
        )

    # ── LLM summarisers ───────────────────────────────────────────────────────

    @staticmethod
    def _flatten_spans(trace: TraceEntry) -> list[dict]:
        """Strip internal IDs and timing noise before passing spans to the LLM."""
        return [
            {
                "service": s.service,
                "operation": s.name,
                "duration_ms": s.duration_ms,
                "status": s.status,
                "error": s.attributes.get("error") or s.attributes.get("exception.message"),
            }
            for s in trace.spans
        ]

    async def _llm_summarise_trace(self, trace: TraceEntry) -> tuple[TraceSummary, int]:
        import json

        prompt = (
            f"Analyse this distributed trace and populate the summary fields.\n\n"
            f"Trace ID: {trace.trace_id}\n"
            f"Root service: {trace.root_service} - {trace.root_operation}\n"
            f"Duration: {trace.duration_ms}ms | Status: {trace.status}\n\n"
            f"Spans:\n{json.dumps(self._flatten_spans(trace), indent=2)}"
        )
        t0 = time.perf_counter()
        try:
            agent = self._get_agent()
            await self._acquire_token()
            async with self._sem:
                result = await agent.run(prompt)
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            summary = result.output
            summary.used_llm = True
            log.info("summarise_trace", path="llm", trace_id=trace.trace_id, ms=elapsed_ms, severity=summary.severity)
            return summary, elapsed_ms
        except Exception as exc:
            log.error("llm_summarise_trace_failed", trace_id=trace.trace_id, error=str(exc))
            # Graceful fallback
            summary = self._template_summarise_trace(trace)
            return summary, None

    async def _llm_summarise_log(self, entry: LogEntry) -> tuple[TraceSummary, int]:
        import json

        context_str = json.dumps(entry.metadata, indent=2) if entry.metadata else "none"
        prompt = (
            f"Analyse this log event and populate the summary fields.\n\n"
            f"Service: {entry.service} | Severity: {entry.severity}\n"
            f"Message: {entry.message}\n"
            f"Context: {context_str}"
        )
        t0 = time.perf_counter()
        try:
            agent = self._get_agent()
            await self._acquire_token()
            async with self._sem:
                result = await agent.run(prompt)
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            summary = result.output
            summary.used_llm = True
            log.info("summarise_log", path="llm", service=entry.service, severity=entry.severity, ms=elapsed_ms)
            return summary, elapsed_ms
        except Exception as exc:
            log.error("llm_summarise_log_failed", service=entry.service, error=str(exc))
            summary = self._template_summarise_log(entry)
            return summary, None
