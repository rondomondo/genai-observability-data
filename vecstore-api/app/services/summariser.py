"""Summarisation service.

Uses a pydantic-ai agent for error/anomaly traces (structured output).
Falls back to a fast template summariser for normal traces and all log lines.

The agent returns a TraceSummary with typed fields so callers can route
on severity, affected_services etc. without parsing free text.
"""

import asyncio
import time

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

from app.config import Settings
from app.logging_config import get_logger
from app.models.ingest import LogEntry, TraceEntry
from app.models.summarise import TraceSummary

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are an SRE assistant summarising observability data for a RAG index.
Be specific: use actual service names, error messages, and span names from the data.
Never invent details not present in the input.
Return only the structured JSON — no preamble, no markdown fences."""


class Summariser:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._agent: Agent[None, TraceSummary] | None = None
        self._sem = asyncio.Semaphore(settings.summariser_concurrency)

    def _get_agent(self) -> "Agent[None, TraceSummary]":
        if self._agent is None:
            model = GeminiModel(self._settings.summariser_model)
            self._agent = Agent(
                model=model,
                result_type=TraceSummary,
                system_prompt=_SYSTEM_PROMPT,
            )
        return self._agent

    # ── Public API ────────────────────────────────────────────────────────────

    async def summarise_trace(self, trace: TraceEntry, force_llm: bool = False) -> tuple[TraceSummary, int | None]:
        """Returns (TraceSummary, llm_ms | None)."""
        if force_llm or self._should_use_llm(trace):
            return await self._llm_summarise_trace(trace)
        else:
            summary = self._template_summarise_trace(trace)
            return summary, None

    async def summarise_log(self, entry: LogEntry) -> tuple[TraceSummary, int | None]:
        """Logs use template by default — LLM only for ERROR/FATAL."""
        if entry.severity in ("ERROR", "FATAL"):
            return await self._llm_summarise_log(entry)
        summary = self._template_summarise_log(entry)
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

    async def _llm_summarise_trace(self, trace: TraceEntry) -> tuple[TraceSummary, int]:
        import json

        prompt = (
            f"Summarise this distributed trace for a RAG index.\n\n"
            f"{json.dumps(trace.model_dump(), indent=2, default=str)}"
        )
        t0 = time.perf_counter()
        try:
            agent = self._get_agent()
            async with self._sem:
                result = await agent.run(prompt)
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            summary = result.data
            summary.used_llm = True
            log.info("llm_summarise_trace", trace_id=trace.trace_id, ms=elapsed_ms, severity=summary.severity)
            return summary, elapsed_ms
        except Exception as exc:
            log.error("llm_summarise_trace_failed", trace_id=trace.trace_id, error=str(exc))
            # Graceful fallback
            summary = self._template_summarise_trace(trace)
            return summary, None

    async def _llm_summarise_log(self, entry: LogEntry) -> tuple[TraceSummary, int]:
        import json

        prompt = (
            f"Summarise this log entry for a RAG index.\n\n{json.dumps(entry.model_dump(), indent=2, default=str)}"
        )
        t0 = time.perf_counter()
        try:
            agent = self._get_agent()
            async with self._sem:
                result = await agent.run(prompt)
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            summary = result.data
            summary.used_llm = True
            log.info("llm_summarise_log", service=entry.service, severity=entry.severity, ms=elapsed_ms)
            return summary, elapsed_ms
        except Exception as exc:
            log.error("llm_summarise_log_failed", service=entry.service, error=str(exc))
            summary = self._template_summarise_log(entry)
            return summary, None
