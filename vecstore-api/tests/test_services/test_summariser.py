"""Tests for the Summariser service.

Template paths are pure Python -- no mocking needed.
LLM paths are tested by patching the pydantic-ai Agent so no API key is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.models.ingest import LogEntry, Span, TraceEntry
from app.models.summarise import TraceSummary
from app.services.summariser import Summariser


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        gemini_api_key="fake",
        summariser_llm_threshold_duration_ms=1000,
        otel_enabled=False,
    )


@pytest.fixture()
def summariser(settings: Settings) -> Summariser:
    return Summariser(settings)


def _ok_trace(duration_ms: int = 200, status: str = "ok") -> TraceEntry:
    return TraceEntry(
        trace_id="tid-001",
        root_operation="GET /api/data",
        root_service="api-gateway",
        duration_ms=duration_ms,
        status=status,  # type: ignore[arg-type]
        spans=[
            Span(
                name="db.select",
                span_id="sp-001",
                service="postgres",
                start_time_ms=0,
                duration_ms=50,
                status="ok",
            )
        ],
    )


def _error_trace() -> TraceEntry:
    return TraceEntry(
        trace_id="tid-err",
        root_operation="POST /payment",
        root_service="payment-svc",
        duration_ms=450,
        status="error",
        spans=[
            Span(
                name="stripe.charge",
                span_id="sp-pay",
                service="stripe",
                start_time_ms=0,
                duration_ms=300,
                status="error",
            )
        ],
    )


def _log_entry(severity: str = "INFO", message: str = "App started") -> LogEntry:
    return LogEntry(message=message, service="auth-svc", severity=severity)  # type: ignore[arg-type]


def _fake_summary() -> TraceSummary:
    return TraceSummary(
        narrative="LLM-generated narrative",
        root_cause="Connection refused",
        affected_services=["payment-svc"],
        severity="high",
        used_llm=False,  # will be set to True by the caller
    )


class TestShouldUseLlm:
    def test_error_status_triggers_llm(self, summariser: Summariser) -> None:
        assert summariser._should_use_llm(_error_trace()) is True

    def test_error_span_triggers_llm(self, summariser: Summariser) -> None:
        trace = _ok_trace()
        trace.spans[0].status = "error"
        assert summariser._should_use_llm(trace) is True

    def test_slow_trace_triggers_llm(self, summariser: Summariser) -> None:
        assert summariser._should_use_llm(_ok_trace(duration_ms=1500)) is True

    def test_fast_ok_trace_does_not_trigger_llm(self, summariser: Summariser) -> None:
        assert summariser._should_use_llm(_ok_trace(duration_ms=200)) is False

    def test_exactly_at_threshold_does_not_trigger_llm(self, summariser: Summariser) -> None:
        assert summariser._should_use_llm(_ok_trace(duration_ms=1000)) is False


class TestTraceToText:
    def test_includes_service_and_operation(self, summariser: Summariser) -> None:
        text = summariser.trace_to_text(_ok_trace())
        assert "api-gateway" in text
        assert "GET /api/data" in text

    def test_includes_duration_and_status(self, summariser: Summariser) -> None:
        text = summariser.trace_to_text(_ok_trace(duration_ms=350))
        assert "350ms" in text
        assert "status=ok" in text

    def test_error_spans_mentioned(self, summariser: Summariser) -> None:
        trace = _ok_trace()
        trace.spans[0].status = "error"
        text = summariser.trace_to_text(trace)
        assert "errors" in text
        assert "db.select" in text

    def test_slow_spans_mentioned(self, summariser: Summariser) -> None:
        trace = _ok_trace()
        trace.spans[0].duration_ms = 2000
        text = summariser.trace_to_text(trace)
        assert "slow_spans" in text

    def test_clean_trace_has_no_error_section(self, summariser: Summariser) -> None:
        text = summariser.trace_to_text(_ok_trace())
        assert "errors" not in text
        assert "slow_spans" not in text


class TestLogToText:
    def test_format(self, summariser: Summariser) -> None:
        entry = _log_entry(severity="WARN", message="Cache miss rate high")
        text = summariser.log_to_text(entry)
        assert text == "auth-svc [WARN] Cache miss rate high"


class TestTemplateSummariseTrace:
    def test_ok_trace_severity_low(self, summariser: Summariser) -> None:
        result = summariser._template_summarise_trace(_ok_trace())
        assert result.severity == "low"
        assert result.used_llm is False
        assert "api-gateway" in result.narrative

    def test_error_trace_severity_high(self, summariser: Summariser) -> None:
        result = summariser._template_summarise_trace(_error_trace())
        assert result.severity == "high"
        assert "POST /payment" in result.narrative

    def test_slow_trace_severity_medium(self, summariser: Summariser) -> None:
        result = summariser._template_summarise_trace(_ok_trace(duration_ms=1500))
        assert result.severity == "medium"

    def test_affected_services_includes_root(self, summariser: Summariser) -> None:
        result = summariser._template_summarise_trace(_ok_trace())
        assert "api-gateway" in result.affected_services

    def test_error_span_mentioned_in_narrative(self, summariser: Summariser) -> None:
        trace = _error_trace()
        result = summariser._template_summarise_trace(trace)
        assert "stripe.charge" in result.narrative

    def test_root_cause_is_none(self, summariser: Summariser) -> None:
        result = summariser._template_summarise_trace(_ok_trace())
        assert result.root_cause is None


class TestTemplateSummariseLog:
    @pytest.mark.parametrize(
        "severity,expected",
        [
            ("DEBUG", "low"),
            ("INFO", "low"),
            ("WARN", "medium"),
            ("ERROR", "high"),
            ("FATAL", "critical"),
        ],
    )
    def test_severity_mapping(self, summariser: Summariser, severity: str, expected: str) -> None:
        result = summariser._template_summarise_log(_log_entry(severity=severity))
        assert result.severity == expected

    def test_narrative_contains_service_and_message(self, summariser: Summariser) -> None:
        result = summariser._template_summarise_log(_log_entry(severity="WARN", message="High memory"))
        assert "auth-svc" in result.narrative
        assert "High memory" in result.narrative
        assert "WARN" in result.narrative

    def test_used_llm_is_false(self, summariser: Summariser) -> None:
        result = summariser._template_summarise_log(_log_entry())
        assert result.used_llm is False


class TestSummariseTraceRouting:
    async def test_ok_fast_trace_uses_template(self, summariser: Summariser) -> None:
        result, llm_ms = await summariser.summarise_trace(_ok_trace())
        assert result.used_llm is False
        assert llm_ms is None

    async def test_force_llm_calls_agent(self, summariser: Summariser) -> None:
        fake = _fake_summary()
        mock_result = MagicMock()
        mock_result.data = fake

        with patch.object(summariser, "_get_agent") as mock_get:
            agent = MagicMock()
            agent.run = AsyncMock(return_value=mock_result)
            mock_get.return_value = agent

            result, llm_ms = await summariser.summarise_trace(_ok_trace(), force_llm=True)

        assert result.used_llm is True
        assert llm_ms is not None
        agent.run.assert_awaited_once()

    async def test_error_trace_calls_agent(self, summariser: Summariser) -> None:
        fake = _fake_summary()
        mock_result = MagicMock()
        mock_result.data = fake

        with patch.object(summariser, "_get_agent") as mock_get:
            agent = MagicMock()
            agent.run = AsyncMock(return_value=mock_result)
            mock_get.return_value = agent

            result, _ = await summariser.summarise_trace(_error_trace())

        assert result.used_llm is True

    async def test_llm_failure_falls_back_to_template(self, summariser: Summariser) -> None:
        with patch.object(summariser, "_get_agent") as mock_get:
            agent = MagicMock()
            agent.run = AsyncMock(side_effect=RuntimeError("API down"))
            mock_get.return_value = agent

            result, llm_ms = await summariser.summarise_trace(_error_trace())

        assert result.used_llm is False
        assert llm_ms is None
        assert result.severity in ("low", "medium", "high", "critical")


class TestSummariseLogRouting:
    async def test_info_log_uses_template(self, summariser: Summariser) -> None:
        result, llm_ms = await summariser.summarise_log(_log_entry(severity="INFO"))
        assert result.used_llm is False
        assert llm_ms is None

    async def test_warn_log_uses_template(self, summariser: Summariser) -> None:
        result, llm_ms = await summariser.summarise_log(_log_entry(severity="WARN"))
        assert result.used_llm is False

    async def test_error_log_calls_agent(self, summariser: Summariser) -> None:
        fake = _fake_summary()
        mock_result = MagicMock()
        mock_result.data = fake

        with patch.object(summariser, "_get_agent") as mock_get:
            agent = MagicMock()
            agent.run = AsyncMock(return_value=mock_result)
            mock_get.return_value = agent

            result, llm_ms = await summariser.summarise_log(_log_entry(severity="ERROR"))

        assert result.used_llm is True
        assert llm_ms is not None

    async def test_fatal_log_calls_agent(self, summariser: Summariser) -> None:
        fake = _fake_summary()
        mock_result = MagicMock()
        mock_result.data = fake

        with patch.object(summariser, "_get_agent") as mock_get:
            agent = MagicMock()
            agent.run = AsyncMock(return_value=mock_result)
            mock_get.return_value = agent

            result, _ = await summariser.summarise_log(_log_entry(severity="FATAL"))

        assert result.used_llm is True

    async def test_error_log_llm_failure_falls_back(self, summariser: Summariser) -> None:
        with patch.object(summariser, "_get_agent") as mock_get:
            agent = MagicMock()
            agent.run = AsyncMock(side_effect=Exception("timeout"))
            mock_get.return_value = agent

            result, llm_ms = await summariser.summarise_log(_log_entry(severity="ERROR"))

        assert result.used_llm is False
        assert llm_ms is None
