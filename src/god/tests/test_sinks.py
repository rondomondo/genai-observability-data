"""Tests for output sinks — file, stdout, OTLP, multi."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from god.models import LogEntry, LogLevel, Trace
from god.orchestrator import generate_trace
from god.services import ServiceRegistry
from god.sinks import FileSink, MultiSink, OTLPHttpSink, StdoutSink, _logs_to_loki, _logs_to_otlp


def _sample_trace_and_logs() -> tuple[list[Trace], list[LogEntry]]:
    reg = ServiceRegistry()
    rng = np.random.default_rng(0)
    trace, logs = generate_trace(reg, start_ms=0.0, rng=rng)
    return [trace], logs


def test_stdout_sink_writes_logfmt(capsys) -> None:
    traces, logs = _sample_trace_and_logs()
    sink = StdoutSink(include_traces=False)
    sink.flush(traces, logs)
    out = capsys.readouterr().out
    assert "level=" in out
    assert "service=" in out


def test_stdout_sink_include_traces(capsys) -> None:
    traces, logs = _sample_trace_and_logs()
    sink = StdoutSink(include_traces=True)
    sink.flush(traces, logs)
    out = capsys.readouterr().out
    assert "trace_id=" in out


def test_file_sink_writes_logfmt() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        traces, logs = _sample_trace_and_logs()
        sink = FileSink(output_dir=tmpdir, formats=["logfmt"])
        sink.flush(traces, logs)
        lines = Path(tmpdir, "logs.logfmt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(logs)
        assert all("level=" in line for line in lines)


def test_file_sink_writes_otlp_traces() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        traces, logs = _sample_trace_and_logs()
        sink = FileSink(output_dir=tmpdir, formats=["otlp_traces"])
        sink.flush(traces, logs)
        content = Path(tmpdir, "traces.otlp.jsonl").read_text(encoding="utf-8")
        obj = json.loads(content.strip().splitlines()[0])
        assert "resourceSpans" in obj


def test_file_sink_writes_otlp_logs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        traces, logs = _sample_trace_and_logs()
        sink = FileSink(output_dir=tmpdir, formats=["otlp_logs"])
        sink.flush(traces, logs)
        content = Path(tmpdir, "logs.otlp.jsonl").read_text(encoding="utf-8")
        obj = json.loads(content.strip().splitlines()[0])
        assert "resourceLogs" in obj


def test_file_sink_writes_loki() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        traces, logs = _sample_trace_and_logs()
        sink = FileSink(output_dir=tmpdir, formats=["loki"])
        sink.flush(traces, logs)
        content = Path(tmpdir, "logs.loki.jsonl").read_text(encoding="utf-8")
        obj = json.loads(content.strip().splitlines()[0])
        assert "streams" in obj


def test_file_sink_gzip_compression() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        traces, logs = _sample_trace_and_logs()
        sink = FileSink(output_dir=tmpdir, formats=["logfmt"], compress=True)
        sink.flush(traces, logs)
        gz_path = Path(tmpdir, "logs.logfmt.gz")
        assert gz_path.exists()
        with gzip.open(str(gz_path), "rt", encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert len(lines) == len(logs)


def test_file_sink_appends_across_calls() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        traces, logs = _sample_trace_and_logs()
        sink = FileSink(output_dir=tmpdir, formats=["logfmt"], append=True)
        sink.flush(traces, logs)
        sink.flush(traces, logs)
        lines = Path(tmpdir, "logs.logfmt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(logs) * 2


def test_multi_sink_fans_out() -> None:
    called: list[tuple[str] | tuple[str, int, int]] = []

    class MockSink:
        def flush(self, traces, logs) -> None:
            called.append(("flush", len(traces), len(logs)))

        def close(self) -> None:
            called.append(("close",))

    traces, logs = _sample_trace_and_logs()
    sink = MultiSink([MockSink(), MockSink()])  # type: ignore[list-item]
    sink.flush(traces, logs)
    sink.close()

    assert called.count(("flush", len(traces), len(logs))) == 2
    assert called.count(("close",)) == 2


def test_logs_to_otlp_groups_by_service() -> None:
    logs = [
        LogEntry(0, "svc-a", LogLevel.INFO, "msg a"),
        LogEntry(1, "svc-b", LogLevel.INFO, "msg b"),
        LogEntry(2, "svc-a", LogLevel.INFO, "msg a2"),
    ]
    result = _logs_to_otlp(logs)
    assert len(result["resourceLogs"]) == 2
    total_records = sum(len(rl["scopeLogs"][0]["logRecords"]) for rl in result["resourceLogs"])
    assert total_records == 3


def test_logs_to_loki_groups_by_service() -> None:
    logs = [
        LogEntry(100, "svc-a", LogLevel.INFO, "a"),
        LogEntry(200, "svc-a", LogLevel.ERROR, "a2"),
        LogEntry(300, "svc-b", LogLevel.DEBUG, "b"),
    ]
    result = _logs_to_loki(logs)
    assert "streams" in result
    service_keys = {s["stream"]["service"] for s in result["streams"]}
    assert "svc-a" in service_keys
    assert "svc-b" in service_keys


def test_otlp_http_sink_posts_to_endpoint() -> None:
    traces, logs = _sample_trace_and_logs()
    sink = OTLPHttpSink(endpoint="http://localhost:4318")

    with patch("god.sinks.urllib_request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_resp
        sink.flush(traces, logs)

    assert mock_open.call_count >= 1


def test_otlp_http_sink_handles_connection_error(capsys) -> None:
    from urllib.error import URLError

    traces, logs = _sample_trace_and_logs()
    sink = OTLPHttpSink(endpoint="http://no-such-host:4318")

    with patch("god.sinks.urllib_request.urlopen", side_effect=URLError("refused")):
        sink.flush(traces, logs)  # should not raise

    err = capsys.readouterr().err
    assert "failed" in err
