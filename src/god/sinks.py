"""
Output sinks for generated traces and logs.

Supported formats and destinations:
  - stdout          logfmt (human-readable)
  - file            OTLP JSON (traces + logs), logfmt, Loki push format
  - http            OTLP HTTP exporter (sends to any OTLP-compatible backend)

All sinks implement the Sink protocol: flush(traces, logs).
"""
from __future__ import annotations

import gzip
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence
from urllib import request as urllib_request
from urllib.error import URLError


from god.models import LogEntry, Trace


# ---------------------------------------------------------------------------
# Sink protocol
# ---------------------------------------------------------------------------

class Sink(ABC):
    @abstractmethod
    def flush(self, traces: list[Trace], logs: list[LogEntry]) -> None:
        ...

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Stdout sink (logfmt, human-readable)
# ---------------------------------------------------------------------------

class StdoutSink(Sink):
    """
    Writes log entries as logfmt lines and a brief trace summary to stdout.
    Good for development and piping to tools like lnav or grep.
    """

    def __init__(self, include_traces: bool = False):
        self.include_traces = include_traces

    def flush(self, traces: list[Trace], logs: list[LogEntry]) -> None:
        for log in sorted(logs, key=lambda l: l.timestamp_unix_nano):
            print(log.to_logfmt(), flush=True)

        if self.include_traces:
            for trace in traces:
                root = trace.root_span()
                name = root.name if root else "?"
                svc = root.service_name if root else "?"
                spans = len(trace.spans)
                errors = sum(1 for s in trace.spans if s.status.value == "ERROR")
                duration = (
                    (max(s.end_time_unix_nano for s in trace.spans) -
                     min(s.start_time_unix_nano for s in trace.spans)) / 1_000_000
                    if trace.spans else 0
                )
                status = "ERROR" if errors else "OK"
                print(
                    f"trace trace_id={trace.trace_id} root={name} service={svc} "
                    f"spans={spans} errors={errors} duration_ms={duration:.1f} status={status}",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# File sink
# ---------------------------------------------------------------------------

class FileSink(Sink):
    """
    Writes to one or more files. Supports:
      otlp_traces  → OTLP JSON (resourceSpans format)
      otlp_logs    → OTLP JSON (resourceLogs format)
      logfmt       → one log line per entry
      loki         → Loki push API JSON
    All files can be optionally gzip-compressed.
    """

    def __init__(
        self,
        output_dir: str | Path = ".",
        formats: list[str] | None = None,
        compress: bool = False,
        append: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats or ["otlp_traces", "otlp_logs", "logfmt"]
        self.compress = compress
        self.append = append

    def _open(self, name: str):
        path = self.output_dir / name
        if self.compress:
            return gzip.open(str(path) + ".gz", "at", encoding="utf-8")
        mode = "a" if self.append else "w"
        return open(path, mode, encoding="utf-8")

    def flush(self, traces: list[Trace], logs: list[LogEntry]) -> None:
        if "otlp_traces" in self.formats and traces:
            with self._open("traces.otlp.jsonl") as f:
                for trace in traces:
                    f.write(json.dumps(trace.to_otlp_dict()) + "\n")

        if "otlp_logs" in self.formats and logs:
            with self._open("logs.otlp.jsonl") as f:
                otlp_batch = _logs_to_otlp(logs)
                f.write(json.dumps(otlp_batch) + "\n")

        if "logfmt" in self.formats and logs:
            with self._open("logs.logfmt") as f:
                for log in sorted(logs, key=lambda l: l.timestamp_unix_nano):
                    f.write(log.to_logfmt() + "\n")

        if "loki" in self.formats and logs:
            with self._open("logs.loki.jsonl") as f:
                loki_batch = _logs_to_loki(logs)
                f.write(json.dumps(loki_batch) + "\n")


# ---------------------------------------------------------------------------
# OTLP HTTP sink
# ---------------------------------------------------------------------------

class OTLPHttpSink(Sink):
    """
    Sends traces and logs to an OTLP-compatible HTTP endpoint.
    Compatible with: OTel Collector, Grafana Tempo, Honeycomb, Jaeger.

    endpoint example: "http://localhost:4318"
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ):
        self.traces_url = f"{endpoint.rstrip('/')}/v1/traces"
        self.logs_url = f"{endpoint.rstrip('/')}/v1/logs"
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout_s = timeout_s

    def _post(self, url: str, payload: dict) -> bool:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(url, data=data, headers=self.headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status in (200, 204)
        except URLError as e:
            print(f"[OTLPHttpSink] POST {url} failed: {e}", file=sys.stderr)
            return False

    def flush(self, traces: list[Trace], logs: list[LogEntry]) -> None:
        if traces:
            all_resource_spans: list[dict] = []
            for trace in traces:
                all_resource_spans.extend(trace.to_otlp_dict().get("resourceSpans", []))
            self._post(self.traces_url, {"resourceSpans": all_resource_spans})

        if logs:
            self._post(self.logs_url, _logs_to_otlp(logs))


# ---------------------------------------------------------------------------
# Multi-sink (fan-out)
# ---------------------------------------------------------------------------

class MultiSink(Sink):
    def __init__(self, sinks: list[Sink]):
        self.sinks = sinks

    def flush(self, traces: list[Trace], logs: list[LogEntry]) -> None:
        for sink in self.sinks:
            sink.flush(traces, logs)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


# ---------------------------------------------------------------------------
# OTLP format helpers
# ---------------------------------------------------------------------------

def _logs_to_otlp(logs: list[LogEntry]) -> dict:
    """
    Convert log entries to OTLP resourceLogs format.
    Groups by service for efficient wire format.
    """
    by_service: dict[str, list[LogEntry]] = {}
    for log in logs:
        by_service.setdefault(log.service_name, []).append(log)

    resource_logs = []
    for svc_name, svc_logs in by_service.items():
        resource_logs.append({
            "resource": {
                "attributes": svc_logs[0].resource_attributes
                    and [{"key": k, "value": {"stringValue": str(v)}}
                         for k, v in svc_logs[0].resource_attributes.items()]
                    or [],
            },
            "scopeLogs": [{
                "scope": {"name": f"synthetic.{svc_name}", "version": "1.0.0"},
                "logRecords": [l.to_otlp_dict() for l in svc_logs],
            }],
        })
    return {"resourceLogs": resource_logs}


def _logs_to_loki(logs: list[LogEntry]) -> dict:
    """
    Loki push API format.
    Groups by service stream label.
    """
    streams: dict[str, dict] = {}
    for log in sorted(logs, key=lambda l: l.timestamp_unix_nano):
        key = log.service_name
        if key not in streams:
            streams[key] = log.to_loki_dict()
        else:
            streams[key]["values"].extend(log.to_loki_dict()["values"])

    return {"streams": list(streams.values())}
