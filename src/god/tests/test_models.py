"""Tests for core data models — serialisation, enums, helpers."""
from __future__ import annotations

import json

import pytest

from god.models import (
    FinishReason,
    GenAIOperation,
    LogEntry,
    LogLevel,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    Trace,
    _attrs_to_otlp,
    _severity_number,
    new_id,
)


def test_new_id_length_default() -> None:
    assert len(new_id()) == 16


def test_new_id_length_custom() -> None:
    assert len(new_id(32)) == 32


def test_new_id_hex() -> None:
    assert all(c in "0123456789abcdef" for c in new_id())


def test_span_duration_ms() -> None:
    span = Span(
        trace_id="abc",
        span_id="def",
        parent_span_id=None,
        name="test",
        kind=SpanKind.SERVER,
        service_name="svc",
        start_time_unix_nano=0,
        end_time_unix_nano=1_000_000,  # 1 ms
    )
    assert span.duration_ms == pytest.approx(1.0)


def test_span_to_otlp_dict_structure() -> None:
    span = Span(
        trace_id="abc123",
        span_id="def456",
        parent_span_id="parent789",
        name="GET /api",
        kind=SpanKind.SERVER,
        service_name="api-gw",
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=2_000_000_000,
        attributes={"http.status_code": 200, "flag": True, "score": 0.5, "tags": ["a", "b"]},
    )
    d = span.to_otlp_dict()
    assert d["traceId"] == "abc123"
    assert d["spanId"] == "def456"
    assert d["parentSpanId"] == "parent789"
    assert d["startTimeUnixNano"] == "1000000000"
    assert d["endTimeUnixNano"] == "2000000000"
    assert isinstance(d["attributes"], list)
    assert isinstance(d["events"], list)


def test_span_no_parent_is_empty_string() -> None:
    span = Span(
        trace_id="t", span_id="s", parent_span_id=None,
        name="root", kind=SpanKind.SERVER, service_name="svc",
        start_time_unix_nano=0, end_time_unix_nano=1,
    )
    assert "parentSpanId" not in span.to_otlp_dict()


def test_trace_root_span() -> None:
    root = Span(
        trace_id="t", span_id="root", parent_span_id=None,
        name="root", kind=SpanKind.SERVER, service_name="svc",
        start_time_unix_nano=0, end_time_unix_nano=10,
    )
    child = Span(
        trace_id="t", span_id="child", parent_span_id="root",
        name="child", kind=SpanKind.CLIENT, service_name="svc2",
        start_time_unix_nano=1, end_time_unix_nano=9,
    )
    trace = Trace(trace_id="t", spans=[root, child])
    assert trace.root_span() is root


def test_trace_to_otlp_dict_groups_by_service() -> None:
    def _make(sid: str, svc: str, pid: str | None = None) -> Span:
        return Span(
            trace_id="t", span_id=sid, parent_span_id=pid,
            name="x", kind=SpanKind.INTERNAL, service_name=svc,
            start_time_unix_nano=0, end_time_unix_nano=1,
        )

    trace = Trace(trace_id="t", spans=[
        _make("s1", "svc-a"),
        _make("s2", "svc-a", "s1"),
        _make("s3", "svc-b", "s1"),
    ])
    d = trace.to_otlp_dict()
    services = {rs["scopeSpans"][0]["scope"]["name"] for rs in d["resourceSpans"]}
    assert "synthetic.svc-a" in services
    assert "synthetic.svc-b" in services


def test_log_entry_to_logfmt() -> None:
    log = LogEntry(
        timestamp_unix_nano=1_000_000_000,
        service_name="my-svc",
        level=LogLevel.INFO,
        body="hello world",
        trace_id="abc",
        span_id="def",
    )
    fmt = log.to_logfmt()
    assert "level=INFO" in fmt
    assert "service=my-svc" in fmt
    assert 'msg="hello world"' in fmt
    assert "trace_id=abc" in fmt


def test_log_entry_to_loki_dict() -> None:
    log = LogEntry(
        timestamp_unix_nano=9_000_000_000,
        service_name="svc",
        level=LogLevel.ERROR,
        body="boom",
        resource_attributes={"environment": "prod", "region": "eu-west-1"},
    )
    d = log.to_loki_dict()
    assert d["stream"]["service"] == "svc"
    assert d["stream"]["level"] == "ERROR"
    assert d["stream"]["environment"] == "prod"
    assert len(d["values"]) == 1


def test_log_entry_to_otlp_dict() -> None:
    log = LogEntry(
        timestamp_unix_nano=5_000,
        service_name="svc",
        level=LogLevel.WARN,
        body="watch out",
    )
    d = log.to_otlp_dict()
    assert d["severityText"] == "WARN"
    assert d["severityNumber"] == 13
    assert d["body"]["stringValue"] == "watch out"


def test_attrs_to_otlp_types() -> None:
    attrs = _attrs_to_otlp({
        "bool_val": True,
        "int_val": 42,
        "float_val": 3.14,
        "str_val": "hello",
        "list_val": [1, 2],
    })
    by_key = {a["key"]: a["value"] for a in attrs}
    assert by_key["bool_val"] == {"boolValue": True}
    assert by_key["int_val"] == {"intValue": "42"}
    assert by_key["float_val"] == {"doubleValue": 3.14}
    assert by_key["str_val"] == {"stringValue": "hello"}
    assert "arrayValue" in by_key["list_val"]


def test_severity_numbers() -> None:
    assert _severity_number(LogLevel.DEBUG) == 5
    assert _severity_number(LogLevel.INFO) == 9
    assert _severity_number(LogLevel.WARN) == 13
    assert _severity_number(LogLevel.ERROR) == 17
    assert _severity_number(LogLevel.FATAL) == 21


def test_span_event_to_otlp() -> None:
    ev = SpanEvent(name="exception", time_unix_nano=1234, attributes={"msg": "oops"})
    d = ev.to_otlp_dict()
    assert d["name"] == "exception"
    assert d["timeUnixNano"] == "1234"
    assert isinstance(d["attributes"], list)
