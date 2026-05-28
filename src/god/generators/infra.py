"""
Infrastructure span and log generators.

Covers HTTP/gRPC services, databases, caches, and queues.
All timing uses log-normal distributions for realism.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from god.models import (
    LogEntry,
    LogLevel,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    new_id,
)
from god.services import ServiceDef, ServiceType

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# HTTP status code distributions
# ---------------------------------------------------------------------------

_HTTP_SUCCESS = [200, 200, 200, 200, 200, 201, 204]
_HTTP_CLIENT_ERR = [400, 401, 403, 404, 422, 429]
_HTTP_SERVER_ERR = [500, 502, 503, 504]

_GRPC_SUCCESS = [0]  # OK
_GRPC_ERRORS = [2, 4, 5, 13, 14]  # UNKNOWN, DEADLINE_EXCEEDED, NOT_FOUND, INTERNAL, UNAVAILABLE

_DB_OPERATIONS = [
    ("SELECT", "users", "SELECT * FROM users WHERE id = ?"),
    ("SELECT", "sessions", "SELECT * FROM sessions WHERE token = ?"),
    ("SELECT", "transactions", "SELECT * FROM transactions WHERE account_id = ?"),
    ("INSERT", "audit_log", "INSERT INTO audit_log (user_id, action, ts) VALUES (?, ?, ?)"),
    ("UPDATE", "accounts", "UPDATE accounts SET balance = ? WHERE id = ?"),
    ("BEGIN", None, "BEGIN"),
    ("COMMIT", None, "COMMIT"),
]

_QUEUE_TOPICS = ["payments.events", "notifications.dispatch", "audit.trail", "analytics.raw"]

_CACHE_KEYS = [
    "user:{user_id}:profile",
    "session:{session_id}:data",
    "rate_limit:{ip}:count",
    "feature_flags:global",
    "config:service:payments",
]

_LOG_TEMPLATES: dict[ServiceType, list[tuple[LogLevel, str]]] = {
    ServiceType.HTTP_API: [
        (LogLevel.INFO, "Incoming {method} {path} from {ip}"),
        (LogLevel.INFO, "Request completed in {duration_ms:.1f}ms status={status}"),
        (LogLevel.WARN, "Slow request detected: {duration_ms:.1f}ms > threshold {threshold}ms"),
        (LogLevel.ERROR, "Upstream {upstream} returned {status}: {error}"),
        (LogLevel.ERROR, "Request timed out after {timeout_ms}ms calling {upstream}"),
    ],
    ServiceType.GRPC_SERVICE: [
        (LogLevel.INFO, "gRPC {method} called by {caller}"),
        (LogLevel.INFO, "gRPC {method} completed code={code}"),
        (LogLevel.ERROR, "gRPC {method} failed: {error}"),
    ],
    ServiceType.DATABASE: [
        (LogLevel.DEBUG, "Query: {sql} rows={rows} duration={duration_ms:.2f}ms"),
        (LogLevel.WARN, "Slow query: {sql} duration={duration_ms:.2f}ms"),
        (LogLevel.ERROR, "Connection pool exhausted: active={active} max={max_conns}"),
        (LogLevel.ERROR, "Query failed: {error} sql={sql}"),
        (LogLevel.WARN, "Replica lag: {lag_ms}ms exceeds threshold"),
    ],
    ServiceType.QUEUE: [
        (LogLevel.INFO, "Published message to {topic} partition={partition} offset={offset}"),
        (LogLevel.INFO, "Consumer lag on {topic}: {lag} messages"),
        (LogLevel.WARN, "Producer retry #{attempt} for {topic}: {error}"),
        (LogLevel.ERROR, "Failed to publish to {topic} after {retries} retries: {error}"),
    ],
    ServiceType.CACHE: [
        (LogLevel.DEBUG, "Cache {result} for key {key} ttl={ttl}s"),
        (LogLevel.WARN, "Cache miss rate elevated: {miss_rate:.1%}"),
        (LogLevel.ERROR, "Redis connection error: {error}"),
    ],
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _sample_latency(svc: ServiceDef, rng: np.random.Generator) -> float:
    """Sample a latency in ms from the service's log-normal profile."""
    mu, sigma = svc.effective_latency.mu_sigma()
    ms = float(rng.lognormal(mu, sigma))
    return min(ms, svc.effective_latency.timeout_ms * 1.05)


def _is_timeout(latency_ms: float, svc: ServiceDef) -> bool:
    return latency_ms >= svc.effective_latency.timeout_ms


def _should_error(svc: ServiceDef, rng: np.random.Generator) -> bool:
    return rng.random() < svc.effective_error_rate


def _ns(ts_ms: float) -> int:
    """Convert millisecond float to nanoseconds int."""
    return int(ts_ms * 1_000_000)


def _http_method_path() -> tuple[str, str]:
    choices = [
        ("GET", "/api/v1/users/{id}"),
        ("POST", "/api/v1/auth/token"),
        ("GET", "/api/v1/search"),
        ("POST", "/api/v1/payments"),
        ("GET", "/api/v1/health"),
        ("GET", "/api/v1/rag/query"),
        ("POST", "/api/v1/rag/index"),
    ]
    m, p = random.choice(choices)
    return m, p.replace("{id}", str(random.randint(1000, 9999)))


# ---------------------------------------------------------------------------
# Span builders
# ---------------------------------------------------------------------------


def build_http_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
) -> tuple[Span, float]:
    """Build an HTTP server span. Returns (span, end_time_ms)."""
    span_id = new_id(16)
    duration = _sample_latency(svc, rng)
    end_ms = start_ms + duration
    is_err = _should_error(svc, rng) or _is_timeout(duration, svc)

    method, path = _http_method_path()
    status_code = random.choice(_HTTP_SERVER_ERR) if is_err else random.choice(_HTTP_SUCCESS)

    attrs = {
        "http.request.method": method,
        "url.path": path,
        "http.response.status_code": status_code,
        "http.route": path,
        "network.protocol.version": "1.1",
        "server.address": svc.name,
        "server.port": 8080,
        "http.request.duration_ms": round(duration, 2),
        "client.address": f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}",
    }
    if is_err:
        attrs["error.type"] = "HTTPError"

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"{method} {path}",
        kind=SpanKind.SERVER,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        status_message="Internal Server Error" if is_err else "",
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    if is_err:
        span.events.append(
            SpanEvent(
                name="exception",
                time_unix_nano=_ns(start_ms + duration * 0.9),
                attributes={
                    "exception.type": "RuntimeError",
                    "exception.message": random.choice(
                        [
                            "upstream timeout",
                            "connection refused",
                            "rate limit exceeded",
                            "internal processing error",
                        ]
                    ),
                    "exception.stacktrace": (
                        f'  File "{svc.name}/handler.py", line {random.randint(50, 300)}\n    raise RuntimeError(...)'
                    ),
                },
            )
        )
    return span, end_ms


def build_grpc_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
) -> tuple[Span, float]:
    span_id = new_id(16)
    duration = _sample_latency(svc, rng)
    end_ms = start_ms + duration
    is_err = _should_error(svc, rng)

    grpc_method = random.choice(["Authenticate", "ValidateToken", "RefreshToken", "GetPermissions"])
    code = random.choice(_GRPC_ERRORS) if is_err else 0

    attrs = {
        "rpc.system": "grpc",
        "rpc.service": svc.name.replace("-", ".").title(),
        "rpc.method": grpc_method,
        "rpc.grpc.status_code": code,
        "server.address": svc.name,
        "server.port": 9090,
        "network.transport": "tcp",
    }

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"{svc.name}/{grpc_method}",
        kind=SpanKind.CLIENT,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


def build_db_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
) -> tuple[Span, float]:
    span_id = new_id(16)
    duration = _sample_latency(svc, rng)
    end_ms = start_ms + duration
    is_err = _should_error(svc, rng) or _is_timeout(duration, svc)

    op, table, sql = random.choice(_DB_OPERATIONS)
    rows = int(rng.integers(0, 200)) if op == "SELECT" else 1

    attrs = {
        "db.system.name": "postgresql",
        "db.namespace": svc.name.replace("-", "_"),
        "db.operation.name": op,
        "db.query.text": sql,
        "db.response.returned_rows": rows,
        "server.address": svc.name,
        "server.port": 5432,
        "db.query.duration_ms": round(duration, 3),
    }
    if table:
        attrs["db.collection.name"] = table
    if is_err:
        attrs["error.type"] = "DBError"
        attrs["db.error.message"] = random.choice(
            [
                "connection timeout",
                "deadlock detected",
                "connection pool exhausted",
                "query cancelled",
            ]
        )

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"{op} {table or 'db'}",
        kind=SpanKind.CLIENT,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


def build_queue_span(
    svc: ServiceDef,
    trace_id: str,
    parent_span_id: str | None,
    start_ms: float,
    rng: np.random.Generator,
    is_producer: bool = True,
) -> tuple[Span, float]:
    span_id = new_id(16)
    duration = _sample_latency(svc, rng)
    end_ms = start_ms + duration
    is_err = _should_error(svc, rng)

    topic = random.choice(_QUEUE_TOPICS)
    partition = int(rng.integers(0, 12))
    offset = int(rng.integers(10000, 999999))

    attrs = {
        "messaging.system": "kafka",
        "messaging.destination.name": topic,
        "messaging.destination.partition.id": str(partition),
        "messaging.kafka.offset": offset,
        "messaging.operation.name": "publish" if is_producer else "receive",
        "server.address": svc.name,
        "server.port": 9092,
        "messaging.message.body.size": int(rng.integers(100, 8192)),
    }
    if is_err:
        attrs["error.type"] = "MessagingError"

    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=f"{'publish' if is_producer else 'receive'} {topic}",
        kind=SpanKind.PRODUCER if is_producer else SpanKind.CONSUMER,
        service_name=svc.name,
        start_time_unix_nano=_ns(start_ms),
        end_time_unix_nano=_ns(end_ms),
        status=SpanStatus.ERROR if is_err else SpanStatus.OK,
        attributes=attrs,
        resource_attributes=svc.resource_attributes(),
    )
    return span, end_ms


# ---------------------------------------------------------------------------
# Log generators
# ---------------------------------------------------------------------------


def generate_logs_for_span(
    span: Span,
    svc: ServiceDef,
    rng: np.random.Generator,
) -> list[LogEntry]:
    """Generate 1–3 correlated log entries for a span."""
    logs: list[LogEntry] = []
    templates = _LOG_TEMPLATES.get(svc.type, [])
    if not templates:
        return logs

    is_err = span.status == SpanStatus.ERROR
    duration_ms = span.duration_ms

    candidates = (
        [t for t in templates if t[0] in (LogLevel.ERROR, LogLevel.WARN)]
        if is_err
        else [t for t in templates if t[0] in (LogLevel.INFO, LogLevel.DEBUG)]
    )
    if not candidates:
        candidates = templates

    # Pick 1-2 templates
    chosen = rng.choice(len(candidates), size=min(2, len(candidates)), replace=False)

    for idx in chosen:
        level, template = candidates[int(idx)]

        # Slightly noisy timestamp relative to span
        offset_ns = int(rng.uniform(0, span.end_time_unix_nano - span.start_time_unix_nano))
        ts_ns = span.start_time_unix_nano + offset_ns

        ctx = {
            "method": span.attributes.get("http.request.method", "GET"),
            "path": span.attributes.get("url.path", "/"),
            "ip": span.attributes.get("client.address", "10.0.0.1"),
            "status": span.attributes.get("http.response.status_code", 200),
            "duration_ms": duration_ms,
            "threshold": svc.latency.p99_ms * 0.5,
            "timeout_ms": svc.effective_latency.timeout_ms,
            "upstream": random.choice(svc.dependencies) if svc.dependencies else "upstream",
            "error": span.attributes.get("error.type", "unknown error"),
            "code": span.attributes.get("rpc.grpc.status_code", 0),
            "caller": "api-gateway",
            "sql": span.attributes.get("db.query.text", "SELECT ..."),
            "rows": span.attributes.get("db.response.returned_rows", 0),
            "active": random.randint(8, 20),
            "max_conns": 20,
            "lag_ms": int(rng.uniform(500, 10000)),
            "topic": span.attributes.get("messaging.destination.name", "events"),
            "partition": span.attributes.get("messaging.destination.partition.id", "0"),
            "offset": span.attributes.get("messaging.kafka.offset", 0),
            "lag": int(rng.uniform(100, 10000)),
            "attempt": random.randint(1, 3),
            "retries": 3,
            "result": "HIT" if rng.random() > 0.3 else "MISS",
            "key": random.choice(_CACHE_KEYS)
            .replace("{user_id}", str(random.randint(1, 9999)))
            .replace("{session_id}", new_id(8))
            .replace("{ip}", "10.0.0.1"),
            "ttl": random.choice([60, 300, 3600]),
            "miss_rate": float(rng.uniform(0.3, 0.8)),
        }

        try:
            body = template.format(**ctx)
        except KeyError:
            body = template

        logs.append(
            LogEntry(
                timestamp_unix_nano=ts_ns,
                service_name=svc.name,
                level=level,
                body=body,
                trace_id=span.trace_id,
                span_id=span.span_id,
                attributes={
                    "service.type": svc.type.value,
                },
                resource_attributes=svc.resource_attributes(),
            )
        )

    return logs
