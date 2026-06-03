"""Correlation ID middleware.

Reads X-Request-ID from incoming headers (or generates one).
Reads X-Trace-ID if provided by caller (e.g. from their OTEL context).
Populates contextvars so all log lines within the request carry both IDs.
Echoes X-Request-ID in the response.
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import get_logger, request_id_var, trace_id_var

log = get_logger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        trace_id = request.headers.get("x-trace-id", "-")

        token_rid = request_id_var.set(request_id)
        token_tid = trace_id_var.set(trace_id)

        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log.info(
                "request",
                method=request.method,
                http_path=request.url.path,
                status=response.status_code if response is not None else 500,
                duration_ms=duration_ms,
            )
            request_id_var.reset(token_rid)
            trace_id_var.reset(token_tid)

        response.headers["x-request-id"] = request_id
        return response
