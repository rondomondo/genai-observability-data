"""Structured logging configuration using structlog.

JSON in production, coloured console in dev.
Every log record automatically gets request_id and trace_id
from contextvars (populated by correlation middleware).
"""

import logging
import sys
from contextvars import ContextVar

import structlog
from structlog.types import EventDict, Processor

# Context vars — set by correlation middleware, read by log processors
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def _add_request_context(logger: logging.Logger, method: str, event_dict: EventDict) -> EventDict:
    event_dict["request_id"] = request_id_var.get()
    event_dict["trace_id"] = trace_id_var.get()
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_request_context,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quieten noisy libraries
    for noisy in ("uvicorn.access", "httpx", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
