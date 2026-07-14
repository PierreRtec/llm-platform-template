"""Structured logging setup (structlog).

JSON in `ci`/`prod`, a readable console renderer in `local`. Trace
correlation is prepared here via `contextvars` so a future OpenTelemetry
integration (T8) can bind the real `trace_id` into the logging context
without this module ever depending on OTel.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import structlog

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog (and the stdlib logging it wraps) for the process."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer()
        if settings.APP_ENV == "local"
        else structlog.processors.JSONRenderer()
    )

    min_level = logging.getLevelNamesMapping().get(settings.LOG_LEVEL.upper(), logging.INFO)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(min_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@contextmanager
def bind_trace_id(trace_id: str) -> Iterator[None]:
    """Bind `trace_id` into the structlog context for the duration of a block.

    Placeholder correlation mechanism ahead of the OTel integration in T8:
    once OpenInference/OTel is wired, the real span trace id will be bound
    here on every request instead of a caller-supplied one.
    """
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars("trace_id")
