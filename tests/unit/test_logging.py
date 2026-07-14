"""Unit tests for app.core.logging: renderer selection and trace correlation."""

import structlog

from app.core.config import Settings
from app.core.logging import bind_trace_id, configure_logging


def test_configure_logging_uses_console_renderer_in_local_env() -> None:
    settings = Settings(APP_API_KEY="test-key", APP_ENV="local")

    configure_logging(settings)

    processors = structlog.get_config()["processors"]
    assert isinstance(processors[-1], structlog.dev.ConsoleRenderer)


def test_configure_logging_uses_json_renderer_outside_local_env() -> None:
    settings = Settings(APP_API_KEY="test-key", APP_ENV="prod")

    configure_logging(settings)

    processors = structlog.get_config()["processors"]
    assert isinstance(processors[-1], structlog.processors.JSONRenderer)


def test_bind_trace_id_is_visible_inside_the_context_and_cleared_after() -> None:
    settings = Settings(APP_API_KEY="test-key")
    configure_logging(settings)

    assert "trace_id" not in structlog.contextvars.get_contextvars()

    with bind_trace_id("trace-123"):
        assert structlog.contextvars.get_contextvars()["trace_id"] == "trace-123"

    assert "trace_id" not in structlog.contextvars.get_contextvars()
