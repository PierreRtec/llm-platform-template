"""Unit tests for app.core.telemetry (T8: OpenInference -> OTel -> Langfuse).

No real network call anywhere in this file: `setup_telemetry` without keys
never builds an exporter at all, and the "with fake keys" tests either (a)
only construct an `OTLPSpanExporter` (which does not connect at construction
time, only when a batch is actually flushed, which never happens here since
no span processor is ever force-flushed against it) or (b) inject an
in-memory `TracerProvider`/`InMemorySpanExporter` that never touches OTLP or
the real Langfuse host at all.

`LangChainInstrumentor` and OTel's global tracer provider are both
process-wide singletons (see `app/core/telemetry.py` module docstring), so
every test that instruments here must uninstrument in teardown
(`_reset_telemetry_state` below) to stay independent of test order and to
avoid leaking an instrumented LangChain callback manager into unrelated
tests (e.g. `tests/unit/test_graph.py`, which invokes real LangChain
`Runnable`s).
"""

from collections.abc import Iterator

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import __version__
from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.core import telemetry as telemetry_module
from app.core.config import Settings
from app.core.telemetry import (
    _langchain_instrumentor,
    build_tracer_provider,
    setup_telemetry,
    shutdown_telemetry,
)

THREAD_CONFIG = {"configurable": {"thread_id": "telemetry-test-thread"}}


@pytest.fixture(autouse=True)
def _reset_telemetry_state() -> Iterator[None]:
    """Uninstrument LangChain after every test in this module.

    Does not (and cannot) reset OTel's global tracer provider back to the
    untouched default: the OTel API only allows setting it once per
    process. That is exactly why the "spans are actually emitted" test
    below injects its own `TracerProvider` via `setup_telemetry`'s
    `tracer_provider` parameter rather than relying on the global one.
    """
    yield
    shutdown_telemetry()
    assert not _langchain_instrumentor.is_instrumented_by_opentelemetry


class TestSetupTelemetryNoOp:
    def test_does_not_raise_when_langfuse_keys_are_empty(self) -> None:
        settings = Settings(APP_API_KEY="test-key", LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="")

        setup_telemetry(settings)  # must not raise

        assert not _langchain_instrumentor.is_instrumented_by_opentelemetry

    def test_no_op_is_logged_clearly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Assert on the log call directly via a stub logger.

        Not `structlog.testing.capture_logs()`: `configure_logging`'s
        `cache_logger_on_first_use=True` means this module's module-level
        `logger` proxy permanently caches whichever processor chain was
        active the *first* time anything logs through it in the whole test
        session (structlog's documented caching behavior), which may well
        predate this test and its `capture_logs()` context entirely once
        this file runs alongside the rest of `tests/unit`. Patching the
        `logger` attribute itself sidesteps that entirely.
        """
        settings = Settings(APP_API_KEY="test-key", LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="")
        logged: list[tuple[str, dict[str, object]]] = []

        class _StubLogger:
            def info(self, event: str, **kwargs: object) -> None:
                logged.append((event, kwargs))

        monkeypatch.setattr(telemetry_module, "logger", _StubLogger())

        setup_telemetry(settings)

        assert logged
        event, kwargs = logged[0]
        assert event == "telemetry_disabled"
        assert "reason" in kwargs

    def test_no_ops_when_only_the_secret_key_is_missing(self) -> None:
        settings = Settings(
            APP_API_KEY="test-key", LANGFUSE_PUBLIC_KEY="pub-only", LANGFUSE_SECRET_KEY=""
        )

        setup_telemetry(settings)

        assert not _langchain_instrumentor.is_instrumented_by_opentelemetry


class TestBuildTracerProvider:
    def test_resource_carries_the_required_attributes(self) -> None:
        settings = Settings(
            APP_API_KEY="test-key",
            APP_ENV="ci",
            LANGFUSE_PUBLIC_KEY="fake-public",
            LANGFUSE_SECRET_KEY="fake-secret",
        )

        provider = build_tracer_provider(settings)

        attributes = provider.resource.attributes
        assert attributes["service.name"] == "llm-platform"
        assert attributes["service.version"] == __version__
        assert attributes["deployment.environment"] == "ci"


class TestSetupTelemetryWithFakeKeysEmitsLangChainSpans:
    """The most valuable test: a real graph invocation, with a fake LLM and
    zero network, actually produces OpenInference spans once telemetry is
    set up. Uses `setup_telemetry`'s `tracer_provider` injection to bind
    `LangChainInstrumentor` straight to an `InMemorySpanExporter`, so this
    never touches the real OTLP exporter or OTel's global tracer provider.
    """

    def test_graph_invocation_produces_at_least_one_span(self) -> None:
        settings = Settings(
            APP_API_KEY="test-key",
            LANGFUSE_PUBLIC_KEY="fake-public",
            LANGFUSE_SECRET_KEY="fake-secret",
        )
        exporter = InMemorySpanExporter()
        provider = build_tracer_provider(settings)
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        setup_telemetry(settings, tracer_provider=provider)
        assert _langchain_instrumentor.is_instrumented_by_opentelemetry

        llm = FakeMessagesListChatModel(responses=[AIMessage(content="Voici une reponse.")])
        graph = build_graph(llm=llm, checkpointer=InMemorySaver())
        state = AgentState(
            messages=[HumanMessage(content="Quelles aides existent ?")],
            user_id="user-1",
            input_flags=[],
        )

        result = graph.invoke(state, config=THREAD_CONFIG)

        assert result["messages"][-1].content == "Voici une reponse."
        finished_spans = exporter.get_finished_spans()
        assert len(finished_spans) > 0, "expected at least one OpenInference span"

    def test_no_op_setup_never_instruments_even_with_a_provider_injected(self) -> None:
        """Empty keys must still short-circuit before the injected provider is used."""
        settings = Settings(APP_API_KEY="test-key", LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="")
        provider = TracerProvider()

        setup_telemetry(settings, tracer_provider=provider)

        assert not _langchain_instrumentor.is_instrumented_by_opentelemetry


class TestShutdownTelemetry:
    def test_shutdown_without_prior_setup_does_not_raise(self) -> None:
        shutdown_telemetry()  # must not raise even when nothing was set up

    def test_shutdown_uninstruments_langchain(self) -> None:
        settings = Settings(
            APP_API_KEY="test-key",
            LANGFUSE_PUBLIC_KEY="fake-public",
            LANGFUSE_SECRET_KEY="fake-secret",
        )
        provider = TracerProvider()
        setup_telemetry(settings, tracer_provider=provider)
        assert _langchain_instrumentor.is_instrumented_by_opentelemetry

        shutdown_telemetry()

        assert not _langchain_instrumentor.is_instrumented_by_opentelemetry
