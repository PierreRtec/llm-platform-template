"""OpenInference -> OpenTelemetry -> Langfuse tracing (T8).

Wires the OpenInference `LangChainInstrumentor` to an OTel `TracerProvider`
that exports spans over OTLP/HTTP straight to Langfuse's own ingestion
endpoint (`{LANGFUSE_HOST}/api/public/otel/v1/traces`), authenticated with
HTTP Basic auth (`base64("public_key:secret_key")`) the way Langfuse's
OTel-native ingestion (self-hosted v3) expects from callers that are not
using its own SDK.

No-op by design (logged, never raised) when `LANGFUSE_PUBLIC_KEY` or
`LANGFUSE_SECRET_KEY` is empty: this is the shape both `docker-compose.yml`
(`${LANGFUSE_PUBLIC_KEY:-}`, defaults to empty) and `docker-compose.ci.yml`
(Langfuse never starts) hand the app when observability is not configured,
so the app must boot and serve traffic identically either way, e.g. in CI.

`BatchSpanProcessor` (not the synchronous `SimpleSpanProcessor`): spans are
batched and exported off the request path. `shutdown_telemetry()` flushes
that batch and closes the exporter, and is expected to run from the
lifespan's shutdown phase so buffered spans are not lost on process exit.

Log correlation and per-request attributes (`user_id`, `thread_id`, prompt
version/hash) are deliberately *not* handled here: see
`app/api/routes/chat.py`, which opens the request-level span and binds
`app.core.logging.bind_trace_id` around the graph invocation.
"""

from __future__ import annotations

import base64
from typing import Final

import structlog
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app import __version__
from app.core.config import Settings

logger = structlog.get_logger(__name__)

# Langfuse's OTLP/HTTP traces ingestion path, appended to `LANGFUSE_HOST`.
# Explicitly passing `endpoint=` to `OTLPSpanExporter` (rather than relying on
# the `OTEL_EXPORTER_OTLP_ENDPOINT` env var) skips the SDK's own "append
# v1/traces to the base endpoint" convenience, so this module appends it.
_OTLP_TRACES_PATH: Final[str] = "/api/public/otel/v1/traces"

# `BaseInstrumentor` (opentelemetry-instrumentation) is itself a singleton
# (`__new__` returns the same instance for every `LangChainInstrumentor()`
# call), so this module-level handle and the process-wide instrumented/not
# state it wraps are one and the same either way; naming it here just makes
# that state explicit and gives `shutdown_telemetry` something to call.
_langchain_instrumentor = LangChainInstrumentor()

# The `TracerProvider` this module last wired up via `setup_telemetry`
# (either the real, Langfuse-backed one, or a test's injected one), or
# `None` when nothing is currently configured. Tracked separately from
# OTel's own process-wide global (`trace.get_tracer_provider()`) for two
# reasons: (1) the OTel API silently ignores every `set_tracer_provider`
# call after the first one, so a second real `setup_telemetry` call would
# otherwise build and instrument an orphaned second provider that nothing
# reads from, while any tracer obtained earlier (e.g. `chat.py`'s
# module-level `tracer`) stays bound to the first, possibly already
# shut-down provider; (2) tests inject their own `TracerProvider` without
# ever touching `trace.set_tracer_provider`, so `shutdown_telemetry` cannot
# rely on the global to find the provider it needs to flush and close.
_active_provider: TracerProvider | None = None


def _otlp_traces_endpoint(langfuse_host: str) -> str:
    """Build Langfuse's OTLP/HTTP traces endpoint from its base host."""
    return f"{langfuse_host.rstrip('/')}{_OTLP_TRACES_PATH}"


def _basic_auth_header(public_key: str, secret_key: str) -> str:
    """Return the `Authorization` header value for HTTP Basic `public:secret`."""
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return f"Basic {token}"


def build_tracer_provider(settings: Settings) -> TracerProvider:
    """Build a bare `TracerProvider` carrying the resource attributes T8 requires.

    Deliberately pure: no exporter, no span processor, no global
    registration (`trace.set_tracer_provider`). Split out of
    `setup_telemetry` so tests can assert on the resource attributes
    directly, without touching OTel's process-wide global tracer provider
    (which, per the OTel API, can only be set once per process: see
    `setup_telemetry`'s `tracer_provider` parameter for how tests avoid that
    entirely for anything beyond resource-attribute assertions).
    """
    resource = Resource.create(
        {
            "service.name": "llm-platform",
            "service.version": __version__,
            "deployment.environment": settings.APP_ENV,
        }
    )
    return TracerProvider(resource=resource)


def setup_telemetry(settings: Settings, *, tracer_provider: TracerProvider | None = None) -> None:
    """Wire OpenInference's `LangChainInstrumentor` to Langfuse over OTLP/HTTP.

    No-ops (logs `telemetry_disabled`, never raises) when either
    `settings.LANGFUSE_PUBLIC_KEY` or `settings.LANGFUSE_SECRET_KEY` is
    empty: the app must run identically with or without observability
    configured (see module docstring).

    `tracer_provider` exists purely for tests (the same injection pattern as
    `app/agent/graph.py`'s `build_graph(llm=..., checkpointer=...)`): pass a
    `TracerProvider` wired to an `InMemorySpanExporter` to assert on the
    spans `LangChainInstrumentor` emits for a real graph invocation, without
    ever touching the real OTLP exporter or OTel's global tracer provider
    registration. When omitted (the real, lifespan-driven path), a fresh
    `build_tracer_provider(settings)` is built, given a `BatchSpanProcessor`
    exporting to Langfuse, and registered as the process-wide provider via
    `trace.set_tracer_provider`.

    Idempotent for the real path: if this is called again with
    `tracer_provider=None` while a provider from a previous real call is
    still active (`_active_provider` is not `None`), it logs
    `telemetry_already_configured` and returns without rebuilding anything.
    Without this guard, a second real call would build and instrument an
    orphaned second `TracerProvider` (OTel's global `set_tracer_provider`
    silently ignores every call after the first), while tracers obtained
    earlier stay bound to the first provider: two disconnected providers,
    broken traces. Call `shutdown_telemetry()` first to allow a fresh setup.
    """
    global _active_provider

    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.info(
            "telemetry_disabled",
            reason="LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY are empty",
        )
        return

    if tracer_provider is None and _active_provider is not None:
        logger.warning(
            "telemetry_already_configured",
            reason="setup_telemetry was called again while a provider is still active",
        )
        return

    provider = tracer_provider
    if provider is None:
        provider = build_tracer_provider(settings)
        exporter = OTLPSpanExporter(
            endpoint=_otlp_traces_endpoint(settings.LANGFUSE_HOST),
            headers={
                "Authorization": _basic_auth_header(
                    settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY
                )
            },
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    _langchain_instrumentor.instrument(tracer_provider=provider)
    _active_provider = provider
    logger.info(
        "telemetry_enabled",
        langfuse_host=settings.LANGFUSE_HOST,
        service_version=__version__,
        deployment_environment=settings.APP_ENV,
    )


def shutdown_telemetry() -> None:
    """Uninstrument LangChain and flush/close the module's active tracer provider.

    Meant for the lifespan's shutdown phase, so any spans still sitting in
    the `BatchSpanProcessor`'s buffer are exported before the process exits
    rather than silently dropped. Safe to call even when `setup_telemetry`
    no-op'd: uninstrumenting an instrumentor that was never instrumented is
    itself a no-op, and there is simply no `_active_provider` to shut down.

    Uses `_active_provider` (this module's own bookkeeping) rather than
    `trace.get_tracer_provider()` (OTel's global): the global reflects
    whatever provider won the process-wide `set_tracer_provider` race, which
    is not necessarily the one this module last configured (see
    `_active_provider`'s module-level docstring), and tests that inject a
    `TracerProvider` never call `set_tracer_provider` at all. Always resets
    `_active_provider` to `None` afterwards so a following `setup_telemetry`
    call can reconfigure cleanly.
    """
    global _active_provider

    if _langchain_instrumentor.is_instrumented_by_opentelemetry:
        _langchain_instrumentor.uninstrument()

    if _active_provider is not None:
        _active_provider.shutdown()
        _active_provider = None
