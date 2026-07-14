"""Application factory and lifespan for the FastAPI service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

from app import __version__
from app.agent.graph import build_graph
from app.api.routes import chat, health
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.telemetry import setup_telemetry, shutdown_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot and shutdown hooks for the app.

    Today: load and validate settings, configure structured logging, wire
    OpenInference/OTel/Langfuse telemetry (T8, no-op without Langfuse
    keys), then compile the agent graph once and stash it on
    `app.state.agent_graph` (read back per-request via
    `app.api.deps.get_agent_graph`). Telemetry is set up *before* the graph
    is compiled so `LangChainInstrumentor`'s callback-manager patch (see
    `app/core/telemetry.py`) is already in place for every LangChain
    `Runnable` the graph builds, not just ones invoked after the first
    request.

    `build_graph` and the `yield` are wrapped in the same `try`/`finally` as
    `shutdown_telemetry`: if compiling the graph raises (bad config, a
    dependency not reachable at boot), `setup_telemetry` has already run, so
    `shutdown_telemetry` still needs to flush/close it. Without this, a
    boot failure here would leak whatever `setup_telemetry` configured
    (buffered spans never flushed, `LangChainInstrumentor` left instrumented
    for the next `lifespan` attempt in the same process).

    The checkpointer backing that graph is an `InMemorySaver`: MVP scope
    (docs/DESIGN.md section 7) accepts this explicitly. It is process-local
    memory only, lost on restart and never shared across replicas.
    `AsyncPostgresSaver` (T4) replaces it here without changing
    `app/api/routes/chat.py`'s contract.

    Left as a clean, intentional extension point for a later task (no dead
    code, no premature stub):

    - T4: open the Postgres connection pool, run
      `AsyncPostgresSaver`/`AsyncPostgresStore` `.setup()`, stash both on
      `app.state`, and close the pool on shutdown.
    """
    settings = get_settings()
    configure_logging(settings)
    setup_telemetry(settings)
    try:
        checkpointer = InMemorySaver()
        app.state.agent_graph = build_graph(checkpointer=checkpointer)
        yield
    finally:
        shutdown_telemetry()


def create_app() -> FastAPI:
    """Build the FastAPI application. Does not read settings by itself.

    Settings are only resolved when a request or the lifespan actually needs
    them (via `Depends(get_settings)` / `get_settings()`), so constructing
    the app is safe even before the environment is fully configured, e.g.
    when building the app in a test with `app.dependency_overrides`.
    """
    app = FastAPI(title="llm-platform-template", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
