"""Application factory and lifespan for the FastAPI service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import health
from app.core.config import get_settings
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot and shutdown hooks for the app.

    Today: load and validate settings, then configure structured logging.
    Left as clean, intentional extension points for later tasks (no dead
    code, no premature stubs):

    - T4: open the Postgres connection pool, run
      `AsyncPostgresSaver`/`AsyncPostgresStore` `.setup()`, stash both on
      `app.state`, and close the pool on shutdown.
    - T8: start the OpenInference `LangChainInstrumentor` and the OTel
      tracer provider exporting to Langfuse.
    """
    settings = get_settings()
    configure_logging(settings)
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application. Does not read settings by itself.

    Settings are only resolved when a request or the lifespan actually needs
    them (via `Depends(get_settings)` / `get_settings()`), so constructing
    the app is safe even before the environment is fully configured, e.g.
    when building the app in a test with `app.dependency_overrides`.
    """
    app = FastAPI(title="llm-platform-template", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    return app


app = create_app()
