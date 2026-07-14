"""Unit tests for the app factory and lifespan."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import create_app


def test_create_app_does_not_require_env() -> None:
    """Building the app must not resolve settings eagerly."""
    app = create_app()

    assert isinstance(app, FastAPI)
    assert app.title == "llm-platform-template"


def test_lifespan_boots_with_a_valid_environment(app_env: None) -> None:
    """Entering the lifespan (startup) must succeed once APP_API_KEY is set."""
    app = create_app()

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200


def test_shutdown_telemetry_runs_even_when_build_graph_fails(
    app_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FINDING 2 (post-review): a `build_graph` failure at boot must not skip
    `shutdown_telemetry`. Before the fix, `setup_telemetry` ran, then
    `build_graph()` was called outside the `try`/`finally` guarding
    `shutdown_telemetry`, so a boot failure there leaked whatever telemetry
    had just been configured.

    The tracking wrapper below calls through to the *real*
    `shutdown_telemetry` (rather than replacing it with a bare no-op): this
    process's `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are not guaranteed
    to be empty here (a pytest plugin loaded elsewhere in this suite may
    have populated `os.environ` from the repo's `.env`), so `setup_telemetry`
    may take the real, instrumenting path here. Discarding the real
    teardown would leak that instrumentation into every test that runs
    after this one.
    """
    shutdown_calls: list[None] = []
    original_shutdown = main_module.shutdown_telemetry

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("boot failure")

    def _tracking_shutdown() -> None:
        shutdown_calls.append(None)
        original_shutdown()

    monkeypatch.setattr(main_module, "build_graph", _boom)
    monkeypatch.setattr(main_module, "shutdown_telemetry", _tracking_shutdown)

    app = create_app()

    with pytest.raises(RuntimeError, match="boot failure"), TestClient(app):
        pass

    assert shutdown_calls == [None]
