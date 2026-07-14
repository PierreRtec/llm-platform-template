"""Shared pytest fixtures for the test suite."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Ensure `get_settings()` re-reads the environment on every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A minimal, valid environment: only the mandatory `APP_API_KEY`."""
    monkeypatch.setenv("APP_API_KEY", "test-api-key")


@pytest.fixture
def app(app_env: None) -> FastAPI:
    """A freshly built app with a valid environment."""
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """A TestClient with the lifespan executed (startup/shutdown events)."""
    with TestClient(app) as test_client:
        yield test_client
