"""Unit tests for the app factory and lifespan."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

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
