"""Unit tests for the X-API-Key auth dependency (app.api.deps.verify_api_key).

Exercised against a small standalone route (not /health, which is
intentionally unauthenticated) so the dependency is tested in isolation.
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import verify_api_key
from app.core.config import get_settings


@pytest.fixture
def protected_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_API_KEY", "correct-key")
    get_settings.cache_clear()

    test_app = FastAPI()

    @test_app.get("/protected", dependencies=[Depends(verify_api_key)])
    async def protected() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(test_app)


def test_missing_api_key_is_rejected(protected_client: TestClient) -> None:
    response = protected_client.get("/protected")

    assert response.status_code == 401


def test_wrong_api_key_is_rejected(protected_client: TestClient) -> None:
    response = protected_client.get("/protected", headers={"X-API-Key": "wrong-key"})

    assert response.status_code == 401


def test_correct_api_key_is_accepted(protected_client: TestClient) -> None:
    response = protected_client.get("/protected", headers={"X-API-Key": "correct-key"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
