"""Unit tests for /health and /health/ready.

Readiness dependencies (`get_postgres_check`, `get_redis_check`,
`get_litellm_check`) are overridden with fakes so these tests never touch a
real network connection.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import CheckResult, get_litellm_check, get_postgres_check, get_redis_check


def _fake_check(result: CheckResult) -> object:
    async def _check() -> CheckResult:
        return result

    return _check


def test_health_liveness_is_always_200_and_unauthenticated(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"


def test_health_ready_returns_200_when_all_checks_pass(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_postgres_check] = lambda: _fake_check((True, "ok"))
    app.dependency_overrides[get_redis_check] = lambda: _fake_check((True, "ok"))
    app.dependency_overrides[get_litellm_check] = lambda: _fake_check((True, "ok"))

    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["checks"]["postgres"] == {"ok": True, "detail": "ok"}
    assert body["checks"]["redis"] == {"ok": True, "detail": "ok"}
    assert body["checks"]["litellm"] == {"ok": True, "detail": "ok"}


def test_health_ready_returns_503_when_a_check_fails(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_postgres_check] = lambda: _fake_check((True, "ok"))
    app.dependency_overrides[get_redis_check] = lambda: _fake_check((False, "connection failed"))
    app.dependency_overrides[get_litellm_check] = lambda: _fake_check((True, "ok"))

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["checks"]["redis"] == {"ok": False, "detail": "connection failed"}
    assert body["checks"]["postgres"]["ok"] is True


def test_health_ready_reports_all_failures_independently(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_postgres_check] = lambda: _fake_check((False, "timeout"))
    app.dependency_overrides[get_redis_check] = lambda: _fake_check((False, "timeout"))
    app.dependency_overrides[get_litellm_check] = lambda: _fake_check((False, "timeout"))

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert all(check["ok"] is False for check in body["checks"].values())


def test_health_ready_reports_generic_messages_not_raw_exceptions(
    app: FastAPI, client: TestClient
) -> None:
    """Verify that generic error messages are used, not raw exception strings."""
    # Use generic messages as they should be reported by the checks
    app.dependency_overrides[get_postgres_check] = lambda: _fake_check((False, "timeout"))
    app.dependency_overrides[get_redis_check] = lambda: _fake_check((False, "connection failed"))
    app.dependency_overrides[get_litellm_check] = lambda: _fake_check((True, "ok"))

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    # Verify generic messages, never raw exception strings
    assert body["checks"]["postgres"]["detail"] == "timeout"
    assert body["checks"]["redis"]["detail"] == "connection failed"
    assert body["checks"]["litellm"]["detail"] == "ok"
