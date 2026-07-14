"""Shared FastAPI dependencies: API key auth and injectable readiness checks.

Each readiness check is built from `Settings` behind a small provider
function (`get_postgres_check`, `get_redis_check`, `get_litellm_check`) so
tests can override them with `app.dependency_overrides` and never touch a
real network connection.
"""

import hmac
from collections.abc import Awaitable, Callable

import httpx
import psycopg
import redis.asyncio as redis
import structlog
from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings

READINESS_TIMEOUT_SECONDS = 1.0
# libpq rounds any connect_timeout below 2s up to 2s, so 2 is the effective floor
POSTGRES_CONNECT_TIMEOUT_SECONDS = 2

# (ok, human-readable detail: "ok" or an error message)
CheckResult = tuple[bool, str]
ReadinessCheck = Callable[[], Awaitable[CheckResult]]


async def verify_api_key(
    settings: Settings = Depends(get_settings),
    x_api_key: str = Header(default=""),
) -> None:
    """Require a valid `X-API-Key` header, compared in constant time."""
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.APP_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )


def _build_postgres_check(settings: Settings) -> ReadinessCheck:
    async def _check() -> CheckResult:
        logger = structlog.get_logger(__name__)
        try:
            conn = await psycopg.AsyncConnection.connect(
                settings.DATABASE_URL,
                connect_timeout=POSTGRES_CONNECT_TIMEOUT_SECONDS,
            )
            await conn.close()
        except TimeoutError as exc:  # readiness must report, never crash the route
            logger.warning("postgres readiness check timeout", check="postgres", exc_info=exc)
            return False, "timeout"
        except Exception as exc:  # readiness must report, never crash the route
            logger.warning("postgres readiness check failed", check="postgres", exc_info=exc)
            return False, "connection failed"
        return True, "ok"

    return _check


def _build_redis_check(settings: Settings) -> ReadinessCheck:
    async def _check() -> CheckResult:
        logger = structlog.get_logger(__name__)
        client = redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=READINESS_TIMEOUT_SECONDS,
            socket_timeout=READINESS_TIMEOUT_SECONDS,
        )
        try:
            await client.ping()
        except TimeoutError as exc:  # readiness must report, never crash the route
            logger.warning("redis readiness check timeout", check="redis", exc_info=exc)
            return False, "timeout"
        except Exception as exc:  # readiness must report, never crash the route
            logger.warning("redis readiness check failed", check="redis", exc_info=exc)
            return False, "connection failed"
        finally:
            await client.aclose()
        return True, "ok"

    return _check


def _build_litellm_check(settings: Settings) -> ReadinessCheck:
    async def _check() -> CheckResult:
        health_url = httpx.URL(settings.LITELLM_BASE_URL).copy_with(path="/health/liveliness")
        try:
            async with httpx.AsyncClient(timeout=READINESS_TIMEOUT_SECONDS) as http_client:
                response = await http_client.get(health_url)
                response.raise_for_status()
        except Exception as exc:  # readiness must report, never crash the route
            return False, str(exc)
        return True, "ok"

    return _check


def get_postgres_check(settings: Settings = Depends(get_settings)) -> ReadinessCheck:
    return _build_postgres_check(settings)


def get_redis_check(settings: Settings = Depends(get_settings)) -> ReadinessCheck:
    return _build_redis_check(settings)


def get_litellm_check(settings: Settings = Depends(get_settings)) -> ReadinessCheck:
    return _build_litellm_check(settings)
