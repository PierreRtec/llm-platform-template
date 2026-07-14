"""Health check routes.

`GET /health` is an unauthenticated liveness probe: it always returns 200 if
the process is up and able to serve requests. `GET /health/ready` is a
readiness probe that verifies Postgres, Redis and LiteLLM are reachable,
each with a short timeout, and returns 503 with per-check detail if any of
them is not.
"""

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app import __version__
from app.api.deps import ReadinessCheck, get_litellm_check, get_postgres_check, get_redis_check

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict[str, str]:
    """Liveness probe: always 200, no auth, no dependency checks."""
    return {"status": "ok", "version": __version__}


@router.get("/health/ready")
async def readiness(
    postgres_check: ReadinessCheck = Depends(get_postgres_check),
    redis_check: ReadinessCheck = Depends(get_redis_check),
    litellm_check: ReadinessCheck = Depends(get_litellm_check),
) -> JSONResponse:
    """Readiness probe: 200 with all checks ok, 503 with detail otherwise."""
    names = ("postgres", "redis", "litellm")
    results = await asyncio.gather(postgres_check(), redis_check(), litellm_check())
    checks = {
        name: {"ok": ok, "detail": detail}
        for name, (ok, detail) in zip(names, results, strict=True)
    }
    all_ok = all(ok for ok, _ in results)
    return JSONResponse(status_code=200 if all_ok else 503, content={"checks": checks})
