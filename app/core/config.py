"""Application settings.

Settings are sourced strictly from environment variables and validated once
at process boot (see CLAUDE.md non-negotiable #4: no hardcoded config, no
reading arbitrary files at runtime for secrets). Field names intentionally
match the environment variable names 1:1 (see design doc section 3.1 and
docker-compose.yml's `app` service `environment` block) so there is a single
obvious source of truth for what the app expects to receive.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the app.

    Only `APP_API_KEY` is mandatory: it is the app's own auth secret and has
    no safe default. Every other variable has a default reasonable for a
    bare `docker compose up` / local `uv run uvicorn` run.
    """

    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")

    # --- Process ---
    APP_ENV: Literal["local", "ci", "prod"] = "local"
    LOG_LEVEL: str = "INFO"

    # --- Postgres (single instance, three logical databases) ---
    DATABASE_URL: str = "postgresql://platform:platform@localhost:5432/app"

    # --- Redis (LiteLLM cache today; readiness check reuses it) ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- LiteLLM gateway (self-hosted, OpenAI-compatible) ---
    LITELLM_BASE_URL: str = "http://localhost:4000/v1"
    LITELLM_API_KEY: str = "changeme-litellm-master-key"

    # --- Langfuse (tracing; consumed starting T8, config accepted from T2) ---
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str = "changeme-langfuse-public-key"
    LANGFUSE_SECRET_KEY: str = "changeme-langfuse-secret-key"

    # --- App auth (no default: must be set explicitly, never a placeholder) ---
    APP_API_KEY: str

    # --- Chat streaming (POST /v1/chat) ---
    # Upper bound on the *total* wall-clock duration of one SSE stream (the
    # whole astream_events consumption plus the final aget_state call), not a
    # per-attempt timeout (see app.agent.llm.DEFAULT_TIMEOUT_SECONDS for that
    # layer). Guards against a graph that hangs (stuck tool call, unhealthy
    # upstream that neither errors nor completes) holding the connection open
    # forever.
    CHAT_STREAM_TIMEOUT_SECONDS: float = 120.0


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton, built once from the env."""
    return Settings()
