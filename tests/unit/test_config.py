"""Unit tests for app.core.config.Settings.

Settings are sourced strictly from environment variables (no .env file
reading at runtime); see CLAUDE.md non-negotiable #4 and the design doc
section 3.1 for the variable list.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_settings_requires_app_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """APP_API_KEY has no default: booting without it must fail clearly."""
    monkeypatch.delenv("APP_API_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)  # type: ignore[call-arg]

    assert "APP_API_KEY" in str(exc_info.value)


def test_settings_loads_from_env_with_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only APP_API_KEY is mandatory; everything else has a sane local default."""
    monkeypatch.setenv("APP_API_KEY", "test-key")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.APP_API_KEY == "test-key"
    assert settings.APP_ENV == "local"
    assert settings.LOG_LEVEL == "INFO"
    assert settings.DATABASE_URL.startswith("postgresql://")
    assert settings.REDIS_URL.startswith("redis://")
    assert settings.LITELLM_BASE_URL.startswith("http://")
    assert settings.LANGFUSE_HOST.startswith("http://")


def test_settings_reads_all_section_3_1_variables_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every variable the app receives per docker-compose.yml (section 3.1) is parsed."""
    env = {
        "APP_API_KEY": "app-key",
        "APP_ENV": "ci",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": "postgresql://u:p@postgres:5432/app",
        "REDIS_URL": "redis://redis:6379/0",
        "LITELLM_BASE_URL": "http://litellm:4000/v1",
        "LITELLM_API_KEY": "litellm-key",
        "LANGFUSE_HOST": "http://langfuse-web:3000",
        "LANGFUSE_PUBLIC_KEY": "pub-key",
        "LANGFUSE_SECRET_KEY": "secret-key",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    for key, value in env.items():
        assert getattr(settings, key) == value


def test_settings_rejects_invalid_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_API_KEY", "test-key")
    monkeypatch.setenv("APP_ENV", "not-a-real-env")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() must be a process-wide singleton (parsed once at boot)."""
    monkeypatch.setenv("APP_API_KEY", "test-key")
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
    get_settings.cache_clear()
