# syntax=docker/dockerfile:1

# --- Builder: resolve and install the locked dependency set with uv ---
FROM ghcr.io/astral-sh/uv:0.9-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (cached separately from application code) so
# code-only changes do not bust the dependency layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md ./
COPY app ./app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# --- Runtime: slim image, non-root user, no build toolchain ---
FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /app --no-create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
