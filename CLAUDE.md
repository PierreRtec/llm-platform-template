# CLAUDE.md

Guidance for Claude Code (or any coding agent) working in this repository.

## What this repo is

`llm-platform-template`: a local-first template for a production-shaped LLM
agent platform.

### Target architecture

FastAPI + LangGraph agent, talking to models only through a self-hosted
LiteLLM gateway (model groups `sovereign-cheap`, `sovereign-premium`,
`frontier`; zero provider SDK in application code), Postgres-backed memory
(checkpointer + pgvector store), layered guardrails, human-in-the-loop
approval, and tracing into Langfuse via OpenInference/OTel. `docker compose
up` is the nominal path; Terraform (Scaleway) is a bonus, not a prerequisite.
See `docs/DESIGN.md` for the full design and rationale.

### Current MVP state

`InMemorySaver` backs the checkpointer today (Postgres checkpointer lands in
T4); no HITL yet (full guardrail/approval flow lands with T6 full). See the
roadmap in `README.md` for what is done vs. still planned.

## Language

- This repository is public. **Everything is in English**: code, comments,
  docstrings, commit messages, README, ADRs, PR descriptions. No French
  anywhere in the tracked files, except `docs/DESIGN.md` (working design
  notes, in French by design) and product-facing prompts (`app/agent/prompts.py`),
  which are in French because the agent talks to end users in French.
- No em dash (—) or double-dash-as-punctuation anywhere, in any language, in
  any file this repo tracks. Use a period, comma, or parentheses instead.

## Non-negotiables

1. **TDD.** Write the failing test first, then the implementation. This
   applies to every unit under `app/`: no production code lands without a
   test that exercised it failing beforehand. See
   `superpowers:test-driven-development` if using Claude Code with the
   superpowers plugin.
2. **Conventional Commits.** Every commit message follows
   `type(scope): summary` (`feat`, `fix`, `chore`, `docs`, `test`, `refactor`,
   `ci`, `build`). Body explains why, not just what, when non-obvious.
3. **Zero secrets, ever.** No real API key, token, password, or connection
   string in any tracked file, commit message, or log fixture. Only
   `.env.example` is committed, with obvious `changeme-*` placeholders.
   `gitleaks` runs locally (pre-commit) and in CI (`security.yml`) against
   full history. If you are about to paste a real key into a file to "test
   something," stop and use an env var instead.
4. **Strict typing and linting.** `ruff check` and `ruff format --check` must
   be clean. `mypy --strict` must pass on `app/` with no `# type: ignore`
   unless a comment justifies it (e.g. an untyped third-party stub). Do not
   silence errors by loosening `pyproject.toml` mypy settings.
5. **No provider SDK in application code.** The agent only ever talks to
   `LITELLM_BASE_URL` via `ChatOpenAI`-shaped clients. If you find yourself
   importing `anthropic`, `mistralai`, or a Scaleway SDK in `app/`, that is a
   design violation, not a shortcut.

## Common commands (Makefile)

```
make up        # docker compose up -d --wait (local infra + app)
make down      # docker compose down
make dev       # uv run uvicorn app.main:app --reload
make test      # uv run pytest -m "not integration"
make test-int  # docker compose (+ci override) up, then pytest -m integration
make lint      # uv run ruff check . && uv run ruff format --check . && uv run mypy app
make eval      # uv run deepeval test run evals/deepeval
make seed      # uv run python scripts/seed_aids.py
make smoke     # ./scripts/smoke.sh
```

Targets that depend on infrastructure not built yet in the current task wave
fail on purpose with a clear `not implemented yet (Tn)` message rather than
silently no-op-ing. If you hit one of those, check the roadmap in `README.md`
to see which task unlocks it.

## Testing conventions

- Unit tests (`tests/unit/`) never touch the network or a real LLM. Use
  `GenericFakeChatModel` / `FakeListChatModel` from
  `langchain_core.language_models.fake_chat_models` and `respx` for HTTP.
  They must stay fast (whole unit suite well under 10s).
- Integration tests (`tests/integration/`) are marked `@pytest.mark.integration`
  and require `docker compose -f docker-compose.yml -f docker-compose.ci.yml
  up -d --wait` (LiteLLM in `mock_response` mode, no real API key needed).
- Table-driven tests (`pytest.mark.parametrize`) for guardrails and registry
  validation; do not hand-roll loops of asserts where parametrize reads
  cleaner.
- Before claiming a task done: run the actual verification commands
  (`make lint`, `make test`, and `make test-int` when relevant) and look at
  their output. Do not assert green without having seen it.

## Style notes

- `app/` is the package root (no `src/` layout), Python 3.12, `pydantic` v2.
- Settings only ever come from environment variables via
  `pydantic-settings`, validated at boot in `app/core/config.py`. No
  hardcoded config, no reading arbitrary files at runtime for secrets.
- Prefer pure functions for guardrails (`app/agent/guardrails/`) so they stay
  trivially unit-testable without a running graph.
- Keep tools declared through `ToolRegistry` (`app/agent/tools/registry.py`,
  planned, T5); the graph must refuse to call anything not registered there.
