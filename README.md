# llm-platform-template

[![ci](https://github.com/PierreRtec/llm-platform-template/actions/workflows/ci.yml/badge.svg)](https://github.com/PierreRtec/llm-platform-template/actions/workflows/ci.yml)
[![security](https://github.com/PierreRtec/llm-platform-template/actions/workflows/security.yml/badge.svg)](https://github.com/PierreRtec/llm-platform-template/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local-first, sovereignty-aware template for production-grade LLM agent platforms. FastAPI serves a
LangGraph agent that talks to models only through a self-hosted LiteLLM gateway (never a provider SDK
directly), with Postgres-backed memory, layered guardrails, human-in-the-loop approval, and full tracing
into Langfuse. `docker compose up` is the nominal path; a Terraform module for Scaleway is a bonus, not a
prerequisite.

## Status: work in progress

This repository is being built incrementally, task by task. Working design notes live in
`docs/DESIGN.md`; polished English architecture docs and ADRs land with T13. Roadmap:

- [x] T0 - Bootstrap repo and hygiene (tooling, lint, hooks, skeleton)
- [x] T1 - Local infra via compose (postgres, redis, litellm, langfuse)
- [x] T2 - FastAPI core (config, logging, health)
- [x] T3 - Gateway-aware LLM layer (retries, fallback)
- [ ] T4 - Postgres memory (checkpointer + store)
- [ ] T5 - Tool registry and tools
- [x] T6 - LangGraph graph + guardrails + HITL (MVP: no HITL/guard_output yet)
- [x] T7 - Chat API (SSE) + threads
- [x] T8 - Observability: OpenInference -> OTel -> Langfuse
- [ ] T9 - DeepEval quality gate
- [ ] T10 - Ragas (out of CI)
- [x] T11 - GitHub Actions workflows
- [ ] T12 - Terraform on Scaleway (thin module)
- [ ] T13 - Final docs

### Telemetry troubleshooting

If `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set but Langfuse itself is
down or unreachable, the app still boots and serves traffic normally: the
OTel SDK just logs periodic export failures in the background while it
retries. To disable telemetry entirely (e.g. in CI, or to silence those
logs), leave both keys empty.

## Quickstart

```bash
cp .env.example .env   # fill in real API keys for manual testing
docker compose up -d --wait
curl -s -X POST localhost:8000/v1/chat -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "What financial aid exists for someone under 25?"}'
```

## Known limitations (v1 scope)

This template is intentionally stubbed in a few places, called out here rather than left implicit:

- **Memory is process-local.** `InMemorySaver` backs the LangGraph checkpointer today: state is lost
  on restart and not shared across replicas. `AsyncPostgresSaver` (T4) replaces it without changing
  the chat API's contract.
- **Guardrails are heuristic.** Input/output guardrails are regex and keyword-based (PII patterns,
  injection markers, a disclaimer), not a dedicated classifier. Wiring in Presidio or Llama Guard is
  a known follow-up iteration, not planned for this template.
- **No HITL, no `guard_output` node yet.** The graph is agent + tools only; the `interrupt()`
  approval flow before sensitive tools and the output guardrail node land with the full T6.
- **One Postgres instance, three logical databases** (`app`, `litellm`, `langfuse`). A deliberate
  demo trade-off, not a production topology; see `docs/DESIGN.md` section 6.
- **Auth is a static API key.** No OAuth, no multi-tenant story, no rate limiting in the app itself
  (delegated to LiteLLM).
- **Evals and infra-as-code are not shipped yet.** DeepEval/Ragas quality gates (T9/T10) and the
  Scaleway Terraform module (T12) are on the roadmap, not in this repository yet.
- **Token-level streaming is not exercised against a real provider.** The MVP agent node calls the
  LLM with a single `ainvoke`, not a token-by-token `.astream()`; the SSE contract still guarantees
  one `done` event with the full answer regardless of how granular (or absent) upstream token
  streaming is. See `app/api/routes/chat.py` module docstring for the detailed rationale.

## Development

```bash
uv sync
make lint
make test
```

See `CLAUDE.md` for repository conventions (commands, TDD, commit style, secret hygiene).
