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

This repository is being built incrementally, task by task. Full design notes will land in
`docs/SPEC.md` and `docs/adr/` as the corresponding tasks complete. Roadmap:

- [x] T0 - Bootstrap repo and hygiene (tooling, lint, hooks, skeleton)
- [ ] T1 - Local infra via compose (postgres, redis, litellm, langfuse)
- [ ] T2 - FastAPI core (config, logging, health)
- [ ] T3 - Gateway-aware LLM layer (retries, fallback)
- [ ] T4 - Postgres memory (checkpointer + store)
- [ ] T5 - Tool registry and tools
- [ ] T6 - LangGraph graph + guardrails + HITL
- [ ] T7 - Chat API (SSE) + threads
- [ ] T8 - Observability: OpenInference -> OTel -> Langfuse
- [ ] T9 - DeepEval quality gate
- [ ] T10 - Ragas (out of CI)
- [ ] T11 - GitHub Actions workflows
- [ ] T12 - Terraform on Scaleway (thin module)
- [ ] T13 - Final docs

## Quickstart (placeholder, will work end-to-end after T7/T8)

```bash
cp .env.example .env   # fill in real API keys for manual testing
docker compose up -d --wait
curl -s -X POST localhost:8000/v1/chat -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "What financial aid exists for someone under 25?"}'
```

## Development

```bash
uv sync
make lint
make test
```

See `CLAUDE.md` for repository conventions (commands, TDD, commit style, secret hygiene).
