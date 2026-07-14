# Security Policy

## Scope

This repository is a **template**: a demo-grade local-first LLM agent platform
(FastAPI + LangGraph + LiteLLM + Langfuse). It is not a production service and
holds no real user data. Reported issues in scope include:

- Secret leakage in the repository (committed keys, tokens, credentials).
- Vulnerabilities in the code under `app/`, `gateway/`, `scripts/`, or the
  Docker/Compose/Terraform configuration that would let an attacker escalate
  privileges, exfiltrate data, or bypass the guardrails/auth layer in a
  deployment that follows this template as-is.
- Supply-chain issues (malicious or compromised dependency pinned in
  `uv.lock` or a GitHub Actions workflow).

Out of scope: findings that only apply to intentionally-stubbed v1 pieces
documented in the README ("known limitations / v1 scope"), such as the
absence of OAuth, rate limiting delegated to LiteLLM, or the single shared
Postgres instance used for the demo.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability. Instead:

1. Open a [private GitHub security advisory](../../security/advisories/new)
   on this repository, or
2. Email the maintainer directly (see the GitHub profile of the repository
   owner, PierreRtec) with a description, reproduction steps, and impact.

Expect an acknowledgement within a few days. This is a solo-maintained
personal project, so response times are best-effort, not SLA-backed.

## Secret Handling

- No real secret is ever committed. `.env` is gitignored; only `.env.example`
  (obvious placeholder values, `changeme-*`) is tracked.
- `gitleaks` runs in CI (`security.yml`) against the full git history on
  every push and pull request. `.gitleaks.toml` only allowlists the
  placeholder values from `.env.example`, nothing else.
- If a real secret is ever accidentally committed: rotate it immediately at
  the provider, then rewrite history to remove it. Rotation always comes
  first; history rewriting alone does not protect an already-leaked key.
