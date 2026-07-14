## Summary

<!-- What changed and why, in a sentence or two. -->

## Checklist

- [ ] `make lint && make test` pass locally (or CI is green: lint, typecheck, test)
- [ ] Zero secrets committed (`.env`, real API keys, tokens) — `gitleaks` job is green
- [ ] Coverage did not drop below the `--cov-fail-under` gate in `ci.yml`
- [ ] Eval impact considered: label `run-evals` added if this touches `app/agent/**` or prompts
- [ ] Commit messages follow Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `ci:`, ...)
