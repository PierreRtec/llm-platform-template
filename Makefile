.PHONY: up down dev test test-int lint eval seed smoke

# Targets below fail on purpose with a clear message when the files they
# depend on have not been built yet by a later task (Tn) in the roadmap
# (see README.md). This keeps `make <target>` honest instead of silently
# no-op-ing on a fresh clone.

up:
	@test -f docker-compose.yml || { echo "make up: not implemented yet (T1)"; exit 1; }
	docker compose up -d --wait

down:
	@test -f docker-compose.yml || { echo "make down: not implemented yet (T1)"; exit 1; }
	docker compose down

dev:
	@test -f app/main.py || { echo "make dev: not implemented yet (T2)"; exit 1; }
	uv run uvicorn app.main:app --reload

test:
	uv run pytest -m "not integration"

test-int:
	@test -f docker-compose.ci.yml || { echo "make test-int: not implemented yet (T1/T7)"; exit 1; }
	docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --wait
	uv run pytest -m integration

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy app

eval:
	@test -f evals/deepeval/test_agent_quality.py || { echo "make eval: not implemented yet (T9)"; exit 1; }
	uv run deepeval test run evals/deepeval

seed:
	@test -f scripts/seed_aids.py || { echo "make seed: not implemented yet (T4)"; exit 1; }
	uv run python scripts/seed_aids.py

smoke:
	@test -f scripts/smoke.sh || { echo "make smoke: not implemented yet (T7)"; exit 1; }
	./scripts/smoke.sh
