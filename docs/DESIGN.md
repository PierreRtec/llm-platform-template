Working design notes for this template (in French). Polished English architecture docs and ADRs
land with T13.

# Design de reference : llm-platform-template (14/07/2026)

Sortie de l'agent d'architecture, validee par Pierre. Source unique pour la spec et le build du repo 1.
Contexte projet : Finley 2.0, 4 repos publics (llm-platform-template, finley-2, finley-mcp, finley-web), doc Phase 0 : ~/prodev/liberty/tmp/2026-07-14-finley-phase0.md.
GitHub : compte PierreRtec, repo public des le premier commit. Emplacement local : ~/prodev/finley2/llm-platform-template/.

## 0. Principes directeurs

- Local-first : `docker compose up` est le chemin nominal. Terraform Scaleway est un bonus demontrable, jamais un prerequis.
- LiteLLM deploye, pas reecrit : l'app ne connait que des noms de model groups (`sovereign-cheap`, `sovereign-premium`, `frontier`) via une base URL OpenAI-compatible. Zero SDK provider dans le code applicatif.
- Un seul Postgres, trois bases : `app` (checkpointer + store pgvector), `litellm` (spend tracking), `langfuse` (traces). Un script d'init cree les trois. Choix assume de template de demo (en prod : instances separees, documente dans un ADR).
- Testable sans cle API : tout le graphe est testable avec des fakes LangChain, et l'integration compose tourne avec `mock_response` LiteLLM. Les cles reelles ne servent qu'aux evals et a l'usage manuel.
- Domaine minimal : un tool `search_aids` sur un corpus JSON fictif d'aides financieres (5 a 10 entrees inventees). Le vrai RAG est hors scope (repo `finley-2`). Ici on prouve le harness, pas le retrieval.
- Package Python : `app/` a la racine (convention fastapi-template), Python 3.12, uv, pydantic v2, mypy strict, ruff.

## 1. Arborescence complete

```
llm-platform-template/
├── README.md                        # Pitch, schema mermaid, quickstart 5 commandes, badges CI
├── CLAUDE.md                        # Conventions repo pour agents (commandes, style, TDD, zero secret)
├── LICENSE                          # MIT
├── SECURITY.md                      # Politique de report, perimetre, gestion des secrets
├── .env.example                     # Toutes les variables, valeurs factices, commentees par bloc
├── .gitignore                       # Python, .env, .terraform, artefacts eval
├── .gitleaks.toml                   # Allowlist (valeurs factices de .env.example)
├── .pre-commit-config.yaml          # ruff, ruff-format, gitleaks, mypy (hook local), commitizen check
├── pyproject.toml                   # uv, deps, config ruff/mypy strict/pytest/coverage
├── uv.lock                          # Lockfile commite
├── Makefile                         # up, down, dev, test, test-int, lint, eval, seed, smoke
├── Dockerfile                       # Multi-stage uv (builder + runtime slim, user non-root)
├── docker-compose.yml               # postgres, redis, litellm, app + include langfuse
├── docker-compose.langfuse.yml      # langfuse-web, langfuse-worker, clickhouse, minio (via include)
├── docker-compose.ci.yml            # Override CI : litellm en mock_response, pas de langfuse
│
├── app/
│   ├── __init__.py
│   ├── main.py                      # App factory + lifespan (pools pg, checkpointer.setup, OTel)
│   ├── core/
│   │   ├── config.py                # Settings pydantic-settings (env only, validees au boot)
│   │   ├── logging.py               # structlog JSON, correlation trace_id
│   │   └── telemetry.py             # OpenInference instrumentor -> OTLP -> Langfuse
│   ├── api/
│   │   ├── deps.py                  # Auth API key (header), injection graph/store
│   │   └── routes/
│   │       ├── health.py            # GET /health (liveness) + /health/ready (pg, redis, litellm)
│   │       ├── chat.py              # POST /v1/chat (SSE streaming, thread_id, user_id)
│   │       └── threads.py           # GET /v1/threads/{id} (etat checkpointer), POST resume HITL
│   ├── agent/
│   │   ├── state.py                 # AgentState (messages, user_id, budget tokens, flags guardrails)
│   │   ├── graph.py                 # build_graph() : guard_in -> agent <-> tools -> guard_out, interrupt HITL
│   │   ├── llm.py                   # get_llm(group) : ChatOpenAI vers LiteLLM + retry tenacity + fallback groupe
│   │   ├── prompts.py               # Prompts versionnes (constantes + hash logge en trace)
│   │   ├── memory.py                # AsyncPostgresSaver + AsyncPostgresStore (pgvector HNSW), namespaces
│   │   ├── tools/
│   │   │   ├── registry.py          # ToolRegistry Pydantic : version, schema args, flag requires_approval
│   │   │   ├── search_aids.py       # Recherche semantique dans le Store (corpus aides fictives)
│   │   │   └── get_aid_details.py   # Fiche detaillee d'une aide (lookup exact, tool "sensible" HITL demo)
│   │   └── guardrails/
│   │       ├── input.py             # Longueur, langue, patterns injection, PII regex FR (NIR, IBAN)
│   │       ├── output.py            # Disclaimer IA (art. 50), refus decision d'eligibilite, redaction PII
│   │       └── pipeline.py          # Chainage en couches, chaque couche = fonction pure testee
│
├── gateway/
│   └── litellm.config.yaml          # 3 model groups + embed, simple-shuffle, fallbacks, spend pg, cache redis
│
├── data/
│   └── aids_sample.json             # 8 aides financieres fictives (nom, conditions, montants, source)
│
├── scripts/
│   ├── init-db.sh                   # Cree les bases app/litellm/langfuse + extension pgvector
│   ├── seed_aids.py                 # Embedde data/aids_sample.json dans le Store (via sovereign-embed)
│   └── smoke.sh                     # curl /health/ready + un POST /v1/chat, exit code propre
│
├── evals/
│   ├── datasets/
│   │   ├── golden_smoke.jsonl       # 12 cas (input, expected_facts, contexte) pour le gate PR
│   │   └── golden_full.jsonl        # ~50 cas pour le run nightly
│   ├── deepeval/
│   │   ├── conftest.py              # Juge epingle temp 0, sampling EVAL_SAMPLE_SIZE, cache
│   │   └── test_agent_quality.py    # AnswerRelevancy, Faithfulness, GEval "pas de decision d'eligibilite"
│   └── ragas/
│       └── run_ragas.py             # Script hors CI : evaluation retriever + TestsetGenerator (squelette)
│
├── tests/
│   ├── conftest.py                  # Fixtures : settings test, FakeListChatModel, InMemorySaver
│   ├── unit/
│   │   ├── test_config.py           # Parsing/validation settings
│   │   ├── test_registry.py         # Versionnage, schemas, rejet tool non declare
│   │   ├── test_guardrails.py       # Table-driven : injections, PII FR, disclaimer
│   │   ├── test_llm_fallback.py     # Retries tenacity + bascule de groupe (mocks)
│   │   └── test_graph.py            # Topologie compilee, routage conditionnel, interrupt HITL (fake LLM)
│   └── integration/
│       ├── test_chat_api.py         # E2E compose : POST /v1/chat via LiteLLM mock, SSE
│       └── test_memory.py           # Persistance checkpointer entre deux tours + Store put/search
│
├── infra/terraform/
│   ├── versions.tf                  # provider scaleway epingle, required_version
│   ├── main.tf                      # Compose les modules
│   ├── variables.tf                 # project_id, region fr-par, tailles
│   ├── outputs.tf                   # URLs containers, host RDB
│   ├── environments/dev.tfvars.example
│   └── modules/
│       ├── database/                # RDB Postgres manage + pgvector
│       ├── cache/                   # Redis manage
│       └── containers/              # Serverless Containers : app, litellm, langfuse-web
│
├── docs/
│   ├── SPEC.md                      # La spec du repo (copiee depuis specs/SPEC-llm-platform-template.md)
│   ├── architecture.md              # Diagramme mermaid des flux (requete, trace, spend)
│   └── adr/
│       ├── 0001-litellm-gateway.md  # Pourquoi LiteLLM self-host vs OpenRouter (souverainete)
│       ├── 0002-postgres-memory.md  # Pourquoi checkpointer+store Postgres vs framework memoire
│       ├── 0003-eval-gate.md        # DeepEval en CI vs Ragas hors CI, juge epingle
│       └── 0004-single-postgres.md  # Trade-off demo : 1 instance, 3 bases
│
└── .github/
    ├── dependabot.yml               # pip + github-actions, weekly
    ├── PULL_REQUEST_TEMPLATE.md     # Checklist : tests, secrets, impact eval
    └── workflows/
        ├── ci.yml                   # lint + mypy + pytest unit + coverage
        ├── security.yml             # gitleaks full history
        ├── eval.yml                 # gate DeepEval (PR echantillonne, nightly full)
        └── terraform.yml            # fmt -check + validate + tflint
```

Coupe volontairement : CONTRIBUTING.md, devcontainer, helm/k8s, Alembic (les `setup()` LangGraph + init SQL suffisent), frontend, module Terraform network dedie.

## 2. Decoupage en taches et vagues

```
Vague 1 : T0
Vague 2 : T1, T2, T3, T5, T12        (independantes entre elles, dependent de T0)
Vague 3 : T4, T6                     (T4 dep T1+T2 ; T6 dep T3+T5)
Vague 4 : T7, T8                     (dep T4+T6)
Vague 5 : T9, T11                    (T9 dep T7 ; T11 dep T0, s'enrichit avec T9/T12)
Vague 6 : T10, T13                   (finitions)
```

### T0 : Bootstrap repo et hygiene (bloquant, sequentiel)
- Livrables : `pyproject.toml` (deps : fastapi, uvicorn, langgraph, langgraph-checkpoint-postgres, langchain-openai, pydantic-settings, structlog, tenacity, httpx, openinference-instrumentation-langchain, opentelemetry-sdk/exporter-otlp ; dev : pytest, pytest-asyncio, respx, mypy, ruff, deepeval, pre-commit), `.pre-commit-config.yaml`, `.gitleaks.toml`, `.env.example`, `.gitignore`, `LICENSE`, `SECURITY.md`, `CLAUDE.md`, `Makefile`, README squelette, arborescence vide avec `__init__.py`.
- Done : `uv sync && uv run ruff check . && uv run mypy app && uv run pytest` (0 test ou placeholder, exit 0) et `pre-commit run --all-files` passent.

### T1 : Infra locale compose (postgres, redis, litellm, langfuse)
- Livrables : `docker-compose.yml`, `docker-compose.langfuse.yml`, `scripts/init-db.sh`, `gateway/litellm.config.yaml`, `gateway/litellm.ci.yaml`, `docker-compose.ci.yml`.
- Done : `docker compose up -d --wait` tous services healthy, puis `curl -f localhost:4000/health/liveliness` (LiteLLM) et `curl -f localhost:3000/api/public/health` (Langfuse).

### T2 : Socle FastAPI (config, logging, health)
- Livrables : `app/core/config.py`, `app/core/logging.py`, `app/main.py`, `app/api/routes/health.py`, `app/api/deps.py` (auth API key simple), `Dockerfile`.
- Done : `uv run pytest tests/unit/test_config.py` vert et `uv run uvicorn app.main:app` puis `curl -f localhost:8000/health`.

### T3 : Couche LLM gateway-aware (retries, fallback)
- Livrables : `app/agent/llm.py` (factory `get_llm(model_group)` -> `ChatOpenAI(base_url=LITELLM_URL, model=group)`, retry tenacity exponentiel sur 429/5xx puis bascule `sovereign-cheap -> sovereign-premium -> frontier` cote applicatif, en plus des fallbacks LiteLLM), `tests/unit/test_llm_fallback.py` (respx/mocks).
- Done : `uv run pytest tests/unit/test_llm_fallback.py` vert. Aucun appel reseau reel dans les tests.

### T4 : Memoire Postgres (checkpointer + store)
- Livrables : `app/agent/memory.py` : `AsyncPostgresSaver.from_conn_string` (court terme) + `AsyncPostgresStore` avec index pgvector HNSW (long terme, namespace `("aids",)` corpus et `("memories", user_id)` utilisateur), `setup()` dans le lifespan, `scripts/seed_aids.py`, `tests/integration/test_memory.py`.
- Done : compose up puis `uv run pytest -m integration tests/integration/test_memory.py` vert.

### T5 : Tool registry et tools
- Livrables : `app/agent/tools/registry.py` (modele Pydantic `ToolSpec` : nom, version semver, schema args, `requires_approval: bool` ; registre qui refuse tout tool non declare et loggue la version en trace), `search_aids.py`, `get_aid_details.py` (requires_approval=True pour la demo HITL), `data/aids_sample.json`, `tests/unit/test_registry.py`.
- Done : `uv run pytest tests/unit/test_registry.py` vert.

### T6 : Graphe LangGraph + guardrails + HITL
- Livrables : `app/agent/state.py`, `app/agent/graph.py` (noeuds : `guard_input` -> `agent` -> arete conditionnelle vers `tools` ou `guard_output` ; `interrupt()` avant tout tool `requires_approval`), `app/agent/guardrails/` (3 fichiers, fonctions pures), `app/agent/prompts.py`, `tests/unit/test_graph.py` et `test_guardrails.py` avec `FakeListChatModel`/`GenericFakeChatModel` et `InMemorySaver`.
- Done : `uv run pytest tests/unit/test_graph.py tests/unit/test_guardrails.py` vert (dont un test qui verifie que le graphe s'interrompt sur `get_aid_details` et reprend via `Command(resume=...)`).

### T7 : API chat SSE + threads
- Livrables : `app/api/routes/chat.py` (POST `/v1/chat` : `{message, thread_id?, user_id}` -> stream SSE `astream_events`, creation thread_id si absent), `app/api/routes/threads.py` (GET etat, POST `/v1/threads/{id}/resume`), `tests/integration/test_chat_api.py`.
- Done : `docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --wait` puis `uv run pytest -m integration tests/integration/test_chat_api.py` vert, et `./scripts/smoke.sh` exit 0.

### T8 : Observabilite OpenInference -> OTel -> Langfuse
- Livrables : `app/core/telemetry.py` (`LangChainInstrumentor` OpenInference, exporter OTLP HTTP vers `LANGFUSE_HOST/api/public/otel`, auth basic public/secret key, resource attributes service.name/version), correlation `trace_id` dans les logs structlog, propagation `user_id`/`thread_id` en attributs de trace.
- Done : compose up complet, `./scripts/smoke.sh`, puis trace agent complete (spans LLM + tool + cout) visible dans Langfuse localhost:3000. Critere automatisable partiel : `GET /api/public/traces` non vide.

### T9 : Evals DeepEval
- Livrables : `evals/datasets/golden_smoke.jsonl` (12 cas dont 3 adversariaux : demande de decision d'eligibilite, injection, PII), `golden_full.jsonl`, `evals/deepeval/conftest.py` (juge epingle via LiteLLM, temperature 0, sous-echantillonnage seede via `EVAL_SAMPLE_SIZE`), `test_agent_quality.py` (AnswerRelevancy >= 0.7, Faithfulness >= 0.8, GEval custom "refuse toute decision d'eligibilite" >= 0.9).
- Done : `EVAL_SAMPLE_SIZE=4 uv run deepeval test run evals/deepeval` vert en local avec cles.

### T10 : Ragas hors CI (squelette assume)
- Livrables : `evals/ragas/run_ragas.py` : golden set -> context_precision/context_recall sur `search_aids`, sortie markdown. Pas de gate, pas de CI.
- Done : `uv run python evals/ragas/run_ragas.py --dataset evals/datasets/golden_smoke.jsonl` produit un rapport.

### T11 : GitHub Actions
- Livrables : 4 workflows (section 5), `dependabot.yml`, `PULL_REQUEST_TEMPLATE.md`, branch protection documentee dans le README.
- Done : push d'une branche + PR de test : `ci` et `security` verts ; `eval` vert avec le label `run-evals` ; `terraform` vert.

### T12 : Terraform Scaleway (module mince)
- Livrables : tout `infra/terraform/` : RDB Postgres (note pgvector), Redis manage, 3 Serverless Containers (app, litellm image ghcr.io/berriai/litellm, langfuse-web), secrets via variables sensibles, `dev.tfvars.example`. Pas de state distant par defaut (documente).
- Done : `terraform -chdir=infra/terraform init -backend=false && terraform -chdir=infra/terraform validate && terraform fmt -check -recursive infra/terraform` (en CI, terraform absent en local).

### T13 : Docs finales
- Livrables : README complet (badges, mermaid, quickstart, tableau model groups, section FinOps et Responsible AI), `docs/architecture.md`, 4 ADRs, section "limites connues / v1 scope".
- Done : le quickstart du README fonctionne tel quel sur machine propre.

## 3. Docker Compose et config LiteLLM

### 3.1 docker-compose.yml

```yaml
name: llm-platform

include:
  - docker-compose.langfuse.yml

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-platform}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-platform}
      POSTGRES_MULTIPLE_DATABASES: app,litellm,langfuse
    volumes:
      - ./scripts/init-db.sh:/docker-entrypoint-initdb.d/init-db.sh:ro
      - pgdata:/var/lib/postgresql/data
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d app"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  litellm:
    image: ghcr.io/berriai/litellm:main-stable
    command: ["--config", "/etc/litellm/config.yaml", "--port", "4000"]
    volumes:
      - ./gateway/litellm.config.yaml:/etc/litellm/config.yaml:ro
    environment:
      LITELLM_MASTER_KEY: ${LITELLM_MASTER_KEY}
      DATABASE_URL: postgresql://${POSTGRES_USER:-platform}:${POSTGRES_PASSWORD:-platform}@postgres:5432/litellm
      REDIS_HOST: redis
      SCALEWAY_API_KEY: ${SCALEWAY_API_KEY:-}
      MISTRAL_API_KEY: ${MISTRAL_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
    ports: ["4000:4000"]
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:4000/health/liveliness || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 12

  app:
    build: .
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-platform}:${POSTGRES_PASSWORD:-platform}@postgres:5432/app
      LITELLM_BASE_URL: http://litellm:4000/v1
      LITELLM_API_KEY: ${LITELLM_MASTER_KEY}
      LANGFUSE_HOST: http://langfuse-web:3000
      LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:-}
      LANGFUSE_SECRET_KEY: ${LANGFUSE_SECRET_KEY:-}
      APP_API_KEY: ${APP_API_KEY}
    ports: ["8000:8000"]
    depends_on:
      postgres: { condition: service_healthy }
      litellm: { condition: service_healthy }
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:8000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 12

volumes:
  pgdata:
```

### 3.2 docker-compose.langfuse.yml (Langfuse v3 self-host)

| Service | Image | Healthcheck |
|---|---|---|
| langfuse-web | langfuse/langfuse:3 | wget /api/public/health |
| langfuse-worker | langfuse/langfuse-worker:3 | interne (pas expose) |
| clickhouse | clickhouse/clickhouse-server:24 | wget localhost:8123/ping |
| minio | minio/minio | curl :9000/minio/health/live |

langfuse-web et worker pointent sur la base `langfuse` du Postgres partage, sur `redis` partage (DB Redis 1 pour eviter la collision avec le cache LiteLLM), ClickHouse et MinIO. Variables `LANGFUSE_INIT_ORG_ID`, `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`/`SECRET_KEY` pour provisionner les cles au premier boot.

docker-compose.ci.yml : override qui remplace le volume config LiteLLM par `gateway/litellm.ci.yaml` (chaque model group avec `mock_response: "Reponse mock CI"`) et neutralise Langfuse (profil manual). Aucune cle API necessaire en CI d'integration.

### 3.3 gateway/litellm.config.yaml

```yaml
model_list:
  # sovereign-cheap : Scaleway Generative APIs (OpenAI-compatible, fr-par)
  - model_name: sovereign-cheap
    litellm_params:
      model: openai/llama-3.3-70b-instruct          # verifier le catalogue au build (rotation rapide)
      api_base: https://api.scaleway.ai/v1
      api_key: os.environ/SCALEWAY_API_KEY
  - model_name: sovereign-cheap
    litellm_params:
      model: openai/mistral-small-3.2-24b-instruct-2506
      api_base: https://api.scaleway.ai/v1
      api_key: os.environ/SCALEWAY_API_KEY

  # sovereign-premium : Mistral La Plateforme
  - model_name: sovereign-premium
    litellm_params:
      model: mistral/mistral-large-latest
      api_key: os.environ/MISTRAL_API_KEY

  # frontier : Anthropic (escalade documentee, hors PII)
  - model_name: frontier
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY

  # embeddings souverains (Store pgvector + seed)
  - model_name: sovereign-embed
    litellm_params:
      model: openai/bge-multilingual-gemma2
      api_base: https://api.scaleway.ai/v1
      api_key: os.environ/SCALEWAY_API_KEY

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 2
  timeout: 60
  fallbacks:
    - sovereign-cheap: ["sovereign-premium"]
    - sovereign-premium: ["frontier"]
  redis_host: os.environ/REDIS_HOST
  redis_port: 6379

litellm_settings:
  drop_params: true
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: 6379
    ttl: 300

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  store_model_in_db: false
```

Points d'entretien encodes ici : deux deploiements physiques derriere un meme alias logique (sovereign-cheap), fallback inter-groupes = cascade de souverainete (Scaleway -> Mistral -> Anthropic en dernier recours), spend Postgres consultable via /spend et l'UI LiteLLM.

## 4. Strategie de test

### 4.1 Unitaires (sans LLM, sans reseau, < 10 s au total)
- Marqueur : defaut, `pytest -m "not integration"` en CI.
- LLM : `GenericFakeChatModel` / `FakeListChatModel` de `langchain_core.language_models.fake_chat_models`, y compris `tool_calls` scriptes. HTTP mocke avec respx pour la couche tenacity (429 puis 200, verifier la bascule de groupe).
- Graphe : compile avec `InMemorySaver`, assertions topologie (`graph.get_graph().nodes`), flux (fake LLM demande get_aid_details -> `__interrupt__` -> `Command(resume={"approved": True})` -> reponse finale), guardrails (disclaimer present, "ai-je droit au RSA ?" declenche la reformulation orientation).
- Guardrails : table-driven (parametres pytest) : NIR valide/invalide, IBAN FR, injections connues.
- Registry : validation Pydantic, semver, refus d'un tool non enregistre.

### 4.2 Integration (-m integration, necessite compose)
- Prerequis : `docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d --wait`. LiteLLM en mock_response : zero token, zero cle.
- test_memory.py : deux tours meme thread_id -> historique visible ; store.aput puis store.asearch (embeddings mockes ou dimension fixe pour ne pas dependre du provider).
- test_chat_api.py : POST /v1/chat -> SSE bien forme, thread_id retourne, 401 sans API key, flux resume HITL complet.

### 4.3 DeepEval en CI sans exploser le budget
- Sous-echantillonnage PR : EVAL_SAMPLE_SIZE=12 (smoke set entier), selection deterministe (seed = numero de PR) si echantillonnage du full set. Nightly : golden_full.jsonl complet (~50 cas).
- Juge epingle : ID de modele date (ex. claude-sonnet-4-5-20250929), temperature=0, appele via LiteLLM (spend du juge tracke = argument FinOps). Jamais de latest.
- Seuils calibres : premiere execution sur main = baseline commitee, seuils sous la baseline (marge de bruit du juge). Documente dans ADR 0003.
- Cache DeepEval : `deepeval test run -c`.
- Budget estime : 12 cas x (1 generation + 3 metriques juge) = quelques dizaines de milliers de tokens par PR. Garde-fou : timeout-minutes: 15.
- Securite secrets : le job eval ne tourne jamais sur les PR de forks (condition sur head.repo + label run-evals).

### 4.4 Ragas
Hors CI, script manuel pour le tuning retriever. Assume squelette en v1.

## 5. Workflows GitHub Actions

### ci.yml
- on: push main + pull_request, concurrency cancel-in-progress.
- jobs : lint (astral-sh/setup-uv -> uv sync --frozen -> ruff check + ruff format --check), typecheck (mypy app tests), test (pytest -m "not integration" --cov=app --cov-fail-under=80), integration (compose CI up --wait -> pytest -m integration, logs en cas d'echec). lint/typecheck/test en parallele, integration apres test.

### security.yml
- on push + PR : actions/checkout fetch-depth 0 -> gitleaks/gitleaks-action@v2 (historique complet).

### eval.yml
- on : PR (types labeled, synchronize) + schedule nightly 3h + workflow_dispatch.
- Condition : label run-evals ou event != PR. timeout-minutes: 15. EVAL_SAMPLE_SIZE=12 en PR, 0 (full) en nightly. compose up litellm reel -> deepeval test run -c.

### terraform.yml
- on : PR paths infra/terraform/**. hashicorp/setup-terraform -> fmt -check -recursive -> init -backend=false -> validate -> tflint (soft fail).

## 6. Volontairement stubbe ou minimal en v1 (a assumer en entretien)

1. Auth : API key statique en header, pas d'OAuth/JWT ni multi-tenant. Rate limiting delegue a LiteLLM.
2. Guardrails : regex + heuristiques + disclaimer, pas de Llama Guard ni NeMo ni Presidio complet. Architecture en couches prete ; brancher Presidio = iteration connue.
3. Ragas : squelette fonctionnel, pas de TestsetGenerator branche.
4. RAG : recherche semantique naive dans le Store pgvector, pas d'hybrid BM25+RRF ni reranking. Perimetre de finley-2, dit dans le README.
5. Un seul Postgres, trois bases : trade-off demo assume (ADR 0004).
6. Terraform : validate en CI seulement, pas de plan/apply automatise ni state distant.
7. Langfuse : self-host brut, pas de dashboards ni evaluators online provisionnes.
8. Pas de multi-agent (single agent = choix mature, munition 15x tokens), pas de Temporal, pas de semantic cache applicatif.
9. Corpus : 8 aides fictives inventees, aucune donnee reelle.

## 7. MVP premiere session

Sous-ensemble : T0 complet + T1 complet + T2 + T3 (retry simple si la cascade complete ne rentre pas) + T6 reduit (graphe agent+tools sans HITL ni guard_output, tool search_aids sur lookup JSON en memoire, pas encore le Store) + T7 reduit (POST /v1/chat SSE simple, InMemorySaver acceptable en fin de session, sinon AsyncPostgresSaver direct) + T8 (telemetrie) + ci.yml et security.yml.

Sequence : T0 -> (T1 en parallele de T2) -> T3 -> T6 reduit -> T7 reduit -> T8 -> CI -> premier push public -> workflows verts.

Critere de reussite MVP :
```bash
cp .env.example .env   # + renseigner une cle reelle au minimum pour le test manuel
docker compose up -d --wait
curl -s -X POST localhost:8000/v1/chat -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Quelles aides existent pour un jeune de moins de 25 ans ?"}'
# -> reponse generee via LiteLLM (sovereign-cheap), trace visible sur localhost:3000
```

Sessions suivantes : T4+T5, T6/T7 complets (HITL, guardrails), T9+eval.yml (differenciateur entretien n°1), T12+terraform.yml, T10+T13.
