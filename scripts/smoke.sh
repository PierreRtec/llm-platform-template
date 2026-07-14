#!/usr/bin/env bash
# End-to-end smoke test against a running `docker compose` stack (real or
# CI/mock LiteLLM, see docker-compose.ci.yml). Two steps:
#
# 1. Poll GET /health/ready until it reports 200 (postgres/redis/litellm all
#    reachable), retrying up to 30 times, 2 seconds apart (T7 done criterion,
#    design doc section 7).
# 2. POST /v1/chat with a valid X-API-Key, parse the SSE response, and check
#    it contains a terminal `done` event with a non-empty `content` and a
#    `thread_id`.
#
# Usage: ./scripts/smoke.sh
# Requires: curl, python3 (both already required elsewhere in this repo).
# Exit codes: 0 on success, non-zero (with a clear message on stderr) on any
# failure, so this is safe to call from CI (`make smoke` / `make test-int`).

set -euo pipefail

APP_HOST="${APP_HOST:-localhost}"
APP_PORT="${APP_PORT:-8000}"
BASE_URL="http://${APP_HOST}:${APP_PORT}"
APP_API_KEY="${APP_API_KEY:?smoke.sh: APP_API_KEY must be set in the environment}"

READY_RETRIES=30
READY_INTERVAL_SECONDS=2

echo "smoke: waiting for ${BASE_URL}/health/ready ..." >&2

ready=0
for attempt in $(seq 1 "${READY_RETRIES}"); do
    if curl -fsS --max-time 10 -o /dev/null "${BASE_URL}/health/ready"; then
        ready=1
        echo "smoke: /health/ready OK (attempt ${attempt}/${READY_RETRIES})" >&2
        break
    fi
    echo "smoke: /health/ready not ready yet (attempt ${attempt}/${READY_RETRIES}), retrying in ${READY_INTERVAL_SECONDS}s ..." >&2
    sleep "${READY_INTERVAL_SECONDS}"
done

if [ "${ready}" -ne 1 ]; then
    echo "smoke: FAILED - /health/ready never returned 200 after ${READY_RETRIES} attempts" >&2
    exit 1
fi

echo "smoke: POST ${BASE_URL}/v1/chat ..." >&2

response_file="$(mktemp)"
trap 'rm -f "${response_file}"' EXIT

http_status=$(curl -sS --max-time 90 -o "${response_file}" -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/chat" \
    -H "X-API-Key: ${APP_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"message": "Quelles aides existent pour un jeune de moins de 25 ans ?"}')

if [ "${http_status}" != "200" ]; then
    echo "smoke: FAILED - POST /v1/chat returned HTTP ${http_status}" >&2
    cat "${response_file}" >&2
    exit 1
fi

if ! python3 - "${response_file}" <<'PYEOF'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    raw = handle.read()

done_events = []
for block in raw.split("\n\n"):
    block = block.strip()
    if not block or not block.startswith("data: "):
        continue
    payload = json.loads(block.removeprefix("data: "))
    if payload.get("type") == "done":
        done_events.append(payload)

if not done_events:
    print("smoke: FAILED - no 'done' event found in the SSE response", file=sys.stderr)
    print(raw, file=sys.stderr)
    sys.exit(1)

done_event = done_events[-1]
if not done_event.get("thread_id"):
    print(f"smoke: FAILED - 'done' event has no thread_id: {done_event}", file=sys.stderr)
    sys.exit(1)
if not done_event.get("content"):
    print(f"smoke: FAILED - 'done' event has empty content: {done_event}", file=sys.stderr)
    sys.exit(1)

print(f"smoke: OK - thread_id={done_event['thread_id']!r} content={done_event['content']!r}")
PYEOF
then
    exit 1
fi

echo "smoke: PASSED" >&2
