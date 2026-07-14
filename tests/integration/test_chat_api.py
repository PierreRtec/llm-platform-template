"""Integration tests for `POST /v1/chat` against the real, running stack.

Requires `docker compose -f docker-compose.yml -f docker-compose.ci.yml up
-d --wait` (LiteLLM in `mock_response` mode: zero provider API keys, zero
tokens spent, see gateway/litellm.ci.yaml). These tests talk to the `app`
service over HTTP exactly as a real client would; no `TestClient`, no
dependency overrides, no fakes: this is what proves the whole stack (app,
litellm, postgres, redis) actually works together, including the async
execution path the MVP agent node bridges through
(`app.agent.graph._FallbackAgentLLM`, see that class's docstring on why the
sync-vs-async bridge is a real risk under FastAPI).
"""

import json
import os
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration

APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8000")
APP_API_KEY = os.environ.get("APP_API_KEY", "changeme-app-api-key")


def _parse_sse_events(raw: str) -> list[dict[str, Any]]:
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"unexpected non-SSE line: {block!r}"
        events.append(json.loads(block.removeprefix("data: ")))
    return events


@pytest.fixture
def http_client() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=APP_BASE_URL, timeout=30.0) as client:
        yield client


def test_chat_without_api_key_is_rejected(http_client: httpx.Client) -> None:
    response = http_client.post("/v1/chat", json={"message": "Bonjour"})

    assert response.status_code == 401


def test_chat_with_api_key_streams_sse_with_a_done_event_and_thread_id(
    http_client: httpx.Client,
) -> None:
    response = http_client.post(
        "/v1/chat",
        headers={"X-API-Key": APP_API_KEY},
        json={"message": "Quelles aides existent pour un jeune de moins de 25 ans ?"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    assert events, "expected at least one SSE event"

    done_events = [event for event in events if event.get("type") == "done"]
    assert len(done_events) == 1, f"expected exactly one 'done' event, got: {events}"

    done_event = done_events[0]
    assert done_event["content"], "the 'done' event must carry the full generated answer"
    # thread_id was not supplied: the API must generate a valid uuid4.
    assert uuid.UUID(done_event["thread_id"]).version == 4


def test_second_turn_on_the_same_thread_id_succeeds(http_client: httpx.Client) -> None:
    """Conversational memory smoke check: same thread_id across two turns.

    With LiteLLM in `mock_response` mode the reply text is a fixed string,
    so this cannot assert the model actually "remembered" anything: it
    verifies the technical contract instead (second turn on the same
    thread_id is accepted and returns that same thread_id back), which is
    what T7 owns. Content-level memory assertions belong to
    tests/integration/test_memory.py (T4) once a real checkpointer/LLM pair
    can express them.
    """
    first_response = http_client.post(
        "/v1/chat",
        headers={"X-API-Key": APP_API_KEY},
        json={"message": "Quelles aides existent pour un jeune actif ?"},
    )
    assert first_response.status_code == 200
    first_done = [e for e in _parse_sse_events(first_response.text) if e.get("type") == "done"][0]
    thread_id = first_done["thread_id"]

    second_response = http_client.post(
        "/v1/chat",
        headers={"X-API-Key": APP_API_KEY},
        json={"message": "Et pour le logement ?", "thread_id": thread_id},
    )

    assert second_response.status_code == 200
    second_done = [e for e in _parse_sse_events(second_response.text) if e.get("type") == "done"][0]
    assert second_done["thread_id"] == thread_id
    assert second_done["content"]


def test_chat_response_is_the_happy_path_technical_contract(http_client: httpx.Client) -> None:
    """Documents the MVP contract explicitly (design doc section 4.2/7).

    In mock mode LiteLLM always answers the same fixed string regardless of
    the question asked, so this cannot check the answer *matches* the
    corpus (e.g. a specific aid for "under 25"). What it does check is the
    full technical round trip an eval/manual test would rely on later:
    a 200, a well-formed `done` event, non-empty content, and a thread_id
    usable for a follow-up turn.
    """
    response = http_client.post(
        "/v1/chat",
        headers={"X-API-Key": APP_API_KEY},
        json={"message": "Quelles aides existent pour un jeune de moins de 25 ans ?"},
    )

    assert response.status_code == 200
    done_event = [e for e in _parse_sse_events(response.text) if e.get("type") == "done"][0]
    assert isinstance(done_event["content"], str)
    assert len(done_event["content"]) > 0
    assert isinstance(done_event["thread_id"], str)
    assert len(done_event["thread_id"]) > 0
