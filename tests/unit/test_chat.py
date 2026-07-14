"""Unit tests for the SSE chat endpoint (app.api.routes.chat).

The pure event-building/formatting functions (`build_token_event`,
`build_done_event`, `format_sse_event`, `_extract_final_text`) are tested
directly, no app, no network. The route itself is exercised through
`TestClient` with the compiled agent graph dependency (`get_agent_graph`)
overridden by a small async fake, so these tests never touch the network,
a real LLM, or a real checkpointer.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.llm import LLMCascadeExhaustedError, ModelGroup
from app.api.deps import get_agent_graph
from app.api.routes.chat import (
    _extract_final_text,
    _stream_chat_events,
    build_done_event,
    build_error_event,
    build_token_event,
    format_sse_event,
)


class TestBuildTokenEvent:
    def test_wraps_content_with_type_token(self) -> None:
        assert build_token_event("Bonjour") == {"type": "token", "content": "Bonjour"}


class TestBuildDoneEvent:
    def test_wraps_thread_id_and_full_content(self) -> None:
        event = build_done_event("thread-abc", "Full answer.")

        assert event == {"type": "done", "thread_id": "thread-abc", "content": "Full answer."}


class TestFormatSseEvent:
    def test_formats_as_a_data_line_with_a_trailing_blank_line(self) -> None:
        line = format_sse_event({"type": "token", "content": "hi"})

        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line.removeprefix("data: ").strip())
        assert payload == {"type": "token", "content": "hi"}

    def test_round_trips_non_ascii_content(self) -> None:
        line = format_sse_event({"type": "token", "content": "éàçù"})

        payload = json.loads(line.removeprefix("data: ").strip())
        assert payload["content"] == "éàçù"


class TestBuildErrorEvent:
    def test_wraps_message_with_type_error(self) -> None:
        assert build_error_event("the request timed out") == {
            "type": "error",
            "message": "the request timed out",
        }


class TestExtractFinalText:
    def test_returns_the_last_ai_message_content(self) -> None:
        state_values = {
            "messages": [
                HumanMessage(content="Question ?"),
                AIMessage(content="Premiere reponse."),
                AIMessage(content="Reponse finale."),
            ]
        }

        assert _extract_final_text(state_values) == "Reponse finale."

    def test_returns_empty_string_when_no_ai_message_present(self) -> None:
        assert _extract_final_text({"messages": [HumanMessage(content="Question ?")]}) == ""

    def test_returns_empty_string_when_messages_key_is_missing(self) -> None:
        assert _extract_final_text({}) == ""


class _FakeStateSnapshot:
    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = values


class _FakeGraph:
    """Minimal async stand-in for a compiled LangGraph graph.

    Yields a couple of `on_chat_model_stream` events (proving the token
    path is wired end to end), then `aget_state` returns a final `AIMessage`
    so `/v1/chat` always has a full answer for the `done` event, matching
    the real graph's contract even when a provider streams coarsely or not
    at all (see chat.py module docstring on the mock-LiteLLM case).
    """

    def __init__(self, final_text: str = "Reponse complete.") -> None:
        self.final_text = final_text
        self.received_configs: list[Any] = []
        self.received_inputs: list[Any] = []

    async def astream_events(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,  # noqa: A002
    ) -> AsyncIterator[dict[str, Any]]:
        self.received_configs.append(config)
        self.received_inputs.append(input)
        for chunk_text in ("Repon", "se comp", "lete."):
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessage(content=chunk_text)},
            }

    async def aget_state(self, config: Any, **kwargs: Any) -> _FakeStateSnapshot:
        return _FakeStateSnapshot(
            {
                "messages": [
                    HumanMessage(content="hi"),
                    AIMessage(content=self.final_text),
                ]
            }
        )


def _parse_sse_events(raw: str) -> list[dict[str, Any]]:
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: ")
        events.append(json.loads(block.removeprefix("data: ")))
    return events


def test_missing_api_key_returns_401(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"message": "Bonjour"})

    assert response.status_code == 401


def test_wrong_api_key_returns_401(client: TestClient) -> None:
    response = client.post(
        "/v1/chat", json={"message": "Bonjour"}, headers={"X-API-Key": "wrong-key"}
    )

    assert response.status_code == 401


def test_valid_request_streams_sse_ending_with_a_done_event(
    app: FastAPI, client: TestClient
) -> None:
    fake_graph = _FakeGraph(final_text="Voici les aides disponibles.")
    app.dependency_overrides[get_agent_graph] = lambda: fake_graph

    response = client.post(
        "/v1/chat",
        json={"message": "Quelles aides pour un jeune ?"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert any(event["type"] == "token" for event in events)
    done_event = events[-1]
    assert done_event["type"] == "done"
    assert done_event["content"] == "Voici les aides disponibles."
    # thread_id was not supplied: the route must have generated a fresh uuid4.
    assert uuid.UUID(done_event["thread_id"]).version == 4


def test_explicit_thread_id_is_echoed_back_and_passed_to_the_graph(
    app: FastAPI, client: TestClient
) -> None:
    fake_graph = _FakeGraph()
    app.dependency_overrides[get_agent_graph] = lambda: fake_graph
    thread_id = str(uuid.uuid4())

    response = client.post(
        "/v1/chat",
        json={"message": "Bonjour", "thread_id": thread_id},
        headers={"X-API-Key": "test-api-key"},
    )

    events = _parse_sse_events(response.text)
    assert events[-1]["thread_id"] == thread_id
    assert fake_graph.received_configs[0]["configurable"]["thread_id"] == thread_id


def test_non_uuid_thread_id_is_rejected_with_422(client: TestClient) -> None:
    response = client.post(
        "/v1/chat",
        json={"message": "Bonjour", "thread_id": "my-thread-1"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 422


def test_missing_user_id_defaults_to_a_placeholder(app: FastAPI, client: TestClient) -> None:
    fake_graph = _FakeGraph()
    app.dependency_overrides[get_agent_graph] = lambda: fake_graph

    client.post(
        "/v1/chat",
        json={"message": "Bonjour"},
        headers={"X-API-Key": "test-api-key"},
    )

    sent_state = fake_graph.received_inputs[0]
    assert sent_state["user_id"]


def test_blank_message_is_rejected_with_422(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"message": ""}, headers={"X-API-Key": "test-api-key"})

    assert response.status_code == 422


class _RaisingGraph:
    """Fake graph whose `astream_events` raises partway (or immediately)."""

    def __init__(self, exc: Exception, tokens_before_failure: int = 0) -> None:
        self.exc = exc
        self.tokens_before_failure = tokens_before_failure
        self.aget_state_was_called = False

    async def astream_events(
        self,
        input: Any,  # noqa: A002
        config: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        for _ in range(self.tokens_before_failure):
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessage(content="partial ")},
            }
        raise self.exc

    async def aget_state(self, config: Any, **kwargs: Any) -> Any:
        # Must never be reached: the exception above should short-circuit
        # the generator straight to the error-handling branch.
        self.aget_state_was_called = True
        raise AssertionError("aget_state should not be called after astream_events raised")


class _HangingGraph:
    """Fake graph whose `astream_events` never completes (simulates a stuck upstream)."""

    def __init__(self, hang_seconds: float = 999.0) -> None:
        self.hang_seconds = hang_seconds
        self.aget_state_was_called = False

    async def astream_events(
        self,
        input: Any,  # noqa: A002
        config: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        await asyncio.sleep(self.hang_seconds)
        yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="never")}}

    async def aget_state(self, config: Any, **kwargs: Any) -> Any:
        self.aget_state_was_called = True
        raise AssertionError("aget_state should not be called after a stream timeout")


async def _collect(agen: AsyncIterator[str]) -> list[dict[str, Any]]:
    lines = [line async for line in agen]
    return _parse_sse_events("".join(lines))


class TestStreamChatEventsErrorHandling:
    """Direct generator-level tests: no HTTP, no app, matching the pure-function
    tests above (`build_token_event` etc.) but for `_stream_chat_events`'s
    error/timeout boundary, which is not otherwise reachable without either a
    real graph or an over-engineered dependency override."""

    async def test_internal_exception_yields_a_generic_error_event_and_does_not_raise(
        self,
    ) -> None:
        fake_graph = _RaisingGraph(RuntimeError("secret-internal-detail"), tokens_before_failure=1)

        events = await _collect(
            _stream_chat_events(
                fake_graph,
                message="Bonjour",
                thread_id="thread-1",
                user_id="user-1",
                timeout_seconds=5.0,
            )
        )

        assert events[-1] == {
            "type": "error",
            "message": "an internal error occurred, please retry",
        }
        assert not any(event.get("type") == "done" for event in events)

    async def test_llm_cascade_exhausted_yields_the_temporarily_unavailable_message(
        self,
    ) -> None:
        cascade_error = LLMCascadeExhaustedError(
            groups_tried=[ModelGroup.SOVEREIGN_CHEAP, ModelGroup.SOVEREIGN_PREMIUM],
            last_error=RuntimeError("gateway unreachable"),
        )
        fake_graph = _RaisingGraph(cascade_error)

        events = await _collect(
            _stream_chat_events(
                fake_graph,
                message="Bonjour",
                thread_id="thread-1",
                user_id="user-1",
                timeout_seconds=5.0,
            )
        )

        assert events == [
            {
                "type": "error",
                "message": "the assistant is temporarily unavailable, please retry",
            }
        ]

    async def test_hanging_graph_times_out_and_yields_a_timeout_error_event(self) -> None:
        fake_graph = _HangingGraph(hang_seconds=999.0)

        events = await _collect(
            _stream_chat_events(
                fake_graph,
                message="Bonjour",
                thread_id="thread-1",
                user_id="user-1",
                timeout_seconds=0.05,
            )
        )

        assert events == [{"type": "error", "message": "the request timed out"}]
        assert not fake_graph.aget_state_was_called

    async def test_internal_exception_text_never_leaks_into_the_sse_stream(self) -> None:
        marker = "secret-internal-detail"
        fake_graph = _RaisingGraph(RuntimeError(marker))

        raw_lines = [
            line
            async for line in _stream_chat_events(
                fake_graph,
                message="Bonjour",
                thread_id="thread-1",
                user_id="user-1",
                timeout_seconds=5.0,
            )
        ]

        assert marker not in "".join(raw_lines)


@pytest.fixture(autouse=True)
def _clear_graph_override(app: FastAPI) -> Iterator[None]:
    yield
    app.dependency_overrides.pop(get_agent_graph, None)
