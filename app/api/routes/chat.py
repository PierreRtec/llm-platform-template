"""POST /v1/chat: SSE-streamed chat over the compiled agent graph.

MVP scope (docs/DESIGN.md section 7 / task T7 reduced): only the chat endpoint
itself. `GET /v1/threads/{id}` (checkpointer state) and
`POST /v1/threads/{id}/resume` (HITL resume) are out of scope here; they
land once the full graph (interrupt before `requires_approval` tools) and
`ToolRegistry` exist.

Streaming contract: every SSE line is `data: <json>\\n\\n`. Three event
shapes:

- `{"type": "token", "content": <str>}`: one streamed LLM chunk, emitted for
  every `on_chat_model_stream` event `astream_events` surfaces from the
  agent node's LLM call.
- `{"type": "done", "thread_id": <str>, "content": <str>}`: the terminal
  event on success, `content` is the full final answer (not a concatenation
  of the token chunks above).
- `{"type": "error", "message": <str>}`: the terminal event on failure
  (an unhandled exception while streaming, or the stream exceeding
  `Settings.CHAT_STREAM_TIMEOUT_SECONDS`). Emitted instead of, never
  alongside, a `done` event. `message` is always one of a small set of
  generic, user-safe strings (never raw exception text: the exception is
  logged server-side via structlog and never leaves the process). The
  generator ends cleanly right after, no exception propagates out of it.

The `done` event's `content` is read back from the compiled graph's own
state after streaming finishes (`graph.aget_state`), not accumulated from
`token` events. This is deliberate, not a shortcut: the MVP agent node
(`app.agent.graph._FallbackAgentLLM`) bridges to
`app.agent.llm.ainvoke_with_fallback`, a plain `ainvoke` call rather than a
real token-by-token `.astream()` against the gateway, so with the CI
mock-LiteLLM setup (and possibly some real providers) zero
`on_chat_model_stream` events may ever fire. The docs/DESIGN.md explicitly
accepts this (token-by-token streaming can end up coarse-grained): the
minimal contract is at least one `done` event carrying the complete answer
and the thread id, which reading the graph's own post-run state guarantees
regardless of how granular (or absent) the upstream token stream was.

Memory note: the checkpointer wired in the lifespan (`app/main.py`) is an
`InMemorySaver`, per-process only, lost on restart, not shared across
replicas. `AsyncPostgresSaver` (T4) replaces it without changing this
route's contract.

Telemetry (T8): `_stream_chat_events` opens one OTel span (`chat_request`)
around the whole graph invocation, tagged with `user_id`, `thread_id`, and
the system prompt's version/hash (`app/agent/prompts.py`). This is the
request's own span, not one `LangChainInstrumentor` creates: OpenInference's
LangChain tracer (`openinference.instrumentation.langchain._tracer`)
deliberately never attaches its spans to OTel's ambient/"current" context
(its own comment: doing so "can be hazardous" if a callback never fires to
detach it), so `opentelemetry.trace.get_current_span()` would never see one
of its spans from here regardless of timing. It *does* parent a run's span
on the ambient current span when that run has no LangChain-level parent
(the graph invocation's own root run) via `context.get_current()`, so
opening `chat_request` as the current span here still nests every
LangChain/tool span the graph produces underneath it, in the same trace.
The span's own trace id is what `app.core.logging.bind_trace_id` binds into
structlog for the duration (a no-op when telemetry itself is disabled: the
span is then OTel's default no-op span, `get_span_context().is_valid` is
`False`, and no trace id is bound). `user_id`/`thread_id`/prompt version and
hash are *also* passed as LangChain `RunnableConfig` metadata below, which
`LangChainInstrumentor` attaches to every nested span it creates for this
run (`thread_id` becomes the `session.id` attribute Langfuse groups traces
by; the rest lands in each span's `metadata` attribute). The two
mechanisms are complementary, not redundant: one span (ours) vs. every span
(theirs).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import ExitStack
from typing import Any, Final

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace
from pydantic import BaseModel, Field, field_validator

from app.agent.llm import LLMCascadeExhaustedError
from app.agent.prompts import PROMPT_VERSION, system_prompt_hash
from app.agent.state import AgentState
from app.api.deps import AgentGraph, get_agent_graph, verify_api_key
from app.core.config import Settings, get_settings
from app.core.logging import bind_trace_id

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Generic, user-safe error messages for the `error` SSE event. Never the raw
# exception text: that is only ever logged server-side (see
# `_stream_chat_events`).
_CASCADE_EXHAUSTED_MESSAGE: Final[str] = "the assistant is temporarily unavailable, please retry"
_TIMEOUT_MESSAGE: Final[str] = "the request timed out"
_INTERNAL_ERROR_MESSAGE: Final[str] = "an internal error occurred, please retry"

router = APIRouter(
    prefix="/v1",
    tags=["chat"],
    dependencies=[Depends(verify_api_key)],
)

DEFAULT_USER_ID: Final[str] = "anonymous"


class ChatRequest(BaseModel):
    """Body of `POST /v1/chat`."""

    message: str = Field(..., min_length=1, description="The user's message, in French.")
    thread_id: str | None = Field(
        default=None,
        description="Existing conversation thread to continue. A fresh uuid4 is "
        "generated and returned when omitted.",
    )
    user_id: str | None = Field(default=None, description="Caller-supplied user identifier.")

    @field_validator("thread_id")
    @classmethod
    def _thread_id_must_be_a_valid_uuid(cls, value: str | None) -> str | None:
        """Reject a client-supplied thread_id that is not a valid UUID (422).

        Debt (T4): this only validates *shape*. Once the checkpointer is
        shared (Postgres, T4) rather than per-process `InMemorySaver`, add an
        ownership check (does this thread_id actually belong to user_id)
        before letting a request read or extend that thread's state.
        """
        if value is None:
            return value
        try:
            uuid.UUID(value)
        except ValueError as exc:
            raise ValueError("thread_id must be a valid UUID") from exc
        return value


def build_token_event(content: str) -> dict[str, str]:
    """SSE payload for one streamed LLM chunk."""
    return {"type": "token", "content": content}


def build_done_event(thread_id: str, content: str) -> dict[str, str]:
    """SSE payload for the terminal event on success: full answer plus thread id."""
    return {"type": "done", "thread_id": thread_id, "content": content}


def build_error_event(message: str) -> dict[str, str]:
    """SSE payload for the terminal event on failure.

    `message` must always be a generic, user-safe string, never raw
    exception text (see the module docstring's streaming contract).
    """
    return {"type": "error", "message": message}


def format_sse_event(event: Mapping[str, str]) -> str:
    """Format one event mapping as a single `data: <json>\\n\\n` SSE line."""
    return f"data: {json.dumps(dict(event))}\n\n"


def _extract_final_text(state_values: Mapping[str, Any]) -> str:
    """Return the last `AIMessage`'s text content from a graph state snapshot.

    Empty string if there are no messages or no `AIMessage` among them (e.g.
    guard_input blocked the input before the agent ever ran... though
    guard_input's own refusal is itself an `AIMessage`, see
    `app/agent/graph.py::guard_input`, so this only stays empty in
    genuinely degenerate cases).
    """
    messages = state_values.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _classify_error(exc: Exception) -> str:
    """Map an exception to the generic, user-safe `error` event message."""
    if isinstance(exc, LLMCascadeExhaustedError):
        return _CASCADE_EXHAUSTED_MESSAGE
    if isinstance(exc, TimeoutError):
        return _TIMEOUT_MESSAGE
    return _INTERNAL_ERROR_MESSAGE


async def _stream_chat_events(
    graph: AgentGraph,
    *,
    message: str,
    thread_id: str,
    user_id: str,
    timeout_seconds: float,
) -> AsyncIterator[str]:
    """Drive one graph invocation, yielding SSE lines, ending with `done` or `error`.

    The whole invocation (the `astream_events` consumption plus the closing
    `aget_state` call) is bounded by `timeout_seconds` of total wall-clock
    time: a deadline is computed once, and each step of the underlying async
    iterator is awaited with `asyncio.wait_for` against the *remaining*
    budget, not a fresh per-step timeout. This is what stops a hung graph
    (stuck tool call, unhealthy upstream that neither errors nor completes)
    from holding the SSE connection open forever.

    Any exception, whether raised by the graph itself or a timeout, is
    caught here: it is logged with the full traceback server-side
    (structlog `chat_stream_failed`), never re-raised, and turned into a
    single terminal `error` event with a generic message (see
    `_classify_error` and the module docstring). The generator always ends
    cleanly, on every path.
    """
    prompt_hash = system_prompt_hash()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        # Picked up by `LangChainInstrumentor` on every nested span it
        # creates for this run (see the module docstring): `thread_id`
        # becomes the `session.id` attribute Langfuse groups traces by, the
        # rest lands in each span's `metadata` attribute.
        "metadata": {
            "thread_id": thread_id,
            "user_id": user_id,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
        },
    }
    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        user_id=user_id,
        input_flags=[],
    )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    with tracer.start_as_current_span("chat_request") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("thread_id", thread_id)
        span.set_attribute("prompt.version", PROMPT_VERSION)
        span.set_attribute("prompt.hash", prompt_hash)

        with ExitStack() as correlation:
            span_context = span.get_span_context()
            if span_context.is_valid:
                correlation.enter_context(bind_trace_id(format(span_context.trace_id, "032x")))

            try:
                events_iter = graph.astream_events(
                    initial_state, config=config, version="v2"
                ).__aiter__()
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"chat stream exceeded CHAT_STREAM_TIMEOUT_SECONDS ({timeout_seconds}s)"
                        )
                    try:
                        event = await asyncio.wait_for(events_iter.__anext__(), timeout=remaining)
                    except StopAsyncIteration:
                        break

                    if event["event"] != "on_chat_model_stream":
                        continue
                    chunk = event["data"].get("chunk")
                    content = getattr(chunk, "content", None) if chunk is not None else None
                    if content:
                        text = content if isinstance(content, str) else str(content)
                        yield format_sse_event(build_token_event(text))

                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"chat stream exceeded CHAT_STREAM_TIMEOUT_SECONDS ({timeout_seconds}s)"
                    )
                final_state = await asyncio.wait_for(graph.aget_state(config), timeout=remaining)
                final_text = _extract_final_text(final_state.values)
                yield format_sse_event(build_done_event(thread_id, final_text))
            except Exception as exc:  # deliberately broad: this is the stream's error boundary
                logger.error(
                    "chat_stream_failed",
                    thread_id=thread_id,
                    user_id=user_id,
                    error_type=type(exc).__name__,
                    exc_info=exc,
                )
                yield format_sse_event(build_error_event(_classify_error(exc)))


@router.post("/chat")
async def chat(
    body: ChatRequest,
    graph: AgentGraph = Depends(get_agent_graph),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream a chat turn as SSE. Creates `thread_id` (uuid4) when omitted."""
    thread_id = body.thread_id or str(uuid.uuid4())
    user_id = body.user_id or DEFAULT_USER_ID

    events = _stream_chat_events(
        graph,
        message=body.message,
        thread_id=thread_id,
        user_id=user_id,
        timeout_seconds=settings.CHAT_STREAM_TIMEOUT_SECONDS,
    )
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
