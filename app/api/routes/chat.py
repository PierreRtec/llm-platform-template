"""POST /v1/chat: SSE-streamed chat over the compiled agent graph.

MVP scope (design doc section 7 / task T7 reduced): only the chat endpoint
itself. `GET /v1/threads/{id}` (checkpointer state) and
`POST /v1/threads/{id}/resume` (HITL resume) are out of scope here; they
land once the full graph (interrupt before `requires_approval` tools) and
`ToolRegistry` exist.

Streaming contract: every SSE line is `data: <json>\\n\\n`. Two event
shapes:

- `{"type": "token", "content": <str>}`: one streamed LLM chunk, emitted for
  every `on_chat_model_stream` event `astream_events` surfaces from the
  agent node's LLM call.
- `{"type": "done", "thread_id": <str>, "content": <str>}`: always the last
  event, `content` is the full final answer (not a concatenation of the
  token chunks above).

The `done` event's `content` is read back from the compiled graph's own
state after streaming finishes (`graph.aget_state`), not accumulated from
`token` events. This is deliberate, not a shortcut: the MVP agent node
(`app.agent.graph._FallbackAgentLLM`) bridges to
`app.agent.llm.ainvoke_with_fallback`, a plain `ainvoke` call rather than a
real token-by-token `.astream()` against the gateway, so with the CI
mock-LiteLLM setup (and possibly some real providers) zero
`on_chat_model_stream` events may ever fire. The design doc explicitly
accepts this ("le streaming token par token peut etre grossier"): the
minimal contract is at least one `done` event carrying the complete answer
and the thread id, which reading the graph's own post-run state guarantees
regardless of how granular (or absent) the upstream token stream was.

Memory note: the checkpointer wired in the lifespan (`app/main.py`) is an
`InMemorySaver`, per-process only, lost on restart, not shared across
replicas. `AsyncPostgresSaver` (T4) replaces it without changing this
route's contract.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any, Final

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.api.deps import AgentGraph, get_agent_graph, verify_api_key

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


def build_token_event(content: str) -> dict[str, str]:
    """SSE payload for one streamed LLM chunk."""
    return {"type": "token", "content": content}


def build_done_event(thread_id: str, content: str) -> dict[str, str]:
    """SSE payload for the terminal event: the full answer plus the thread id."""
    return {"type": "done", "thread_id": thread_id, "content": content}


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


async def _stream_chat_events(
    graph: AgentGraph,
    *,
    message: str,
    thread_id: str,
    user_id: str,
) -> AsyncIterator[str]:
    """Drive one graph invocation, yielding SSE lines, ending with `done`."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        user_id=user_id,
        input_flags=[],
    )

    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        if event["event"] != "on_chat_model_stream":
            continue
        chunk = event["data"].get("chunk")
        content = getattr(chunk, "content", None) if chunk is not None else None
        if content:
            text = content if isinstance(content, str) else str(content)
            yield format_sse_event(build_token_event(text))

    final_state = await graph.aget_state(config)
    final_text = _extract_final_text(final_state.values)
    yield format_sse_event(build_done_event(thread_id, final_text))


@router.post("/chat")
async def chat(
    body: ChatRequest,
    graph: AgentGraph = Depends(get_agent_graph),
) -> StreamingResponse:
    """Stream a chat turn as SSE. Creates `thread_id` (uuid4) when omitted."""
    thread_id = body.thread_id or str(uuid.uuid4())
    user_id = body.user_id or DEFAULT_USER_ID

    events = _stream_chat_events(graph, message=body.message, thread_id=thread_id, user_id=user_id)
    return StreamingResponse(events, media_type="text/event-stream")
