"""Builds the MVP agent graph: guard_input -> agent <-> tools -> END.

MVP scope (design doc section 7): no human-in-the-loop `interrupt()` and no
`guard_output` node. The full graph (`guard_input -> agent -> tools/END`,
with an `interrupt()` before any `requires_approval` tool, then
`guard_output` before the final answer) lands once the `ToolRegistry` and a
"sensitive" tool such as `get_aid_details` exist (T5 completion). Extension
points are called out inline below.

Node/edge shape:

    guard_input --(blocked)--------------> END (polite refusal, agent never called)
    guard_input --(clear)----------------> agent
    agent       --(tool_calls, budget ok)-> tools
    agent       --(tool_calls, budget hit)-> budget_exceeded
    agent       --(no tool_calls)--------> END
    tools       --------------------------> agent (loop back for the final answer)
    budget_exceeded -----------------------> END (polite French fallback)
"""

from __future__ import annotations

import asyncio
from typing import Any, Final, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.guardrails.pipeline import BLOCKED_FLAG_PREFIX, run_input_guardrails
from app.agent.llm import ainvoke_with_fallback
from app.agent.prompts import BUDGET_EXCEEDED_MESSAGE_FR, REFUSAL_MESSAGE_FR, SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.tools.search_aids import search_aids
from app.core.config import Settings, get_settings

# Registered tools available to the agent node. MVP scope: `search_aids`
# only. `get_aid_details` (a deliberately "sensitive" tool requiring HITL
# approval) arrives with the full `ToolRegistry` (T5); at that point this
# list is replaced by `ToolRegistry.enabled_tools()` so the graph refuses to
# call anything not declared there (CLAUDE.md style note).
TOOLS: tuple[BaseTool, ...] = (search_aids,)

# Hard cap on agent <-> tools round trips per invocation. LangGraph's own
# default recursion limit (~25 super-steps) would eventually raise a raw
# `GraphRecursionError` if the model kept requesting tools forever; this
# constant lets `route_after_agent` intercept that failure mode well before
# the recursion limit and end the graph on a polite message instead.
MAX_TOOL_ROUNDS: Final[int] = 5

_LlmRunnable = Runnable[LanguageModelInput, BaseMessage]


def _to_messages(model_input: LanguageModelInput) -> list[BaseMessage]:
    """Normalize a `LanguageModelInput` into a plain `list[BaseMessage]`.

    Mirrors `BaseChatModel._convert_input`'s cases (`PromptValue`, `str`, or a
    message sequence) rather than assuming the input shape, even though in
    this module the only caller (`call_agent`) always passes a message list.
    """
    if isinstance(model_input, PromptValue):
        return model_input.to_messages()
    if isinstance(model_input, str):
        return convert_to_messages([model_input])
    return convert_to_messages(model_input)


class _FallbackAgentLLM(Runnable[LanguageModelInput, BaseMessage]):
    """Routes the default agent node through the sovereignty fallback cascade.

    LangGraph's synchronous `.invoke()` entry point cannot execute a
    coroutine node function directly (`langgraph.pregel` raises `RuntimeError:
    In an sync context async tasks cannot be called`), so `call_agent` itself
    must stay a plain sync function calling `resolved_llm.invoke(...)`. This
    adapter is where the bridge to the async `ainvoke_with_fallback` (T3)
    cascade lives instead: `invoke` runs the coroutine via `asyncio.run`
    (safe here, there is no already-running loop in the calling thread,
    whether that thread is the main thread for a sync `.invoke()` graph call
    or a worker thread LangGraph offloads sync nodes to under `.ainvoke()`),
    while `ainvoke` is native for callers that are already async.
    """

    def __init__(self, settings: Settings, tools: tuple[BaseTool, ...]) -> None:
        self._settings = settings
        self._tools = tools

    async def ainvoke(
        self,
        input: LanguageModelInput,  # noqa: A002 - matches Runnable.ainvoke's signature
        config: RunnableConfig | None = None,
        **kwargs: object,
    ) -> BaseMessage:
        messages = _to_messages(input)
        return await ainvoke_with_fallback(messages, settings=self._settings, tools=self._tools)

    def invoke(
        self,
        input: LanguageModelInput,  # noqa: A002 - matches Runnable.invoke's signature
        config: RunnableConfig | None = None,
        **kwargs: object,
    ) -> BaseMessage:
        return asyncio.run(self.ainvoke(input, config, **kwargs))


def _default_llm() -> _LlmRunnable:
    """Build the default LLM: the sovereignty fallback cascade bound to tools.

    Uses `ainvoke_with_fallback` (T3) instead of a single
    `get_llm(...).bind_tools(...)` client so the agent node benefits from the
    same retry/escalation cascade as any other cascade call site, rather than
    a bare `max_retries=0` client with no resilience above the LiteLLM
    router's own `num_retries`/`fallbacks`.
    """
    settings = get_settings()
    return _FallbackAgentLLM(settings, TOOLS)


def guard_input(state: AgentState) -> dict[str, object]:
    """Run input guardrails on the latest message and update `input_flags`.

    On a blocking failure (length or injection), also appends a polite
    French refusal `AIMessage` directly: `route_after_guard_input` then
    sends the graph straight to `END`, so the agent/LLM is never invoked for
    a blocked input.
    """
    messages = state["messages"]
    latest_text = _latest_text(messages)
    result = run_input_guardrails(latest_text)

    update: dict[str, object] = {"input_flags": list(result.flags)}
    if not result.ok:
        update["messages"] = [AIMessage(content=REFUSAL_MESSAGE_FR)]
    return update


def _latest_text(messages: list[BaseMessage]) -> str:
    if not messages:
        return ""
    content = messages[-1].content
    return content if isinstance(content, str) else str(content)


def route_after_guard_input(state: AgentState) -> str:
    """Send blocked input straight to `END`; clear input proceeds to `agent`."""
    if any(flag.startswith(BLOCKED_FLAG_PREFIX) for flag in state["input_flags"]):
        return END
    return "agent"


def route_after_agent(state: AgentState) -> str:
    """Route the agent node's output: tool call, final answer, or budget cutoff.

    Delegates to `tools_condition` (langgraph prebuilt) for the base decision
    (does the latest `AIMessage` actually request a tool call). The
    `tool_rounds` budget only comes into play on top of a "yes" from that
    base decision, so a direct final answer is never redirected to
    `budget_exceeded`, even after `MAX_TOOL_ROUNDS` prior tool rounds.
    """
    base_route = tools_condition(cast(dict[str, Any], state))
    if base_route != "tools":
        return base_route
    if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
        return "budget_exceeded"
    return "tools"


def budget_exceeded(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Terminal node reached when the agent <-> tools loop hits `MAX_TOOL_ROUNDS`.

    Produces a polite French fallback (`BUDGET_EXCEEDED_MESSAGE_FR`, see
    `app/agent/prompts.py`) so the graph always ends cleanly on a
    conversational answer, never on a raw `GraphRecursionError`.
    """
    return {"messages": [AIMessage(content=BUDGET_EXCEEDED_MESSAGE_FR)]}


def build_graph(
    llm: _LlmRunnable | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the MVP agent graph.

    `llm` is injectable: defaults to the sovereignty fallback cascade
    (`ainvoke_with_fallback`, T3) started on `ModelGroup.SOVEREIGN_CHEAP` and
    bound to `TOOLS` (see `_default_llm`/`_FallbackAgentLLM`). Tests pass a
    `FakeMessagesListChatModel` (or similar) instead so the graph is
    exercised with zero network calls and zero real API key.

    `checkpointer` is injectable: `None` (no persistence) by default,
    `InMemorySaver` in tests, `AsyncPostgresSaver` once T4's Postgres-backed
    memory lands.

    The `tools` node wraps langgraph's `ToolNode` to also increment
    `AgentState["tool_rounds"]` on every pass; `route_after_agent` uses that
    counter (capped at `MAX_TOOL_ROUNDS`) to route to the `budget_exceeded`
    terminal node instead of looping back to `agent` forever.
    """
    resolved_llm = llm if llm is not None else _default_llm()
    tool_node = ToolNode(list(TOOLS))

    def call_agent(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = resolved_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def run_tools(state: AgentState) -> dict[str, object]:
        result: dict[str, object] = tool_node.invoke(state)
        return {**result, "tool_rounds": state.get("tool_rounds", 0) + 1}

    graph = StateGraph(AgentState)
    graph.add_node("guard_input", guard_input)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", run_tools)
    graph.add_node("budget_exceeded", budget_exceeded)

    graph.set_entry_point("guard_input")
    graph.add_conditional_edges(
        "guard_input",
        route_after_guard_input,
        {"agent": "agent", END: END},
    )
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "budget_exceeded": "budget_exceeded", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("budget_exceeded", END)

    return graph.compile(checkpointer=checkpointer)
