"""Builds the MVP agent graph: guard_input -> agent <-> tools -> END.

MVP scope (design doc section 7): no human-in-the-loop `interrupt()` and no
`guard_output` node. The full graph (`guard_input -> agent -> tools/END`,
with an `interrupt()` before any `requires_approval` tool, then
`guard_output` before the final answer) lands once the `ToolRegistry` and a
"sensitive" tool such as `get_aid_details` exist (T5 completion). Extension
points are called out inline below.

Node/edge shape:

    guard_input --(blocked)--> END (polite refusal, agent never called)
    guard_input --(clear)----> agent
    agent       --(tool_calls)-> tools
    agent       --(no tool_calls)-> END
    tools       -----------------> agent (loop back for the final answer)
"""

from __future__ import annotations

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.guardrails.pipeline import BLOCKED_FLAG_PREFIX, run_input_guardrails
from app.agent.llm import ModelGroup, get_llm
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.tools.search_aids import search_aids
from app.core.config import get_settings

# Registered tools available to the agent node. MVP scope: `search_aids`
# only. `get_aid_details` (a deliberately "sensitive" tool requiring HITL
# approval) arrives with the full `ToolRegistry` (T5); at that point this
# list is replaced by `ToolRegistry.enabled_tools()` so the graph refuses to
# call anything not declared there (CLAUDE.md style note).
TOOLS: tuple[BaseTool, ...] = (search_aids,)

REFUSAL_MESSAGE_FR: str = (
    "Je ne peux pas traiter cette demande telle quelle. Reformulez votre question sur les "
    "aides financieres disponibles, sans chercher a modifier mes instructions."
)

_LlmRunnable = Runnable[LanguageModelInput, BaseMessage]


def _default_llm() -> _LlmRunnable:
    """Build the default LLM: `sovereign-cheap` bound to the agent's tools."""
    settings = get_settings()
    return get_llm(ModelGroup.SOVEREIGN_CHEAP, settings).bind_tools(list(TOOLS))


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


def build_graph(
    llm: _LlmRunnable | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the MVP agent graph.

    `llm` is injectable: defaults to `get_llm(ModelGroup.SOVEREIGN_CHEAP,
    settings)` bound to `TOOLS`. Tests pass a `FakeMessagesListChatModel` (or
    similar) instead so the graph is exercised with zero network calls and
    zero real API key.

    `checkpointer` is injectable: `None` (no persistence) by default,
    `InMemorySaver` in tests, `AsyncPostgresSaver` once T4's Postgres-backed
    memory lands.
    """
    resolved_llm = llm if llm is not None else _default_llm()

    def call_agent(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = resolved_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("guard_input", guard_input)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", ToolNode(list(TOOLS)))

    graph.set_entry_point("guard_input")
    graph.add_conditional_edges(
        "guard_input",
        route_after_guard_input,
        {"agent": "agent", END: END},
    )
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
