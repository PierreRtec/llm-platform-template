"""Graph state schema for the agent.

MVP scope: no `guard_output` verdict field yet (guard_output itself is out
of scope, design doc section 7); it would land alongside
`app/agent/guardrails/output.py` in a later task.
"""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State threaded through `build_graph`'s nodes.

    `input_flags` carries both blocking refusals (prefixed
    `guardrails.pipeline.BLOCKED_FLAG_PREFIX`) and non-blocking PII flags
    (prefixed `guardrails.pipeline.PII_FLAG_PREFIX`) produced by the
    `guard_input` node; see that module for why a single list covers both.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    input_flags: list[str]
    # Simple per-conversation token budget, optional: not enforced in the
    # MVP graph, a placeholder for a future budget-aware node/guard.
    token_budget: NotRequired[int]
    # Number of times the `tools` node has run for this invocation. Incremented
    # by the `tools` node itself; the conditional edge after `agent` checks it
    # against `MAX_TOOL_ROUNDS` (app/agent/graph.py) to terminate the agent
    # <-> tools loop gracefully instead of hitting LangGraph's own recursion
    # limit. Absent from the initial state (existing tests construct
    # `AgentState` without it); treated as 0 when missing.
    tool_rounds: NotRequired[int]
