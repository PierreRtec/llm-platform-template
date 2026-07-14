"""Gateway-aware LLM layer: model groups, retries, sovereignty fallback cascade.

Every model in this template is reached through a single self-hosted LiteLLM
gateway, addressed only by *model group* name (see design doc section 3.3 and
`gateway/litellm.config.yaml`): `sovereign-cheap`, `sovereign-premium`,
`frontier`, `sovereign-embed`. The application never imports a provider SDK
(`anthropic`, `mistralai`, a Scaleway client, ...); it only ever talks
`ChatOpenAI` to `LITELLM_BASE_URL`. See CLAUDE.md non-negotiable #5.

Two independent retry/fallback layers exist on purpose, at different levels:

- The LiteLLM router (`router_settings.num_retries` / `router_settings.fallbacks`
  in `gateway/litellm.config.yaml`) retries and falls back *within* and
  *across* physical deployments behind a group alias (e.g. two different
  Scaleway models both aliased to `sovereign-cheap`).
- This module retries transient errors on top of that (defense in depth
  against a slow/unhealthy gateway), then escalates *across* model groups in
  cascading sovereignty order when a whole group keeps failing: sovereign
  compute first, third-party frontier models only as a last resort. This is
  the interview-relevant point: the cascade encodes a sovereignty policy, not
  just "keep trying until something answers".
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

import structlog
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from app.core.config import Settings


class ModelGroup(StrEnum):
    """The four LiteLLM model group aliases exposed by the gateway.

    Values must match `model_name` entries in `gateway/litellm.config.yaml`
    exactly: they are sent verbatim as the `model` field of chat completion
    requests.
    """

    SOVEREIGN_CHEAP = "sovereign-cheap"
    SOVEREIGN_PREMIUM = "sovereign-premium"
    FRONTIER = "frontier"
    SOVEREIGN_EMBED = "sovereign-embed"


# The sovereignty escalation order used by `ainvoke_with_fallback`. Chosen
# deliberately: cheapest/most sovereign compute first, third-party frontier
# model last. `sovereign-embed` is intentionally excluded, it is an
# embeddings-only group (used by the Store/seed script), never a chat
# completion target, so it can never appear in this cascade.
_CASCADE: Final[tuple[ModelGroup, ...]] = (
    ModelGroup.SOVEREIGN_CHEAP,
    ModelGroup.SOVEREIGN_PREMIUM,
    ModelGroup.FRONTIER,
)

DEFAULT_TEMPERATURE: Final[float] = 0.2
DEFAULT_MAX_TOKENS: Final[int] = 1024
# Kept low on purpose: this is a per-attempt timeout, and the retry/escalation
# layers above already provide the resilience budget. A high per-attempt
# timeout would let a single hung group stall the whole cascade.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 20.0
DEFAULT_MAX_ATTEMPTS_PER_GROUP: Final[int] = 3

logger = structlog.get_logger(__name__)


def get_llm(
    model_group: ModelGroup,
    settings: Settings,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ChatOpenAI:
    """Build a `ChatOpenAI` client for `model_group`, pointed at the gateway.

    `max_retries=0` is hardcoded and load-bearing, not a tunable default: the
    openai SDK client underlying `ChatOpenAI` performs its own connection-level
    retries on 429/5xx unless this is forced to 0. Leaving it non-zero would
    stack an untraceable retry layer underneath both `ainvoke_with_fallback`'s
    tenacity retries and the LiteLLM router's own `num_retries`, making actual
    attempt counts and latency budgets impossible to reason about. Callers
    that need raw client access (bypassing the fallback cascade) get this same
    guarantee for free.
    """
    return ChatOpenAI(
        base_url=settings.LITELLM_BASE_URL,
        api_key=settings.LITELLM_API_KEY,
        model=model_group.value,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=0,
    )


def _is_transient(exc: BaseException) -> bool:
    """Classify an error as retryable/escalatable (True) or not (False).

    Transient: request timeouts, connection failures, HTTP 429, and any HTTP
    5xx from the gateway. Not transient: any other 4xx (400 bad request, 401
    unauthorized, 404 not found, ...). A non-transient error means the
    request itself is wrong or misconfigured; retrying or escalating to a
    different model group would not fix it and would only hide a real bug or
    a credentials problem, so those propagate to the caller immediately.
    """
    if isinstance(exc, APITimeoutError | APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


class LLMCascadeExhaustedError(RuntimeError):
    """Raised when every allowed group in the sovereignty cascade failed.

    Carries `groups_tried` (in cascade order) and `last_error` (the final
    transient exception from the last group attempted) so callers/logs get a
    precise picture of what was tried, not just "it failed".
    """

    def __init__(self, groups_tried: Sequence[ModelGroup], last_error: Exception) -> None:
        self.groups_tried: tuple[ModelGroup, ...] = tuple(groups_tried)
        self.last_error = last_error
        tried = " -> ".join(group.value for group in self.groups_tried)
        super().__init__(
            f"LLM sovereignty cascade exhausted after trying: {tried} "
            f"(last error: {type(last_error).__name__}: {last_error})"
        )


async def _invoke_group(
    settings: Settings,
    group: ModelGroup,
    messages: Sequence[BaseMessage],
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_attempts: int,
) -> BaseMessage:
    """Invoke a single model group with tenacity retries on transient errors.

    On a non-transient error, `retry_if_exception(_is_transient)` makes
    tenacity give up immediately (first attempt only). On retry exhaustion,
    `reraise=True` re-raises the original exception (not `tenacity.RetryError`)
    so the caller (the cascade loop) can classify it with the same
    `_is_transient` check.
    """
    llm = get_llm(group, settings, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    def _log_retry(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "llm_retry",
            group=group.value,
            attempt=retry_state.attempt_number,
            error_type=type(exc).__name__ if exc is not None else "unknown",
        )

    retryer: AsyncRetrying = AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception(_is_transient),
        before_sleep=_log_retry,
        reraise=True,
    )
    result: BaseMessage = await retryer(llm.ainvoke, messages)
    return result


async def ainvoke_with_fallback(
    messages: Sequence[BaseMessage],
    *,
    settings: Settings,
    start_group: ModelGroup = ModelGroup.SOVEREIGN_CHEAP,
    allowed_groups: Sequence[ModelGroup] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts_per_group: int = DEFAULT_MAX_ATTEMPTS_PER_GROUP,
) -> BaseMessage:
    """Invoke the gateway with a sovereignty-first escalation cascade.

    This is an application-level escalation layered *on top of* (not instead
    of) the LiteLLM router's own `num_retries`/`fallbacks`
    (`gateway/litellm.config.yaml` `router_settings`). The router already
    retries and falls back *within* a group alias, across physical
    deployments; this function additionally escalates *across* groups when a
    whole group is unavailable, in cascading sovereignty order:
    `sovereign-cheap` -> `sovereign-premium` -> `frontier`. That ordering is
    the interview-relevant argument: escalation is a sovereignty policy
    (sovereign compute exhausted first, third-party frontier model only as a
    last resort), not an accident of whichever fallback happened to answer.
    The cascade stops at `frontier`: there is nothing to escalate to past it.

    Each group is retried up to `max_attempts_per_group` times (tenacity,
    exponential backoff with jitter) before the cascade escalates to the next
    group. Only transient errors (429, 5xx, timeout, connection failure)
    trigger a retry or an escalation; every retry and every escalation is
    logged via structlog (`llm_retry`, `llm_group_escalation`) with the group,
    attempt number, and error type.

    Non-transient errors (any 4xx other than 429: bad request, unauthorized,
    not found, ...) are never retried and never trigger escalation: they
    indicate a bug or misconfiguration that retrying or escalating would only
    mask, so they propagate to the caller immediately, from whichever group
    raised them.

    `allowed_groups` restricts which groups the cascade may visit (cascade
    order is preserved regardless of the order passed in). This is
    preparation for the finley-2 PII lock: a caller handling PII-tagged input
    can pass `allowed_groups=(ModelGroup.SOVEREIGN_CHEAP,
    ModelGroup.SOVEREIGN_PREMIUM)` to guarantee a request never reaches
    `frontier` (a non-sovereign, third-party-hosted model), regardless of how
    many transient failures happen upstream.

    Raises:
        ValueError: `start_group` is not part of the (possibly restricted)
            cascade, or `allowed_groups` is empty.
        LLMCascadeExhaustedError: every allowed group from `start_group`
            onward failed transiently; wraps the last transient error.
        Exception: the original, unwrapped exception for any non-transient
            error encountered along the way.
    """
    effective_allowed = _CASCADE if allowed_groups is None else allowed_groups
    if not effective_allowed:
        raise ValueError("allowed_groups must not be empty")
    cascade = [group for group in _CASCADE if group in effective_allowed]
    if start_group not in cascade:
        raise ValueError(
            f"start_group {start_group.value!r} is not in the allowed cascade "
            f"{[group.value for group in cascade]!r}"
        )
    start_index = cascade.index(start_group)

    groups_tried: list[ModelGroup] = []
    last_error: Exception | None = None
    remaining = cascade[start_index:]
    for position, group in enumerate(remaining):
        groups_tried.append(group)
        try:
            return await _invoke_group(
                settings,
                group,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_attempts=max_attempts_per_group,
            )
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last_error = exc
            has_next = position + 1 < len(remaining)
            if has_next:
                logger.warning(
                    "llm_group_escalation",
                    from_group=group.value,
                    to_group=remaining[position + 1].value,
                    error_type=type(exc).__name__,
                )

    assert last_error is not None  # cascade is non-empty, so this branch always sets it
    raise LLMCascadeExhaustedError(groups_tried, last_error) from last_error
