"""Unit tests for the gateway-aware LLM layer (app.agent.llm).

All HTTP to the LiteLLM gateway's OpenAI-compatible endpoint is mocked with
`respx`: no real network call, no real API key, ever. These tests exercise:

1. A direct success on the starting group (no retry, no escalation).
2. Tenacity retries on transient errors (429) within the same group.
3. Escalation across the sovereignty cascade when a whole group keeps
   failing transiently (`sovereign-cheap` -> `sovereign-premium`).
4. Immediate propagation of a non-transient error (401): no retry, no
   escalation, exactly one HTTP call.
5. A clear exception when the entire allowed cascade is exhausted.
6. `allowed_groups` restricting the cascade so no escalation happens even
   on transient failure.

Plus focused tests that `get_llm` builds a client wired to the right
`base_url`/`model`, with `max_retries=0` so tenacity is the only retry layer
(verified both by attribute and by an actual respx call count).
"""

import httpx
import pytest
import respx
from langchain_core.messages import BaseMessage, HumanMessage

from app.agent.llm import (
    LLMCascadeExhaustedError,
    ModelGroup,
    ainvoke_with_fallback,
    get_llm,
)
from app.core.config import Settings

GATEWAY_URL = "http://litellm.test/v1"
CHAT_ENDPOINT = f"{GATEWAY_URL}/chat/completions"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        APP_API_KEY="test-app-key",
        LITELLM_BASE_URL=GATEWAY_URL,
        LITELLM_API_KEY="test-litellm-key",
    )


def _chat_completion_body(model: str, content: str = "hello") -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _error_body(message: str, code: str) -> dict[str, object]:
    return {"error": {"message": message, "type": code, "code": code}}


MESSAGES: list[BaseMessage] = [HumanMessage(content="What aids exist?")]


def test_get_llm_builds_client_with_gateway_base_url_and_group_model(
    settings: Settings,
) -> None:
    llm = get_llm(ModelGroup.SOVEREIGN_CHEAP, settings)

    assert llm.openai_api_base == GATEWAY_URL
    assert llm.model_name == ModelGroup.SOVEREIGN_CHEAP.value
    assert llm.max_retries == 0


async def test_get_llm_client_makes_exactly_one_http_call_per_ainvoke(
    settings: Settings,
) -> None:
    """Behavioral proof that `max_retries=0` holds end to end: the openai SDK
    performs its own internal retries on 429/5xx unless `max_retries` is
    forced to 0. Assert on the respx call count (not just the attribute) so
    a regression is caught even if the attribute stops reflecting real
    client behavior."""
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(CHAT_ENDPOINT).mock(
            return_value=httpx.Response(500, json=_error_body("boom", "server_error"))
        )
        llm = get_llm(ModelGroup.SOVEREIGN_CHEAP, settings)

        with pytest.raises(Exception):  # noqa: B017 - only proving call count here
            await llm.ainvoke(MESSAGES)

    assert route.call_count == 1


async def test_ainvoke_with_fallback_succeeds_directly_on_sovereign_cheap(
    settings: Settings,
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(CHAT_ENDPOINT).mock(
            return_value=httpx.Response(200, json=_chat_completion_body("sovereign-cheap"))
        )

        result = await ainvoke_with_fallback(MESSAGES, settings=settings)

    assert result.content == "hello"
    assert route.call_count == 1
    assert route.calls.last.request.content is not None
    assert b'"model":"sovereign-cheap"' in route.calls.last.request.content


async def test_ainvoke_with_fallback_retries_429_then_succeeds_same_group(
    settings: Settings,
) -> None:
    responses = [
        httpx.Response(429, json=_error_body("rate limited", "rate_limit_error")),
        httpx.Response(429, json=_error_body("rate limited", "rate_limit_error")),
        httpx.Response(200, json=_chat_completion_body("sovereign-cheap")),
    ]

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(CHAT_ENDPOINT).mock(side_effect=responses)

        result = await ainvoke_with_fallback(MESSAGES, settings=settings)

    assert result.content == "hello"
    assert route.call_count == 3
    # every call must have targeted the same group: no escalation on retry
    for call in route.calls:
        assert b'"model":"sovereign-cheap"' in call.request.content


async def test_ainvoke_with_fallback_escalates_to_premium_after_persistent_500(
    settings: Settings,
) -> None:
    def _route_by_model(request: httpx.Request) -> httpx.Response:
        body = request.content
        if b'"model":"sovereign-cheap"' in body:
            return httpx.Response(500, json=_error_body("internal error", "server_error"))
        if b'"model":"sovereign-premium"' in body:
            return httpx.Response(200, json=_chat_completion_body("sovereign-premium"))
        raise AssertionError(f"unexpected request body: {body!r}")

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(CHAT_ENDPOINT).mock(side_effect=_route_by_model)

        result = await ainvoke_with_fallback(MESSAGES, settings=settings, max_attempts_per_group=2)

    assert result.content == "hello"
    cheap_calls = [c for c in route.calls if b'"model":"sovereign-cheap"' in c.request.content]
    premium_calls = [c for c in route.calls if b'"model":"sovereign-premium"' in c.request.content]
    assert len(cheap_calls) == 2  # max_attempts_per_group, then escalates
    assert len(premium_calls) == 1  # succeeds on first try after escalation


async def test_ainvoke_with_fallback_propagates_401_immediately_no_retry_no_escalation(
    settings: Settings,
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(CHAT_ENDPOINT).mock(
            return_value=httpx.Response(
                401, json=_error_body("invalid api key", "invalid_request_error")
            )
        )

        with pytest.raises(Exception) as exc_info:  # noqa: B017 - real openai error type asserted below
            await ainvoke_with_fallback(MESSAGES, settings=settings, max_attempts_per_group=3)

    assert getattr(exc_info.value, "status_code", None) == 401
    # exactly one HTTP call: no tenacity retry, no cascade escalation
    assert route.call_count == 1


async def test_ainvoke_with_fallback_raises_clear_error_when_cascade_exhausted(
    settings: Settings,
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(CHAT_ENDPOINT).mock(
            return_value=httpx.Response(503, json=_error_body("unavailable", "server_error"))
        )

        with pytest.raises(LLMCascadeExhaustedError) as exc_info:
            await ainvoke_with_fallback(MESSAGES, settings=settings, max_attempts_per_group=2)

    error = exc_info.value
    assert error.groups_tried == (
        ModelGroup.SOVEREIGN_CHEAP,
        ModelGroup.SOVEREIGN_PREMIUM,
        ModelGroup.FRONTIER,
    )
    assert error.last_error is not None
    # 3 groups x 2 attempts each = 6 calls total, all exhausted
    assert route.call_count == 6


async def test_ainvoke_with_fallback_allowed_groups_restricts_cascade_no_escalation(
    settings: Settings,
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(CHAT_ENDPOINT).mock(
            return_value=httpx.Response(500, json=_error_body("internal error", "server_error"))
        )

        with pytest.raises(LLMCascadeExhaustedError) as exc_info:
            await ainvoke_with_fallback(
                MESSAGES,
                settings=settings,
                allowed_groups=[ModelGroup.SOVEREIGN_CHEAP],
                max_attempts_per_group=2,
            )

    assert exc_info.value.groups_tried == (ModelGroup.SOVEREIGN_CHEAP,)
    for call in route.calls:
        assert b'"model":"sovereign-cheap"' in call.request.content
    assert route.call_count == 2  # only the retries within the single allowed group


def test_model_group_values_match_gateway_config() -> None:
    assert ModelGroup.SOVEREIGN_CHEAP.value == "sovereign-cheap"
    assert ModelGroup.SOVEREIGN_PREMIUM.value == "sovereign-premium"
    assert ModelGroup.FRONTIER.value == "frontier"
    assert ModelGroup.SOVEREIGN_EMBED.value == "sovereign-embed"


async def test_ainvoke_with_fallback_rejects_explicitly_empty_allowed_groups(
    settings: Settings,
) -> None:
    """An explicitly empty `allowed_groups=[]` must be rejected, not silently
    treated as "no restriction" (which would defeat the whole point of the
    parameter, e.g. the finley-2 PII lock use case)."""
    with pytest.raises(ValueError, match="allowed_groups must not be empty"):
        await ainvoke_with_fallback(MESSAGES, settings=settings, allowed_groups=[])
