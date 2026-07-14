"""Chains the individual input guardrail checks into one aggregated result.

Kept as thin orchestration over `input.py`'s pure checks: no regex, no
heuristics live here, only composition. `guard_output`/the corresponding
output pipeline (disclaimer, eligibility-verdict redaction) is out of MVP
scope (docs/DESIGN.md section 7) and will live in `app/agent/guardrails/output.py`
plus a symmetrical `run_output_guardrails` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.agent.guardrails.input import (
    MAX_INPUT_LENGTH,
    CheckResult,
    check_injection,
    check_length,
    check_pii,
)

# Prefixes tag each flag with where it came from, so `AgentState.input_flags`
# is both a human-readable audit trail and a machine-checkable routing
# signal (the graph's `guard_input` conditional edge looks for the
# `BLOCKED_FLAG_PREFIX`) without needing a separate state field.
BLOCKED_FLAG_PREFIX: Final[str] = "blocked:"
PII_FLAG_PREFIX: Final[str] = "pii:"


@dataclass(frozen=True)
class GuardrailPipelineResult:
    """Aggregated outcome of running all input guardrail checks on one input."""

    ok: bool
    refusal_reason: str | None
    flags: tuple[str, ...]


def run_input_guardrails(
    text: str,
    *,
    max_length: int = MAX_INPUT_LENGTH,
) -> GuardrailPipelineResult:
    """Run every input check on `text` and aggregate the outcome.

    Blocking checks (`check_length`, `check_injection`) short-circuit the
    pipeline's `ok` verdict on the first failure, in the order listed.
    Non-blocking checks (`check_pii`) never affect `ok`; their reasons are
    always collected into `flags`.
    """
    blocking_results: list[CheckResult] = [
        check_length(text, max_length=max_length),
        check_injection(text),
    ]
    pii_result = check_pii(text)

    first_failure = next((result for result in blocking_results if not result.ok), None)

    flags: list[str] = []
    if first_failure is not None:
        flags.append(f"{BLOCKED_FLAG_PREFIX}{first_failure.reason}")
    if pii_result.reason:
        flags.append(f"{PII_FLAG_PREFIX}{pii_result.reason}")

    return GuardrailPipelineResult(
        ok=first_failure is None,
        refusal_reason=first_failure.reason if first_failure is not None else None,
        flags=tuple(flags),
    )
