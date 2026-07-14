"""Pure, composable input guardrail checks.

Each check is a small pure function: `text -> CheckResult`. No I/O, no
state, trivially unit-testable in isolation (CLAUDE.md style note: "prefer
pure functions for guardrails"). `pipeline.py` chains them into a single
aggregated result for the graph's `guard_input` node.

MVP scope note: regex and heuristics only, no Presidio/NeMo/Llama Guard.
Architecture is layered so a stronger PII/injection detector can replace
`check_pii`/`check_injection` internals later without touching the pipeline
or the graph (design doc section 6, point 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

MAX_INPUT_LENGTH: Final[int] = 4000


@dataclass(frozen=True)
class CheckResult:
    """Typed result of a single guardrail check.

    `blocking=True` (the default) means `ok=False` should stop the pipeline
    and trigger a refusal. `blocking=False` checks never fail the pipeline;
    `reason`, when set, is a flag for downstream routing or redaction (e.g.
    PII detection), not a rejection.
    """

    ok: bool
    reason: str | None = None
    blocking: bool = True


def check_length(text: str, *, max_length: int = MAX_INPUT_LENGTH) -> CheckResult:
    """Reject input longer than `max_length` characters."""
    if len(text) > max_length:
        return CheckResult(
            ok=False,
            reason=f"input exceeds max length of {max_length} characters",
        )
    return CheckResult(ok=True)


# Known prompt-injection phrasings, in both English and French (the product
# is French-facing, but injection attempts are commonly copy-pasted in
# English). Matched case-insensitively against the raw input. This is a
# heuristic allowlist-of-badness, not a complete defense; see module
# docstring.
_INJECTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "ignore/disregard/forget [up to 3 qualifier words] instructions",
        # loose enough to catch "ignore all previous instructions" as well as
        # "ignore les instructions precedentes" without one pattern per
        # phrasing.
        r"ignore\s+(?:\w+\s+){0,3}instructions",
        r"disregard\s+(?:\w+\s+){0,3}instructions",
        r"forget\s+(?:\w+\s+){0,3}instructions",
        r"oublie\s+(?:\w+\s+){0,3}instructions",
        r"reveal (your |the )?system prompt",
        r"(show|print|display) (me )?your (system )?prompt",
        r"affiche (ton|le) prompt (systeme|système)",
        r"you are now",
        r"act as (if you|a|an)",
        r"override your instructions",
        r"jailbreak",
        r"\bdan mode\b",
        r"nouvelle instruction\s*:",
    )
)


def check_injection(text: str) -> CheckResult:
    """Reject input matching a known prompt-injection pattern."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return CheckResult(
                ok=False,
                reason=f"potential prompt injection pattern detected: {pattern.pattern!r}",
            )
    return CheckResult(ok=True)


# French social security number (NIR): 13 digits (sex, year, month,
# department, commune, order) + a 2-digit check key, optionally separated by
# single spaces in common display formats (e.g. "1 85 03 76 116 001 42").
_NIR_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(\d(?:\s?\d){12})\s?(\d{2})\b")

# French IBAN: "FR" + 2 check digits + 23 alphanumeric characters (5 groups
# of 4 + a final group of 3), optionally space-separated.
_IBAN_FR_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bFR\d{2}(?:\s?[A-Z0-9]{4}){5}\s?[A-Z0-9]{3}\b",
    re.IGNORECASE,
)


def _nir_key_is_plausible(digits_13: str, key: str) -> bool:
    """Check the NIR's 2-digit key against the standard mod-97 formula.

    Simplified: assumes a purely numeric, metropolitan department code.
    Corsican department codes (2A/2B) and Reunion/overseas key computation
    variants are not handled here; a false negative on those is acceptable
    for a non-blocking flag used only to trigger a routing/redaction
    decision downstream, not to validate the number.
    """
    number = int(digits_13.replace(" ", ""))
    expected_key = 97 - (number % 97)
    return int(key) == expected_key


def check_pii(text: str) -> CheckResult:
    """Flag (never block) plausible French NIR and/or IBAN in `text`.

    `blocking=False`: detecting PII here never stops the pipeline. It only
    surfaces a flag (`nir_detected`, `iban_fr_detected`) that the graph
    records on `AgentState.input_flags` for future routing/redaction (design
    doc: "pas de blocage du NIR : flag pour routage/redaction future").
    """
    flags: list[str] = []

    nir_match = _NIR_PATTERN.search(text)
    if nir_match and _nir_key_is_plausible(nir_match.group(1), nir_match.group(2)):
        flags.append("nir_detected")

    if _IBAN_FR_PATTERN.search(text):
        flags.append("iban_fr_detected")

    return CheckResult(ok=True, reason=", ".join(flags) if flags else None, blocking=False)
