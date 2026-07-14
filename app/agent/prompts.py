"""Versioned system prompt for the aid-information agent.

The prompt content itself is French: it is the product, addressed to a
French-speaking end user asking about financial aid. Everything around it
(this module's code, comments, docstrings) stays English per CLAUDE.md.

`PROMPT_VERSION` and `system_prompt_hash()` exist so prompt changes are
traceable: T8 (telemetry) attaches the hash as a trace attribute, and the
eval gate (T9) can pin expectations to a specific version/hash instead of a
raw string compare.
"""

from __future__ import annotations

import hashlib
from typing import Final

PROMPT_VERSION: Final[str] = "v1"

# Responsible AI posture, encoded directly in the prompt rather than left to
# model judgement:
# - inform and orient on financial aid, never issue an eligibility verdict
#   ("vous avez droit" / "vous n'avez pas droit").
# - always point to the competent organism for an authoritative answer.
# - answer the end user in French.
SYSTEM_PROMPT: Final[str] = """\
Tu es un assistant d'information sur les aides financieres francaises.

Ton role est d'INFORMER et d'ORIENTER, jamais de decider :
- Tu presentes les aides qui semblent correspondre a la situation decrite par l'utilisateur,
  avec leurs conditions generales, leur montant indicatif et leur source.
- Tu ne rends JAMAIS de verdict d'eligibilite. Tu ne dis jamais "vous avez droit a cette aide"
  ni "vous n'y avez pas droit". Les conditions reelles dependent de la situation complete de
  la personne et de l'appreciation de l'organisme concerne.
- Pour toute question d'eligibilite precise, tu renvoies systematiquement l'utilisateur vers
  l'organisme competent indique dans la source de l'aide, ou vers un travailleur social, pour
  une instruction officielle de son dossier.
- Tu utilises l'outil de recherche disponible pour trouver les aides pertinentes avant de
  repondre plutot que d'inventer des informations.
- Tu reponds toujours en francais, de maniere claire et concise, a l'utilisateur final.
- Le corpus d'aides utilise ici est un jeu de demonstration invente (aides fictives, aucune
  donnee reelle) : tu ne dois jamais presenter une aide de ce corpus comme une aide reelle et
  verifiee aupres d'un organisme officiel.
"""


def system_prompt_hash(prompt: str = SYSTEM_PROMPT) -> str:
    """Return a short, stable hash of `prompt` for use as a trace attribute.

    Truncated to 12 hex chars: enough to detect any prompt drift between
    traces without needing the full digest in logs/dashboards.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


# Polite French refusal appended by `guard_input` (app/agent/graph.py) when a
# blocking guardrail (length or injection) fires: the graph routes straight
# to END without ever invoking the agent/LLM for this message.
REFUSAL_MESSAGE_FR: Final[str] = (
    "Je ne peux pas traiter cette demande telle quelle. Reformulez votre question sur les "
    "aides financieres disponibles, sans chercher a modifier mes instructions."
)

# Polite French fallback appended by the `budget_exceeded` node (app/agent/
# graph.py) when the agent <-> tools loop hits `MAX_TOOL_ROUNDS` without
# reaching a final answer: the graph terminates on this message instead of
# letting LangGraph's own recursion limit raise `GraphRecursionError`.
BUDGET_EXCEEDED_MESSAGE_FR: Final[str] = (
    "Je n'ai pas reussi a aboutir a une reponse sur les aides financieres correspondant a "
    "votre demande. Reformulez votre question de maniere plus precise, ou contactez "
    "directement l'organisme concerne pour une instruction officielle de votre situation."
)
