"""`search_aids` tool: keyword lookup over the demo financial-aid corpus.

MVP scope (design doc section 7): this is a simple, in-memory, accent- and
case-insensitive keyword match over `data/aids_sample.json`, not semantic
search. The real corpus lookup (pgvector `Store`, embeddings via the
`sovereign-embed` model group) lands with T4/T5; at that point this module's
`_load_corpus` is replaced by a `Store.asearch` call and `search_aids_corpus`
by a similarity ranking, but the tool's public signature (`query`,
`category`) and result shape are meant to stay stable across that swap.

Every aid record is a synthetic, invented example (see `data/aids_sample.json`,
`"synthetic": true` on each entry): no real French welfare data is used.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final, TypedDict

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# app/agent/tools/search_aids.py -> parents[3] is the repo root.
DEFAULT_AIDS_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "aids_sample.json"

MAX_RESULTS: Final[int] = 3
SNIPPET_LENGTH: Final[int] = 160
# Title matches count more than body matches: a query word landing in the
# aid's title is a stronger signal of relevance than the same word appearing
# once in a longer description/conditions block.
TITLE_MATCH_WEIGHT: Final[int] = 2


class AidRecord(TypedDict):
    """Shape of one entry in `data/aids_sample.json`."""

    id: str
    title: str
    category: str
    description: str
    conditions: str
    amount: str
    source_url: str
    synthetic: bool


class AidSearchResult(TypedDict):
    """One row of `search_aids` output: enough to let the agent cite an aid."""

    id: str
    title: str
    snippet: str


class SearchAidsArgs(BaseModel):
    """Args schema for the `search_aids` tool.

    Kept as an explicit Pydantic model (rather than relying on `@tool`'s
    signature inference) so the schema advertised to the model is stable and
    reviewable independently of the Python function signature, and so the
    upcoming `ToolRegistry` (T5) has a concrete `args_schema` to validate
    against.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Free-text query in French describing the user's situation or need.",
    )
    category: str | None = Field(
        default=None,
        description="Optional category filter, e.g. 'logement', 'emploi', 'formation'.",
    )


def _strip_accents(text: str) -> str:
    """Remove diacritics so 'etudiant' and 'étudiant' match the same token."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _normalize(text: str) -> str:
    return _strip_accents(text).lower()


_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(_normalize(text)))


@lru_cache(maxsize=1)
def _load_corpus(path: Path = DEFAULT_AIDS_PATH) -> tuple[AidRecord, ...]:
    """Load and cache the demo aid corpus from disk.

    Cached with `lru_cache` keyed on `path` (a `Path` is hashable): tests can
    inject a different corpus file by passing an explicit `path` without
    disturbing the cached default, and can call `_load_corpus.cache_clear()`
    to force a re-read of the default file if needed.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(raw)


def _matches_category(record: AidRecord, category: str) -> bool:
    return _normalize(category) in _normalize(record["category"])


def _score(record: AidRecord, query_tokens: set[str]) -> int:
    title_tokens = _tokenize(record["title"])
    body_tokens = _tokenize(record["description"]) | _tokenize(record["conditions"])
    return TITLE_MATCH_WEIGHT * len(query_tokens & title_tokens) + len(query_tokens & body_tokens)


def _snippet(record: AidRecord) -> str:
    description = record["description"]
    if len(description) <= SNIPPET_LENGTH:
        return description
    return description[:SNIPPET_LENGTH].rstrip() + "..."


def search_aids_corpus(
    query: str,
    category: str | None,
    corpus: Sequence[AidRecord],
) -> list[AidSearchResult]:
    """Pure search function: keyword-score `corpus`, return up to 3 best matches.

    Kept separate from the `@tool`-decorated `search_aids` below so it can be
    unit tested directly against an arbitrary in-memory corpus, without going
    through tool invocation or the on-disk default corpus.
    """
    candidates = (
        [record for record in corpus if _matches_category(record, category)]
        if category
        else list(corpus)
    )
    query_tokens = _tokenize(query)
    scored = [(record, _score(record, query_tokens)) for record in candidates]
    matched = (pair for pair in scored if pair[1] > 0)
    ranked = sorted(matched, key=lambda pair: pair[1], reverse=True)
    return [
        AidSearchResult(id=record["id"], title=record["title"], snippet=_snippet(record))
        for record, _score_value in ranked[:MAX_RESULTS]
    ]


@tool("search_aids", args_schema=SearchAidsArgs)
def search_aids(query: str, category: str | None = None) -> list[AidSearchResult]:
    """Search the demo financial-aid corpus by keyword and return up to 3 matches.

    Matching is case- and accent-insensitive keyword overlap against the
    aid's title, description and conditions. `category`, if given, filters
    the corpus before scoring. Every result is a synthetic demo aid, never
    real French welfare data.
    """
    corpus = _load_corpus()
    return search_aids_corpus(query=query, category=category, corpus=corpus)
