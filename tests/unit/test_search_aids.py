"""Unit tests for the `search_aids` tool (app.agent.tools.search_aids).

Pure in-memory keyword matching, no network, no LLM: `search_aids_corpus` is
tested directly against small fixture corpora, and `_load_corpus` /
`search_aids` (the tool wrapper) are tested against the real
`data/aids_sample.json` on disk.
"""

import json
from pathlib import Path

import pytest

from app.agent.tools.search_aids import (
    DEFAULT_AIDS_PATH,
    AidRecord,
    _load_corpus,
    search_aids,
    search_aids_corpus,
)


def _aid(
    aid_id: str,
    title: str,
    category: str,
    description: str = "",
    conditions: str = "",
) -> AidRecord:
    return AidRecord(
        id=aid_id,
        title=title,
        category=category,
        description=description or f"Description de {title}.",
        conditions=conditions or "Conditions generiques.",
        amount="100 EUR",
        source_url="https://www.exemple.fr/aides/test",
        synthetic=True,
    )


FIXTURE_CORPUS: tuple[AidRecord, ...] = (
    _aid(
        "aide-001",
        "Aide demo Etudiant",
        "logement",
        description="Aide au loyer pour les etudiants boursiers.",
        conditions="Etre etudiant boursier.",
    ),
    _aid(
        "aide-002",
        "Aide demo Emploi Jeune",
        "emploi",
        description="Aide pour les jeunes en recherche d'emploi.",
        conditions="Avoir moins de 25 ans.",
    ),
    _aid(
        "aide-003",
        "Aide demo Formation Numerique",
        "formation",
        description="Formation aux competences numeriques pour adultes.",
        conditions="Etre salarie ou demandeur d'emploi.",
    ),
)


class TestSearchAidsCorpus:
    def test_matches_by_keyword_in_description(self) -> None:
        results = search_aids_corpus("etudiant loyer", None, FIXTURE_CORPUS)

        assert [r["id"] for r in results] == ["aide-001"]

    def test_ranks_title_matches_above_body_only_matches(self) -> None:
        # "emploi" appears in aide-002's title (weighted higher) and only in
        # aide-003's conditions ("demandeur d'emploi").
        results = search_aids_corpus("emploi", None, FIXTURE_CORPUS)

        assert results[0]["id"] == "aide-002"

    def test_returns_at_most_three_results(self) -> None:
        big_corpus = FIXTURE_CORPUS + tuple(
            _aid(f"aide-extra-{i}", f"Aide demo Numero {i}", "divers", description="numero test")
            for i in range(5)
        )

        results = search_aids_corpus("numero", None, big_corpus)

        assert len(results) <= 3

    def test_no_match_returns_empty_list(self) -> None:
        results = search_aids_corpus("xyzabc inexistant", None, FIXTURE_CORPUS)

        assert results == []

    def test_category_filter_restricts_candidates(self) -> None:
        results = search_aids_corpus("aide", "emploi", FIXTURE_CORPUS)

        assert all(r["id"] == "aide-002" for r in results)

    def test_category_filter_excluding_everything_returns_empty(self) -> None:
        results = search_aids_corpus("aide", "categorie-inexistante", FIXTURE_CORPUS)

        assert results == []

    @pytest.mark.parametrize(
        ("query", "expected_id"),
        [
            ("etudiant", "aide-001"),
            ("ETUDIANT", "aide-001"),
            ("étudiant", "aide-001"),
            ("ÉTUDIANT", "aide-001"),
        ],
    )
    def test_matching_is_case_and_accent_insensitive(self, query: str, expected_id: str) -> None:
        results = search_aids_corpus(query, None, FIXTURE_CORPUS)

        assert results
        assert results[0]["id"] == expected_id

    def test_result_shape_has_id_title_snippet_only(self) -> None:
        results = search_aids_corpus("etudiant", None, FIXTURE_CORPUS)

        assert results
        assert set(results[0].keys()) == {"id", "title", "snippet"}


class TestLoadCorpus:
    def test_default_path_points_at_repo_data_file(self) -> None:
        assert DEFAULT_AIDS_PATH.name == "aids_sample.json"
        assert DEFAULT_AIDS_PATH.exists()

    def test_loads_eight_synthetic_aids_from_default_corpus(self) -> None:
        corpus = _load_corpus()

        assert len(corpus) == 8
        assert all(record["synthetic"] is True for record in corpus)

    def test_injected_path_bypasses_default_cache(self, tmp_path: Path) -> None:
        custom_path = tmp_path / "custom_aids.json"
        custom_path.write_text(
            json.dumps(
                [
                    _aid("custom-1", "Aide demo Custom", "test"),
                ]
            ),
            encoding="utf-8",
        )

        corpus = _load_corpus(path=custom_path)

        assert len(corpus) == 1
        assert corpus[0]["id"] == "custom-1"
        # the default corpus is untouched by loading a custom path
        assert len(_load_corpus()) == 8


class TestSearchAidsTool:
    def test_tool_invoke_returns_matches_from_real_corpus(self) -> None:
        result = search_aids.invoke({"query": "jeune emploi"})

        assert isinstance(result, list)
        assert result
        assert all({"id", "title", "snippet"} <= set(row.keys()) for row in result)

    def test_tool_has_pydantic_args_schema_with_query_and_category(self) -> None:
        schema = search_aids.args_schema
        assert schema is not None
        fields = schema.model_fields  # type: ignore[union-attr]
        assert "query" in fields
        assert "category" in fields
