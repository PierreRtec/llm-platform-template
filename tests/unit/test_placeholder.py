"""Placeholder test so the suite is green from the very first commit.

Replaced by real unit tests starting at T2 (config), T3 (llm fallback),
T5 (tool registry), and T6 (graph + guardrails).
"""

from app import __version__


def test_app_package_has_a_version() -> None:
    assert __version__ == "0.1.0"
