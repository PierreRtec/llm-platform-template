#!/usr/bin/env python3
"""Lightweight Conventional Commits check, used as a pre-commit commit-msg hook.

Deliberately dependency-free (no commitizen): a single regex is enough for
what this repo needs, and it keeps the pre-commit environment fast.
"""

from __future__ import annotations

import re
import sys

CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(feat|fix|chore|docs|style|refactor|perf|test|build|ci|revert)"
    r"(\([a-z0-9./_-]+\))?!?: .+"
)

# Merge commits and fixup/squash commits are exempt.
EXEMPT_PREFIXES = ("Merge ", "fixup!", "squash!", 'Revert "')


def main() -> int:
    if len(sys.argv) < 2:
        print("check_commit_msg: missing commit message file argument", file=sys.stderr)
        return 1

    commit_msg_path = sys.argv[1]
    with open(commit_msg_path, encoding="utf-8") as handle:
        first_line = handle.readline().strip()

    if not first_line or first_line.startswith(EXEMPT_PREFIXES):
        return 0

    if CONVENTIONAL_COMMIT_RE.match(first_line):
        return 0

    print(
        "check_commit_msg: commit message must follow Conventional Commits "
        '(e.g. "feat(agent): add search_aids tool").\n'
        f"Got: {first_line!r}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
