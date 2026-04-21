#!/usr/bin/env python3
"""Block committing generated environments and docs build artifacts."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "docs/.doctrees/",
    "docs/_build/",
    "scratch/.venv_docs/",
    ".venv-ci/",
    ".conda-envs/",
    ".conda-pkgs/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
)


def _normalize(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _iter_tracked_files() -> Iterable[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    for raw_path in result.stdout.split(b"\0"):
        if raw_path:
            yield raw_path.decode()


def _find_forbidden(paths: Iterable[str]) -> list[str]:
    matches: list[str] = []
    for path in paths:
        normalized = _normalize(path)
        if not Path(normalized).exists():
            continue
        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            matches.append(normalized)
    return sorted(set(matches))


def main(argv: list[str]) -> int:
    """Check the provided paths, or tracked files, for forbidden artifacts."""
    paths = argv if argv else list(_iter_tracked_files())
    matches = _find_forbidden(paths)

    if not matches:
        print("Success: no forbidden generated artifacts are tracked.")
        return 0

    print("Forbidden generated artifacts detected:", file=sys.stderr)
    for match in matches:
        print(f"  - {match}", file=sys.stderr)
    print(
        "Remove these from git tracking and keep them ignored before committing.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
