#!/usr/bin/env python3
"""Validate release artifacts before publishing."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_PARTS = {
    ".agent",
    ".harness",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "docs_internal",
    "scratch",
    "temp_logs",
    "tmp",
}

FORBIDDEN_SUFFIXES = (".pyc", ".pyo")


def _is_forbidden(member_name: str) -> bool:
    path = Path(member_name.replace("\\", "/"))
    return any(part in FORBIDDEN_PARTS for part in path.parts) or member_name.endswith(
        FORBIDDEN_SUFFIXES
    )


def _iter_wheel_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def _iter_sdist_members(path: Path) -> list[str]:
    with tarfile.open(path, "r:*") as archive:
        return archive.getnames()


def _check_members(path: Path, members: list[str]) -> list[str]:
    return [f"{path.name}:{name}" for name in members if _is_forbidden(name)]


def check_artifacts(dist_dir: Path) -> list[str]:
    """Return forbidden paths found in built artifacts."""
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    problems: list[str] = []

    if len(wheels) != 1:
        problems.append(
            f"Expected exactly one wheel in {dist_dir}, found {len(wheels)}"
        )
    if len(sdists) != 1:
        problems.append(
            f"Expected exactly one sdist in {dist_dir}, found {len(sdists)}"
        )

    for wheel in wheels:
        problems.extend(_check_members(wheel, _iter_wheel_members(wheel)))
    for sdist in sdists:
        problems.extend(_check_members(sdist, _iter_sdist_members(sdist)))

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?", default="dist")
    args = parser.parse_args(argv)

    problems = check_artifacts(Path(args.dist_dir))
    if not problems:
        print("Release artifacts passed hygiene checks.")
        return 0

    print("Release artifact hygiene failures:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
