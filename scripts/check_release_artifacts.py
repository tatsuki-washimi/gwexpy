#!/usr/bin/env python3
"""Validate release artifacts before publishing."""

from __future__ import annotations

import argparse
import stat
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_PARTS = {
    ".agent",
    ".harness",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "docs_internal",
    "htmlcov",
    "scratch",
    "temp_logs",
    "tmp",
}

FORBIDDEN_NAMES = {
    ".coverage",
    ".DS_Store",
    "Thumbs.db",
}

FORBIDDEN_SUFFIXES = (".pyc", ".pyo")


class ArtifactMember:
    """Release artifact member metadata used for hygiene checks."""

    def __init__(self, name: str, is_symlink: bool = False) -> None:
        self.name = name
        self.is_symlink = is_symlink


def _normalized_parts(member_name: str) -> tuple[str, ...]:
    path = Path(member_name.replace("\\", "/"))
    parts = path.parts
    if parts and _is_sdist_root(parts[0]):
        return parts[1:]
    return parts


def _is_sdist_root(part: str) -> bool:
    return part.startswith("gwexpy-") and not part.endswith((".data", ".dist-info"))


def _is_forbidden(member_name: str) -> bool:
    parts = _normalized_parts(member_name)
    name = parts[-1] if parts else ""
    if any(part in FORBIDDEN_PARTS for part in parts):
        return True
    if name in FORBIDDEN_NAMES:
        return True
    if member_name.endswith(FORBIDDEN_SUFFIXES):
        return True
    return _is_package_internal_test(parts) or _is_package_internal_markdown(parts)


def _is_package_internal_test(parts: tuple[str, ...]) -> bool:
    package_parts = _package_parts(parts)
    return len(package_parts) >= 3 and "tests" in package_parts[1:]


def _is_package_internal_markdown(parts: tuple[str, ...]) -> bool:
    # Public docs live outside gwexpy/; package-internal Markdown is development
    # reference material and must not ship in release artifacts.
    package_parts = _package_parts(parts)
    return len(package_parts) >= 2 and package_parts[-1].endswith(".md")


def _package_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    if parts and parts[0] == "gwexpy":
        return parts
    for index, part in enumerate(parts[:-1]):
        if part in {"purelib", "platlib"} and parts[index + 1] == "gwexpy":
            return parts[index + 1 :]
    return ()


def _iter_wheel_members(path: Path) -> list[ArtifactMember]:
    with zipfile.ZipFile(path) as archive:
        return [
            ArtifactMember(
                info.filename,
                stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK,
            )
            for info in archive.infolist()
        ]


def _iter_sdist_members(path: Path) -> list[ArtifactMember]:
    with tarfile.open(path, "r:*") as archive:
        return [
            ArtifactMember(info.name, info.issym() or info.islnk())
            for info in archive.getmembers()
        ]


def _check_members(path: Path, members: list[ArtifactMember]) -> list[str]:
    problems: list[str] = []
    for member in members:
        if member.is_symlink:
            problems.append(f"{path.name}:{member.name} is a symlink")
            continue
        if _is_forbidden(member.name):
            problems.append(f"{path.name}:{member.name}")
    return problems


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
