"""Regression guards for the conda-forge onboarding roadmap."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
ROADMAP = ROOT / "docs/developers/plans/2026-04-28-conda-forge-roadmap.md"


def _dependency_name(requirement: str) -> str:
    name = re.split(r"[\s<>=!~;,\[]", requirement, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _roadmap_core_dependencies() -> set[str]:
    text = ROADMAP.read_text(encoding="utf-8")
    dependencies: set[str] = set()
    in_table = False

    for line in text.splitlines():
        if line.startswith("| PyPI dependency | Conda-forge package |"):
            in_table = True
            continue

        if not in_table:
            continue

        if not line.startswith("|"):
            break

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0].startswith("---"):
            continue

        match = re.search(r"`([^`]+)`", cells[0])
        if match:
            dependencies.add(_dependency_name(match.group(1)))

    return dependencies


def test_conda_forge_roadmap_covers_project_core_dependencies():
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    expected = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["dependencies"]
    }

    assert expected <= _roadmap_core_dependencies()


def test_conda_forge_roadmap_lists_console_entry_points():
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    roadmap = ROADMAP.read_text(encoding="utf-8")

    for script, target in pyproject["project"]["scripts"].items():
        assert f"- {script} = {target}" in roadmap
