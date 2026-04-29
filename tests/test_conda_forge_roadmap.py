"""Regression guards for the conda-forge onboarding roadmap."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
ROADMAP = ROOT / "docs/developers/plans/active/2026-04-28-conda-forge-roadmap.md"


def _dependency_name(requirement: str) -> str:
    name = re.split(r"[\s<>=!~;,\[]", requirement, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _roadmap_core_dependency_map() -> dict[str, str]:
    text = ROADMAP.read_text(encoding="utf-8")
    dependencies: dict[str, str] = {}
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

        pyproject_match = re.search(r"`([^`]+)`", cells[0])
        conda_match = re.search(r"`([^`]+)`", cells[1])
        if pyproject_match and conda_match:
            dependencies[_dependency_name(pyproject_match.group(1))] = (
                conda_match.group(1)
            )

    return dependencies


def _roadmap_initial_recipe() -> str:
    text = ROADMAP.read_text(encoding="utf-8")
    match = re.search(
        r"## Initial Recipe Model.*?```yaml\n(.*?)\n```",
        text,
        flags=re.S,
    )
    assert match is not None, "Initial recipe model block not found"
    return match.group(1)


def test_conda_forge_roadmap_covers_project_core_dependencies():
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    expected = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["dependencies"]
    }

    assert expected <= set(_roadmap_core_dependency_map())


def test_conda_forge_roadmap_records_conda_package_name_translations():
    dependency_map = _roadmap_core_dependency_map()

    assert dependency_map["matplotlib"] == "matplotlib-base"
    assert dependency_map["typing-extensions"] == "typing-extensions"


def test_conda_forge_roadmap_uses_build_entry_points_for_console_scripts():
    recipe = _roadmap_initial_recipe()

    assert "entry_points:" in recipe
    assert "- gwexpy = gwexpy.cli:main" in recipe
    assert "gwexpy.gui = gwexpy.gui.pyaggui:main" not in recipe
    assert "build:\n  python:" not in recipe


def test_pyproject_excludes_experimental_gui_console_script_from_first_pypi():
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts == {"gwexpy": "gwexpy.cli:main"}
    assert "gui" in pyproject["project"]["optional-dependencies"]
    assert "gui" not in pyproject["project"]["optional-dependencies"]["all"]


def test_conda_forge_roadmap_records_noarch_python_recipe_test_coverage():
    recipe = _roadmap_initial_recipe()

    assert "pip_check: true" in recipe
    assert "python_version:" in recipe
    assert "3.11.*" in recipe
    assert '"*"' in recipe
