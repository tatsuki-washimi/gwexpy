"""Boundary tests ensuring legacy GUI does not re-enter the repository."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def test_no_legacy_gui_directories():
    """Ensure gwexpy/gui and tests/gui directories do not exist."""
    assert not (ROOT / "gwexpy" / "gui").exists(), "gwexpy/gui must not exist"
    assert not (ROOT / "tests" / "gui").exists(), "tests/gui must not exist"


def test_gwexpy_gui_not_importable():
    """Ensure gwexpy.gui is not an importable module or package."""
    with pytest.raises((ModuleNotFoundError, ImportError)):
        importlib.import_module("gwexpy.gui")


def test_pyproject_gui_boundary_cleanliness():
    """Ensure pyproject.toml has no GUI dependencies, markers, or ruff ignores."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    # Check dependencies
    dev_deps = pyproject["project"]["optional-dependencies"].get("dev", [])
    assert not any("qt" in dep.lower() or "gui" in dep.lower() for dep in dev_deps)

    # Check pytest markers
    pytest_markers = (
        pyproject.get("tool", {})
        .get("pytest", {})
        .get("ini_options", {})
        .get("markers", [])
    )
    for marker in pytest_markers:
        assert not marker.startswith("gui:"), f"Stale GUI marker found: {marker}"
        assert not marker.startswith("pyautogui:"), (
            f"Stale pyautogui marker found: {marker}"
        )

    # Check ruff per-file-ignores
    ruff_ignores = (
        pyproject.get("tool", {})
        .get("ruff", {})
        .get("lint", {})
        .get("per-file-ignores", {})
    )
    assert "gwexpy/gui/*" not in ruff_ignores
