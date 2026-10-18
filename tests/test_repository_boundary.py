"""Repository split guard: GUI application code belongs in external repositories."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import gwexpy

ROOT = Path(__file__).resolve().parents[1]


def test_gwexpy_package_exposes_no_legacy_gui() -> None:
    assert importlib.util.find_spec("gwexpy.gui") is None


def test_repository_boundary_gate_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_repository_boundaries.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
