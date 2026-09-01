"""Regression checks for dependency-sensitive redesigned-site notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _code_cell_containing(relative_path: str, needle: str) -> str:
    notebook = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if needle in source:
            return source
    raise AssertionError(f"No code cell contains {needle!r}")


def test_advanced_correlation_handles_statsmodels_verbose_removal() -> None:
    source = _code_cell_containing(
        "docs_redesign/how-to/fitting/advanced_correlation.ipynb",
        "granger_causality",
    )

    assert "except TypeError as exc:" in source
    assert "unexpected keyword argument 'verbose'" in source
    assert "raise" in source
