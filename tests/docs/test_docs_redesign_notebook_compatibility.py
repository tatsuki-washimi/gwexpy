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


def test_advanced_correlation_relies_on_product_granger_compatibility() -> None:
    source = _code_cell_containing(
        "docs_redesign/how-to/fitting/advanced_correlation.ipynb",
        "granger_causality",
    )

    assert "except ImportError:" in source
    assert "statsmodels package is not installed" in source
    assert "except TypeError" not in source
    assert "unexpected keyword argument 'verbose'" not in source
