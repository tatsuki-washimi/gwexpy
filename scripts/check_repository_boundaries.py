#!/usr/bin/env python3
"""Prevent legacy GUI source and packaging from re-entering GWexpy."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATHS = (
    "gwexpy/gui",
    "tests/gui",
    "tests/e2e/test_gui_smoke.py",
    "tests/nds/test_gui_nds_smoke.py",
    "tests/run_gui_tests.sh",
    "tests/run_gui_nds_tests.sh",
    "docs/web/en/reference/api/gui.rst",
    "docs/web/ja/reference/api/gui.rst",
    "docs_redesign/reference/api/gui.rst",
)


def main() -> int:
    """Fail when GUI implementation, tests or packaging settings return."""
    found = [path for path in FORBIDDEN_PATHS if (ROOT / path).exists()]
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for needle in ('"gwexpy.gui*"', '"gwexpy/gui/*"', '"pytest-qt"', '"pytest-xvfb"'):
        if needle in pyproject:
            found.append(f"pyproject.toml: {needle}")
    if found:
        for item in found:
            print(f"Legacy GUI boundary violation: {item}")
        return 1
    print("Success: GWexpy contains no embedded GUI implementation/configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
