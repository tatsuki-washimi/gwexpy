#!/usr/bin/env python3
"""Enforce the GWexpy analysis-library / external-GUI repository boundary (#645)."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOVED_PATHS = (
    "gwexpy/gui",
    "tests/gui",
    "tests/e2e/test_gui_smoke.py",
    "tests/plot/test_gui_plotting_normalize_series.py",
    "tests/run_gui_tests.sh",
    "tests/nds/test_gui_nds_smoke.py",
    "tests/nds/test_nds_channel_list_live_smoke.py",
    "tests/nds/test_nds_thread_live_smoke.py",
)
LIVE_GUIDES = (
    "docs/web/en/user_guide/gui.md",
    "docs/web/ja/user_guide/gui.md",
    "docs_redesign/how-to/gui.md",
)
GUI_ONLY_DEPENDENCIES = {"pyqt5", "pyside6", "pyqtgraph", "qtpy", "pytest-qt", "pytest-xvfb", "pyautogui"}


def main() -> int:
    """Fail if obsolete GUI code, imports, or packaging dependencies return."""
    failures: list[str] = []
    for relative in REMOVED_PATHS:
        if (ROOT / relative).exists():
            failures.append(f"Obsolete GUI path is present: {relative}")

    with (ROOT / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    optional = config["project"].get("optional-dependencies", {})
    if "gui" in optional:
        failures.append("The gui optional dependency extra must remain external")
    requirements = [
        dep.split(";", 1)[0].split("[", 1)[0]
        for group in optional.values()
        for dep in group
    ]
    installed = {
        re.split(r"[<>=!~\s]", dep, maxsplit=1)[0].lower().replace("_", "-")
        for dep in requirements
    }
    for name in sorted(GUI_ONLY_DEPENDENCIES & installed):
        failures.append(f"GUI dependency returned to gwexpy extras: {name}")

    package_find = config.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
    if any("gui" in pattern for pattern in package_find.get("exclude", ())):
        failures.append("The old package-discovery GUI exclusion is still present")

    legacy_import = re.compile(r"^\s*(?:from\s+gwexpy\.gui\b|import\s+gwexpy\.gui\b)", re.M)
    for folder in ("gwexpy", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if legacy_import.search(path.read_text(encoding="utf-8")):
                failures.append(f"Obsolete GUI import in {path.relative_to(ROOT)}")

    for relative in LIVE_GUIDES:
        target = ROOT / relative
        if target.exists():
            text = target.read_text(encoding="utf-8")
            for outdated in ("python -m gwexpy.gui", "from gwexpy.gui"):
                if outdated in text:
                    failures.append(f"Live guide includes removed command: {relative} ({outdated})")

    if failures:
        for problem in failures:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print("GWexpy/GUI repository boundary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
