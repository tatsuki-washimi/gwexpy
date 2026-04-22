#!/usr/bin/env python3
"""Strip transient outputs from tracked notebooks before commit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPLAY_ONLY_TAG = "display-only"
TRACKED_NOTEBOOK_PREFIXES = ("docs/web/", "examples/")


def _normalize(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_display_only(notebook: dict) -> bool:
    if not notebook.get("cells"):
        return False
    metadata = notebook["cells"][0].get("metadata", {})
    tags = metadata.get("tags", [])
    return isinstance(tags, list) and DISPLAY_ONLY_TAG in tags


def strip_notebook(path: Path) -> bool:
    """Strip outputs and execution counts from one tracked source notebook."""
    notebook = _load_notebook(path)
    if _is_display_only(notebook):
        return False

    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if changed:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return changed


def main(argv: list[str]) -> int:
    """Strip transient output from tracked notebook paths listed in ``argv``."""
    changed_paths: list[str] = []

    for original_path in argv:
        normalized_path = _normalize(original_path)
        if not normalized_path.endswith(".ipynb"):
            continue
        if not normalized_path.startswith(TRACKED_NOTEBOOK_PREFIXES):
            continue

        path = REPO_ROOT / normalized_path
        if not path.exists():
            continue

        if strip_notebook(path):
            changed_paths.append(normalized_path)

    if not changed_paths:
        return 0

    print("Stripped outputs from tracked notebooks:")
    for path in changed_paths:
        print(f"  - {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
