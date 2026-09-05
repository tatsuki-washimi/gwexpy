#!/usr/bin/env python3
"""Prepare disposable public docs with canonical notebook code and build identity."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_cells(notebook: dict) -> list[dict]:
    """Select executable lesson cells, excluding legacy installation bootstrap."""
    return [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and not any(token in "".join(cell["source"]) for token in ("%pip", "!pip"))
    ]


def canonicalize(public: dict, canonical: dict) -> dict:
    """Keep public prose/gettext identities while deriving execution cells."""
    targets, sources = code_cells(public), code_cells(canonical)
    if len(targets) != len(sources):
        raise ValueError(
            "Canonical/public code-cell count differs; reconcile lesson structure first"
        )
    for target, source in zip(targets, sources):
        target["source"] = source["source"]
        target["outputs"] = []
        target["execution_count"] = None
        target.get("metadata", {}).pop("execution", None)
    return public


def prepare(output: Path, root: Path = ROOT) -> None:
    """Copy sources outside the checkout, then resolve canonical code and revision."""
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Use an output directory outside the repository")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source = root / "docs_redesign"
    shutil.copytree(
        source, output, ignore=shutil.ignore_patterns("_build", "__pycache__", "*.mo")
    )
    shutil.copy2(root / "CHANGELOG.md", output.parent / "CHANGELOG.md")
    notebooks = {}
    for path in sorted(output.rglob("*.ipynb")):
        canonical = root / "docs/web/en/user_guide/tutorials" / path.name
        public = json.loads(path.read_text())
        public = canonicalize(public, json.loads(canonical.read_text()))
        path.write_text(json.dumps(public, ensure_ascii=False, indent=1) + "\n")
        notebooks[path.relative_to(output).as_posix()] = canonical.relative_to(
            root
        ).as_posix()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    source_ref = (
        os.environ.get("GITHUB_REF_NAME")
        or subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip()
    )
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    )
    info = {
        "source_revision": revision,
        "source_ref": source_ref,
        "dirty": dirty,
        "notebook_sources": notebooks,
    }
    (output / "_build_identity.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Prepared {len(notebooks)} canonical notebooks at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
