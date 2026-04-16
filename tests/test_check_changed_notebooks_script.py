from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "notebook_gen" / "check_changed_notebooks.py"
    spec = importlib.util.spec_from_file_location("check_changed_notebooks", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_notebook(path: Path, tags: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"tags": tags or []},
                "source": [],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def test_classify_notebook_tags(tmp_path: Path):
    module = load_script_module()

    light = tmp_path / "light.ipynb"
    heavy = tmp_path / "heavy.ipynb"
    display_only = tmp_path / "display.ipynb"

    write_notebook(light)
    write_notebook(heavy, tags=["ci-heavy"])
    write_notebook(display_only, tags=["display-only"])

    assert module.classify_notebook(light) == "light"
    assert module.classify_notebook(heavy) == "heavy"
    assert module.classify_notebook(display_only) == "display-only"


def test_filter_changed_notebooks_skips_build_and_missing(tmp_path: Path):
    module = load_script_module()

    keep = tmp_path / "docs" / "ok.ipynb"
    skip_build = tmp_path / "docs" / "_build" / "skip.ipynb"
    skip_checkpoint = tmp_path / ".ipynb_checkpoints" / "skip.ipynb"

    write_notebook(keep)
    write_notebook(skip_build)
    write_notebook(skip_checkpoint)

    filtered = module.filter_changed_notebooks(
        [
            str(keep.relative_to(tmp_path)),
            str(skip_build.relative_to(tmp_path)),
            str(skip_checkpoint.relative_to(tmp_path)),
            "docs/missing.ipynb",
            "docs/not_a_notebook.txt",
        ],
        repo_root=tmp_path,
    )

    assert filtered == [keep]
