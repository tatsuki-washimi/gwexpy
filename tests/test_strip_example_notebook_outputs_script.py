from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "strip_example_notebook_outputs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "strip_example_notebook_outputs", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_notebook(
    path: Path,
    *,
    tags: list[str] | None = None,
    outputs: list[dict] | None = None,
    execution_count: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": execution_count,
                "metadata": {"tags": tags or []},
                "outputs": outputs or [],
                "source": ["print('ok')\n"],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_strip_notebook_clears_public_docs_outputs(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "docs" / "web" / "en" / "user_guide" / "tutorials" / "sample.ipynb"
    write_notebook(
        notebook_path,
        outputs=[{"output_type": "stream", "name": "stdout", "text": "ok\n"}],
        execution_count=3,
    )

    changed = module.strip_notebook(notebook_path)
    notebook = read_notebook(notebook_path)

    assert changed is True
    assert notebook["cells"][0]["outputs"] == []
    assert notebook["cells"][0]["execution_count"] is None


def test_strip_notebook_clears_example_outputs(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "examples" / "sample.ipynb"
    write_notebook(
        notebook_path,
        outputs=[{"output_type": "stream", "name": "stdout", "text": "ok\n"}],
        execution_count=3,
    )

    changed = module.strip_notebook(notebook_path)
    notebook = read_notebook(notebook_path)

    assert changed is True
    assert notebook["cells"][0]["outputs"] == []
    assert notebook["cells"][0]["execution_count"] is None


def test_strip_notebook_leaves_display_only_example_intact(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "examples" / "display_only.ipynb"
    outputs = [{"output_type": "stream", "name": "stdout", "text": "ok\n"}]
    write_notebook(
        notebook_path,
        tags=["display-only"],
        outputs=outputs,
        execution_count=3,
    )

    changed = module.strip_notebook(notebook_path)
    notebook = read_notebook(notebook_path)

    assert changed is False
    assert notebook["cells"][0]["outputs"] == outputs
    assert notebook["cells"][0]["execution_count"] == 3
