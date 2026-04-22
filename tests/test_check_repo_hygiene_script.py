from __future__ import annotations

import importlib.util
import json
from pathlib import Path

TUTORIALS_ROOT = ("docs", "web", "en", "user_guide", "tutorials")
JA_TUTORIALS_ROOT = ("docs", "web", "ja", "user_guide", "tutorials")


def load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_repo_hygiene.py"
    )
    spec = importlib.util.spec_from_file_location("check_repo_hygiene", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_notebook(
    path: Path,
    *,
    tags: list[str] | None = None,
    cell_metadata: dict | None = None,
    outputs: list[dict] | None = None,
    execution_count: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": execution_count,
                "metadata": {"tags": tags or [], **(cell_metadata or {})},
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


def test_clean_notebook_has_no_violations(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path.joinpath(*TUTORIALS_ROOT, "clean.ipynb")
    write_notebook(notebook_path)

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert violations == []


def test_public_docs_notebook_flags_persisted_outputs_by_default(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path.joinpath(*TUTORIALS_ROOT, "with_outputs.ipynb")
    write_notebook(
        notebook_path,
        outputs=[
            {
                "output_type": "stream",
                "name": "stdout",
                "text": "small output\n",
            }
        ],
    )

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert len(violations) == 1
    assert violations[0].rule == "notebook-outputs-present"
    assert "docs/web/" in violations[0].message


def test_example_notebook_flags_persisted_outputs_by_default(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "examples" / "with_outputs.ipynb"
    write_notebook(
        notebook_path,
        outputs=[
            {
                "output_type": "stream",
                "name": "stdout",
                "text": "small output\n",
            }
        ],
    )

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert len(violations) == 1
    assert violations[0].rule == "notebook-outputs-present"
    assert "examples/" in violations[0].message


def test_display_only_notebook_can_keep_small_outputs(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "examples" / "display_only.ipynb"
    write_notebook(
        notebook_path,
        tags=["display-only"],
        outputs=[
            {
                "output_type": "stream",
                "name": "stdout",
                "text": "small output\n",
            }
        ],
    )

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert violations == []


def test_display_only_notebook_still_flags_oversized_output(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "examples" / "large_display_only.ipynb"
    write_notebook(
        notebook_path,
        tags=["display-only"],
        outputs=[
            {
                "output_type": "stream",
                "name": "stdout",
                "text": "x" * (module.DEFAULT_MAX_OUTPUT_JSON_BYTES + 200),
            }
        ],
    )

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert len(violations) == 1
    assert violations[0].rule == "notebook-output-too-large"
    assert "output[0]" in violations[0].message


def test_notebook_flags_forbidden_cell_metadata(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path.joinpath(*TUTORIALS_ROOT, "metadata.ipynb")
    write_notebook(
        notebook_path,
        cell_metadata={
            "trusted": True,
        },
    )

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert len(violations) == 1
    assert violations[0].rule == "notebook-forbidden-cell-metadata"
    assert "trusted" in violations[0].message


def test_dot_prefixed_forbidden_path_is_reported(tmp_path: Path):
    module = load_script_module()
    artifact_path = tmp_path / ".pytest_cache" / "CACHEDIR.TAG"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "Signature: 8a477f597d28d172789f06886806bc55",
        encoding="utf-8",
    )

    violations = module.check_paths(
        [str(artifact_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert len(violations) == 1
    assert violations[0].rule == "forbidden-artifact-path"
    assert ".pytest_cache/" in violations[0].message


def test_notebook_flags_execution_count_on_clean_source(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "examples" / "execution_count.ipynb"
    write_notebook(notebook_path, execution_count=1)

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert len(violations) == 1
    assert violations[0].rule == "notebook-execution-count-present"
    assert "execution_count" in violations[0].message


def test_public_docs_notebook_flags_execution_count_on_clean_source(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path.joinpath(*JA_TUTORIALS_ROOT, "executed.ipynb")
    write_notebook(notebook_path, execution_count=1)

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert len(violations) == 1
    assert violations[0].rule == "notebook-execution-count-present"


def test_non_docs_notebook_is_not_hygiene_checked(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "tests" / "fixtures" / "retained.ipynb"
    write_notebook(
        notebook_path,
        outputs=[
            {
                "output_type": "stream",
                "name": "stdout",
                "text": "fixture output\n",
            }
        ],
        execution_count=1,
    )

    violations = module.check_paths(
        [str(notebook_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert violations == []


def test_forbidden_generated_artifact_is_reported(tmp_path: Path):
    module = load_script_module()
    artifact_path = tmp_path / "docs" / "_build" / "index.html"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("<html></html>", encoding="utf-8")

    violations = module.check_paths(
        [str(artifact_path.relative_to(tmp_path))],
        repo_root=tmp_path,
    )

    assert len(violations) == 1
    assert violations[0].rule == "forbidden-artifact-path"
    assert "docs/_build/" in violations[0].message
