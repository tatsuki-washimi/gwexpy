from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "notebook_gen"
        / "prepare_docs_notebook_tree.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_docs_notebook_tree", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_output_root_outside_repository_is_accepted(tmp_path: Path):
    module = load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    module._check_output_root(repo_root, tmp_path / "build")


@pytest.mark.parametrize("layout", ["same", "inside", "parent"])
def test_output_root_overlapping_repository_is_refused(tmp_path: Path, layout: str):
    """The output tree is removed before copying, so overlaps must fail closed."""
    module = load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output_root = {
        "same": repo_root,
        "inside": repo_root / "docs",
        "parent": tmp_path,
    }[layout]

    with pytest.raises(ValueError, match="outside the repository"):
        module._check_output_root(repo_root, output_root)
