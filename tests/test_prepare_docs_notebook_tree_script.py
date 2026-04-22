from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

TUTORIALS_ROOT = ("docs", "web", "en", "user_guide", "tutorials")


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
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_notebook(path: Path, *, tags: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"tags": tags or []},
                "outputs": [],
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


def mutate_notebook(path: Path, marker: str) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [f"print('{marker}')\n"]
    path.write_text(json.dumps(notebook), encoding="utf-8")


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.com"],
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "CI"], cwd=path, check=True)


def commit_all(path: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_iter_docs_notebooks_skips_display_only(tmp_path: Path):
    module = load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    init_git_repo(repo_root)

    write_notebook(repo_root.joinpath(*TUTORIALS_ROOT, "keep.ipynb"))
    write_notebook(
        repo_root.joinpath(*TUTORIALS_ROOT, "display.ipynb"),
        tags=["display-only"],
    )
    commit_all(repo_root, "init")

    notebooks = module._iter_docs_notebooks(repo_root)

    assert notebooks == ["docs/web/en/user_guide/tutorials/keep.ipynb"]


def test_list_changed_docs_notebooks_filters_non_docs_and_display_only(tmp_path: Path):
    module = load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    init_git_repo(repo_root)

    tracked = repo_root / "docs" / "web" / "en" / "user_guide" / "tutorials"
    write_notebook(tracked / "keep.ipynb")
    write_notebook(tracked / "display.ipynb", tags=["display-only"])
    write_notebook(repo_root / "examples" / "sample.ipynb")
    commit_all(repo_root, "init")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    mutate_notebook(tracked / "keep.ipynb", "changed")
    mutate_notebook(tracked / "display.ipynb", "changed-display")
    mutate_notebook(repo_root / "examples" / "sample.ipynb", "changed-example")
    commit_all(repo_root, "update")

    changed = module._list_changed_docs_notebooks(repo_root, base, "HEAD")

    assert changed == ["docs/web/en/user_guide/tutorials/keep.ipynb"]


def test_copy_repo_tree_uses_tracked_files_only(tmp_path: Path):
    module = load_script_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    init_git_repo(repo_root)

    tracked_file = repo_root / "docs" / "conf.py"
    tracked_file.parent.mkdir(parents=True, exist_ok=True)
    tracked_file.write_text("project = 'test'\n", encoding="utf-8")
    commit_all(repo_root, "init")
    untracked_file = repo_root / "docs" / "scratch.txt"
    untracked_file.write_text("ignore me\n", encoding="utf-8")

    output_root = tmp_path / "out"
    module._copy_repo_tree(repo_root, output_root)

    assert (output_root / "docs" / "conf.py").exists()
    assert (output_root / "docs" / "scratch.txt").exists() is False


def test_sanitize_notebook_for_ci_replaces_bootstrap_install_cells(tmp_path: Path):
    module = load_script_module()
    notebook_path = tmp_path / "bootstrap.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [{"output_type": "stream", "text": "ok\n"}],
                "source": [
                    "# Install gwexpy with pinned versions of core dependencies for reproducibility on Colab\n",
                    "%pip install -q \"gwexpy[all]\" \"gwpy<5.0.0\" \"numpy<2.0.0\" \"scipy<1.13.0\" \"astropy<7.0.0\"\n",
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    changed = module._sanitize_notebook_for_ci(notebook_path)
    rewritten = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert changed is True
    assert rewritten["cells"][0]["source"] == [module.CI_SKIP_BOOTSTRAP_COMMENT]
    assert rewritten["cells"][0]["outputs"] == []
    assert rewritten["cells"][0]["execution_count"] is None
