"""Behavioral tests for verify_workflow_notebooks.py runner using temporary fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from scripts.verify_workflow_notebooks import verify_notebook

ROOT = Path(__file__).resolve().parents[2]


def _create_fixture_notebook(tmp_path: Path, code: str) -> Path:
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell(source=code)
    cell["id"] = "test-cell-001"
    nb.cells.append(cell)
    nb_path = tmp_path / "fixture.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return nb_path


def test_runner_success_fixture(tmp_path: Path) -> None:
    code = """
import os, json
out_dir = os.environ.get("GWEXPY_DOCS_OUTPUT_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "output.txt"), "w") as f:
    f.write("ok")
metrics = {"status": "passed", "checks": {"chk1": {"passed": True}}}
with open(os.path.join(out_dir, "validation-metrics.json"), "w") as f:
    json.dump(metrics, f)
"""
    nb_path = _create_fixture_notebook(tmp_path, code)
    entry = {
        "id": "TTEST_OK",
        "public": nb_path.name,
        "canonical": nb_path.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": ["output.txt", "validation-metrics.json"],
    }
    out_dir = tmp_path / "out"
    res = verify_notebook(
        entry=entry,
        source_root=tmp_path,
        output_dir=out_dir,
    )
    assert res["execution_passed"] is True
    assert res["required_outputs_passed"] is True
    assert res["numerical_checks_passed"] is True
    assert (out_dir / "TTEST_OK" / "output.txt").exists()


def test_runner_assert_failure_fixture(tmp_path: Path) -> None:
    code = "assert False, 'Expected failure'"
    nb_path = _create_fixture_notebook(tmp_path, code)
    entry = {
        "id": "TTEST_FAIL",
        "public": nb_path.name,
        "canonical": nb_path.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": [],
    }
    out_dir = tmp_path / "out"
    res = verify_notebook(
        entry=entry,
        source_root=tmp_path,
        output_dir=out_dir,
    )
    assert res["execution_passed"] is False
    assert "Expected failure" in (res["error"] or "")


def test_runner_missing_artifact_fixture(tmp_path: Path) -> None:
    code = "x = 42"
    nb_path = _create_fixture_notebook(tmp_path, code)
    entry = {
        "id": "TTEST_MISSING",
        "public": nb_path.name,
        "canonical": nb_path.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": ["nonexistent_artifact.csv"],
    }
    out_dir = tmp_path / "out"
    res = verify_notebook(
        entry=entry,
        source_root=tmp_path,
        output_dir=out_dir,
    )
    assert res["execution_passed"] is True
    assert res["required_outputs_passed"] is False
    assert "nonexistent_artifact.csv" in res["missing_outputs"]


def test_runner_offline_blocks_remote_socket(tmp_path: Path) -> None:
    code = """
import socket
s = socket.socket()
s.connect(("8.8.8.8", 53))
"""
    nb_path = _create_fixture_notebook(tmp_path, code)
    entry = {
        "id": "TTEST_OFFLINE",
        "public": nb_path.name,
        "canonical": nb_path.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": [],
    }
    out_dir = tmp_path / "out"
    res = verify_notebook(
        entry=entry,
        source_root=tmp_path,
        output_dir=out_dir,
        offline=True,
    )
    assert res["execution_passed"] is False
    assert "Offline execution policy" in (res["error"] or "")
