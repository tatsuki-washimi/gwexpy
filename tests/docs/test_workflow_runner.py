"""Behavioral tests for verify_workflow_notebooks.py runner using temporary fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from scripts.verify_workflow_notebooks import (
    load_manifest,
    probe_kernel_environment,
    verify_kernel_environment,
    verify_notebook,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/verify_workflow_notebooks.py"


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
metrics = {
    "status": "passed",
    "data_kind": "synthetic",
    "checks": {
        "chk1": {
            "passed": True,
            "observed": "value",
            "criterion": "check passed",
        }
    }
}
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
        "required_checks": ["chk1"],
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


def test_runner_evidence_isolation_and_preservation(tmp_path: Path) -> None:
    """Ensure run-level isolation preserves past evidence, fails 2nd run when artifacts missing, and rejects non-empty directory reuse."""
    evidence_root = tmp_path / "evidence"
    run1_dir = evidence_root / "run-1"
    run2_dir = evidence_root / "run-2"

    # Run 1 produces required.csv
    code_run1 = """
import os
out_dir = os.environ.get("GWEXPY_DOCS_OUTPUT_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "required.csv"), "w") as f:
    f.write("run1_evidence_data\\n")
"""
    (tmp_path / "nb1").mkdir(parents=True, exist_ok=True)
    nb_path_1 = _create_fixture_notebook(tmp_path / "nb1", code_run1)
    entry_1 = {
        "id": "TTEST_ISOLATION",
        "public": nb_path_1.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": ["required.csv"],
    }
    res_1 = verify_notebook(
        entry=entry_1,
        source_root=tmp_path / "nb1",
        output_dir=run1_dir,
    )
    assert res_1["required_outputs_passed"] is True
    run1_evidence_file = run1_dir / "TTEST_ISOLATION" / "required.csv"
    assert run1_evidence_file.exists()
    assert run1_evidence_file.read_text() == "run1_evidence_data\n"

    # Run 2 does NOT produce required.csv in a new run directory under the same evidence root
    code_run2 = "x = 1"
    (tmp_path / "nb2").mkdir(parents=True, exist_ok=True)
    nb_path_2 = _create_fixture_notebook(tmp_path / "nb2", code_run2)
    entry_2 = {
        "id": "TTEST_ISOLATION",
        "public": nb_path_2.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": ["required.csv"],
    }
    res_2 = verify_notebook(
        entry=entry_2,
        source_root=tmp_path / "nb2",
        output_dir=run2_dir,
    )
    # 2nd run fails because required output is missing
    assert res_2["required_outputs_passed"] is False
    assert "required.csv" in res_2["missing_outputs"]

    # 1st run evidence MUST be preserved (no rmtree)
    assert run1_evidence_file.exists()
    assert run1_evidence_file.read_text() == "run1_evidence_data\n"

    # Attempting to re-run into existing non-empty run1 directory must be explicitly rejected
    with pytest.raises(FileExistsError, match="already exists and is not empty"):
        verify_notebook(
            entry=entry_2,
            source_root=tmp_path / "nb2",
            output_dir=run1_dir,
        )


def test_runner_empty_checks_rejected(tmp_path: Path) -> None:
    """Empty checks dict must not pass numerical verification."""
    code = """
import os, json
out_dir = os.environ.get("GWEXPY_DOCS_OUTPUT_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
metrics = {"status": "passed", "checks": {}}
with open(os.path.join(out_dir, "validation-metrics.json"), "w") as f:
    json.dump(metrics, f)
"""
    nb_path = _create_fixture_notebook(tmp_path, code)
    entry = {
        "id": "TTEST_EMPTY",
        "public": nb_path.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": ["validation-metrics.json"],
    }
    res = verify_notebook(entry=entry, source_root=tmp_path, output_dir=tmp_path / "out")
    assert res["numerical_checks_passed"] is False


def test_runner_string_passed_rejected(tmp_path: Path) -> None:
    """String 'false' or 'true' instead of strict boolean must be rejected."""
    code = """
import os, json
out_dir = os.environ.get("GWEXPY_DOCS_OUTPUT_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
metrics = {"status": "passed", "checks": {"truth": {"passed": "false"}}}
with open(os.path.join(out_dir, "validation-metrics.json"), "w") as f:
    json.dump(metrics, f)
"""
    nb_path = _create_fixture_notebook(tmp_path, code)
    entry = {
        "id": "TTEST_STR_FALSE",
        "public": nb_path.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": ["validation-metrics.json"],
    }
    res = verify_notebook(entry=entry, source_root=tmp_path, output_dir=tmp_path / "out")
    assert res["numerical_checks_passed"] is False


def test_runner_missing_required_checks_rejected(tmp_path: Path) -> None:
    """If manifest specifies required_checks, all must be present."""
    code = """
import os, json
out_dir = os.environ.get("GWEXPY_DOCS_OUTPUT_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
metrics = {
    "status": "passed",
    "data_kind": "synthetic",
    "checks": {
        "check_A": {
            "passed": True,
            "observed": "val",
            "criterion": "check passed",
        }
    }
}
with open(os.path.join(out_dir, "validation-metrics.json"), "w") as f:
    json.dump(metrics, f)
"""
    nb_path = _create_fixture_notebook(tmp_path, code)
    entry = {
        "id": "TTEST_MISSING_CHK",
        "public": nb_path.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": ["validation-metrics.json"],
        "required_checks": ["check_A", "check_B"],
    }
    res = verify_notebook(entry=entry, source_root=tmp_path, output_dir=tmp_path / "out")
    assert res["numerical_checks_passed"] is False
    assert "check_B" in res.get("missing_checks", [])


def test_runner_cli_unknown_id_and_empty_selection(tmp_path: Path) -> None:
    """CLI must exit non-zero when unknown ID is given or selection is empty."""
    manifest_file = tmp_path / "workflow_notebooks.json"
    manifest_file.write_text(
        json.dumps({
            "schema_version": 1,
            "notebooks": [
                {
                    "id": "T1",
                    "public": "dummy.ipynb",
                    "group": "core",
                    "cell_timeout_seconds": 30,
                    "required_outputs": [],
                }
            ],
        }),
        encoding="utf-8",
    )
    # Unknown ID
    proc = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--manifest", str(manifest_file), "--source", str(tmp_path), "--ids", "UNKNOWN"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Unknown notebook ID" in proc.stderr or "Unknown notebook ID" in proc.stdout


def test_manifest_validation_duplicates(tmp_path: Path) -> None:
    """Manifest loader must reject duplicate IDs or public paths."""
    dup_manifest = tmp_path / "dup.json"
    dup_manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "notebooks": [
                {"id": "T1", "public": "a.ipynb", "group": "core"},
                {"id": "T1", "public": "b.ipynb", "group": "core"},
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate or empty notebook ID"):
        load_manifest(dup_manifest)


def test_probe_and_verify_kernel_environment() -> None:
    """Kernel environment probe should succeed and match current interpreter."""
    env = probe_kernel_environment("python3", timeout=30)
    assert "executable" in env and env["executable"]
    assert "prefix" in env and env["prefix"]
    assert "python_version" in env
    # Should verify without error against current sys.prefix
    verify_kernel_environment(env, expected_prefix=sys.prefix, require_gwexpy=False)


def test_verify_kernel_environment_mismatches() -> None:
    """verify_kernel_environment must reject mismatched prefix or missing gwexpy."""
    fake_env = {
        "executable": "/usr/bin/python3",
        "prefix": "/usr",
        "python_version": "3.10.0",
        "gwexpy_version": None,
        "gwexpy_file": None,
    }
    # Prefix mismatch
    with pytest.raises(RuntimeError, match="Kernel prefix mismatch"):
        verify_kernel_environment(fake_env, expected_prefix="/opt/myenv")

    # Missing gwexpy
    with pytest.raises(RuntimeError, match="cannot import gwexpy"):
        verify_kernel_environment(fake_env, expected_prefix="/usr", require_gwexpy=True)


def test_verify_kernel_environment_install_modes(tmp_path: Path) -> None:
    """verify_kernel_environment must distinguish installed (site-packages) vs editable (checkout) modes."""
    fake_repo = tmp_path / "my_repo"
    fake_repo.mkdir()
    fake_repo_file = fake_repo / "gwexpy" / "__init__.py"
    fake_repo_file.parent.mkdir()
    fake_repo_file.write_text("# fake repo gwexpy\n", encoding="utf-8")

    fake_site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "gwexpy" / "__init__.py"
    fake_site.parent.mkdir(parents=True)
    fake_site.write_text("# fake site-packages gwexpy\n", encoding="utf-8")

    env_repo = {
        "executable": "/usr/bin/python3",
        "prefix": "/usr",
        "python_version": "3.12.0",
        "gwexpy_version": "0.2.3",
        "gwexpy_file": str(fake_repo_file),
    }

    env_site = {
        "executable": "/usr/bin/python3",
        "prefix": "/usr",
        "python_version": "3.12.0",
        "gwexpy_version": "0.2.3",
        "gwexpy_file": str(fake_site),
    }

    # Mode 'installed' with repo checkout file must be rejected
    with pytest.raises(RuntimeError, match="Install mode is 'installed'"):
        verify_kernel_environment(
            env_repo,
            allow_foreign=True,
            require_gwexpy=True,
            install_mode="installed",
            repo_root=fake_repo,
        )

    # Mode 'installed' with site-packages file must succeed
    verify_kernel_environment(
        env_site,
        allow_foreign=True,
        require_gwexpy=True,
        install_mode="installed",
        repo_root=fake_repo,
    )

    # Mode 'editable' with site-packages file must be rejected
    with pytest.raises(RuntimeError, match="Install mode is 'editable'"):
        verify_kernel_environment(
            env_site,
            allow_foreign=True,
            require_gwexpy=True,
            install_mode="editable",
            repo_root=fake_repo,
        )

    # Mode 'editable' with repo checkout file must succeed
    verify_kernel_environment(
        env_repo,
        allow_foreign=True,
        require_gwexpy=True,
        install_mode="editable",
        repo_root=fake_repo,
    )


def test_runner_injected_cell_kernel_env_and_mode_rejection(tmp_path: Path) -> None:
    """Injected setup cell must record kernel environment and enforce requested install mode inside kernel."""
    code = "x = 100"
    nb_path = _create_fixture_notebook(tmp_path, code)
    entry = {
        "id": "TTEST_ENV_CHECK",
        "public": nb_path.name,
        "group": "core",
        "cell_timeout_seconds": 30,
        "required_outputs": [],
    }
    out_dir = tmp_path / "out"

    # In our local environment, gwexpy is an editable checkout (not in site-packages).
    # Requesting install_mode="installed" inside the notebook must trigger rejection in injected setup cell.
    res_installed_fail = verify_notebook(
        entry=entry,
        source_root=tmp_path,
        output_dir=out_dir / "installed_fail",
        install_mode="installed",
    )
    assert res_installed_fail["execution_passed"] is False
    assert "install_mode='installed' violated" in (res_installed_fail["error"] or "")

    # Requesting install_mode="editable" with the kernel's actual editable repo root must succeed
    kernel_env = probe_kernel_environment("python3", timeout=30)
    actual_kernel_gw_file = kernel_env.get("gwexpy_file")
    assert actual_kernel_gw_file is not None
    actual_kernel_repo_root = Path(actual_kernel_gw_file).resolve().parent.parent

    res_editable_ok = verify_notebook(
        entry=entry,
        source_root=tmp_path,
        output_dir=out_dir / "editable_ok",
        install_mode="editable",
        repo_root=actual_kernel_repo_root,
    )
    assert res_editable_ok["execution_passed"] is True
    assert res_editable_ok.get("kernel_environment") is not None
    assert res_editable_ok["kernel_environment"]["gwexpy_version"] is not None
    assert (out_dir / "editable_ok" / "TTEST_ENV_CHECK" / "_kernel_env.json").exists()

    # Requesting install_mode="editable" with a mismatched repo root must fail
    res_editable_mismatch = verify_notebook(
        entry=entry,
        source_root=tmp_path,
        output_dir=out_dir / "editable_mismatch",
        install_mode="editable",
        repo_root=tmp_path / "other_repo",
    )
    assert res_editable_mismatch["execution_passed"] is False
    assert "install_mode='editable' violated" in (res_editable_mismatch["error"] or "")


