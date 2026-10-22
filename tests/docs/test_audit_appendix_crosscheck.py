"""Regression tests for fail-closed B/C/D audit crosschecks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "deliverables/audit-appendix-artifacts-20261017"
SCRIPT = ARTIFACT_DIR / "crosscheck_audit_appendix_stage2.py"
D_ZIP = REPO_ROOT / "deliverables/agent-deliverables-20261017-175223.zip"
D_MEMBER = "external/pass2-surface-coverage.jsonl"
B_C_FILES = (
    "pass2-population-and-dispositions.json",
    "pass1-population.json",
    "pass2-api-status.jsonl",
    "pass2-module-functions.txt",
)


def _copy_b_c_artifacts(destination: Path) -> Path:
    artifact_dir = destination / "artifacts"
    artifact_dir.mkdir()
    for filename in B_C_FILES:
        shutil.copyfile(ARTIFACT_DIR / filename, artifact_dir / filename)
    return artifact_dir


def _run_crosscheck(tmp_path: Path, artifact_dir: Path, *extra: str):
    report = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-dir",
            str(artifact_dir),
            *extra,
            "--output",
            str(report),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(report.read_text(encoding="utf-8"))


def test_d_ledger_present_passes_complete_crosscheck(tmp_path: Path) -> None:
    completed, report = _run_crosscheck(
        tmp_path,
        ARTIFACT_DIR,
        "--d-zip",
        str(D_ZIP),
    )

    assert completed.returncode == 0, completed.stderr
    assert report["passed"] is True
    assert report["checks"]["b_d_surface_2650"]["d_row_count"] == 2650
    assert report["checks"]["b_d_surface_2650"]["field_mismatch_count"] == 0


def test_d_ledger_absent_fails_and_records_skip(tmp_path: Path) -> None:
    artifact_dir = _copy_b_c_artifacts(tmp_path)

    completed, report = _run_crosscheck(tmp_path, artifact_dir)

    assert completed.returncode != 0
    assert report["passed"] is False
    assert report["checks"]["b_d_surface_2650"]["skipped"] is True


def test_invalid_explicit_d_path_does_not_fall_back_to_default(
    tmp_path: Path,
) -> None:
    missing_zip = tmp_path / "missing-d.zip"
    completed, report = _run_crosscheck(
        tmp_path,
        ARTIFACT_DIR,
        "--d-zip",
        str(missing_zip),
    )

    assert completed.returncode != 0
    assert report["passed"] is False
    assert report["checks"]["b_d_surface_2650"]["skipped"] is True
    assert "explicit" in report["checks"]["b_d_surface_2650"]["reason"]
    assert "d_source" not in report


def test_b_d_mismatch_fails(tmp_path: Path) -> None:
    d_jsonl = tmp_path / "d-surface.jsonl"
    with zipfile.ZipFile(D_ZIP) as archive:
        rows = archive.read(D_MEMBER).decode("utf-8").splitlines()
    first = json.loads(rows[0])
    first["id"] = f"mismatch-{first['id']}"
    rows[0] = json.dumps(first)
    d_jsonl.write_text("\n".join(rows) + "\n", encoding="utf-8")

    completed, report = _run_crosscheck(
        tmp_path,
        ARTIFACT_DIR,
        "--d-surface-jsonl",
        str(d_jsonl),
    )

    assert completed.returncode != 0
    assert report["passed"] is False
    assert report["checks"]["b_d_surface_2650"]["field_mismatch_count"] > 0
