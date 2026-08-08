"""Static contracts for v0.1.13 dedicated release gates."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PR_FAST = REPO_ROOT / ".github" / "workflows" / "pr-fast.yml"
SETUP_ACTION = REPO_ROOT / ".github" / "actions" / "setup-gwexpy" / "action.yml"


def test_pr_fast_runs_the_v0113_dedicated_release_gates() -> None:
    workflow = PR_FAST.read_text(encoding="utf-8")

    for gate in ("io-gwf", "io-netcdf", "interop-root"):
        assert f"run: python scripts/ci/run_gate.py {gate}" in workflow

    assert 'extras: "xarray netCDF4"' in workflow
    assert 'conda-packages: "root_base=6.36.*"' in workflow


def test_setup_action_accepts_job_scoped_conda_packages() -> None:
    action = SETUP_ACTION.read_text(encoding="utf-8")

    assert "conda-packages:" in action
    assert "${{ inputs.conda-packages }}" in action
