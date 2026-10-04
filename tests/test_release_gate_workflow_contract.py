"""Static contracts for dedicated release gates."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PR_FAST = REPO_ROOT / ".github" / "workflows" / "pr-fast.yml"
SETUP_ACTION = REPO_ROOT / ".github" / "actions" / "setup-gwexpy" / "action.yml"


def test_pr_fast_runs_the_dedicated_release_gates() -> None:
    workflow = PR_FAST.read_text(encoding="utf-8")
    validate_job = workflow.split("\n  io-contract-gate:", maxsplit=1)[0]

    for gate in ("io-gwf", "io-netcdf", "io-mth5", "interop-root"):
        assert f"run: python scripts/ci/run_gate.py {gate}" in workflow

    assert 'extras: "xarray netCDF4"' in workflow
    assert "name: MTH5 I/O gate (${{ matrix.mth5-spec }})" in workflow
    assert 'mth5-spec: ["mth5==0.6.8", "mth5"]' in workflow
    assert "extras: \"${{ matrix.mth5-spec }} 'obspy<2'\"" in workflow
    assert 'conda-packages: "root_base=6.36.*"' in workflow
    assert "fetch-depth: 0" in validate_job


def test_setup_action_accepts_job_scoped_conda_packages() -> None:
    action = SETUP_ACTION.read_text(encoding="utf-8")

    assert "conda-packages:" in action
    assert "${{ inputs.conda-packages }}" in action


def test_candidate_lanes_consume_one_build_and_original_sidecars() -> None:
    import yaml

    jobs = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/publish-release.yml").read_text()
    )["jobs"]
    for name in ("smoke", "qualify", "diaggui_qualification", "cross_format_io"):
        job = jobs[name]
        assert {"verify", "build"} <= set(job["needs"])
        downloads = [
            step["with"]
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        ]
        assert any(
            download.get("name")
            == "release-payload-${{ needs.verify.outputs.source_sha }}"
            for download in downloads
        )
        assert any(
            download.get("pattern")
            == "release-sidecar-*-${{ needs.verify.outputs.source_sha }}"
            and download.get("merge-multiple") is True
            for download in downloads
        )
        assert not any(
            "python -m build" in step.get("run", "") for step in job["steps"]
        )
    finalizer = jobs["promotion_manifest"]
    assert all(
        "needs." + gate + ".result == 'success'" in finalizer["if"]
        for gate in finalizer["needs"]
    )
    assert "historical_74_gate" not in finalizer["needs"]
