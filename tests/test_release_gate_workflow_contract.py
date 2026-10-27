"""Static contracts for dedicated release gates."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PR_FAST = REPO_ROOT / ".github" / "workflows" / "pr-fast.yml"
SETUP_ACTION = REPO_ROOT / ".github" / "actions" / "setup-gwexpy" / "action.yml"


def test_actionlint_required_check_runs_for_release_pr_targets_without_path_filter() -> (
    None
):
    import yaml

    workflow = yaml.load(
        (REPO_ROOT / ".github/workflows/actionlint.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    events = workflow["on"]

    assert events["pull_request"]["branches"] == ["main", "maint/0.1", "maint/0.2"]
    assert "paths" not in events["pull_request"]
    assert "paths-ignore" not in events["pull_request"]
    assert events["push"]["branches"] == ["main", "maint/0.1", "maint/0.2"]
    assert "paths" not in events["push"]
    assert "paths-ignore" not in events["push"]
    assert "workflow_dispatch" in events

    job = workflow["jobs"]["actionlint"]
    assert job["name"] == "Release workflow static analysis"
    assert any("gh attestation verify" in step.get("run", "") for step in job["steps"])
    assert any(
        '"${{ steps.actionlint.outputs.executable }}" -color' in step.get("run", "")
        for step in job["steps"]
    )


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
            == "release-sidecar*-${{ needs.verify.outputs.source_sha }}"
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


def test_future_promotion_candidate_graph_is_dispatch_only_but_legacy_tag_builds():
    import yaml

    jobs = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/publish-release.yml").read_text()
    )["jobs"]
    candidate_jobs = (
        "build",
        "smoke",
        "qualify",
        "qualification_evidence",
        "diaggui_qualification",
        "diaggui_qualification_evidence",
        "cross_format_io",
        "cross_format_io_evidence",
        "historical_74_gate",
        "evidence",
    )

    def applies(job_name: str, event: str, version: str, promotion: str) -> bool:
        expression = jobs[job_name]["if"].removeprefix("${{").removesuffix("}}").strip()
        values = {
            "github.event_name": event,
            "inputs.release_mode": "candidate",
            "needs.verify.outputs.version": version,
            "needs.verify.outputs.promotion_enabled": promotion,
            "needs.verify.outputs.qualify": "true",
            "needs.verify.outputs.diaggui_qualification": "true",
            "needs.verify.outputs.cross_format_io": "true",
            "needs.verify.result": "success",
            "needs.build.result": "success",
            "needs.diaggui_qualification.result": "success",
            "needs.cross_format_io.result": "success",
        }
        for key, value in sorted(values.items(), key=lambda item: -len(item[0])):
            expression = expression.replace(key, repr(value))
        expression = (
            expression.replace("!cancelled()", "True")
            .replace("&&", " and ")
            .replace("||", " or ")
        )
        return eval(expression, {"__builtins__": {}}, {})

    for name in candidate_jobs:
        assert applies(name, "workflow_dispatch", "99.88.77", "true"), name
        assert not applies(name, "push", "99.88.77", "true"), name

    # Legacy tag runs retain their historical build and qualification paths.
    for name in candidate_jobs:
        assert applies(name, "push", "0.2.5", "false"), name

    assert "promotion_manifest" in jobs
    assert "workflow_dispatch" in jobs["promotion_manifest"]["if"]
    assert "github.event_name == 'push'" in jobs["promotion_verify"]["if"]


def test_release_workflow_hash_bootstraps_pip_before_network_installs() -> None:
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/publish-release.yml").read_text()
    )
    jobs = workflow["jobs"]
    expectations = {
        "build": (
            "python scripts/ci/bootstrap_trusted_pip.py",
            "python -m pip install --require-hashes -r requirements/release-build.txt",
        ),
        "smoke": (
            "control/scripts/ci/bootstrap_trusted_pip.py",
            '"$smoke_dir/venv/bin/python" -m pip install',
        ),
        "qualify": (
            "source/scripts/ci/bootstrap_trusted_pip.py",
            "python -m pip install --disable-pip-version-check",
        ),
        "diaggui_qualification": (
            "source/scripts/ci/bootstrap_trusted_pip.py",
            '"$python" -m pip install --disable-pip-version-check',
        ),
        "cross_format_io": (
            "source/scripts/ci/bootstrap_trusted_pip.py",
            '"$python" -m pip install --disable-pip-version-check',
        ),
    }
    for job_name, (bootstrap, install) in expectations.items():
        combined = "\n".join(step.get("run", "") for step in jobs[job_name]["steps"])
        assert bootstrap in combined, job_name
        assert combined.index(bootstrap) < combined.index(install), job_name

    qualify_run = next(
        step["run"]
        for step in jobs["qualify"]["steps"]
        if "micromamba run -n qualification python -m pip install"
        in step.get("run", "")
    )
    assert qualify_run.count("source/scripts/ci/bootstrap_trusted_pip.py") == 2
    assert qualify_run.index(
        "micromamba run -n qualification python source/scripts/ci/bootstrap_trusted_pip.py"
    ) < qualify_run.index("micromamba run -n qualification python -m pip install")
    assert qualify_run.index(
        "python source/scripts/ci/bootstrap_trusted_pip.py"
    ) < qualify_run.index("python -m pip install --disable-pip-version-check")

    bootstrap_source = (REPO_ROOT / "scripts/ci/bootstrap_trusted_pip.py").read_text()
    assert 'PIP_VERSION = "26.2.2"' in bootstrap_source
    assert (
        "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
        in bootstrap_source
    )
    assert '"--require-hashes"' in bootstrap_source
    assert '"https://pypi.org/simple"' in bootstrap_source
    assert (
        'f"pip=={PIP_VERSION} --hash=sha256:{PIP_WHEEL_SHA256}\\n"' in bootstrap_source
    )

    compatibility_workflow = (
        REPO_ROOT / ".github/workflows/test-compat-gwpy.yml"
    ).read_text()
    assert compatibility_workflow.index(
        "python scripts/ci/bootstrap_trusted_pip.py"
    ) < compatibility_workflow.index("python -m pip install")
