"""Static contract for the manual published-release qualification workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "post-release-qualification.yml"

CHECKOUT = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_PYTHON = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
UPLOAD_ARTIFACT = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
DOWNLOAD_ARTIFACT = "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"
SETUP_MICROMAMBA = "mamba-org/setup-micromamba@f457c30a868e4760d3a6fcea5f25dc655b8edf39"

EXECUTION_JOBS = (
    "install_matrix",
    "gwpy_matrix",
    "sdist_contract",
    "conda_matrix",
    "scientific_extras",
    "docs_en_ja",
)


def read_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def load_workflow() -> dict[str, Any]:
    # BaseLoader keeps the GitHub key ``on`` and boolean-like values as strings.
    loaded = yaml.load(read_workflow(), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def run_text(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def step_named(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_manual_trigger_inputs_permissions_and_concurrency_are_closed() -> None:
    workflow = load_workflow()
    assert set(workflow["on"]) == {"workflow_dispatch"}
    dispatch = workflow["on"]["workflow_dispatch"]
    inputs = dispatch["inputs"]
    assert set(inputs) == {"release_tag", "expected_source_sha", "release_run_id"}
    assert {
        name: (item["type"], item["required"], item["default"])
        for name, item in inputs.items()
    } == {
        "release_tag": ("string", "true", "v0.2.0"),
        "expected_source_sha": (
            "string",
            "true",
            "5c91cf2d1087616c9815d0cbcc082c5f21bb36e9",
        ),
        "release_run_id": ("string", "true", "32935405476"),
    }
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert all("permissions" not in job for job in workflow["jobs"].values())
    assert "id-token" not in read_workflow()
    assert " write" not in read_workflow()
    assert workflow["env"] == {"FORCE_JAVASCRIPT_ACTIONS_TO_NODE24": "true"}
    assert "inputs.release_tag" in workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] == "false"


def test_dispatch_identity_is_validated_before_checkout_or_network_use() -> None:
    identity = load_workflow()["jobs"]["identity"]
    first, second = identity["steps"][:2]
    assert first["name"] == "Validate protected-main dispatch inputs"
    assert second["uses"] == CHECKOUT
    validation = first["run"]
    for token in (
        'github_ref != "refs/heads/main"',
        "flags=re.ASCII",
        'r"v(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)"',
        'r"[0-9a-f]{40}"',
        'r"[1-9][0-9]{0,19}"',
        '"v0.2.0"',
        '"5c91cf2d1087616c9815d0cbcc082c5f21bb36e9"',
        '"32935405476"',
    ):
        assert token in validation


def test_all_jobs_have_timeouts_and_actions_are_exact_sha_pins() -> None:
    workflow = load_workflow()
    assert set(workflow["jobs"]) == {
        "identity",
        "install_matrix",
        "gwpy_matrix",
        "sdist_contract",
        "conda_matrix",
        "scientific_extras",
        "docs_en_ja",
        "aggregate",
        "qualification_gate",
    }
    assert all("timeout-minutes" in job for job in workflow["jobs"].values())

    uses = re.findall(r"^\s*uses:\s*([^\s]+)$", read_workflow(), re.MULTILINE)
    assert uses
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in uses)
    assert set(uses) == {
        CHECKOUT,
        SETUP_PYTHON,
        UPLOAD_ARTIFACT,
        DOWNLOAD_ARTIFACT,
        SETUP_MICROMAMBA,
    }


def test_job_level_environment_uses_only_permitted_contexts() -> None:
    jobs = load_workflow()["jobs"]
    offenders = [
        f"{job_name}.{variable}"
        for job_name, job in jobs.items()
        for variable, value in job.get("env", {}).items()
        if "${{ runner." in value
    ]
    assert offenders == []


def test_identity_proves_tag_run_and_prior_artifacts() -> None:
    identity = load_workflow()["jobs"]["identity"]
    text = run_text(identity)
    checkout = identity["steps"][1]
    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "fetch-depth": "1",
        "persist-credentials": "false",
        "path": "control",
    }
    for token in (
        "git ls-remote",
        "^{commit}",
        "/actions/runs/$RELEASE_RUN_ID",
        'run["path"] != ".github/workflows/publish-release.yml"',
        'run["event"] != "push"',
        'run["head_sha"] != source_sha',
        'run["head_branch"] != release_tag',
        'run["status"] != "completed"',
        'run["conclusion"] != "success"',
    ):
        assert token in text

    downloads = [
        step for step in identity["steps"] if step.get("uses") == DOWNLOAD_ARTIFACT
    ]
    assert len(downloads) == 2
    assert {step["with"]["name"] for step in downloads} == {
        "release-payload-${{ steps.inputs.outputs.expected_source_sha }}",
        "release-sidecars-${{ steps.inputs.outputs.expected_source_sha }}",
    }
    assert all(
        step["with"]["run-id"] == "${{ steps.inputs.outputs.release_run_id }}"
        for step in downloads
    )
    assert all(
        step["with"]["github-token"] == "${{ github.token }}" for step in downloads
    )


def test_identity_validates_public_bytes_attestations_and_readbacks() -> None:
    identity = load_workflow()["jobs"]["identity"]
    text = run_text(identity)
    for token in (
        "https://pypi.org/pypi/gwexpy/$VERSION/json",
        "Cache-Control: no-cache",
        "Pragma: no-cache",
        "validate_pypi_json",
        "validate_payload_sidecar",
        "verify_artifact_directory",
        "filecmp.cmp",
        "pypi-attestations==0.0.30",
        "pypi-attestations --version",
        "pypi-attestations verify pypi --repository https://github.com/tatsuki-washimi/gwexpy",
        "/releases/tags/$RELEASE_TAG",
        "/git/ref/tags/$RELEASE_TAG",
        "verification",
        "unsigned",
        "https://api.anaconda.org/release/conda-forge/gwexpy/$VERSION",
        "https://zenodo.org/api/records/22106588",
        'zenodo["id"] != 22106588',
    ):
        assert token in text

    uploads = [
        step for step in identity["steps"] if step.get("uses") == UPLOAD_ARTIFACT
    ]
    assert [step["with"]["name"] for step in uploads] == [
        "post-release-public-payload-${{ steps.inputs.outputs.expected_source_sha }}",
        "post-release-identity-evidence-${{ steps.inputs.outputs.expected_source_sha }}",
    ]
    assert uploads[0]["with"]["path"] == "${{ runner.temp }}/public-payload"
    assert uploads[1]["with"]["path"] == "${{ runner.temp }}/identity-evidence"


def test_install_matrix_has_exact_twelve_explicit_cells() -> None:
    job = load_workflow()["jobs"]["install_matrix"]
    assert job["strategy"]["fail-fast"] == "false"
    include = job["strategy"]["matrix"]["include"]
    observed = {
        (item["os"], item["python"], item["artifact"], item["cell"]) for item in include
    }
    expected = {
        *(
            ("ubuntu-latest", version, artifact, f"install-ubuntu-{version}-{artifact}")
            for version in ("3.11", "3.12", "3.13", "3.14")
            for artifact in ("wheel", "sdist")
        ),
        ("macos-latest", "3.11", "wheel", "install-macos-3.11-wheel"),
        ("macos-latest", "3.14", "wheel", "install-macos-3.14-wheel"),
        ("windows-latest", "3.11", "wheel", "install-windows-3.11-wheel"),
        ("windows-latest", "3.14", "wheel", "install-windows-3.14-wheel"),
    }
    assert len(include) == 12
    assert observed == expected


def test_specialized_matrices_and_cell_ledger_are_exact() -> None:
    jobs = load_workflow()["jobs"]
    gwpy = jobs["gwpy_matrix"]["strategy"]
    conda = jobs["conda_matrix"]["strategy"]
    assert gwpy["fail-fast"] == conda["fail-fast"] == "false"
    assert {
        (item["python"], item["gwpy"], item["cell"])
        for item in gwpy["matrix"]["include"]
    } == {
        ("3.11", "4.0.1", "gwpy-4.0.1-wheel"),
        ("3.11", "4.0.2", "gwpy-4.0.2-wheel"),
    }
    assert {
        (item["python"], item["cell"], item["packages"])
        for item in conda["matrix"]["include"]
    } == {
        ("3.11", "conda-3.11", "pytest mne lalsuite"),
        ("3.14", "conda-3.14", "pytest"),
    }

    observed = {
        item["cell"]
        for job_name in ("install_matrix", "gwpy_matrix", "conda_matrix")
        for item in jobs[job_name]["strategy"]["matrix"]["include"]
    }
    observed.update(
        jobs[name]["env"]["CELL_ID"]
        for name in ("sdist_contract", "scientific_extras", "docs_en_ja")
    )
    assert observed == {
        "install-ubuntu-3.11-wheel",
        "install-ubuntu-3.11-sdist",
        "install-ubuntu-3.12-wheel",
        "install-ubuntu-3.12-sdist",
        "install-ubuntu-3.13-wheel",
        "install-ubuntu-3.13-sdist",
        "install-ubuntu-3.14-wheel",
        "install-ubuntu-3.14-sdist",
        "install-macos-3.11-wheel",
        "install-macos-3.14-wheel",
        "install-windows-3.11-wheel",
        "install-windows-3.14-wheel",
        "gwpy-4.0.1-wheel",
        "gwpy-4.0.2-wheel",
        "sdist-3.12-claims",
        "conda-3.11",
        "conda-3.14",
        "scientific-3.11-wheel",
        "docs-en-ja-3.11-wheel",
    }


def test_execution_jobs_avoid_source_shadow_and_upload_reports() -> None:
    workflow = read_workflow()
    jobs = load_workflow()["jobs"]
    lowered = workflow.lower()
    assert "continue-on-error" not in lowered
    assert "pip install -e" not in lowered
    assert "--editable" not in lowered
    assert "git push" not in lowered
    assert "gh workflow run" not in lowered
    assert "/dispatches" not in lowered

    for name in EXECUTION_JOBS:
        job = jobs[name]
        text = run_text(job)
        assert "python -m pip check" in text
        assert 'qualify_published_release.py" run-cell' in text
        assert '--repo-root "$CONTROL_ROOT"' in text
        prepare = step_named(job, "Prepare empty execution directory")
        assert prepare["if"] == "always()"
        run_cell_step = step_named(job, "Run strict qualification cell")
        assert run_cell_step["if"] == "always()"
        assert run_cell_step["working-directory"] == (
            "${{ runner.temp }}/qualification-empty"
        )
        upload = step_named(job, "Upload qualification cell report")
        assert upload["if"] == "always()"
        assert upload["uses"] == UPLOAD_ARTIFACT
        assert upload["with"]["if-no-files-found"] == "error"


def test_conda_scientific_and_docs_lanes_are_strict() -> None:
    jobs = load_workflow()["jobs"]
    conda = jobs["conda_matrix"]
    conda_text = run_text(conda)
    micromamba = next(
        step for step in conda["steps"] if step.get("uses") == SETUP_MICROMAMBA
    )
    assert "channel_priority: strict" in micromamba["with"]["condarc"]
    assert "gwexpy=0.2.0" in micromamba["with"]["create-args"]
    assert "python=${{ matrix.python }}" in micromamba["with"]["create-args"]
    assert "${{ matrix.packages }}" in micromamba["with"]["create-args"]
    assert "micromamba run -n qualification python -m pip check" in conda_text
    assert "pip install" not in conda_text

    scientific = run_text(jobs["scientific_extras"])
    for token in (
        "mne",
        "lalsuite",
        "pytest",
        "import lal",
        "import mne",
        "from gwpy.io.gwf import lalframe",
    ):
        assert token in scientific
    docs = jobs["docs_en_ja"]
    assert docs["env"]["MPLBACKEND"] == "Agg"
    assert "matplotlib" in run_text(docs)
    assert "gwexpy-0.2.0-py3-none-any.whl" in run_text(docs)


def test_aggregate_collects_exact_ledger_and_strictly_sets_output() -> None:
    jobs = load_workflow()["jobs"]
    aggregate = jobs["aggregate"]
    assert aggregate["if"] == "always()"
    assert set(aggregate["needs"]) == {
        "identity",
        "install_matrix",
        "gwpy_matrix",
        "sdist_contract",
        "conda_matrix",
        "scientific_extras",
        "docs_en_ja",
    }
    assert aggregate["outputs"] == {
        "all_passed": "${{ steps.final.outputs.all_passed }}"
    }
    text = run_text(aggregate)
    for token in (
        'qualify_published_release.py" aggregate',
        '--pypi-json "$IDENTITY_DIR/pypi.json"',
        '--payload-sidecar "$IDENTITY_DIR/distribution-sha256.json"',
        "aggregate-exit-code.txt",
        'aggregate["passed"] is True',
        'len(aggregate["required_cells"]) == 19',
        'value == "success"',
        "all_passed = exit_code == 0 and passed and needs_passed",
    ):
        assert token in text
    reports_download = step_named(aggregate, "Download all qualification reports")
    assert reports_download["if"] == "always()"
    assert reports_download["with"]["pattern"] == "post-release-report-*"
    assert reports_download["with"]["merge-multiple"] == "true"
    upload = step_named(aggregate, "Upload aggregate qualification evidence")
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"
    final = step_named(aggregate, "Assert complete qualification ledger")
    assert final["if"] == "always()"


def test_stable_gate_requires_success_and_literal_true() -> None:
    gate = load_workflow()["jobs"]["qualification_gate"]
    assert gate["if"] == "always()"
    assert gate["needs"] == "aggregate"
    assert gate["env"] == {
        "AGGREGATE_RESULT": "${{ needs.aggregate.result }}",
        "ALL_PASSED": "${{ needs.aggregate.outputs.all_passed }}",
    }
    text = run_text(gate)
    assert '[[ "$AGGREGATE_RESULT" == "success" ]]' in text
    assert '[[ "$ALL_PASSED" == "true" ]]' in text
