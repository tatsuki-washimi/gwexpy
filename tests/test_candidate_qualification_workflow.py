"""Contract tests for the v0.2.1 private-candidate qualification workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "candidate-qualification.yml"
)


def test_candidate_qualification_is_a_push_only_private_artifact_matrix() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert set(workflow[True]) == {"push"}
    assert workflow[True]["push"]["branches"] == [
        "test/v021-codec-candidate-qualification"
    ]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["build"]["permissions"] == {"contents": "read"}
    assert "publish" not in WORKFLOW.read_text(encoding="utf-8").lower()


def test_candidate_qualification_has_exactly_nineteen_digest_checked_cells() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["qualify"]["strategy"]["matrix"]["include"]

    assert len(matrix) == 19
    assert {entry["cell"] for entry in matrix} == {
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
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "hashlib.sha256" in text
    assert "sha256sum -c" not in text
    assert "CANDIDATE_SOURCE_SHA" in text
    assert "97d4eb77e" in text
    assert "qualification/v021-codec-candidate-source" in text
    assert "/^__version__/" in text
    assert "actions/download-artifact@" in text
    assert "actions/upload-artifact@" in text


def test_candidate_qualification_runs_candidate_source_contract_tests() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["qualify"]["steps"]

    candidate_checkout = next(
        step for step in steps if step["name"] == "Check out fixed candidate source"
    )
    assert (
        candidate_checkout["with"]["ref"] == "qualification/v021-codec-candidate-source"
    )
    assert candidate_checkout["with"]["path"] == "candidate"

    contract_step = next(
        step
        for step in steps
        if step["name"]
        == "Verify installed candidate provenance and run focused contracts"
    )
    assert (
        contract_step["env"]["TEST_ROOT"] == "${{ github.workspace }}/candidate/tests"
    )
    assert "git -C candidate rev-parse HEAD" in contract_step["run"]
