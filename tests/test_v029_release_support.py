"""Fail-closed release validation for v0.2.9."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = ROOT / "scripts" / "ci" / "release_contract.py"
VALIDATOR_PATH = ROOT / "scripts" / "validate_release.py"
CONTRACTS_PATH = ROOT / "scripts" / "ci" / "release_contracts.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v029_contract_registers_the_separate_release_gates() -> None:
    contracts = _load_module("release_contract_v029", LOADER_PATH)
    contract = contracts.release_contract("v0.2.9")
    previous = contracts.release_contract("v0.2.8")

    assert contract["review_base_sha"] == "82c177d5dab221f15c9c7c81ad86bbd91e2c5e7f"
    assert contract["review_evidence_schema"] == "gwexpy-v029-review-evidence-v2"
    assert contract["payload_schema"] == "gwexpy-v029-release-payload-v1"
    assert contract["integration_evidence_schema"] == (
        "gwexpy-v029-integration-evidence-v1"
    )
    assert contract["artifact_prefix"] == "v029-integration-evidence"
    assert contract["protected_refs"] == ["main"]
    assert set(contract["review_lanes"]) == {
        "documentation",
        "release-security",
        "scientific-data-model",
    }
    assert contract["s_to_r_allowed_paths"] == sorted(
        [contract["plan_path"], contract["review_evidence_path"]]
    )
    assert contract["tagger_date_policy"] == "independent"
    assert contract["source_metadata_date_policy"] == "independent"
    assert contract["publication_tip_policy"] == "qualified_candidate_ancestor"
    expected_promotion = previous["promotion"]
    expected_promotion["release_go"] = {
        "issue_number": 871,
        "owner_login": "tatsuki-washimi",
    }
    assert contract["promotion"] == expected_promotion
    assert contracts.review_evidence_profile(contract["review_evidence_schema"]) == "v2"


def test_v029_source_snapshot_preserves_open_plan_and_placeholder() -> None:
    contracts = _load_module("release_contract_v029_files", LOADER_PATH)
    contract = contracts.release_contract("v0.2.9")
    evidence_path = ROOT / contract["review_evidence_path"]
    candidate_evidence = evidence_path.read_text(encoding="utf-8")
    source_match = re.search(
        r'^\s*"source_sha":\s*"([0-9a-f]{40})"',
        candidate_evidence,
        re.MULTILINE,
    )
    source_sha = (
        source_match.group(1) if source_match else _git(ROOT, "rev-parse", "HEAD")
    )
    source_plan = subprocess.check_output(
        ["git", "show", f"{source_sha}:{contract['plan_path']}"], cwd=ROOT
    ).decode("utf-8")
    source_evidence = subprocess.check_output(
        ["git", "show", f"{source_sha}:{contract['review_evidence_path']}"],
        cwd=ROOT,
    )

    assert "Issue #871" in source_plan
    assert "Release Owner selected **2026-11-07**" in source_plan
    assert (
        "does not claim that GitHub Release, PyPI, or Zenodo publication happened"
        in source_plan
    )
    assert "release_date" not in contract
    assert "publication_date" not in contract
    checkboxes = re.findall(r"^- \[([ xX])\]", source_plan, re.MULTILINE)
    assert checkboxes and set(checkboxes) == {" "}

    validator = _load_module("validate_release_v029_files", VALIDATOR_PATH)
    assert source_evidence == validator._empty_v2_review_evidence_placeholder(
        contract["review_evidence_schema"], contract["review_base_sha"]
    )


def test_v029_requires_release_review_evidence() -> None:
    validator = _load_module("validate_release_v029_required", VALIDATOR_PATH)

    with pytest.raises(
        validator.ReleaseValidationError,
        match="v0.2.9 requires release review evidence",
    ):
        validator.validate_release(ROOT, "0" * 40, "v0.2.9")


def test_v029_contract_is_listed_without_reclassifying_older_releases() -> None:
    document = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    releases = document["releases"]

    assert "v0.2.9" in releases
    assert releases["v0.2.9"]["promotion"]["release_go"]["issue_number"] == 871
    assert all(
        "promotion" not in releases[tag] for tag in ("v0.1.13", "v0.2.3", "v0.2.5")
    )


def _scope_covers(path: str, scope: str) -> bool:
    return path == scope or path.startswith(f"{scope.rstrip('/')}/")


def test_v029_review_lanes_cover_current_main_and_readiness_files() -> None:
    contracts = _load_module("release_contract_v029_coverage", LOADER_PATH)
    contract = contracts.release_contract("v0.2.9")
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            f"{contract['review_base_sha']}..HEAD",
            "--",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    changed_paths = {
        item.decode("utf-8", errors="strict")
        for item in result.stdout.split(b"\0")
        if item
    }
    lane_paths = {path for paths in contract["review_lanes"].values() for path in paths}
    uncovered = {
        path
        for path in changed_paths
        if not any(_scope_covers(path, scope) for scope in lane_paths)
    }

    assert not uncovered
    for path in (
        contract["plan_path"],
        contract["review_evidence_path"],
        "tests/test_v029_release_support.py",
    ):
        assert any(_scope_covers(path, scope) for scope in lane_paths)


def test_v029_human_approval_requires_the_v2_risk_binding() -> None:
    contracts = _load_module("release_contract_v029_approval", LOADER_PATH)
    contract = contracts.release_contract("v0.2.9")
    validator = _load_module(
        "review_evidence_v029_approval",
        ROOT / "scripts" / "ci" / "validate_release_review_evidence.py",
    )
    old_approval_shape = {key: "" for key in validator.V024_HUMAN_APPROVAL_KEYS}
    old_approval_shape["scope_paths"] = []
    old_approval_shape["comment_id"] = 0

    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="human approval keys"
    ):
        validator._validate_human_approval(
            ROOT, old_approval_shape, "a" * 40, contract, "v0.2.9"
        )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def test_v029_source_and_candidate_must_be_distinct_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _load_module("validate_release_v029_distinct", VALIDATOR_PATH)
    repo = tmp_path / "v029-release-repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release-test@example.invalid")
    (repo / "README.md").write_text("release source\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "source")
    source = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(
        validator,
        "_release_contract",
        lambda _tag: {
            "plan_path": "plan.md",
            "s_to_r_allowed_paths": ["plan.md", "manifest.yaml"],
        },
    )

    with pytest.raises(validator.ReleaseValidationError, match="distinct commits"):
        validator.validate_s_to_r(repo, source, source, expected_tag="v0.2.9")
