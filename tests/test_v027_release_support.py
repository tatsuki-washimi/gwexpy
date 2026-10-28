from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v027_contract_allows_main_only_and_frozen_review_base() -> None:
    loader = _load("v027_release_contract", "scripts/ci/release_contract.py")
    contract = loader.release_contract("v0.2.6")
    contract.pop("recovery_dispatch_ref")
    contract["review_evidence_schema"] = "gwexpy-v027-review-evidence-v2"
    contract["review_base_sha"] = "a" * 40
    contract["protected_refs"] = ["main"]

    validated = loader._validate_contract("v0.2.7", contract)

    assert validated["protected_refs"] == ["main"]
    assert validated["review_base_sha"] == "a" * 40


def test_v027_contract_requires_promotion_authority() -> None:
    loader = _load("v027_release_contract_authority", "scripts/ci/release_contract.py")
    contract = loader.release_contract("v0.2.6")
    contract.pop("recovery_dispatch_ref")
    contract["review_evidence_schema"] = "gwexpy-v027-review-evidence-v2"
    contract["review_base_sha"] = "a" * 40
    contract["protected_refs"] = ["main"]
    contract.pop("promotion")

    with pytest.raises(
        loader.ReleaseContractError,
        match="requires a promotion contract for qualified-candidate publication",
    ):
        loader._validate_contract("v0.2.7", contract)


def test_v026_keeps_its_two_protected_refs() -> None:
    loader = _load("v026_release_contract_regression", "scripts/ci/release_contract.py")

    assert loader.protected_refs("v0.2.6") == ["main", "maint/0.2"]
    assert not loader._protected_refs(["main"])


def test_v027_frozen_tip_checks_main_without_legacy_maintenance_branch(
    tmp_path: Path,
) -> None:
    validator = _load("v027_release_validator", "scripts/validate_release.py")
    repo = tmp_path / "frozen-tip-repo"
    repo.mkdir()
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    (repo / "file.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "source"], cwd=repo, check=True)
    main_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "checkout", "-q", "-b", "maint/0.2"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "maintenance advanced"],
        cwd=repo,
        check=True,
    )
    maintenance_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", main_sha],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/maint/0.2", maintenance_sha],
        cwd=repo,
        check=True,
    )
    validator._release_contract = lambda _tag: {"protected_refs": ["main"]}

    validator.validate_frozen_tip(repo, main_sha, expected_tag="v0.2.7")

    validator._release_contract = lambda _tag: {"protected_refs": ["main", "maint/0.2"]}
    with pytest.raises(validator.ReleaseValidationError, match="maint/0.2"):
        validator.validate_frozen_tip(repo, main_sha, expected_tag="v0.2.6")


def test_v027_v2_dispatch_accepts_source_review_fixture(tmp_path: Path) -> None:
    from tests.test_v026_review_evidence_v2 import _fixture

    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    contract["review_evidence_schema"] = "gwexpy-v027-review-evidence-v2"
    data["schema"] = "gwexpy-v027-review-evidence-v2"
    contract["review_base_sha"] = data["review_base_sha"]
    path.write_text(json.dumps(data), encoding="utf-8")
    validator._release_contract = lambda _tag: contract

    result = validator.validate_review_evidence(
        path,
        source_sha,
        set(contract["review_lanes"]),
        repo,
        expected_tag="v0.2.7",
        phase="source-review",
    )

    assert result["source_sha"] == source_sha


def test_v027_candidate_requires_bound_human_approval(tmp_path: Path) -> None:
    from tests.test_v026_review_evidence_v2 import _fixture, _scope_digest

    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    contract["review_evidence_schema"] = "gwexpy-v027-review-evidence-v2"
    contract["review_base_sha"] = data["review_base_sha"]
    data["schema"] = "gwexpy-v027-review-evidence-v2"
    lane_paths = contract["review_lanes"]["scientific-data-model"]
    data["human_approval"] = {
        "approver_login": "tatsuki-washimi",
        "role": "release-owner",
        "reviewed_commit": source_sha,
        "scope_paths": lane_paths,
        "scope_digest": _scope_digest(validator, repo, source_sha, lane_paths),
        "timestamp_utc": "2026-10-23T00:20:00Z",
        "verdict": "APPROVED",
        "comment_id": 123456789,
        "risk_decisions_sha256": validator._canonical_json_sha256([]),
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    validator._release_contract = lambda _tag: contract

    result = validator.validate_review_evidence(
        path,
        source_sha,
        set(contract["review_lanes"]),
        repo,
        expected_tag="v0.2.7",
        phase="candidate",
    )

    assert result["human_approval"]["reviewed_commit"] == source_sha


def test_v027_v2_rejects_mismatched_source_sha(tmp_path: Path) -> None:
    from tests.test_v026_review_evidence_v2 import _fixture

    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    contract["review_evidence_schema"] = "gwexpy-v027-review-evidence-v2"
    contract["review_base_sha"] = data["review_base_sha"]
    data["schema"] = "gwexpy-v027-review-evidence-v2"
    path.write_text(json.dumps(data), encoding="utf-8")
    validator._release_contract = lambda _tag: contract

    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="exact source and review base"
    ):
        validator.validate_review_evidence(
            path,
            "b" * 40,
            set(contract["review_lanes"]),
            repo,
            expected_tag="v0.2.7",
            phase="source-review",
        )


def test_unknown_review_schema_does_not_fall_back_to_legacy(tmp_path: Path) -> None:
    from tests.test_v026_review_evidence_v2 import _fixture

    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    contract["review_evidence_schema"] = "gwexpy-v099-review-evidence-v2"
    data["schema"] = "gwexpy-v099-review-evidence-v2"
    path.write_text(json.dumps(data), encoding="utf-8")
    validator._release_contract = lambda _tag: contract

    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="unsupported review evidence schema"
    ):
        validator.validate_review_evidence(
            path,
            source_sha,
            set(contract["review_lanes"]),
            repo,
            expected_tag="v0.2.7",
        )
