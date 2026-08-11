"""Fail-closed schema contracts for Terra advisory review evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "ci" / "validate_release_review_evidence.py"
CONTRACTS = json.loads(
    (ROOT / "scripts" / "ci" / "release_contracts.json").read_text(encoding="utf-8")
)["releases"]
SOURCE_SHA = "a" * 40
LANE_A_PATHS = sorted(
    {
        ".github/workflows/publish-release.yml",
        "RELEASING.md",
        "requirements/release-build.txt",
        "scripts/ci/assemble_release_evidence.py",
        "scripts/ci/validate_release_payload.py",
        "scripts/ci/validate_release_review_evidence.py",
        "scripts/validate_release.py",
        "tests/test_publish_release_workflow.py",
        "tests/test_release_evidence.py",
        "tests/test_release_payload.py",
        "tests/test_release_review_evidence.py",
        "tests/test_validate_release_script.py",
        "tests/fixtures/release",
    },
    key=lambda item: item.encode("utf-8"),
)


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_release_review_evidence", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def review_evidence() -> dict[str, object]:
    return {
        "schema": "gwexpy-v0113-review-evidence-v1",
        "entries": [
            {
                "lane": "A",
                "role": "reviewer",
                "model": "GPT-5.6-Terra",
                "effort": "high",
                "reviewed_commit": SOURCE_SHA,
                "scope_paths": LANE_A_PATHS,
                "scope_digest": hashlib.sha256("scope".encode()).hexdigest(),
                "verdict": "APPROVED",
                "timestamp_utc": "2026-08-06T00:00:00Z",
                "raw_report_sha256": "b" * 64,
                "finding_ids": ["A-001"],
            }
        ],
    }


def test_review_evidence_rejects_unknown_and_duplicate_lanes(tmp_path: Path):
    validator = load_validator()
    path = tmp_path / "review.json"
    data = review_evidence()
    data["untrusted_url"] = "https://example.invalid"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError):
        validator.validate_review_evidence(path, SOURCE_SHA, {"A"})


def test_review_evidence_rejects_a_missing_lane_scope_path(tmp_path: Path):
    validator = load_validator()
    path = tmp_path / "review.json"
    data = review_evidence()
    data["entries"][0]["scope_paths"] = LANE_A_PATHS[:-1]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError):
        validator.validate_review_evidence(path, SOURCE_SHA, {"A"})


def test_review_evidence_requires_repo_root_and_rejects_absent_scope_path(
    tmp_path: Path,
):
    validator = load_validator()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    for relative in LANE_A_PATHS[:-1]:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("scope\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "scope"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    digest = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--full-tree", commit, "--", *LANE_A_PATHS],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    data = review_evidence()
    data["entries"][0]["reviewed_commit"] = commit
    data["entries"][0]["scope_digest"] = hashlib.sha256(digest).hexdigest()
    evidence = tmp_path / "review.json"
    evidence.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError):
        validator.validate_review_evidence(evidence, commit, {"A"})
    with pytest.raises(validator.ReleaseReviewEvidenceError, match="scope path"):
        validator.validate_review_evidence(evidence, commit, {"A"}, repo)


def test_review_evidence_rejects_duplicate_json_keys(tmp_path: Path):
    validator = load_validator()
    path = tmp_path / "review.json"
    path.write_text(
        '{"schema":"gwexpy-v0113-review-evidence-v1","schema":"gwexpy-v0113-review-evidence-v1","entries":[]}',
        encoding="utf-8",
    )

    with pytest.raises(validator.ReleaseReviewEvidenceError):
        validator.validate_review_evidence(path, SOURCE_SHA, {"A"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "gpt-5.6-terra\nraw report"),
        ("model", "m" * 65),
        ("effort", "high\nraw report"),
        ("effort", "extreme"),
        ("effort", []),
        ("effort", {}),
        ("finding_ids", ["A-001\nraw report"]),
        ("finding_ids", ["F" * 65]),
        ("finding_ids", ["F-001", "F-001"]),
        ("finding_ids", ["F-002", "F-001"]),
        ("finding_ids", [f"F-{index:03d}" for index in range(129)]),
    ],
)
def test_review_evidence_rejects_unbounded_or_free_text_fields(
    tmp_path: Path, field: str, value: object
):
    validator = load_validator()
    path = tmp_path / "review.json"
    data = review_evidence()
    data["entries"][0][field] = value
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="invalid review evidence entry"
    ):
        validator.validate_review_evidence(path, SOURCE_SHA, {"A"}, tmp_path)


def test_review_evidence_rejects_oversized_document(tmp_path: Path):
    validator = load_validator()
    path = tmp_path / "review.json"
    path.write_text(" " * (validator.MAX_REVIEW_DOCUMENT_BYTES + 1), encoding="utf-8")

    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="document is too large"
    ):
        validator.validate_review_evidence(path, SOURCE_SHA, {"A"}, tmp_path)


def test_review_evidence_recomputes_canonical_scope_digest(tmp_path: Path):
    validator = load_validator()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    for relative in LANE_A_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("scope\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "scope"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    data = review_evidence()
    data["entries"][0]["reviewed_commit"] = commit
    data["entries"][0]["scope_digest"] = "d" * 64
    evidence = tmp_path / "review.json"
    evidence.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="scope digest"):
        validator.validate_review_evidence(evidence, commit, {"A"}, repo)


def test_review_evidence_accepts_json_block_in_coordinator_yaml(tmp_path: Path):
    validator = load_validator()
    evidence = tmp_path / "audit-manifest.yaml"
    evidence.write_text(
        "review_evidence_json: |\n"
        '  {"schema": "gwexpy-v0113-review-evidence-v1", "schema": "gwexpy-v0113-review-evidence-v1", "entries": []}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="invalid review evidence"
    ):
        validator.validate_review_evidence(evidence, SOURCE_SHA, {"A"}, tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "unreviewed: before\nreview_evidence_json: |\n  {}\n",
        "review_evidence_json: |\n  {}\nunreviewed: after\n",
    ],
)
def test_audit_yaml_rejects_content_outside_sanitized_json_block(
    tmp_path: Path, source: str
):
    validator = load_validator()
    evidence = tmp_path / "audit-manifest.yaml"
    evidence.write_text(source, encoding="utf-8")

    with pytest.raises(
        validator.ReleaseReviewEvidenceError,
        match="only the review_evidence_json block",
    ):
        validator.validate_review_evidence(evidence, SOURCE_SHA, {"A"}, tmp_path)


def test_v0114_placeholder_is_rejected_by_executable_review_gate(tmp_path: Path):
    validator = load_validator()
    manifest = tmp_path / "audit-manifest-v0.1.14-release-readiness.yaml"
    manifest.write_text(
        "review_evidence_json: |\n"
        '  {"schema": "gwexpy-v0114-review-evidence-v1", "entries": []}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        validator.ReleaseReviewEvidenceError,
        match="exactly one reviewed commit",
    ):
        validator.validate_review_evidence(
            manifest,
            None,
            None,
            tmp_path,
            expected_tag="v0.1.14",
        )


def test_audit_yaml_rejects_duplicate_review_evidence_key_with_any_scalar_form(
    tmp_path: Path,
):
    validator = load_validator()
    evidence = tmp_path / "audit-manifest.yaml"
    evidence.write_text(
        "review_evidence_json: |\n  {}\nreview_evidence_json: >-\n  ignored\n",
        encoding="utf-8",
    )

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="exactly one"):
        validator.validate_review_evidence(evidence, SOURCE_SHA, {"A"}, tmp_path)


def test_review_evidence_uses_v0114_schema_and_configured_lanes(tmp_path: Path):
    validator = load_validator()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    paths = CONTRACTS["v0.1.14"]["review_lanes"]["scientific"]
    for relative in paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("scope\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "scope"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    digest = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--full-tree", commit, "--", *paths],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    evidence = tmp_path / "review.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "gwexpy-v0114-review-evidence-v1",
                "entries": [
                    {
                        "lane": "scientific",
                        "role": "reviewer",
                        "model": "GPT-5.6-Terra",
                        "effort": "high",
                        "reviewed_commit": commit,
                        "scope_paths": paths,
                        "scope_digest": hashlib.sha256(digest).hexdigest(),
                        "verdict": "APPROVED",
                        "timestamp_utc": "2026-08-10T00:00:00Z",
                        "raw_report_sha256": "b" * 64,
                        "finding_ids": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = validator.validate_review_evidence(
        evidence,
        commit,
        {"scientific"},
        repo,
        expected_tag="v0.1.14",
    )

    assert result["schema"] == "gwexpy-v0114-review-evidence-v1"


def test_review_evidence_rejects_unconfigured_tag(tmp_path: Path):
    validator = load_validator()
    evidence = tmp_path / "review.json"
    evidence.write_text(json.dumps(review_evidence()), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="unsupported"):
        validator.validate_review_evidence(
            evidence,
            SOURCE_SHA,
            {"A"},
            tmp_path,
            expected_tag="v0.1.15",
        )
