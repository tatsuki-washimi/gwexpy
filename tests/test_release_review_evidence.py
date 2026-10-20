"""Fail-closed schema contracts for Terra advisory review evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "ci" / "validate_release_review_evidence.py"
RELEASE_SCRIPT_PATH = ROOT / "scripts" / "validate_release.py"
V026_EMPTY_READINESS_BYTES = b"""\
review_evidence_json: |
  {
    "schema": "gwexpy-v026-review-evidence-v1",
    "entries": [],
    "human_approval": {
      "approver_login": "",
      "role": "",
      "reviewed_commit": "",
      "scope_paths": [],
      "scope_digest": "",
      "timestamp_utc": "",
      "verdict": "",
      "comment_id": 0
    }
  }
"""
V026_CHECKBOX_PATTERN = re.compile(r"\[[ xX]\]")
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


def test_promotion_contract_requires_source_approval_for_synthetic_future(
    tmp_path, monkeypatch
):
    validator = load_validator()
    contract = dict(CONTRACTS["v0.1.13"])
    contract["promotion"] = {
        "source_approval": {
            "format": "GWEXPY-SOURCE-APPROVAL-v1",
            "approver": "future-approver",
        }
    }
    monkeypatch.setattr(validator, "_release_contract", lambda tag: contract)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review_evidence()))
    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="unknown or missing"
    ):
        validator.validate_review_evidence(
            path, SOURCE_SHA, {"A"}, tmp_path, expected_tag="v99.88.77"
        )


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


def test_v026_review_document_loader_preserves_raw_bytes_and_approval(
    tmp_path: Path,
):
    validator = load_validator()
    evidence = tmp_path / "audit-manifest-v0.2.6-release-readiness.yaml"
    raw = (
        b"review_evidence_json: |\n"
        b"  {\n"
        b'    "schema": "gwexpy-v026-review-evidence-v1",\n'
        b'    "entries": [],\n'
        b'    "human_approval": {"comment_id": 6265784106}\n'
        b"  }\n"
    )
    evidence.write_bytes(raw)

    document = validator.load_review_document(evidence)

    assert document.raw_bytes == raw
    assert document.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert document.data["schema"] == "gwexpy-v026-review-evidence-v1"
    assert document.data["human_approval"] == {"comment_id": 6265784106}


def load_release_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_release_for_readiness_tests", RELEASE_SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _assert_v026_source_placeholder(validator, path: Path) -> None:
    """S state: the exact unapproved placeholder and no completed gates."""
    assert path.read_bytes() == V026_EMPTY_READINESS_BYTES
    document = validator.load_review_document(path)
    assert document.data["entries"] == []
    approval = document.data["human_approval"]
    assert approval["reviewed_commit"] == ""
    assert approval["scope_digest"] == ""
    assert approval["timestamp_utc"] == ""
    assert approval["verdict"] == ""
    assert approval["comment_id"] == 0
    plan = (ROOT / validator._release_contract("v0.2.6")["plan_path"]).read_text(
        encoding="utf-8"
    )
    checklist = plan.split("## Release Gates Checklist", 1)[1]
    assert not [line for line in checklist.splitlines() if line.startswith("- [x]")]


def _assert_v026_candidate_record(validator, path: Path) -> None:
    """R state: three bound reviews, one approval, and a valid S-to-R binding."""
    contract = validator._release_contract("v0.2.6")
    data = validator.validate_review_evidence(
        path, None, None, ROOT, expected_tag="v0.2.6"
    )
    lanes = {entry["lane"] for entry in data["entries"]}
    commits = {entry["reviewed_commit"] for entry in data["entries"]}
    assert lanes == set(contract["review_lanes"])
    assert len(commits) == 1
    assert data["human_approval"]["reviewed_commit"] == next(iter(commits))

    release_validator = load_release_validator()
    head = _git(ROOT, "rev-parse", "HEAD")
    release_validator.validate_s_to_r(
        ROOT, next(iter(commits)), head, expected_tag="v0.2.6"
    )


def test_v026_readiness_manifest_is_exact_source_placeholder_or_bound_candidate():
    """Fail closed: S must be the exact placeholder; any other state must be R."""
    validator = load_validator()
    contract = validator._release_contract("v0.2.6")
    path = ROOT / contract["review_evidence_path"]

    if path.read_bytes() == V026_EMPTY_READINESS_BYTES:
        _assert_v026_source_placeholder(validator, path)
    else:
        _assert_v026_candidate_record(validator, path)


def test_v026_candidate_checkbox_row_has_one_checkbox_token():
    contract = load_validator()._release_contract("v0.2.6")
    plan = (ROOT / contract["plan_path"]).read_text(encoding="utf-8")
    rows = [line for line in plan.splitlines() if "Create candidate `R`" in line]

    assert len(rows) == 1
    assert len(V026_CHECKBOX_PATTERN.findall(rows[0])) == 1
    assert rows[0].startswith(("- [ ]", "- [x]"))


def _v026_transition_repo(
    tmp_path: Path,
    *,
    source_plan: str,
    candidate_plan: str,
    source_manifest: bytes = V026_EMPTY_READINESS_BYTES,
    candidate_manifest: bytes = b"review_evidence_json: |\n  {}\n",
    extra_path: str | None = None,
) -> tuple[Path, str, str]:
    """Build a throwaway S commit and a candidate R commit with synthetic content."""
    contract = load_validator()._release_contract("v0.2.6")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    plan = repo / contract["plan_path"]
    manifest = repo / contract["review_evidence_path"]
    plan.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(source_plan, encoding="utf-8")
    manifest.write_bytes(source_manifest)
    source = _commit_all(repo, "source")

    plan.write_text(candidate_plan, encoding="utf-8")
    manifest.write_bytes(candidate_manifest)
    if extra_path is not None:
        extra = repo / extra_path
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("unrelated\n", encoding="utf-8")
    candidate = _commit_all(repo, "candidate")
    return repo, source, candidate


V026_SOURCE_PLAN = (
    "## Release Gates Checklist\n\n"
    "- [ ] Create candidate `R` with only permitted readiness evidence population "
    "and the matching plan checkbox transitions from unchecked to checked.\n"
    "- [ ] Execute candidate qualification workflow.\n"
)
V026_CANDIDATE_PLAN = V026_SOURCE_PLAN.replace("- [ ] Create", "- [x] Create", 1)


def test_v026_s_to_r_transition_allows_only_manifest_and_one_checkbox(
    tmp_path: Path,
):
    release_validator = load_release_validator()
    repo, source, candidate = _v026_transition_repo(
        tmp_path,
        source_plan=V026_SOURCE_PLAN,
        candidate_plan=V026_CANDIDATE_PLAN,
    )

    release_validator.validate_s_to_r(repo, source, candidate, expected_tag="v0.2.6")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"extra_path": "README.md"},
            "outside coordinator audit paths",
        ),
        (
            {
                "source_plan": V026_SOURCE_PLAN.replace(
                    "- [ ] Create", "- [ ] Create `[ ]` marker,", 1
                ),
                "candidate_plan": V026_SOURCE_PLAN.replace(
                    "- [ ] Create", "- [x] Create `[ ]` marker,", 1
                ),
            },
            "must only transition",
        ),
        (
            {
                "candidate_plan": V026_CANDIDATE_PLAN.replace(
                    "from unchecked to checked", "from unchecked to done", 1
                ),
            },
            "must only transition",
        ),
        (
            {"source_manifest": b"review_evidence_json: |\n  {}\n"},
            "exact empty review evidence placeholder",
        ),
    ],
)
def test_v026_s_to_r_transition_rejects_out_of_contract_changes(
    tmp_path: Path, kwargs: dict[str, object], message: str
):
    release_validator = load_release_validator()
    values: dict[str, object] = {
        "source_plan": V026_SOURCE_PLAN,
        "candidate_plan": V026_CANDIDATE_PLAN,
    }
    values.update(kwargs)
    repo, source, candidate = _v026_transition_repo(tmp_path, **values)

    with pytest.raises(release_validator.ReleaseValidationError, match=message):
        release_validator.validate_s_to_r(
            repo, source, candidate, expected_tag="v0.2.6"
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"review_evidence_json: |\n  {not-json}\n",
        b"review_evidence_json: |\n  {}\nunreviewed: true\n",
        b"review_evidence_json: |\n  {}\nreview_evidence_json: >-\n  ignored\n",
        b'review_evidence_json: |\n  {"schema":"one","schema":"two"}\n',
    ],
)
def test_v026_common_review_loader_rejects_ambiguous_yaml_and_json(
    tmp_path: Path, raw: bytes
):
    validator = load_validator()
    evidence = tmp_path / "audit-manifest.yaml"
    evidence.write_bytes(raw)

    with pytest.raises(validator.ReleaseReviewEvidenceError):
        validator.load_review_document(evidence)


def test_v026_common_review_loader_rejects_oversized_file(tmp_path: Path):
    validator = load_validator()
    evidence = tmp_path / "audit-manifest.yaml"
    evidence.write_bytes(b"x" * (validator.MAX_REVIEW_DOCUMENT_BYTES + 1))

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="too large"):
        validator.load_review_document(evidence)


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
