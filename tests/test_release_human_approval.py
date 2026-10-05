from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_VALIDATOR = ROOT / "scripts" / "ci" / "validate_release_review_evidence.py"
HUMAN_VERIFIER = ROOT / "scripts" / "ci" / "verify_release_human_approval.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def scope_digest(repo: Path, commit: str, paths: list[str]) -> str:
    result = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit,
            "--",
            *paths,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def make_review_evidence_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], str]:
    repo = tmp_path / "human-review-repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Release Test")
    git(repo, "config", "user.email", "release-test@example.invalid")

    baseline_files = {
        "gwexpy/.keep": "",
        "tests/io/.keep": "",
        "docs/.keep": "",
        "scripts/.keep": "",
        "RELEASING.md": "Release control\n",
    }
    for relative, content in baseline_files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "review baseline")
    base = git(repo, "rev-parse", "HEAD")

    changes = {
        "gwexpy/io.py": "reader change\n",
        "tests/io/test_reader.py": "reader regression\n",
        "docs/plan.md": "- [ ] review\n",
        "docs/readiness.json": "placeholder\n",
        "scripts/validate.py": "release validation\n",
        "RELEASING.md": "Release control updated\n",
    }
    for relative, content in changes.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "reviewed candidate S")
    reviewed_commit = git(repo, "rev-parse", "HEAD")

    lanes = {
        "scientific-data-model": ["gwexpy", "tests/io"],
        "documentation": ["docs"],
        "release-security": ["RELEASING.md", "scripts"],
    }
    contract = {
        "review_evidence_schema": "gwexpy-v024-review-evidence-v1",
        "review_base_sha": base,
        "review_lanes": lanes,
    }
    entries = []
    for lane, paths in lanes.items():
        entries.append(
            {
                "lane": lane,
                "role": "reviewer",
                "model": "gpt-6-sol",
                "effort": "high",
                "reviewed_commit": reviewed_commit,
                "scope_paths": paths,
                "scope_digest": scope_digest(repo, reviewed_commit, paths),
                "verdict": "APPROVED",
                "timestamp_utc": "2026-09-26T03:14:15Z",
                "raw_report_sha256": hashlib.sha256(lane.encode()).hexdigest(),
                "finding_ids": [],
            }
        )
    science_paths = lanes["scientific-data-model"]
    approval = {
        "approver_login": "tatsuki-washimi",
        "role": "release-owner",
        "reviewed_commit": reviewed_commit,
        "scope_paths": science_paths,
        "scope_digest": scope_digest(repo, reviewed_commit, science_paths),
        "timestamp_utc": "2026-09-26T03:14:15Z",
        "verdict": "APPROVED",
        "comment_id": 987654321,
    }
    evidence_path = repo / "docs" / "readiness.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "gwexpy-v024-review-evidence-v1",
                "entries": entries,
                "human_approval": approval,
            }
        ),
        encoding="utf-8",
    )
    return repo, evidence_path, contract, reviewed_commit


def test_v024_review_evidence_validates_typed_human_approval_and_full_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = load_module("v024_evidence_validator_test", EVIDENCE_VALIDATOR)
    repo, evidence_path, contract, reviewed_commit = make_review_evidence_repo(tmp_path)
    monkeypatch.setattr(validator, "_release_contract", lambda _tag: contract)

    evidence = validator.validate_review_evidence(
        evidence_path,
        None,
        None,
        repo,
        expected_tag="v0.2.4",
    )

    assert evidence["human_approval"]["reviewed_commit"] == reviewed_commit
    assert evidence["human_approval"]["comment_id"] == 987654321


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approver_login", "someone-else"),
        ("role", "reviewer"),
        ("reviewed_commit", "0" * 40),
        ("scope_paths", ["gwexpy"]),
        ("scope_digest", "not-a-digest"),
        ("timestamp_utc", "2026-02-30T03:14:15Z"),
        ("verdict", "HOLD"),
        ("comment_id", True),
        ("comment_id", 0),
    ],
)
def test_v024_review_evidence_rejects_invalid_human_approval_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    validator = load_module("v024_evidence_validator_invalid_test", EVIDENCE_VALIDATOR)
    repo, evidence_path, contract, _reviewed_commit = make_review_evidence_repo(
        tmp_path
    )
    monkeypatch.setattr(validator, "_release_contract", lambda _tag: contract)
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    document["human_approval"][field] = value
    evidence_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError):
        validator.validate_review_evidence(
            evidence_path,
            None,
            None,
            repo,
            expected_tag="v0.2.4",
        )


def approval_fixture() -> tuple[dict[str, Any], dict[str, Any], str]:
    reviewed_commit = "a" * 40
    digest = "b" * 64
    approval = {
        "approver_login": "tatsuki-washimi",
        "role": "release-owner",
        "reviewed_commit": reviewed_commit,
        "scope_paths": ["gwexpy", "tests/io"],
        "scope_digest": digest,
        "timestamp_utc": "2026-09-26T03:14:15Z",
        "verdict": "APPROVED",
        "comment_id": 987654321,
    }
    body = "\n".join(
        [
            "GWEXPY-RELEASE-APPROVAL v0.2.4",
            f"S: {reviewed_commit}",
            f"SCOPE: {digest}",
            "VERDICT: APPROVED",
        ]
    )
    comment = {
        "user": {"login": "tatsuki-washimi"},
        "created_at": approval["timestamp_utc"],
        "updated_at": approval["timestamp_utc"],
        "body": body,
    }
    return approval, comment, reviewed_commit


def test_human_verifier_accepts_exact_comment_and_rejects_edited_comments() -> None:
    verifier = load_module("v024_human_verifier_comment_test", HUMAN_VERIFIER)
    approval, comment, reviewed_commit = approval_fixture()
    verifier._validate_comment(comment, approval, reviewed_commit)

    edited = dict(comment, updated_at="2026-09-26T03:15:00Z")
    with pytest.raises(verifier.HumanApprovalError, match="author or timestamp"):
        verifier._validate_comment(edited, approval, reviewed_commit)


def test_v025_human_approval_requires_its_own_canonical_tag_token() -> None:
    verifier = load_module("v025_human_verifier_comment_test", HUMAN_VERIFIER)
    approval, historical, reviewed_commit = approval_fixture()
    approval["disposition_digest"] = "c" * 64
    approval_scope = load_module(
        "v025_owner_approval_comment_test",
        ROOT / "scripts/ci/v025_owner_approval.py",
    )
    current = dict(
        historical,
        body="\n".join(
            approval_scope.canonical_comment_lines(
                reviewed_commit,
                approval["scope_digest"],
                approval["disposition_digest"],
            )
        ),
    )
    verifier._validate_comment(current, approval, reviewed_commit, "v0.2.5")
    with pytest.raises(verifier.HumanApprovalError, match="canonical approval"):
        verifier._validate_comment(historical, approval, reviewed_commit, "v0.2.5")


@pytest.mark.parametrize("line_index", [3, 4, 5, 6, 17, 18])
def test_v025_human_approval_rejects_changed_scope_decisions_or_extra_text(
    line_index: int,
) -> None:
    verifier = load_module("v025_human_verifier_decisions_test", HUMAN_VERIFIER)
    approval, historical, reviewed_commit = approval_fixture()
    approval["disposition_digest"] = "c" * 64
    approval_scope = load_module(
        "v025_owner_approval_decisions_test",
        ROOT / "scripts/ci/v025_owner_approval.py",
    )
    lines = approval_scope.canonical_comment_lines(
        reviewed_commit,
        approval["scope_digest"],
        approval["disposition_digest"],
    )
    assert len(lines) == 19
    lines[line_index] = "EXTRA OR CHANGED APPROVAL"
    historical["body"] = "\n".join(lines)
    with pytest.raises(verifier.HumanApprovalError, match="canonical approval"):
        verifier._validate_comment(historical, approval, reviewed_commit, "v0.2.5")


@pytest.mark.parametrize("mutation", ["omit", "append"])
def test_v025_human_approval_requires_exact_line_count(mutation: str) -> None:
    verifier = load_module("v025_human_verifier_line_count_test", HUMAN_VERIFIER)
    approval, comment, reviewed_commit = approval_fixture()
    approval["disposition_digest"] = "c" * 64
    approval_scope = load_module(
        "v025_owner_approval_line_count_test",
        ROOT / "scripts/ci/v025_owner_approval.py",
    )
    lines = approval_scope.canonical_comment_lines(
        reviewed_commit, approval["scope_digest"], approval["disposition_digest"]
    )
    if mutation == "omit":
        del lines[8]
    else:
        lines.append("EXTRA")
    comment["body"] = "\n".join(lines)
    with pytest.raises(verifier.HumanApprovalError, match="canonical approval"):
        verifier._validate_comment(comment, approval, reviewed_commit, "v0.2.5")


def test_v025_review_evidence_binds_dispositions_to_reviewed_source(
    tmp_path: Path,
) -> None:
    validator = load_module("v025_disposition_validator_test", EVIDENCE_VALIDATOR)
    approval_scope = load_module(
        "v025_owner_approval_digest_test",
        ROOT / "scripts/ci/v025_owner_approval.py",
    )
    repo = tmp_path / "v025-approval-source"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Release Test")
    git(repo, "config", "user.email", "release-test@example.invalid")
    scope_paths = ["benchmarks", "gwexpy", "tests/io"]
    for directory in scope_paths:
        path = repo / directory / ".keep"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    document = repo / approval_scope.DISPOSITION_PATH
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("Twelve individually proposed dispositions\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "reviewed source")
    reviewed_commit = git(repo, "rev-parse", "HEAD")
    digest = approval_scope.disposition_document_sha256(repo, reviewed_commit)
    document.write_text("Unreviewed later edit\n", encoding="utf-8")
    assert approval_scope.disposition_document_sha256(repo, reviewed_commit) == digest
    approval = {
        "approver_login": "tatsuki-washimi",
        "role": "release-owner",
        "reviewed_commit": reviewed_commit,
        "scope_paths": scope_paths,
        "scope_digest": scope_digest(repo, reviewed_commit, scope_paths),
        "disposition_digest": digest,
        "timestamp_utc": "2026-09-29T00:00:00Z",
        "verdict": "APPROVED",
        "comment_id": 987654321,
    }
    contract = {"review_lanes": {"scientific-data-model": scope_paths}}
    validator._validate_human_approval(
        repo, approval, reviewed_commit, contract, "v0.2.5"
    )

    wrong = dict(approval)
    wrong["disposition_digest"] = "0" * 64
    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="disposition digest"
    ):
        validator._validate_human_approval(
            repo, wrong, reviewed_commit, contract, "v0.2.5"
        )
    missing = dict(approval)
    del missing["disposition_digest"]
    with pytest.raises(validator.ReleaseReviewEvidenceError, match="approval keys"):
        validator._validate_human_approval(
            repo, missing, reviewed_commit, contract, "v0.2.5"
        )


@pytest.mark.parametrize(
    ("line_index", "replacement"),
    [
        (1, "S: " + "c" * 40),
        (2, "SCOPE: " + "d" * 64),
    ],
    ids=["wrong-reviewed-source", "wrong-scope-digest"],
)
def test_human_verifier_rejects_wrong_source_or_scope_tokens(
    line_index: int,
    replacement: str,
) -> None:
    verifier = load_module("v024_human_verifier_scope_test", HUMAN_VERIFIER)
    approval, comment, reviewed_commit = approval_fixture()
    lines = comment["body"].splitlines()
    lines[line_index] = replacement
    comment["body"] = "\n".join(lines)

    with pytest.raises(verifier.HumanApprovalError, match="canonical approval tokens"):
        verifier._validate_comment(comment, approval, reviewed_commit)


@pytest.mark.parametrize(
    "change",
    [
        {"user": {"login": "other-user"}},
        {"created_at": "2026-09-26T03:15:00Z"},
        {"body": "PRIVATE RAW BODY"},
    ],
)
def test_human_verifier_rejects_comment_identity_and_body_mismatch(
    change: dict[str, object],
) -> None:
    verifier = load_module("v024_human_verifier_mismatch_test", HUMAN_VERIFIER)
    approval, comment, reviewed_commit = approval_fixture()
    comment.update(change)

    with pytest.raises(verifier.HumanApprovalError) as error:
        verifier._validate_comment(comment, approval, reviewed_commit)
    assert "PRIVATE RAW BODY" not in str(error.value)


def test_human_verifier_fetches_comment_by_id_without_logging_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = load_module("v024_human_verifier_fetch_test", HUMAN_VERIFIER)
    received: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps({"body": "private body stays internal"}).encode()

    def fake_urlopen(request: Any, timeout: int) -> FakeResponse:
        received["request"] = request
        received["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(verifier, "urlopen", fake_urlopen)
    result = verifier._fetch_comment(
        "tatsuki-washimi/gwexpy",
        987654321,
        "test-token",
        "https://api.github.com",
    )

    assert result["body"] == "private body stays internal"
    assert received["request"].full_url.endswith("/issues/comments/987654321")
    assert received["request"].get_header("Authorization") == "Bearer test-token"
    assert received["timeout"] == 20


def test_human_verifier_derives_s_and_uses_configured_evidence_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = load_module("v024_human_verifier_evidence_test", HUMAN_VERIFIER)
    approval, comment, reviewed_commit = approval_fixture()
    observed: dict[str, Any] = {}

    def load_evidence(
        repo_root: Path,
        evidence_path: Path,
        expected_tag: str,
    ) -> dict[str, Any]:
        observed["repo_root"] = repo_root
        observed["evidence_path"] = evidence_path
        observed["expected_tag"] = expected_tag
        return {"human_approval": approval}

    def fetch_comment(
        repository: str,
        comment_id: int,
        token: str,
        api_url: str,
    ) -> dict[str, Any]:
        observed["repository"] = repository
        observed["comment_id"] = comment_id
        observed["token"] = token
        observed["api_url"] = api_url
        return comment

    monkeypatch.setattr(verifier, "_load_and_validate_evidence", load_evidence)
    monkeypatch.setattr(verifier, "_fetch_comment", fetch_comment)

    result = verifier.verify_human_approval(
        repo_root=tmp_path,
        expected_tag="v0.2.4",
        repository="tatsuki-washimi/gwexpy",
        token="test-token",
    )

    assert result == reviewed_commit
    assert observed["repo_root"] == tmp_path.resolve()
    assert observed["evidence_path"] == (
        tmp_path
        / "docs/developers/plans/manifests/audit-manifest-v0.2.4-release-readiness.yaml"
    )
    assert observed["expected_tag"] == "v0.2.4"
    assert observed["comment_id"] == 987654321
    assert observed["repository"] == "tatsuki-washimi/gwexpy"


@pytest.mark.parametrize("mutation", [None, "author", "body", "fetch"])
def test_synthetic_future_source_approval_is_contract_selected(
    tmp_path, monkeypatch, mutation
):
    verifier = load_module("future_human_verifier", HUMAN_VERIFIER)
    approval, comment, source = approval_fixture()
    approval["approver_login"] = "source-approver"
    comment["user"] = {"login": "source-approver"}
    comment["body"] = "\n".join(
        [
            "GWEXPY-SOURCE-APPROVAL-v1",
            "version=v99.88.77",
            f"source_sha={source}",
            f"scope_sha256={approval['scope_digest']}",
            "decision=APPROVED",
        ]
    )
    contract = {
        "review_evidence_path": "review.json",
        "promotion": {
            "source_approval": {
                "format": "GWEXPY-SOURCE-APPROVAL-v1",
                "approver": "source-approver",
            }
        },
    }
    monkeypatch.setattr(verifier, "_contract", lambda tag: contract)
    monkeypatch.setattr(
        verifier, "_canonical_evidence_path", lambda *args: tmp_path / "review.json"
    )
    monkeypatch.setattr(
        verifier,
        "_load_and_validate_evidence",
        lambda *args: {"human_approval": approval},
    )
    if mutation == "author":
        comment["user"] = {"login": "wrong-author"}
    if mutation == "body":
        comment["body"] += "\nextra"

    def fetch(*args):
        if mutation == "fetch":
            raise verifier.HumanApprovalError("approval API failed")
        return comment

    monkeypatch.setattr(verifier, "_fetch_comment", fetch)
    if mutation:
        with pytest.raises(verifier.HumanApprovalError):
            verifier.verify_human_approval(
                repo_root=tmp_path,
                expected_tag="v99.88.77",
                repository="example/gwexpy",
                token="fake",
            )
    else:
        assert (
            verifier.verify_human_approval(
                repo_root=tmp_path,
                expected_tag="v99.88.77",
                repository="example/gwexpy",
                token="fake",
            )
            == source
        )


def test_future_review_evidence_requires_configured_approver(tmp_path, monkeypatch):
    validator = load_module("future_review_validator", EVIDENCE_VALIDATOR)
    repo, path, contract, source = make_review_evidence_repo(tmp_path)
    contract["promotion"] = {
        "source_approval": {
            "approver": "source-approver",
            "format": "GWEXPY-SOURCE-APPROVAL-v1",
        }
    }
    monkeypatch.setattr(validator, "_release_contract", lambda tag: contract)
    with pytest.raises(validator.ReleaseReviewEvidenceError):
        validator.validate_review_evidence(
            path, None, None, repo, expected_tag="v99.88.77"
        )
    data = json.loads(path.read_bytes())
    data["human_approval"]["approver_login"] = "source-approver"
    path.write_text(json.dumps(data))
    assert (
        validator.validate_review_evidence(
            path, None, None, repo, expected_tag="v99.88.77"
        )["human_approval"]["reviewed_commit"]
        == source
    )
    del data["human_approval"]
    path.write_text(json.dumps(data))
    with pytest.raises(validator.ReleaseReviewEvidenceError):
        validator.validate_review_evidence(
            path, None, None, repo, expected_tag="v99.88.77"
        )
