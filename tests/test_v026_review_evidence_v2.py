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
VALIDATOR_PATH = ROOT / "scripts" / "ci" / "validate_release_review_evidence.py"
REVIEW_BASE_SHA = "a" * 40


def _load_validator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "v026_review_evidence_v2_validator", VALIDATOR_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release-test@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _scope_digest(validator: Any, repo: Path, commit: str, paths: list[str]) -> str:
    return validator._scope_digest(repo, commit, paths)


def _model_observation(status: str = "unavailable") -> dict[str, Any]:
    if status == "verified":
        evidence_record = '{"model":"gpt-6.1-sol","effort":"high"}'
        return {
            "status": status,
            "effective_model": "gpt-6.1-sol",
            "effective_effort": "high",
            "evidence_source": "runtime metadata capture",
            "evidence_record": evidence_record,
            "evidence_sha256": hashlib.sha256(
                evidence_record.encode("utf-8")
            ).hexdigest(),
            "limitations": "",
            "self_reported_model": None,
            "self_reported_effort": None,
        }
    if status == "unverified":
        return {
            "status": status,
            "effective_model": None,
            "effective_effort": None,
            "evidence_source": "reviewer self-report only",
            "evidence_record": "Reviewer stated model=gpt-6.1-sol; effort=high.",
            "evidence_sha256": hashlib.sha256(
                b"Reviewer stated model=gpt-6.1-sol; effort=high."
            ).hexdigest(),
            "limitations": "The host did not provide signed runtime metadata.",
            "self_reported_model": "gpt-6.1-sol",
            "self_reported_effort": "high",
        }
    return {
        "status": "unavailable",
        "effective_model": None,
        "effective_effort": None,
        "evidence_source": "subagent result schema",
        "evidence_record": None,
        "evidence_sha256": None,
        "limitations": "The execution result does not expose effective model metadata.",
        "self_reported_model": None,
        "self_reported_effort": None,
    }


def _fixture(tmp_path: Path, status: str = "unavailable"):
    validator = _load_validator()
    repo = tmp_path / "reviewed-repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release-test@example.invalid")

    lanes = {
        "documentation": ["docs"],
        "release-security": ["scripts"],
        "scientific-data-model": ["science"],
    }
    for directory, file_name in (
        ("docs", "old.md"),
        ("scripts", "old.py"),
        ("science", "old.py"),
    ):
        path = repo / directory / file_name
        path.parent.mkdir(parents=True)
        path.write_text("baseline\n", encoding="utf-8")
    base_sha = _commit_all(repo, "review base")

    changed_files = {
        "docs/release.md": "release documentation\n",
        "scripts/validate.py": "release validation\n",
        "science/units.py": "scientific behavior tests\n",
    }
    for relative, content in changed_files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    source_sha = _commit_all(repo, "reviewed source")

    contract = {
        "review_evidence_schema": "gwexpy-v026-review-evidence-v2",
        "review_base_sha": base_sha,
        "review_lanes": lanes,
        "promotion": {
            "source_approval": {
                "format": "GWEXPY-SOURCE-APPROVAL-v2",
                "approver": "tatsuki-washimi",
            }
        },
    }
    changed_path_map = {
        "documentation": ["docs/release.md"],
        "release-security": ["scripts/validate.py"],
        "scientific-data-model": ["science/units.py"],
    }
    entries = []
    for lane in sorted(lanes):
        report = f"Independent {lane} review of {source_sha}.\n"
        paths = sorted(lanes[lane])
        entries.append(
            {
                "lane": lane,
                "role": "reviewer",
                "reviewer_type": "ai-agent",
                "reviewer_id": f"reviewer-{lane}",
                "execution_context_id": f"context-{lane}",
                "execution_environment": "Codex collaboration subagent; read-only source worktree.",
                "started_at_utc": "2026-10-23T00:00:00Z",
                "completed_at_utc": "2026-10-23T00:10:00Z",
                "requested_model": "gpt-6.1-sol",
                "requested_effort": "high",
                "model_observation": _model_observation(status),
                "reviewed_commit": source_sha,
                "scope_paths": paths,
                "scope_digest": _scope_digest(validator, repo, source_sha, paths),
                "path_ledger": [
                    {
                        "path": path,
                        "methods": ["full-diff-and-context"],
                        "status": "reviewed",
                        "note": "Inspected the full diff and relevant source context.",
                    }
                    for path in changed_path_map[lane]
                ],
                "findings": [],
                "verdict": "APPROVED",
                "report": report,
                "raw_report_sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
            }
        )
    data = {
        "schema": "gwexpy-v026-review-evidence-v2",
        "source_sha": source_sha,
        "review_base_sha": base_sha,
        "changed_path_map": changed_path_map,
        "entries": entries,
        "owner_risk_decisions": [],
        "human_approval": {
            "approver_login": "",
            "role": "",
            "reviewed_commit": "",
            "scope_paths": [],
            "scope_digest": "",
            "timestamp_utc": "",
            "verdict": "",
            "comment_id": 0,
            "risk_decisions_sha256": "",
        },
    }
    evidence_path = tmp_path / "source-review.json"
    evidence_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return validator, repo, contract, source_sha, data, evidence_path


def _validate(validator, repo, contract, source_sha, path, *, phase="source-review"):
    validator._release_contract = lambda _tag: contract
    return validator.validate_review_evidence(
        path,
        source_sha,
        set(contract["review_lanes"]),
        repo,
        expected_tag="v0.2.6",
        phase=phase,
    )


@pytest.mark.parametrize("status", ["verified", "unverified", "unavailable"])
def test_v026_v2_accepts_explicit_model_observation_states(tmp_path, status):
    validator, repo, contract, source_sha, _, path = _fixture(tmp_path, status)

    data = _validate(validator, repo, contract, source_sha, path)

    assert data["source_sha"] == source_sha
    assert {entry["model_observation"]["status"] for entry in data["entries"]} == {
        status
    }


def test_v026_v2_accepts_portable_execution_environment_description(tmp_path):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    for entry in data["entries"]:
        entry["execution_environment"] = (
            "Linux/bash; independent read-only review of the exact source SHA; "
            "clean checkout; no repository files modified."
        )
    path.write_text(json.dumps(data), encoding="utf-8")

    result = _validate(validator, repo, contract, source_sha, path)

    assert result["source_sha"] == source_sha


@pytest.mark.parametrize(
    "environment",
    [
        "Linux/bash; read-only checkout at /home/reviewer/work/gwexpy.",
        r"Windows; read-only checkout at C:\Users\reviewer\gwexpy.",
        r"Linux; read-only checkout at \\build-host\work\gwexpy.",
    ],
)
def test_v026_v2_rejects_local_absolute_paths_in_execution_environment(
    tmp_path, environment
):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    data["entries"][0]["execution_environment"] = environment
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        validator.ReleaseReviewEvidenceError,
        match="execution environment must not contain local absolute paths",
    ):
        _validate(validator, repo, contract, source_sha, path)


def test_v026_v2_recomputes_embedded_report_hash(tmp_path):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    data["entries"][0]["report"] += "changed after review\n"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="execution record"):
        _validate(validator, repo, contract, source_sha, path)


def test_v026_v2_rejects_a_requested_value_falsely_recorded_as_effective(
    tmp_path,
):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    data["entries"][0]["model_observation"]["effective_model"] = "gpt-6.1-sol"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="unverified values"):
        _validate(validator, repo, contract, source_sha, path)


def test_v026_v2_recomputes_hash_of_runtime_model_evidence(tmp_path):
    validator, repo, contract, source_sha, data, path = _fixture(
        tmp_path, status="verified"
    )
    data["entries"][0]["model_observation"]["evidence_record"] += "tampered"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="model observation"):
        _validate(validator, repo, contract, source_sha, path)


def test_review_document_loader_translates_parser_recursion_error(
    tmp_path, monkeypatch
):
    validator = _load_validator()
    path = tmp_path / "recursion.json"
    path.write_text('{"nested": 0}', encoding="utf-8")

    def raise_recursion_error(*args, **kwargs):
        raise RecursionError("simulated JSON parser recursion failure")

    monkeypatch.setattr(validator.json, "loads", raise_recursion_error)
    with pytest.raises(
        validator.ReleaseReviewEvidenceError,
        match="invalid review evidence JSON",
    ) as exc_info:
        validator.load_review_document(path, allow_v026_size=True)

    assert isinstance(exc_info.value.__cause__, RecursionError)


def test_deep_json_input_does_not_leak_parser_recursion_error(tmp_path):
    validator = _load_validator()
    path = tmp_path / "deep.json"
    path.write_text('{"nested":' * 1200 + "0" + "}" * 1200, encoding="utf-8")

    try:
        document = validator.load_review_document(path, allow_v026_size=True)
    except validator.ReleaseReviewEvidenceError as exc:
        assert str(exc) == "invalid review evidence JSON"
        assert isinstance(exc.__cause__, RecursionError)
        return
    except RecursionError:
        pytest.fail("JSON parser recursion errors must be translated to a domain error")

    assert isinstance(document.data, dict)


def test_v026_v2_requires_an_exact_changed_path_map_and_path_ledger(tmp_path):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    data["changed_path_map"]["documentation"] = []
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="changed path map"):
        _validate(validator, repo, contract, source_sha, path)

    _, repo, contract, source_sha, data, path = _fixture(tmp_path / "second")
    data["entries"][0]["path_ledger"] = []
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="path review ledger"
    ):
        _validate(validator, repo, contract, source_sha, path)


def test_v026_v2_requires_unique_reviewers_and_execution_contexts(tmp_path):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    data["entries"][1]["reviewer_id"] = data["entries"][0]["reviewer_id"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleaseReviewEvidenceError, match="execution record"):
        _validate(validator, repo, contract, source_sha, path)


def test_v026_v2_reports_owner_risk_gate_and_binds_it_in_candidate_approval(
    tmp_path,
):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    finding = {
        "id": "SDM-001",
        "severity": "medium",
        "required": True,
        "status": "unresolved",
        "summary": "A required scientific limitation remains.",
        "resolution": "",
    }
    data["entries"][2]["findings"] = [finding]
    data["entries"][2]["report"] += "Finding SDM-001 requires Owner judgment.\n"
    data["entries"][2]["raw_report_sha256"] = hashlib.sha256(
        data["entries"][2]["report"].encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")

    _validate(validator, repo, contract, source_sha, path)
    assert validator._v026_status(data) == "HOLD_OWNER_RISK_DECISION"

    risk_decisions = [
        {
            "finding_id": "SDM-001",
            "decision": "ACCEPT_WITH_LIMITATION",
            "rationale": "The limitation is understood and accepted for this patch.",
        }
    ]
    data["owner_risk_decisions"] = risk_decisions
    science_paths = contract["review_lanes"]["scientific-data-model"]
    data["human_approval"] = {
        "approver_login": "tatsuki-washimi",
        "role": "release-owner",
        "reviewed_commit": source_sha,
        "scope_paths": science_paths,
        "scope_digest": _scope_digest(validator, repo, source_sha, science_paths),
        "timestamp_utc": "2026-10-23T00:20:00Z",
        "verdict": "APPROVED",
        "comment_id": 123456789,
        "risk_decisions_sha256": validator._canonical_json_sha256(risk_decisions),
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    result = _validate(validator, repo, contract, source_sha, path, phase="candidate")
    assert result["owner_risk_decisions"] == risk_decisions

    data["human_approval"]["risk_decisions_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(validator.ReleaseReviewEvidenceError, match="risk digest"):
        _validate(validator, repo, contract, source_sha, path, phase="candidate")


@pytest.mark.parametrize(
    ("approval_timestamp", "accepted"),
    [
        ("2026-10-23T00:09:59Z", False),
        ("2026-10-23T00:10:00Z", False),
        ("2026-10-23T00:10:01Z", True),
    ],
)
def test_v026_candidate_approval_must_follow_all_review_completions(
    tmp_path, approval_timestamp, accepted
):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    science_paths = contract["review_lanes"]["scientific-data-model"]
    data["human_approval"] = {
        "approver_login": "tatsuki-washimi",
        "role": "release-owner",
        "reviewed_commit": source_sha,
        "scope_paths": science_paths,
        "scope_digest": _scope_digest(validator, repo, source_sha, science_paths),
        "timestamp_utc": approval_timestamp,
        "verdict": "APPROVED",
        "comment_id": 123456789,
        "risk_decisions_sha256": validator._canonical_json_sha256([]),
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    if accepted:
        _validate(validator, repo, contract, source_sha, path, phase="candidate")
    else:
        with pytest.raises(
            validator.ReleaseReviewEvidenceError,
            match="must follow all review completion timestamps",
        ):
            _validate(validator, repo, contract, source_sha, path, phase="candidate")


def test_v026_v2_rejects_source_review_packet_with_owner_approval(tmp_path):
    validator, repo, contract, source_sha, data, path = _fixture(tmp_path)
    data["human_approval"]["verdict"] = "APPROVED"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        validator.ReleaseReviewEvidenceError, match="keep the owner approval empty"
    ):
        _validate(validator, repo, contract, source_sha, path)
