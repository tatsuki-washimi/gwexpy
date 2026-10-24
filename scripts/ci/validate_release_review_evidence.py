#!/usr/bin/env python3
"""Validate versioned, source-bound AI review evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
FINDING_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
LOCAL_ABSOLUTE_PATH = re.compile(
    r"(?<![\w/])/(?!/)\S+|(?<![\w])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+\\)[^\s,;]+"
)
EFFORTS = {"low", "medium", "high", "xhigh", "max", "ultra"}
MAX_FINDING_IDS = 128
MAX_REVIEW_DOCUMENT_BYTES = 64 * 1024
MAX_V026_REVIEW_DOCUMENT_BYTES = 1024 * 1024
MAX_V026_REPORT_BYTES = 128 * 1024
ENTRY_KEYS = {
    "lane",
    "role",
    "model",
    "effort",
    "reviewed_commit",
    "scope_paths",
    "scope_digest",
    "verdict",
    "timestamp_utc",
    "raw_report_sha256",
    "finding_ids",
}
V024_TAG = "v0.2.4"
STRICT_APPROVAL_TAGS = {"v0.2.4", "v0.2.5", "v0.2.6"}
V024_HUMAN_APPROVAL_KEYS = {
    "approver_login",
    "role",
    "reviewed_commit",
    "scope_paths",
    "scope_digest",
    "timestamp_utc",
    "verdict",
    "comment_id",
}
V025_HUMAN_APPROVAL_KEYS = V024_HUMAN_APPROVAL_KEYS | {"disposition_digest"}
V026_HUMAN_APPROVAL_KEYS = V024_HUMAN_APPROVAL_KEYS | {"risk_decisions_sha256"}
V026_REVIEW_SCHEMA = "gwexpy-v026-review-evidence-v2"
V026_TOP_LEVEL_KEYS = {
    "schema",
    "source_sha",
    "review_base_sha",
    "changed_path_map",
    "entries",
    "owner_risk_decisions",
    "human_approval",
}
V026_ENTRY_KEYS = {
    "lane",
    "role",
    "reviewer_type",
    "reviewer_id",
    "execution_context_id",
    "execution_environment",
    "started_at_utc",
    "completed_at_utc",
    "requested_model",
    "requested_effort",
    "model_observation",
    "reviewed_commit",
    "scope_paths",
    "scope_digest",
    "path_ledger",
    "findings",
    "verdict",
    "report",
    "raw_report_sha256",
}
V026_MODEL_OBSERVATION_KEYS = {
    "status",
    "effective_model",
    "effective_effort",
    "evidence_source",
    "evidence_record",
    "evidence_sha256",
    "limitations",
    "self_reported_model",
    "self_reported_effort",
}
V026_PATH_LEDGER_KEYS = {"path", "methods", "status", "note"}
V026_FINDING_KEYS = {"id", "severity", "required", "status", "summary", "resolution"}
V026_RISK_DECISION_KEYS = {"finding_id", "decision", "rationale"}
V026_PATH_METHODS = {
    "full-diff-and-context",
    "structured-validation",
    "archive-inspection",
    "notebook-cell-review",
    "visual-inspection",
    "test-execution",
    "source-or-workflow-trace",
}
V026_APPROVER_LOGIN = "tatsuki-washimi"
V024_APPROVER_LOGIN = "tatsuki-washimi"


class ReleaseReviewEvidenceError(ValueError):
    """Raised when advisory review evidence is ambiguous or unsafe."""


class _DuplicateJSONKey(ValueError):
    """Raised by the JSON hook when an object has ambiguous keys."""


def _release_contract(expected_tag: str) -> dict[str, Any]:
    path = Path(__file__).with_name("release_contract.py")
    spec = importlib.util.spec_from_file_location("release_contract", path)
    if spec is None or spec.loader is None:
        raise ReleaseReviewEvidenceError("release contract loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        return module.release_contract(expected_tag)
    except module.ReleaseContractError as exc:
        raise ReleaseReviewEvidenceError(str(exc)) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


class ReviewEvidenceDocument:
    """Keep the original file bytes separate from the parsed JSON object."""

    __slots__ = ("raw_bytes", "raw_sha256", "data")

    def __init__(self, raw_bytes: bytes, raw_sha256: str, data: dict[str, Any]):
        self.raw_bytes = raw_bytes
        self.raw_sha256 = raw_sha256
        self.data = data


def load_review_document(
    path: Path | str, *, allow_v026_size: bool = False
) -> ReviewEvidenceDocument:
    """Load bounded JSON evidence or its strict coordinator YAML wrapper.

    ``raw_sha256`` always hashes the exact file bytes. ``data`` is the decoded
    inner JSON object and must never be re-serialized to identify the file.
    """
    evidence_path = Path(path)
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ReleaseReviewEvidenceError("review evidence must be a regular file")
    try:
        with evidence_path.open("rb") as stream:
            read_limit = (
                MAX_V026_REVIEW_DOCUMENT_BYTES
                if allow_v026_size
                else MAX_REVIEW_DOCUMENT_BYTES
            )
            raw = stream.read(read_limit + 1)
        max_bytes = (
            MAX_V026_REVIEW_DOCUMENT_BYTES
            if allow_v026_size
            else MAX_REVIEW_DOCUMENT_BYTES
        )
        if len(raw) > max_bytes:
            raise ReleaseReviewEvidenceError("review evidence document is too large")
        source = raw.decode("utf-8", errors="strict")
    except OSError as exc:
        raise ReleaseReviewEvidenceError("invalid review evidence document") from exc
    except UnicodeDecodeError as exc:
        raise ReleaseReviewEvidenceError("invalid review evidence document") from exc
    if evidence_path.suffix.lower() in {".yaml", ".yml"}:
        key = re.compile(r"^review_evidence_json\s*:", re.MULTILINE)
        if len(key.findall(source)) != 1:
            raise ReleaseReviewEvidenceError(
                "audit YAML must contain exactly one review_evidence_json key"
            )
        marker = re.compile(r"^review_evidence_json:\s*\|\s*$", re.MULTILINE)
        matches = list(marker.finditer(source))
        if len(matches) != 1:
            raise ReleaseReviewEvidenceError(
                "audit YAML must contain exactly one review_evidence_json block"
            )
        lines = source.splitlines()
        if (
            not lines
            or lines[0] != "review_evidence_json: |"
            or len(lines) < 2
            or any(line and not line.startswith("  ") for line in lines[1:])
        ):
            raise ReleaseReviewEvidenceError(
                "audit YAML must contain only the review_evidence_json block"
            )
        source = "\n".join(line[2:] if line else "" for line in lines[1:])
    try:
        data = json.loads(source, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateJSONKey, RecursionError) as exc:
        raise ReleaseReviewEvidenceError("invalid review evidence JSON") from exc
    if not isinstance(data, dict):
        raise ReleaseReviewEvidenceError("review evidence must be a JSON object")
    return ReviewEvidenceDocument(
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        data=data,
    )


def _load_review_document(path: Path) -> dict[str, Any]:
    """Compatibility wrapper for callers that need only the parsed object."""
    return load_review_document(path).data


def _scope_digest(repo_root: Path, commit: str, paths: list[str]) -> str:
    for path in paths:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if exists.returncode:
            raise ReleaseReviewEvidenceError(
                f"review scope path is absent from reviewed tree: {path}"
            )
    result = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--full-tree", commit, "--", *paths],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ReleaseReviewEvidenceError("cannot calculate review scope digest")
    return hashlib.sha256(result.stdout).hexdigest()


def changed_paths_between(repo_root: Path | str, base: str, candidate: str) -> set[str]:
    """Return both sides of every changed path, including rename endpoints."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            "--diff-filter=ACDMRTUXB",
            f"{base}..{candidate}",
            "--",
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ReleaseReviewEvidenceError("cannot enumerate release review changes")
    raw_paths = result.stdout
    if raw_paths and not raw_paths.endswith(b"\0"):
        raise ReleaseReviewEvidenceError("git returned a malformed changed-path list")
    try:
        paths = [
            item.decode("utf-8", errors="strict") for item in raw_paths.split(b"\0")
        ]
    except UnicodeDecodeError as exc:
        raise ReleaseReviewEvidenceError("changed paths are not valid UTF-8") from exc
    if paths[-1:] != [""] or any(not path for path in paths[:-1]):
        raise ReleaseReviewEvidenceError("git returned a malformed changed-path list")
    return set(paths[:-1])


def _scope_covers_path(path: str, scope_paths: set[str]) -> bool:
    return any(path == scope or path.startswith(f"{scope}/") for scope in scope_paths)


def validate_review_scope_coverage(
    repo_root: Path | str,
    reviewed_commit: str,
    contract: dict[str, Any],
) -> None:
    """Require configured release lanes to cover the full release diff."""
    base = contract.get("review_base_sha")
    if not isinstance(base, str) or SHA40.fullmatch(base) is None:
        raise ReleaseReviewEvidenceError("release has no frozen review baseline")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, reviewed_commit],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode:
        raise ReleaseReviewEvidenceError(
            "release review baseline is not an ancestor of reviewed commit"
        )
    changed = changed_paths_between(repo_root, base, reviewed_commit)
    lane_paths = {
        path for paths in dict(contract["review_lanes"]).values() for path in paths
    }
    uncovered = sorted(
        path for path in changed if not _scope_covers_path(path, lane_paths)
    )
    if uncovered:
        raise ReleaseReviewEvidenceError(
            "release review lanes do not cover changed paths: " + ", ".join(uncovered)
        )


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return False
    return True


def _utc_datetime(value: str) -> datetime:
    """Parse a timestamp already checked by ``_valid_utc_timestamp``."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _v025_owner_approval_module() -> Any:
    path = Path(__file__).with_name("v025_owner_approval.py")
    spec = importlib.util.spec_from_file_location("v025_owner_approval", path)
    if spec is None or spec.loader is None:
        raise ReleaseReviewEvidenceError("v0.2.5 approval scope is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_human_approval(
    repo_root: Path,
    approval: object,
    reviewed_commit: str,
    contract: dict[str, Any],
    expected_tag: str,
) -> None:
    keys = (
        V026_HUMAN_APPROVAL_KEYS
        if expected_tag == "v0.2.6"
        else V025_HUMAN_APPROVAL_KEYS
        if expected_tag == "v0.2.5"
        else V024_HUMAN_APPROVAL_KEYS
    )
    if not isinstance(approval, dict) or set(approval) != keys:
        raise ReleaseReviewEvidenceError("invalid human approval keys")
    lane_paths = dict(contract["review_lanes"]).get("scientific-data-model")
    paths = approval["scope_paths"]
    comment_id = approval["comment_id"]
    if (
        approval["approver_login"]
        != contract.get("promotion", {})
        .get("source_approval", {})
        .get("approver", V024_APPROVER_LOGIN)
        or approval["role"] != "release-owner"
        or approval["reviewed_commit"] != reviewed_commit
        or not isinstance(paths, list)
        or not isinstance(lane_paths, list)
        or paths != lane_paths
        or not isinstance(approval["scope_digest"], str)
        or SHA256.fullmatch(approval["scope_digest"]) is None
        or not _valid_utc_timestamp(approval["timestamp_utc"])
        or approval["verdict"] != "APPROVED"
        or isinstance(comment_id, bool)
        or not isinstance(comment_id, int)
        or comment_id <= 0
    ):
        raise ReleaseReviewEvidenceError("invalid human approval evidence")
    if approval["scope_digest"] != _scope_digest(
        repo_root, reviewed_commit, lane_paths
    ):
        raise ReleaseReviewEvidenceError(
            "human approval scope digest does not match the reviewed tree"
        )
    if expected_tag == "v0.2.5":
        digest = approval["disposition_digest"]
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ReleaseReviewEvidenceError("invalid disposition digest")
        try:
            expected_digest = _v025_owner_approval_module().disposition_document_sha256(
                repo_root, reviewed_commit
            )
        except ValueError as exc:
            raise ReleaseReviewEvidenceError(str(exc)) from exc
        if digest != expected_digest:
            raise ReleaseReviewEvidenceError(
                "disposition digest does not match reviewed source"
            )


def _validate_legacy_review_evidence(
    path: Path | str,
    reviewed_commit: str | None,
    required_lanes: set[str] | None,
    repo_root: Path | str | None = None,
    *,
    expected_tag: str = "v0.1.13",
) -> dict[str, Any]:
    """Validate the historical evidence format without changing its rules."""
    evidence_path = Path(path)
    if repo_root is None:
        raise ReleaseReviewEvidenceError(
            "review evidence validation requires --repo-root"
        )
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ReleaseReviewEvidenceError("review evidence must be a regular file")
    contract = _release_contract(expected_tag)
    schema = contract["review_evidence_schema"]
    lane_scope_paths = {
        lane: set(paths) for lane, paths in dict(contract["review_lanes"]).items()
    }
    data = load_review_document(evidence_path).data
    expected_top_level = {"schema", "entries"}
    if expected_tag in STRICT_APPROVAL_TAGS or "promotion" in contract:
        expected_top_level.add("human_approval")
    if (
        not isinstance(data, dict)
        or set(data) != expected_top_level
        or data["schema"] != schema
    ):
        raise ReleaseReviewEvidenceError("unknown or missing review evidence keys")
    entries = data["entries"]
    if not isinstance(entries, list):
        raise ReleaseReviewEvidenceError("review evidence entries must be a list")
    if required_lanes is None:
        required_lanes = set(lane_scope_paths)
    if not required_lanes or not required_lanes <= set(lane_scope_paths):
        raise ReleaseReviewEvidenceError("review evidence has unknown required lanes")
    if reviewed_commit is None:
        commits: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            candidate_commit = entry.get("reviewed_commit")
            if isinstance(candidate_commit, str):
                commits.add(candidate_commit)
        if len(commits) != 1:
            raise ReleaseReviewEvidenceError(
                "review evidence must name exactly one reviewed commit"
            )
        reviewed_commit = commits.pop()
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise ReleaseReviewEvidenceError("unknown or missing review entry keys")
        lane = entry["lane"]
        paths = entry["scope_paths"]
        model = entry["model"]
        effort = entry["effort"]
        finding_ids = entry["finding_ids"]
        if (
            not isinstance(lane, str)
            or lane not in required_lanes
            or lane in seen
            or entry["role"] != "reviewer"
            or entry["verdict"] != "APPROVED"
            or entry["reviewed_commit"] != reviewed_commit
            or not SHA40.fullmatch(reviewed_commit)
            or not isinstance(paths, list)
            or not paths
            or any(
                not isinstance(item, str)
                or not item
                or item.startswith("/")
                or ".." in Path(item).parts
                for item in paths
            )
            or paths != sorted(set(paths), key=lambda item: item.encode("utf-8"))
            or set(paths) != lane_scope_paths[lane]
            or not isinstance(model, str)
            or MODEL_ID.fullmatch(model) is None
            or not isinstance(effort, str)
            or effort not in EFFORTS
            or not isinstance(entry["timestamp_utc"], str)
            or not TIMESTAMP.fullmatch(entry["timestamp_utc"])
            or not isinstance(entry["scope_digest"], str)
            or not SHA256.fullmatch(entry["scope_digest"])
            or not isinstance(entry["raw_report_sha256"], str)
            or not SHA256.fullmatch(entry["raw_report_sha256"])
            or not isinstance(finding_ids, list)
            or len(finding_ids) > MAX_FINDING_IDS
            or any(
                not isinstance(item, str) or FINDING_ID.fullmatch(item) is None
                for item in finding_ids
            )
            or finding_ids
            != sorted(set(finding_ids), key=lambda item: item.encode("utf-8"))
        ):
            raise ReleaseReviewEvidenceError("invalid review evidence entry")
        if entry["scope_digest"] != _scope_digest(
            Path(repo_root), reviewed_commit, paths
        ):
            raise ReleaseReviewEvidenceError(
                "review scope digest does not match the reviewed tree"
            )
        seen.add(lane)
    if seen != required_lanes:
        raise ReleaseReviewEvidenceError("review evidence has missing or extra lanes")
    if expected_tag in STRICT_APPROVAL_TAGS or "promotion" in contract:
        _validate_human_approval(
            Path(repo_root),
            data["human_approval"],
            reviewed_commit,
            contract,
            expected_tag,
        )
        validate_review_scope_coverage(Path(repo_root), reviewed_commit, contract)
    return data


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(encoded).hexdigest()


def _v026_changed_path_map(
    repo_root: Path,
    reviewed_commit: str,
    contract: dict[str, Any],
) -> dict[str, list[str]]:
    base = contract.get("review_base_sha")
    if not isinstance(base, str) or SHA40.fullmatch(base) is None:
        raise ReleaseReviewEvidenceError("release has no frozen review baseline")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, reviewed_commit],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode:
        raise ReleaseReviewEvidenceError(
            "release review baseline is not an ancestor of reviewed commit"
        )
    changed = changed_paths_between(repo_root, base, reviewed_commit)
    lanes = dict(contract["review_lanes"])
    mapping = {
        lane: sorted(
            (path for path in changed if _scope_covers_path(path, set(paths))),
            key=lambda item: item.encode("utf-8"),
        )
        for lane, paths in lanes.items()
    }
    uncovered = sorted(
        path
        for path in changed
        if not any(path in mapped for mapped in mapping.values())
    )
    if uncovered:
        raise ReleaseReviewEvidenceError(
            "release review lanes do not cover changed paths: " + ", ".join(uncovered)
        )
    return mapping


def _validate_v026_approval_placeholder(approval: object) -> None:
    expected = {
        "approver_login": "",
        "role": "",
        "reviewed_commit": "",
        "scope_paths": [],
        "scope_digest": "",
        "timestamp_utc": "",
        "verdict": "",
        "comment_id": 0,
        "risk_decisions_sha256": "",
    }
    if approval != expected:
        raise ReleaseReviewEvidenceError(
            "source-review phase must keep the owner approval empty"
        )


def _validate_v026_model_observation(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != V026_MODEL_OBSERVATION_KEYS:
        return False
    status = value["status"]
    effective_model = value["effective_model"]
    effective_effort = value["effective_effort"]
    evidence_source = value["evidence_source"]
    evidence_record = value["evidence_record"]
    limitations = value["limitations"]
    self_model = value["self_reported_model"]
    evidence_sha256 = value["evidence_sha256"]
    self_effort = value["self_reported_effort"]
    if not isinstance(status, str) or status not in {
        "verified",
        "unverified",
        "unavailable",
    }:
        return False
    if self_model is not None and (
        not isinstance(self_model, str) or MODEL_ID.fullmatch(self_model) is None
    ):
        return False
    if self_effort is not None and (
        not isinstance(self_effort, str) or self_effort not in EFFORTS
    ):
        return False
    if status == "verified":
        return (
            isinstance(effective_model, str)
            and MODEL_ID.fullmatch(effective_model) is not None
            and isinstance(effective_effort, str)
            and effective_effort in EFFORTS
            and isinstance(evidence_source, str)
            and bool(evidence_source.strip())
            and isinstance(evidence_record, str)
            and bool(evidence_record.strip())
            and len(evidence_record.encode("utf-8")) <= 32 * 1024
            and isinstance(evidence_sha256, str)
            and SHA256.fullmatch(evidence_sha256) is not None
            and hashlib.sha256(evidence_record.encode("utf-8")).hexdigest()
            == evidence_sha256
            and limitations == ""
        )
    if status == "unverified":
        return (
            effective_model is None
            and effective_effort is None
            and isinstance(evidence_source, str)
            and bool(evidence_source.strip())
            and (
                (evidence_record is None and evidence_sha256 is None)
                or (
                    isinstance(evidence_record, str)
                    and bool(evidence_record.strip())
                    and len(evidence_record.encode("utf-8")) <= 32 * 1024
                    and isinstance(evidence_sha256, str)
                    and SHA256.fullmatch(evidence_sha256) is not None
                    and hashlib.sha256(evidence_record.encode("utf-8")).hexdigest()
                    == evidence_sha256
                )
            )
            and isinstance(limitations, str)
            and bool(limitations.strip())
        )
    return (
        effective_model is None
        and effective_effort is None
        and self_model is None
        and self_effort is None
        and isinstance(evidence_source, str)
        and bool(evidence_source.strip())
        and evidence_record is None
        and evidence_sha256 is None
        and isinstance(limitations, str)
        and bool(limitations.strip())
    )


def _validate_v026_path_ledger(value: object, expected_paths: list[str]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected_paths):
        return False
    for item, expected_path in zip(value, expected_paths):
        if not isinstance(item, dict) or set(item) != V026_PATH_LEDGER_KEYS:
            return False
        methods = item["methods"]
        status = item["status"]
        if (
            item["path"] != expected_path
            or not isinstance(status, str)
            or status not in {"reviewed", "exception"}
            or not isinstance(methods, list)
            or any(
                not isinstance(method, str) or method not in V026_PATH_METHODS
                for method in methods
            )
            or (status == "reviewed" and not methods)
            or (status == "exception" and methods)
            or methods != sorted(set(methods))
            or not isinstance(item["note"], str)
            or not item["note"].strip()
            or len(item["note"].encode("utf-8")) > 2048
        ):
            return False
    return True


def _validate_v026_findings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_FINDING_IDS:
        raise ReleaseReviewEvidenceError("invalid review findings")
    findings: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != V026_FINDING_KEYS:
            raise ReleaseReviewEvidenceError("invalid review finding keys")
        identifier = item["id"]
        severity = item["severity"]
        required = item["required"]
        status = item["status"]
        if (
            not isinstance(identifier, str)
            or FINDING_ID.fullmatch(identifier) is None
            or not isinstance(severity, str)
            or severity not in {"critical", "high", "medium", "low", "info"}
            or type(required) is not bool
            or not isinstance(status, str)
            or status not in {"resolved", "nonblocking", "unresolved"}
            or not isinstance(item["summary"], str)
            or not item["summary"].strip()
            or len(item["summary"].encode("utf-8")) > 4096
            or not isinstance(item["resolution"], str)
            or len(item["resolution"].encode("utf-8")) > 4096
            or (status == "resolved" and not item["resolution"].strip())
            or (status == "nonblocking" and required)
            or (status == "nonblocking" and not item["resolution"].strip())
        ):
            raise ReleaseReviewEvidenceError("invalid review finding")
        identifiers.append(identifier)
        findings.append(item)
    if identifiers != sorted(set(identifiers), key=lambda item: item.encode("utf-8")):
        raise ReleaseReviewEvidenceError("review finding IDs must be unique and sorted")
    return findings


def _validate_v026_owner_risk_decisions(
    value: object,
    required_unresolved_ids: set[str],
    approver_login: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_FINDING_IDS:
        raise ReleaseReviewEvidenceError("invalid owner risk decisions")
    decisions: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != V026_RISK_DECISION_KEYS:
            raise ReleaseReviewEvidenceError("invalid owner risk decision keys")
        identifier = item["finding_id"]
        if (
            not isinstance(identifier, str)
            or FINDING_ID.fullmatch(identifier) is None
            or item["decision"] != "ACCEPT_WITH_LIMITATION"
            or not isinstance(item["rationale"], str)
            or not item["rationale"].strip()
            or len(item["rationale"].encode("utf-8")) > 4096
        ):
            raise ReleaseReviewEvidenceError("invalid owner risk decision")
        identifiers.append(identifier)
        decisions.append(item)
    if identifiers != sorted(set(identifiers), key=lambda item: item.encode("utf-8")):
        raise ReleaseReviewEvidenceError(
            "owner risk decisions must be unique and sorted"
        )
    if set(identifiers) != required_unresolved_ids:
        raise ReleaseReviewEvidenceError(
            "owner risk decisions must cover exactly the unresolved required findings"
        )
    if decisions and approver_login != V026_APPROVER_LOGIN:
        raise ReleaseReviewEvidenceError(
            "owner risk decisions require Release Owner approval"
        )
    return decisions


def _v026_status(data: dict[str, Any]) -> str:
    if not all(entry["verdict"] == "APPROVED" for entry in data["entries"]):
        return "HOLD_REVIEW_LANE"
    unresolved_required = {
        finding["id"]
        for entry in data["entries"]
        for finding in entry["findings"]
        if finding["required"] and finding["status"] == "unresolved"
    }
    if unresolved_required:
        return "HOLD_OWNER_RISK_DECISION"
    return "READY_FOR_SOURCE_HUMAN_APPROVAL"


def _validate_v026_review_evidence(
    path: Path | str,
    reviewed_commit: str | None,
    required_lanes: set[str] | None,
    repo_root: Path,
    contract: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    document = load_review_document(path, allow_v026_size=True)
    data = document.data
    lanes = dict(contract["review_lanes"])
    all_lanes = set(lanes)
    if required_lanes is not None and required_lanes != all_lanes:
        raise ReleaseReviewEvidenceError(
            "v0.2.6 requires all three independent review lanes"
        )
    if set(data) != V026_TOP_LEVEL_KEYS or data.get("schema") != V026_REVIEW_SCHEMA:
        raise ReleaseReviewEvidenceError(
            "unknown or missing v0.2.6 review evidence keys"
        )
    source_sha = data["source_sha"]
    base_sha = data["review_base_sha"]
    if (
        not isinstance(source_sha, str)
        or SHA40.fullmatch(source_sha) is None
        or (reviewed_commit is not None and source_sha != reviewed_commit)
        or not isinstance(base_sha, str)
        or base_sha != contract.get("review_base_sha")
    ):
        raise ReleaseReviewEvidenceError(
            "review evidence is not bound to the exact source and review base"
        )
    expected_map = _v026_changed_path_map(repo_root, source_sha, contract)
    if data["changed_path_map"] != expected_map:
        raise ReleaseReviewEvidenceError(
            "changed path map does not exactly cover the review-base diff"
        )
    entries = data["entries"]
    if not isinstance(entries, list) or len(entries) != len(all_lanes):
        raise ReleaseReviewEvidenceError("v0.2.6 requires exactly three review entries")
    if not isinstance(data["owner_risk_decisions"], list):
        raise ReleaseReviewEvidenceError("owner risk decisions must be a list")
    if phase not in {"candidate", "source-review"}:
        raise ReleaseReviewEvidenceError("unsupported review validation phase")
    if phase == "source-review":
        _validate_v026_approval_placeholder(data["human_approval"])
        if data["owner_risk_decisions"]:
            raise ReleaseReviewEvidenceError(
                "source-review evidence cannot invent or pre-authorize owner risk decisions"
            )
    elif (
        not isinstance(data["human_approval"], dict)
        or set(data["human_approval"]) != V026_HUMAN_APPROVAL_KEYS
    ):
        raise ReleaseReviewEvidenceError("invalid human approval keys")

    reviewer_ids: set[str] = set()
    execution_ids: set[str] = set()
    all_findings: list[dict[str, Any]] = []
    seen_lanes: set[str] = set()
    if any(not isinstance(item, dict) for item in entries):
        raise ReleaseReviewEvidenceError("review entries must be objects")
    ordered_entries = sorted(entries, key=lambda item: str(item.get("lane", "")))
    if entries != ordered_entries:
        raise ReleaseReviewEvidenceError("review entries must be sorted by lane")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != V026_ENTRY_KEYS:
            raise ReleaseReviewEvidenceError(
                "unknown or missing v0.2.6 review entry keys"
            )
        lane = entry["lane"]
        if not isinstance(lane, str) or lane not in all_lanes or lane in seen_lanes:
            raise ReleaseReviewEvidenceError("unknown or duplicate review lane")
        reviewer_id = entry["reviewer_id"]
        execution_id = entry["execution_context_id"]
        environment = entry["execution_environment"]
        requested_model = entry["requested_model"]
        requested_effort = entry["requested_effort"]
        started_at = entry["started_at_utc"]
        completed_at = entry["completed_at_utc"]
        scope_paths = entry["scope_paths"]
        report = entry["report"]
        report_hash = entry["raw_report_sha256"]
        if (
            entry["role"] != "reviewer"
            or entry["reviewer_type"] != "ai-agent"
            or not isinstance(reviewer_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", reviewer_id)
            or reviewer_id in reviewer_ids
            or not isinstance(execution_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", execution_id)
            or execution_id in execution_ids
            or not isinstance(environment, str)
            or not environment.strip()
            or len(environment.encode("utf-8")) > 1024
            or (
                requested_model is not None
                and (
                    not isinstance(requested_model, str)
                    or MODEL_ID.fullmatch(requested_model) is None
                )
            )
            or (
                requested_effort is not None
                and (
                    not isinstance(requested_effort, str)
                    or requested_effort not in EFFORTS
                )
            )
            or not _valid_utc_timestamp(started_at)
            or not _valid_utc_timestamp(completed_at)
            or completed_at < started_at
            or entry["reviewed_commit"] != source_sha
            or scope_paths != sorted(lanes[lane], key=lambda item: item.encode("utf-8"))
            or not isinstance(entry["verdict"], str)
            or entry["verdict"] not in {"APPROVED", "HOLD"}
            or not isinstance(report, str)
            or not report.strip()
            or len(report.encode("utf-8")) > MAX_V026_REPORT_BYTES
            or not isinstance(report_hash, str)
            or SHA256.fullmatch(report_hash) is None
            or hashlib.sha256(report.encode("utf-8")).hexdigest() != report_hash
            or not isinstance(entry["scope_digest"], str)
            or SHA256.fullmatch(entry["scope_digest"]) is None
        ):
            raise ReleaseReviewEvidenceError("invalid v0.2.6 review execution record")
        if LOCAL_ABSOLUTE_PATH.search(environment):
            raise ReleaseReviewEvidenceError(
                "execution environment must not contain local absolute paths"
            )
        if not _validate_v026_model_observation(entry["model_observation"]):
            raise ReleaseReviewEvidenceError(
                "invalid model observation; unverified values must not be presented as effective"
            )
        if entry["scope_digest"] != _scope_digest(repo_root, source_sha, scope_paths):
            raise ReleaseReviewEvidenceError(
                "review scope digest does not match the reviewed tree"
            )
        if not _validate_v026_path_ledger(entry["path_ledger"], expected_map[lane]):
            raise ReleaseReviewEvidenceError(
                f"changed-path review ledger is incomplete for lane {lane}"
            )
        if entry["verdict"] == "APPROVED" and any(
            item["status"] != "reviewed" for item in entry["path_ledger"]
        ):
            raise ReleaseReviewEvidenceError(
                "an approved lane cannot contain unreviewed path exceptions"
            )
        findings = _validate_v026_findings(entry["findings"])
        if any(
            finding["id"] in {item["id"] for item in all_findings}
            for finding in findings
        ):
            raise ReleaseReviewEvidenceError(
                "finding IDs must be unique across review lanes"
            )
        all_findings.extend(findings)
        reviewer_ids.add(reviewer_id)
        execution_ids.add(execution_id)
        seen_lanes.add(lane)
    if seen_lanes != all_lanes:
        raise ReleaseReviewEvidenceError("review evidence has missing or extra lanes")
    validate_review_scope_coverage(repo_root, source_sha, contract)

    required_unresolved_ids = {
        finding["id"]
        for finding in all_findings
        if finding["required"] and finding["status"] == "unresolved"
    }
    if phase == "candidate":
        decisions = _validate_v026_owner_risk_decisions(
            data["owner_risk_decisions"],
            required_unresolved_ids,
            data["human_approval"].get("approver_login"),
        )
        _validate_human_approval(
            repo_root,
            data["human_approval"],
            source_sha,
            contract,
            "v0.2.6",
        )
        approval_time = _utc_datetime(data["human_approval"]["timestamp_utc"])
        latest_review_completion = max(
            _utc_datetime(entry["completed_at_utc"]) for entry in entries
        )
        if approval_time <= latest_review_completion:
            raise ReleaseReviewEvidenceError(
                "human approval must follow all review completion timestamps"
            )
        if data["human_approval"]["risk_decisions_sha256"] != _canonical_json_sha256(
            decisions
        ):
            raise ReleaseReviewEvidenceError(
                "human approval risk digest does not match owner risk decisions"
            )
        if any(entry["verdict"] != "APPROVED" for entry in entries):
            raise ReleaseReviewEvidenceError(
                "candidate evidence contains a HOLD review"
            )
    return data


def validate_review_evidence(
    path: Path | str,
    reviewed_commit: str | None,
    required_lanes: set[str] | None,
    repo_root: Path | str | None = None,
    *,
    expected_tag: str = "v0.1.13",
    phase: str = "candidate",
) -> dict[str, Any]:
    """Validate source reviews or a candidate record for the configured release."""
    if repo_root is None:
        raise ReleaseReviewEvidenceError(
            "review evidence validation requires --repo-root"
        )
    evidence_path = Path(path)
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ReleaseReviewEvidenceError("review evidence must be a regular file")
    contract = _release_contract(expected_tag)
    if expected_tag == "v0.2.6":
        try:
            return _validate_v026_review_evidence(
                evidence_path,
                reviewed_commit,
                required_lanes,
                Path(repo_root),
                contract,
                phase,
            )
        except UnicodeEncodeError as exc:
            raise ReleaseReviewEvidenceError(
                "review evidence contains text that is not valid UTF-8"
            ) from exc
    if phase != "candidate":
        raise ReleaseReviewEvidenceError(
            "source-review phase is available only for v0.2.6"
        )
    return _validate_legacy_review_evidence(
        evidence_path,
        reviewed_commit,
        required_lanes,
        repo_root,
        expected_tag=expected_tag,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--required-lane", action="append")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--phase", choices=("candidate", "source-review"), default="candidate"
    )
    args = parser.parse_args(argv)
    try:
        validated = validate_review_evidence(
            args.evidence,
            args.reviewed_commit,
            set(args.required_lane) if args.required_lane else None,
            args.repo_root,
            expected_tag=args.expected_tag,
            phase=args.phase,
        )
    except ReleaseReviewEvidenceError as exc:
        parser.error(str(exc))
    if args.phase == "source-review":
        status = _v026_status(validated)
        print(f"source_review_status={status}")
        if status != "READY_FOR_SOURCE_HUMAN_APPROVAL":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
