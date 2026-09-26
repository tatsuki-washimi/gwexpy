#!/usr/bin/env python3
"""Validate sanitized, non-authorizing Terra review evidence."""

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
EFFORTS = {"low", "medium", "high", "xhigh", "max", "ultra"}
MAX_FINDING_IDS = 128
MAX_REVIEW_DOCUMENT_BYTES = 64 * 1024
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


def _load_review_document(path: Path) -> dict[str, Any]:
    """Load JSON evidence, or the strict JSON block in the audit YAML."""
    try:
        if path.stat().st_size > MAX_REVIEW_DOCUMENT_BYTES:
            raise ReleaseReviewEvidenceError("review evidence document is too large")
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseReviewEvidenceError("invalid review evidence document") from exc
    if path.suffix in {".yaml", ".yml"}:
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
    except (json.JSONDecodeError, _DuplicateJSONKey) as exc:
        raise ReleaseReviewEvidenceError("invalid review evidence JSON") from exc
    if not isinstance(data, dict):
        raise ReleaseReviewEvidenceError("review evidence must be a JSON object")
    return data


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
    """Require configured v0.2.4 review lanes to cover the full release diff."""
    base = contract.get("review_base_sha")
    if not isinstance(base, str) or SHA40.fullmatch(base) is None:
        raise ReleaseReviewEvidenceError("v0.2.4 has no frozen review baseline")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, reviewed_commit],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode:
        raise ReleaseReviewEvidenceError(
            "v0.2.3 release source is not an ancestor of reviewed commit"
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


def _validate_v024_human_approval(
    repo_root: Path,
    approval: object,
    reviewed_commit: str,
    contract: dict[str, Any],
) -> None:
    if not isinstance(approval, dict) or set(approval) != V024_HUMAN_APPROVAL_KEYS:
        raise ReleaseReviewEvidenceError("invalid human approval keys")
    lane_paths = dict(contract["review_lanes"]).get("scientific-data-model")
    paths = approval["scope_paths"]
    comment_id = approval["comment_id"]
    if (
        approval["approver_login"] != V024_APPROVER_LOGIN
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


def validate_review_evidence(
    path: Path | str,
    reviewed_commit: str | None,
    required_lanes: set[str] | None,
    repo_root: Path | str | None = None,
    *,
    expected_tag: str = "v0.1.13",
) -> dict[str, Any]:
    """Validate one approved entry per required lane; raw reports are never read."""
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
    data = _load_review_document(evidence_path)
    expected_top_level = {"schema", "entries"}
    if expected_tag == V024_TAG:
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
    if expected_tag == V024_TAG:
        _validate_v024_human_approval(
            Path(repo_root), data["human_approval"], reviewed_commit, contract
        )
        validate_review_scope_coverage(Path(repo_root), reviewed_commit, contract)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--required-lane", action="append")
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_review_evidence(
            args.evidence,
            args.reviewed_commit,
            set(args.required_lane) if args.required_lane else None,
            args.repo_root,
            expected_tag=args.expected_tag,
        )
    except ReleaseReviewEvidenceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
