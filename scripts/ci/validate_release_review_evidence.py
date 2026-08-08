#!/usr/bin/env python3
"""Validate sanitized, non-authorizing Terra review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path
from typing import Any

SCHEMA = "gwexpy-v0113-review-evidence-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
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
LANE_SCOPE_PATHS = {
    "A": {
        ".github/workflows/publish-release.yml",
        "RELEASING.md",
        "requirements/release-build.txt",
        "scripts/ci/assemble_release_evidence.py",
        "scripts/ci/validate_release_payload.py",
        "scripts/ci/validate_release_review_evidence.py",
        "scripts/validate_release.py",
        "tests/fixtures/release",
        "tests/test_publish_release_workflow.py",
        "tests/test_release_evidence.py",
        "tests/test_release_payload.py",
        "tests/test_release_review_evidence.py",
        "tests/test_validate_release_script.py",
    },
    "B": {
        "docs_redesign/about/citation.md",
        "docs_redesign/how-to/cli.md",
        "docs_redesign/locales/ja/LC_MESSAGES/about/citation.po",
        "docs_redesign/locales/ja/LC_MESSAGES/how-to/cli.po",
        "docs_redesign/locales/ja/LC_MESSAGES/tutorials/installation.po",
        "docs_redesign/tutorials/installation.md",
        "docs/web/en/user_guide/citation.md",
        "docs/web/en/user_guide/cli.md",
        "docs/web/en/user_guide/installation.md",
        "docs/web/ja/user_guide/citation.md",
        "docs/web/ja/user_guide/cli.md",
        "docs/web/ja/user_guide/installation.md",
        "tests/docs/test_v0113_public_release_facts.py",
    },
}


class ReleaseReviewEvidenceError(ValueError):
    """Raised when advisory review evidence is ambiguous or unsafe."""


class _DuplicateJSONKey(ValueError):
    """Raised by the JSON hook when an object has ambiguous keys."""


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
        block_lines: list[str] = []
        for line in source[matches[0].end() :].splitlines():
            if line and not line.startswith((" ", "\t")):
                break
            if line:
                block_lines.append(line)
        source = textwrap.dedent("\n".join(block_lines))
    try:
        data = json.loads(source, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateJSONKey) as exc:
        raise ReleaseReviewEvidenceError("invalid review evidence JSON") from exc
    if not isinstance(data, dict):
        raise ReleaseReviewEvidenceError("review evidence must be a JSON object")
    return data


def _scope_digest(repo_root: Path, commit: str, paths: list[str]) -> str:
    import subprocess

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


def validate_review_evidence(
    path: Path | str,
    reviewed_commit: str | None,
    required_lanes: set[str],
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one approved entry per required lane; raw reports are never read."""
    evidence_path = Path(path)
    if repo_root is None:
        raise ReleaseReviewEvidenceError(
            "review evidence validation requires --repo-root"
        )
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ReleaseReviewEvidenceError("review evidence must be a regular file")
    data = _load_review_document(evidence_path)
    if (
        not isinstance(data, dict)
        or set(data) != {"schema", "entries"}
        or data["schema"] != SCHEMA
    ):
        raise ReleaseReviewEvidenceError("unknown or missing review evidence keys")
    entries = data["entries"]
    if not isinstance(entries, list):
        raise ReleaseReviewEvidenceError("review evidence entries must be a list")
    if not required_lanes or not required_lanes <= set(LANE_SCOPE_PATHS):
        raise ReleaseReviewEvidenceError("review evidence has unknown required lanes")
    if reviewed_commit is None:
        commits = {
            entry.get("reviewed_commit")
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("reviewed_commit"), str)
        }
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
            or set(paths) != LANE_SCOPE_PATHS[lane]
            or not all(isinstance(entry[field], str) for field in ("model", "effort"))
            or not isinstance(entry["timestamp_utc"], str)
            or not TIMESTAMP.fullmatch(entry["timestamp_utc"])
            or not isinstance(entry["scope_digest"], str)
            or not SHA256.fullmatch(entry["scope_digest"])
            or not isinstance(entry["raw_report_sha256"], str)
            or not SHA256.fullmatch(entry["raw_report_sha256"])
            or not isinstance(entry["finding_ids"], list)
            or any(
                not isinstance(item, str) or not item for item in entry["finding_ids"]
            )
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
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reviewed-commit")
    parser.add_argument("--required-lane", action="append", required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_review_evidence(
            args.evidence, args.reviewed_commit, set(args.required_lane), args.repo_root
        )
    except ReleaseReviewEvidenceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
