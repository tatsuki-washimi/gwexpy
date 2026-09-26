#!/usr/bin/env python3
"""Verify the v0.2.4 release-owner approval through the GitHub comment API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HumanApprovalError(ValueError):
    """Raised when the recorded GitHub approval cannot be verified."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise HumanApprovalError(f"{name} is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contract(expected_tag: str) -> dict[str, Any]:
    path = Path(__file__).with_name("release_contract.py")
    module = _load_module("release_contract_for_human_approval", path)
    try:
        return module.release_contract(expected_tag)
    except module.ReleaseContractError as exc:
        raise HumanApprovalError(str(exc)) from exc


def _load_and_validate_evidence(
    repo_root: Path,
    evidence_path: Path,
    expected_tag: str,
) -> dict[str, Any]:
    validator_path = Path(__file__).with_name("validate_release_review_evidence.py")
    module = _load_module("release_review_evidence_for_human_approval", validator_path)
    try:
        return module.validate_review_evidence(
            evidence_path,
            None,
            None,
            repo_root,
            expected_tag=expected_tag,
        )
    except module.ReleaseReviewEvidenceError as exc:
        raise HumanApprovalError(f"review evidence validation failed: {exc}") from exc


def _canonical_evidence_path(
    repo_root: Path,
    evidence: Path | None,
    contract: dict[str, Any],
) -> Path:
    configured = (repo_root / str(contract["review_evidence_path"])).resolve()
    selected = (
        evidence
        if evidence is not None
        else Path(str(contract["review_evidence_path"]))
    )
    if not selected.is_absolute():
        selected = repo_root / selected
    resolved = selected.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise HumanApprovalError(
            "review evidence must remain inside repo root"
        ) from exc
    if resolved != configured:
        raise HumanApprovalError(
            "review evidence must use the configured evidence path"
        )
    return resolved


def _fetch_comment(
    repository: str,
    comment_id: int,
    token: str,
    api_url: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise HumanApprovalError("repository must be OWNER/NAME")
    url = f"{api_url.rstrip('/')}/repos/{repository}/issues/comments/{comment_id}"
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "gwexpy-release-approval-verifier",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = response.read(64 * 1024 + 1)
    except HTTPError as exc:
        raise HumanApprovalError(
            f"GitHub comment API returned HTTP {exc.code}"
        ) from exc
    except (OSError, URLError) as exc:
        raise HumanApprovalError("GitHub comment API request failed") from exc
    if len(payload) > 64 * 1024:
        raise HumanApprovalError("GitHub comment response is too large")
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HumanApprovalError("GitHub comment API returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise HumanApprovalError("GitHub comment API returned an invalid object")
    return data


def _validate_comment(
    comment: dict[str, Any],
    approval: dict[str, Any],
    reviewed_commit: str,
) -> None:
    user = comment.get("user")
    created_at = comment.get("created_at")
    updated_at = comment.get("updated_at")
    body = comment.get("body")
    if (
        not isinstance(user, dict)
        or user.get("login") != approval["approver_login"]
        or created_at != approval["timestamp_utc"]
        or updated_at != created_at
        or not isinstance(body, str)
        or len(body.encode("utf-8")) > 16 * 1024
    ):
        raise HumanApprovalError("GitHub comment author or timestamp does not match")
    expected_body = [
        "GWEXPY-RELEASE-APPROVAL v0.2.4",
        f"S: {reviewed_commit}",
        f"SCOPE: {approval['scope_digest']}",
        "VERDICT: APPROVED",
    ]
    if body.replace("\r\n", "\n").splitlines() != expected_body:
        raise HumanApprovalError(
            "GitHub comment does not contain canonical approval tokens"
        )


def verify_human_approval(
    *,
    repo_root: Path | str,
    expected_tag: str,
    repository: str,
    evidence: Path | str | None = None,
    token: str | None = None,
    api_url: str | None = None,
) -> str:
    """Validate release evidence and its referenced owner comment; return S."""
    root = Path(repo_root).resolve()
    contract = _contract(expected_tag)
    evidence_path = _canonical_evidence_path(
        root, Path(evidence) if evidence is not None else None, contract
    )
    evidence_data = _load_and_validate_evidence(root, evidence_path, expected_tag)
    if expected_tag != "v0.2.4":
        raise HumanApprovalError("human approval verification is configured for v0.2.4")
    approval = evidence_data["human_approval"]
    reviewed_commit = approval["reviewed_commit"]
    credential = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
    if not credential:
        raise HumanApprovalError("GITHUB_TOKEN is required for comment verification")
    base_url = api_url or os.environ.get("GITHUB_API_URL", "https://api.github.com")
    comment = _fetch_comment(
        repository,
        approval["comment_id"],
        credential,
        base_url,
    )
    _validate_comment(comment, approval, reviewed_commit)
    return reviewed_commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        reviewed_commit = verify_human_approval(
            repo_root=args.repo_root,
            expected_tag=args.expected_tag,
            repository=args.repository,
            evidence=args.evidence,
        )
    except HumanApprovalError as exc:
        parser.error(str(exc))
    print(f"human_approval=verified\nreviewed_commit={reviewed_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
