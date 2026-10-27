#!/usr/bin/env python3
"""Verify a Release Owner's exact authorization to reuse a qualified candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MARKER = "GWEXPY-RELEASE-RECOVERY-v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ZENODO_CHOICES = {"correct-after-archive", "retain-candidate-date"}
REQUIRED_RECOVERY_PR_PATHS = {
    ".github/workflows/publish-release.yml",
    "RELEASING.md",
    "scripts/check_release_metadata.py",
    "scripts/ci/release_contract.py",
    "scripts/ci/release_contracts.json",
    "scripts/ci/verify_release_recovery.py",
    "scripts/validate_release.py",
    "tests/test_check_release_metadata_script.py",
    "tests/test_release_recovery.py",
    "tests/test_validate_release_script.py",
}


class RecoveryAuthorizationError(ValueError):
    """Raised when an existing candidate lacks exact recovery authorization."""


def _contract(tag: str) -> dict[str, Any]:
    path = Path(__file__).with_name("release_contract.py")
    spec = importlib.util.spec_from_file_location(
        "release_contract_for_release_recovery", path
    )
    if spec is None or spec.loader is None:
        raise RecoveryAuthorizationError("release contract loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        return module.release_contract(tag)
    except module.ReleaseContractError as exc:
        raise RecoveryAuthorizationError(str(exc)) from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryAuthorizationError(message)


def parse_recovery_authorization(body: str) -> dict[str, str]:
    """Parse the canonical, line-oriented authorization record."""
    names = (
        "version",
        "tag",
        "source_sha",
        "tag_object_sha",
        "control_ref",
        "recovery_pr",
        "candidate_run_id",
        "promotion_manifest_sha256",
        "sdist_sha256",
        "wheel_sha256",
        "tagger_date_policy",
        "source_metadata_date_policy",
        "publication_tip_policy",
        "existing_candidate",
        "zenodo_date_disposition",
        "trusted_publishing",
        "publication_decision",
    )
    lines = body.splitlines()
    _require(
        len(lines) == len(names) + 1 and lines[0] == MARKER,
        "recovery authorization is not the canonical record",
    )
    fields: dict[str, str] = {}
    for line, name in zip(lines[1:], names, strict=True):
        key, separator, value = line.partition("=")
        _require(
            bool(separator) and key == name and bool(value),
            "recovery authorization field order is invalid",
        )
        fields[name] = value

    _require(fields["tag"] == "v" + fields["version"], "recovery tag/version mismatch")
    _require(SHA40.fullmatch(fields["source_sha"]) is not None, "invalid source SHA")
    _require(
        SHA40.fullmatch(fields["tag_object_sha"]) is not None, "invalid tag object SHA"
    )
    _require(
        fields["recovery_pr"].isdecimal() and int(fields["recovery_pr"]) > 0,
        "invalid recovery PR number",
    )
    _require(
        fields["candidate_run_id"].isdecimal() and int(fields["candidate_run_id"]) > 0,
        "invalid candidate run ID",
    )
    for name in ("promotion_manifest_sha256", "sdist_sha256", "wheel_sha256"):
        _require(SHA256.fullmatch(fields[name]) is not None, f"invalid {name}")
    _require(
        fields["zenodo_date_disposition"] in ZENODO_CHOICES,
        "invalid Zenodo date disposition",
    )
    _require(
        fields["publication_decision"] in {"GO", "HOLD"},
        "publication decision must be GO or HOLD",
    )
    return fields


def validate_recovery_authorization(
    comments: Sequence[Mapping[str, Any]],
    *,
    owner_login: str,
    expected: Mapping[str, str],
) -> dict[str, str]:
    """Require the latest Owner record for this exact candidate to authorize GO."""
    _require(isinstance(comments, Sequence), "GitHub issue comments are malformed")
    identity_fields = (
        "version",
        "tag",
        "source_sha",
        "tag_object_sha",
        "candidate_run_id",
    )
    matching: list[tuple[int, dict[str, str]]] = []
    for comment in comments:
        if not isinstance(comment, Mapping):
            continue
        user = comment.get("user")
        login = user.get("login") if isinstance(user, Mapping) else None
        if not isinstance(login, str) or login.lower() != owner_login.lower():
            continue
        body = comment.get("body")
        if not isinstance(body, str) or not body.startswith(MARKER + "\n"):
            continue
        try:
            record = parse_recovery_authorization(body)
        except RecoveryAuthorizationError:
            continue
        if all(record[field] == expected[field] for field in identity_fields):
            comment_id = comment.get("id")
            if (
                not isinstance(comment_id, int)
                or isinstance(comment_id, bool)
                or comment_id <= 0
            ):
                raise RecoveryAuthorizationError(
                    "recovery authorization comment ID is invalid"
                )
            _require(
                comment.get("created_at") == comment.get("updated_at"),
                "recovery authorization comment was edited",
            )
            matching.append((comment_id, record))

    _require(bool(matching), "no matching Release Owner recovery authorization exists")
    _, latest = max(matching, key=lambda item: item[0])
    _require(
        latest["publication_decision"] == "GO",
        "latest Release Owner recovery authorization is HOLD",
    )
    for name, value in expected.items():
        _require(latest.get(name) == value, f"recovery authorization {name} mismatch")
    return latest


def _comments(
    repository: str, issue: int, token: str, api_url: str
) -> list[dict[str, Any]]:
    _require(
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is not None,
        "repository must be OWNER/NAME",
    )
    result: list[dict[str, Any]] = []
    page = 1
    while True:
        request = urllib.request.Request(
            f"{api_url.rstrip('/')}/repos/{repository}/issues/{issue}/comments"
            f"?per_page=100&page={page}",
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                page_items = json.load(response)
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryAuthorizationError(
                "could not read Release Owner recovery comments"
            ) from exc
        _require(isinstance(page_items, list), "GitHub comments response is malformed")
        result.extend(item for item in page_items if isinstance(item, dict))
        if len(page_items) < 100:
            return result
        page += 1


def fetch_pull_request(
    repository: str, number: int, token: str, api_url: str
) -> dict[str, Any]:
    """Fetch the selected recovery PR through GitHub's read-only API."""
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/repos/{repository}/pulls/{number}",
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryAuthorizationError(
            "could not read the recovery PR merge record"
        ) from exc
    _require(isinstance(result, dict), "GitHub recovery PR response is malformed")
    return result


def validate_recovery_pull_request(
    pull: Mapping[str, Any],
    *,
    files: Sequence[Mapping[str, Any]],
    repository: str,
    number: int,
    control_sha: str,
) -> None:
    """Bind the dispatch control tag to the Owner-approved merged PR."""
    base = pull.get("base")
    base_repo = base.get("repo") if isinstance(base, Mapping) else None
    base_repository = (
        base_repo.get("full_name") if isinstance(base_repo, Mapping) else None
    )
    _require(
        pull.get("number") == number
        and isinstance(pull.get("number"), int)
        and not isinstance(pull.get("number"), bool),
        "recovery PR number mismatch",
    )
    _require(
        pull.get("merged") is True
        and pull.get("merged_at") is not None
        and isinstance(base, Mapping)
        and base.get("ref") == "main"
        and isinstance(base_repository, str)
        and base_repository.lower() == repository.lower()
        and pull.get("merge_commit_sha") == control_sha,
        "recovery control tag does not identify the approved main merge",
    )
    changed_paths = {
        item.get("filename") for item in files if isinstance(item, Mapping)
    }
    _require(
        REQUIRED_RECOVERY_PR_PATHS <= changed_paths,
        "recovery PR omits required validator, workflow, contract, docs, or tests",
    )


def fetch_pull_request_files(
    repository: str, number: int, token: str, api_url: str
) -> list[dict[str, Any]]:
    """Fetch every changed path from the selected recovery PR."""
    result: list[dict[str, Any]] = []
    page = 1
    while True:
        request = urllib.request.Request(
            f"{api_url.rstrip('/')}/repos/{repository}/pulls/{number}/files"
            f"?per_page=100&page={page}",
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                page_items = json.load(response)
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryAuthorizationError(
                "could not read the recovery PR changed-file list"
            ) from exc
        _require(isinstance(page_items, list), "GitHub recovery PR files are malformed")
        result.extend(item for item in page_items if isinstance(item, dict))
        if len(page_items) < 100:
            return result
        page += 1


def expected_authorization(
    *,
    tag: str,
    source_sha: str,
    tag_object_sha: str,
    recovery_pr: int,
    candidate_run_id: int,
    manifest_sha256: str,
    sdist_sha256: str,
    wheel_sha256: str,
    zenodo_date_disposition: str,
) -> tuple[dict[str, str], str]:
    contract = _contract(tag)
    expected_control_ref = contract.get("recovery_dispatch_ref")
    if not isinstance(expected_control_ref, str):
        raise RecoveryAuthorizationError(
            "release contract does not permit recovery dispatch"
        )
    return (
        {
            "version": tag.removeprefix("v"),
            "tag": tag,
            "source_sha": source_sha,
            "tag_object_sha": tag_object_sha,
            "control_ref": expected_control_ref,
            "recovery_pr": str(recovery_pr),
            "candidate_run_id": str(candidate_run_id),
            "promotion_manifest_sha256": manifest_sha256,
            "sdist_sha256": sdist_sha256,
            "wheel_sha256": wheel_sha256,
            "tagger_date_policy": str(contract["tagger_date_policy"]),
            "source_metadata_date_policy": str(contract["source_metadata_date_policy"]),
            "publication_tip_policy": str(contract["publication_tip_policy"]),
            "existing_candidate": "reuse-qualified-candidate",
            "zenodo_date_disposition": zenodo_date_disposition,
            "trusted_publishing": "existing-pypi-workflow-environment",
            "publication_decision": "GO",
        },
        str(contract["promotion"]["release_go"]["owner_login"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--tag-object-sha", required=True)
    parser.add_argument("--recovery-pr", required=True, type=int)
    parser.add_argument("--control-sha", required=True)
    parser.add_argument("--candidate-run-id", required=True, type=int)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--sdist-sha256", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument(
        "--zenodo-date-disposition",
        choices=sorted(ZENODO_CHOICES),
        required=True,
    )
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    args = parser.parse_args(argv)
    try:
        contract = _contract(args.tag)
        expected, owner = expected_authorization(
            tag=args.tag,
            source_sha=args.source_sha,
            tag_object_sha=args.tag_object_sha,
            recovery_pr=args.recovery_pr,
            candidate_run_id=args.candidate_run_id,
            manifest_sha256=args.manifest_sha256,
            sdist_sha256=args.sdist_sha256,
            wheel_sha256=args.wheel_sha256,
            zenodo_date_disposition=args.zenodo_date_disposition,
        )
        token = os.environ.get("GITHUB_TOKEN")
        _require(bool(token), "GITHUB_TOKEN is required")
        comments = _comments(
            args.repository,
            int(contract["promotion"]["release_go"]["issue_number"]),
            token or "",
            args.api_url,
        )
        validate_recovery_authorization(
            comments,
            owner_login=owner,
            expected=expected,
        )
        pull = fetch_pull_request(
            args.repository,
            args.recovery_pr,
            token or "",
            args.api_url,
        )
        pull_files = fetch_pull_request_files(
            args.repository,
            args.recovery_pr,
            token or "",
            args.api_url,
        )
        validate_recovery_pull_request(
            pull,
            files=pull_files,
            repository=args.repository,
            number=args.recovery_pr,
            control_sha=args.control_sha,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"recovery authorization failed: {exc}", file=sys.stderr)
        return 1
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write("authorized=true\n")
    print("recovery_authorized=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
