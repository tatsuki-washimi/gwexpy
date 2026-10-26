#!/usr/bin/env python3
"""Require the configured Release Owner's exact approval for partial PyPI recovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MARKER = "GWEXPY-PYPI-RECOVERY-v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RecoveryApprovalError(ValueError):
    """Raised when a partial PyPI upload lacks exact Owner authorization."""


def _release_contract(tag: str) -> dict[str, Any]:
    path = Path(__file__).with_name("release_contract.py")
    spec = importlib.util.spec_from_file_location(
        "release_contract_for_pypi_recovery", path
    )
    if spec is None or spec.loader is None:
        raise RecoveryApprovalError("release contract is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.release_contract(tag)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryApprovalError(message)


def _parse_files(value: str) -> dict[str, str]:
    result = {}
    for item in value.split(","):
        parts = item.split(":")
        _require(len(parts) == 2, "recovery approval has malformed approved_files")
        name, digest = parts
        _require(
            name.startswith("gwexpy-")
            and "/" not in name
            and "\\" not in name
            and "\n" not in name
            and "\r" not in name
            and SHA256.fullmatch(digest) is not None
            and name not in result,
            "recovery approval has an invalid or duplicate file record",
        )
        result[name] = digest
    _require(bool(result), "recovery approval has no approved files")
    _require(
        list(result) == sorted(result, key=lambda item: item.encode("utf-8")),
        "recovery approval file records are not canonical",
    )
    return result


def parse_recovery_decision(body: str) -> dict[str, Any]:
    """Parse the exact line-based Release Owner recovery decision format."""
    lines = body.splitlines()
    _require(
        len(lines) == 8 and lines[0] == MARKER,
        "recovery approval has a noncanonical record",
    )
    expected_names = (
        "version",
        "tag",
        "source_sha",
        "candidate_run_id",
        "promotion_manifest_sha256",
        "approved_files",
        "decision",
    )
    fields = {}
    for line, name in zip(lines[1:], expected_names):
        key, separator, value = line.partition("=")
        if not separator or key != name or not value:
            raise RecoveryApprovalError("recovery approval field order is invalid")
        fields[name] = value
    _require(
        fields["tag"] == "v" + fields["version"],
        "recovery approval tag/version mismatch",
    )
    _require(
        SHA40.fullmatch(fields["source_sha"]) is not None, "invalid recovery source SHA"
    )
    _require(
        fields["candidate_run_id"].isdecimal() and int(fields["candidate_run_id"]) > 0,
        "invalid recovery candidate run ID",
    )
    _require(
        SHA256.fullmatch(fields["promotion_manifest_sha256"]) is not None,
        "invalid recovery manifest SHA-256",
    )
    _require(fields["decision"] in {"GO", "HOLD"}, "invalid recovery decision")
    return {
        "version": fields["version"],
        "tag": fields["tag"],
        "source_sha": fields["source_sha"],
        "candidate_run_id": int(fields["candidate_run_id"]),
        "promotion_manifest_sha256": fields["promotion_manifest_sha256"],
        "approved_files": _parse_files(fields["approved_files"]),
        "decision": fields["decision"],
    }


def validate_recovery_decision(
    comments: Sequence[Mapping[str, Any]],
    *,
    owner_login: str,
    expected: Mapping[str, Any],
) -> None:
    """Require the latest Owner record for this candidate to cover missing files."""
    _require(isinstance(comments, Sequence), "GitHub issue comments are malformed")
    matching: list[tuple[int, dict[str, Any]]] = []
    identity_fields = (
        "version",
        "tag",
        "source_sha",
        "candidate_run_id",
        "promotion_manifest_sha256",
    )
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
            decision = parse_recovery_decision(body)
        except RecoveryApprovalError:
            continue
        if all(decision[field] == expected[field] for field in identity_fields):
            comment_id = comment.get("id")
            if (
                not isinstance(comment_id, int)
                or isinstance(comment_id, bool)
                or comment_id <= 0
            ):
                raise RecoveryApprovalError("recovery approval comment ID is invalid")
            matching.append((comment_id, decision))
    _require(bool(matching), "no matching Release Owner recovery decision exists")
    _, latest = max(matching, key=lambda item: item[0])
    _require(
        latest["decision"] == "GO", "latest Release Owner recovery decision is HOLD"
    )
    approved_files = latest["approved_files"]
    candidate_files = expected["candidate_files"]
    _require(
        approved_files
        and all(
            candidate_files.get(name) == digest
            for name, digest in approved_files.items()
        ),
        "recovery approval files differ from the candidate manifest",
    )
    missing_files = expected["missing_files"]
    _require(
        bool(missing_files)
        and all(
            approved_files.get(name) == digest for name, digest in missing_files.items()
        ),
        "recovery approval does not cover every currently missing file",
    )


def _comments(repository: str, token: str, issue: int) -> list[dict[str, Any]]:
    comments = []
    page = 1
    while True:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository}/issues/{issue}/comments?per_page=100&page={page}",
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryApprovalError(
                "could not read Release Owner recovery comments"
            ) from exc
        if not isinstance(result, list):
            raise RecoveryApprovalError("GitHub issue comments response is malformed")
        comments.extend(result)
        if len(result) < 100:
            return comments
        page += 1


def _candidate_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RecoveryApprovalError("promotion manifest artifacts are malformed")
    payloads = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("role") == "payload"
    ]
    _require(len(payloads) == 1, "promotion manifest payload is missing or ambiguous")
    files = payloads[0].get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise RecoveryApprovalError("promotion payload must have two distributions")
    result = {}
    for item in files:
        _require(isinstance(item, Mapping), "promotion payload file is malformed")
        name = item.get("name")
        digest = item.get("sha256")
        _require(
            isinstance(name, str)
            and name.startswith("gwexpy-")
            and "/" not in name
            and "\\" not in name
            and isinstance(digest, str)
            and SHA256.fullmatch(digest) is not None
            and name not in result,
            "promotion payload filename or SHA-256 is invalid",
        )
        result[name] = digest
    _require(
        sum(name.endswith(".whl") for name in result) == 1
        and sum(name.endswith(".tar.gz") for name in result) == 1,
        "promotion payload must contain one wheel and one source distribution",
    )
    return result


def _missing_files(
    directory: Path, candidate_files: Mapping[str, str]
) -> dict[str, str]:
    _require(
        directory.is_dir() and not directory.is_symlink(),
        "missing PyPI payload is not a directory",
    )
    observed = {}
    for path in directory.iterdir():
        _require(
            stat.S_ISREG(path.lstat().st_mode), "missing PyPI payload has a non-file"
        )
        _require(
            path.name in candidate_files,
            "missing PyPI file is outside candidate payload",
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            digest == candidate_files[path.name],
            "missing PyPI file hash differs from candidate",
        )
        observed[path.name] = digest
    _require(bool(observed), "partial PyPI recovery has no missing files")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--candidate-run-id", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--missing-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.manifest.read_bytes()
        manifest = json.loads(raw)
        _require(isinstance(manifest, dict), "promotion manifest is malformed")
        manifest_sha = hashlib.sha256(raw).hexdigest()
        _require(
            args.tag == "v" + args.version
            and manifest.get("version") == args.version
            and manifest.get("tag") == args.tag
            and manifest.get("source_sha") == args.source_sha
            and manifest.get("release_sha") == args.source_sha
            and manifest.get("run_id") == args.candidate_run_id
            and SHA40.fullmatch(args.source_sha) is not None,
            "recovery target differs from the candidate manifest",
        )
        candidate_files = _candidate_files(manifest)
        missing_files = _missing_files(args.missing_dir, candidate_files)
        contract = _release_contract(args.tag)
        authority = contract["promotion"]["release_go"]
        token = os.environ["GITHUB_TOKEN"]
        comments = _comments(args.repository, token, authority["issue_number"])
        validate_recovery_decision(
            comments,
            owner_login=authority["owner_login"],
            expected={
                "version": args.version,
                "tag": args.tag,
                "source_sha": args.source_sha,
                "candidate_run_id": args.candidate_run_id,
                "promotion_manifest_sha256": manifest_sha,
                "candidate_files": candidate_files,
                "missing_files": missing_files,
            },
        )
    except (
        KeyError,
        OSError,
        json.JSONDecodeError,
        RecoveryApprovalError,
        ValueError,
    ) as exc:
        print(f"PyPI recovery authorization rejected: {exc}", file=sys.stderr)
        return 1
    print("Release Owner recovery decision verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
