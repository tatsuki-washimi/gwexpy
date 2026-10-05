#!/usr/bin/env python3
"""Verify an immutable release-owner GO comment against candidate artifacts."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ReleaseGoError(ValueError):
    """Raised when release GO identity, content, or timing is invalid."""


def _promotion_module() -> Any:
    path = Path(__file__).with_name("release_promotion.py")
    spec = importlib.util.spec_from_file_location("release_promotion_for_go", path)
    if spec is None or spec.loader is None:
        raise ReleaseGoError("release promotion parser is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ReleaseGoError(f"{name} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseGoError(f"{name} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ReleaseGoError(f"{name} timestamp must include a timezone")
    return parsed


def fetch_comment(
    repository: str, comment_id: int, *, token: str, api_url: str
) -> dict[str, Any]:
    """Fetch one issue comment; callers can inject a local replacement in tests."""
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ReleaseGoError("repository must be OWNER/NAME")
    request = Request(
        f"{api_url.rstrip('/')}/repos/{repository}/issues/comments/{comment_id}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "gwexpy-release-go-verifier",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read(64 * 1024 + 1)
    except HTTPError as exc:
        raise ReleaseGoError(f"GitHub comment API returned HTTP {exc.code}") from exc
    except (OSError, URLError) as exc:
        raise ReleaseGoError("GitHub comment API request failed") from exc
    if len(raw) > 64 * 1024:
        raise ReleaseGoError("GitHub comment response is too large")
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReleaseGoError("GitHub comment API returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ReleaseGoError("GitHub comment API returned an invalid object")
    return result


def _validate_issue_identity(
    comment: Mapping[str, Any], *, repository: str, api_url: str, expected_number: int
) -> None:
    issue_url = comment.get("issue_url")
    if not isinstance(issue_url, str):
        raise ReleaseGoError("GO comment tracking issue URL is missing")
    expected_url = f"{api_url.rstrip('/')}/repos/{repository}/issues/{expected_number}"
    actual_parts = urlsplit(issue_url)
    expected_parts = urlsplit(expected_url)
    if (
        actual_parts.scheme != expected_parts.scheme
        or actual_parts.netloc != expected_parts.netloc
        or actual_parts.path != expected_parts.path
        or actual_parts.query
        or actual_parts.fragment
        or actual_parts.username is not None
        or actual_parts.password is not None
    ):
        raise ReleaseGoError("GO comment tracking issue URL mismatch")
    synthetic_number = comment.get("issue_number")
    if synthetic_number is not None and (
        isinstance(synthetic_number, bool)
        or not isinstance(synthetic_number, int)
        or synthetic_number != expected_number
    ):
        raise ReleaseGoError("GO comment tracking issue number contradicts URL")


def _distribution_hashes(manifest: Mapping[str, Any]) -> tuple[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReleaseGoError("manifest artifacts are missing")
    payloads = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("role") == "payload"
    ]
    if len(payloads) != 1 or not isinstance(payloads[0].get("files"), list):
        raise ReleaseGoError("manifest payload artifact is missing or ambiguous")
    files = payloads[0]["files"]
    wheel = [
        item.get("sha256")
        for item in files
        if isinstance(item, Mapping) and str(item.get("name", "")).endswith(".whl")
    ]
    sdist = [
        item.get("sha256")
        for item in files
        if isinstance(item, Mapping) and str(item.get("name", "")).endswith(".tar.gz")
    ]
    if len(wheel) != 1 or len(sdist) != 1:
        raise ReleaseGoError("manifest must bind exactly one wheel and sdist")
    return sdist[0], wheel[0]


def verify_release_go(
    *,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_raw: bytes,
    manifest_sha256: str,
    comment_id: int,
    candidate_run: Mapping[str, Any],
    manifest_artifact: Mapping[str, Any],
    manifest_archive: bytes,
    comment: Mapping[str, Any] | None = None,
    repository: str = "tatsuki-washimi/gwexpy",
    token: str | None = None,
    api_url: str = "https://api.github.com",
    comment_fetcher: Callable[..., dict[str, Any]] = fetch_comment,
) -> dict[str, str]:
    """Verify the configured issue comment, package bytes, and strict timing."""
    try:
        promotion = contract["promotion"]["release_go"]
        issue_number, owner = promotion["issue_number"], promotion["owner_login"]
    except (KeyError, TypeError) as exc:
        raise ReleaseGoError("release GO authority is missing from contract") from exc
    if (
        not isinstance(comment_id, int)
        or isinstance(comment_id, bool)
        or comment_id <= 0
    ):
        raise ReleaseGoError("comment ID must be a positive integer")
    if comment is None:
        if not token:
            raise ReleaseGoError("GITHUB_TOKEN is required to fetch the GO comment")
        comment = comment_fetcher(repository, comment_id, token=token, api_url=api_url)
    if isinstance(comment.get("id"), bool) or comment.get("id") != comment_id:
        raise ReleaseGoError("GO comment ID mismatch")
    _validate_issue_identity(
        comment,
        repository=repository,
        api_url=api_url,
        expected_number=issue_number,
    )
    user = comment.get("user")
    if not isinstance(user, Mapping) or user.get("login") != owner:
        raise ReleaseGoError("GO comment author does not match release owner")
    created = _time(comment.get("created_at"), "comment created_at")
    updated = _time(comment.get("updated_at"), "comment updated_at")
    if updated != created:
        raise ReleaseGoError("GO comment was edited after creation")
    parser = _promotion_module()
    if (
        not isinstance(manifest_raw, bytes)
        or parser.sha256(manifest_raw) != manifest_sha256
    ):
        raise ReleaseGoError("raw promotion manifest digest does not match")
    try:
        decoded_manifest = parser.load_manifest(manifest_raw)
    except parser.PromotionManifestError as exc:
        raise ReleaseGoError(f"raw promotion manifest is invalid: {exc}") from exc
    if (
        decoded_manifest != manifest
        or parser.serialize_manifest(manifest) != manifest_raw
    ):
        raise ReleaseGoError(
            "raw manifest content does not match supplied manifest mapping"
        )
    if any(
        key in candidate_run
        for key in ("workflow_id", "path", "event", "head_branch", "head_sha")
    ):
        try:
            parser.validate_candidate_run(
                manifest,
                contract,
                candidate_run,
                repository=repository,
                tag=f"v{manifest.get('version')}",
                source_sha=manifest.get("release_sha"),
            )
        except parser.PromotionManifestError as exc:
            raise ReleaseGoError(f"candidate run identity is invalid: {exc}") from exc
    try:
        parser.validate_manifest_artifact(
            manifest_artifact,
            manifest_archive,
            manifest_raw,
            manifest.get("release_sha"),
            manifest.get("run_id"),
            manifest_sha256,
        )
    except parser.PromotionManifestError as exc:
        raise ReleaseGoError(f"promotion manifest artifact is invalid: {exc}") from exc
    try:
        record = parser.parse_release_go(comment.get("body"))
    except (parser.PromotionManifestError, TypeError) as exc:
        raise ReleaseGoError(str(exc)) from exc
    expected_version = manifest.get("version")
    expected_run = manifest.get("run_id")
    if (
        not isinstance(manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
    ):
        raise ReleaseGoError("manifest digest is malformed")
    if (
        record["version"] != f"v{expected_version}"
        or record["source_sha"] != manifest.get("release_sha")
        or record["candidate_run_id"] != str(expected_run)
        or record["promotion_manifest_sha256"] != manifest_sha256
    ):
        raise ReleaseGoError(
            "GO body does not match version, release SHA, run, or manifest digest"
        )
    try:
        sdist_hash, wheel_hash = _distribution_hashes(manifest)
    except ReleaseGoError:
        raise
    if record["sdist_sha256"] != sdist_hash or record["wheel_sha256"] != wheel_hash:
        raise ReleaseGoError("GO distribution hashes do not match promotion manifest")
    if (
        isinstance(candidate_run.get("id"), bool)
        or candidate_run.get("id") != expected_run
    ):
        raise ReleaseGoError("candidate run ID does not match manifest")
    if (
        candidate_run.get("status") != "completed"
        or candidate_run.get("conclusion") != "success"
        or type(candidate_run.get("run_attempt")) is not int
        or candidate_run.get("run_attempt") != 1
    ):
        raise ReleaseGoError(
            "candidate run must be completed successfully on attempt 1"
        )
    if candidate_run.get("conclusion") not in (None, "success"):
        raise ReleaseGoError("candidate gate/run conclusion is not successful")
    # The workflow-runs REST record has no completed_at field. For a run that
    # GitHub reports as completed, updated_at is a server-observed timestamp
    # at or after completion, so requiring GO after it is a conservative bound.
    completed = _time(candidate_run.get("updated_at"), "candidate updated_at")
    artifact_created = _time(
        manifest_artifact.get("created_at"), "manifest artifact created_at"
    )
    if created <= completed:
        raise ReleaseGoError("GO comment must be later than candidate completion")
    if created <= artifact_created:
        raise ReleaseGoError("GO comment must be later than manifest artifact upload")
    return record
