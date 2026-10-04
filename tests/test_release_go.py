"""Tests for immutable release-owner GO authorization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "verify_release_go.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_release_go_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    release_sha = "a" * 40
    manifest_hash = "b" * 64
    sdist_hash = "c" * 64
    wheel_hash = "d" * 64
    contract = {
        "promotion": {
            "release_go": {"issue_number": 42, "owner_login": "release-owner"}
        }
    }
    manifest = {
        "version": "0.2.6",
        "release_sha": release_sha,
        "run_id": 987654,
        "artifacts": [
            {
                "role": "payload",
                "files": [
                    {"name": "gwexpy-0.2.6.tar.gz", "sha256": sdist_hash},
                    {"name": "gwexpy-0.2.6-py3-none-any.whl", "sha256": wheel_hash},
                ],
            }
        ],
    }
    body = "\n".join(
        [
            "GWEXPY-RELEASE-GO-v1",
            "version=v0.2.6",
            f"source_sha={release_sha}",
            "candidate_run_id=987654",
            f"promotion_manifest_sha256={manifest_hash}",
            f"sdist_sha256={sdist_hash}",
            f"wheel_sha256={wheel_hash}",
            "decision=GO",
        ]
    )
    comment = {
        "id": 12345,
        "issue_number": 42,
        "issue_url": "https://api.github.com/repos/tatsuki-washimi/gwexpy/issues/42",
        "user": {"login": "release-owner"},
        "created_at": "2026-10-05T10:05:00Z",
        "updated_at": "2026-10-05T10:05:00Z",
        "body": body,
    }
    run = {
        "id": 987654,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "completed_at": "2026-10-05T10:04:00Z",
    }
    archive = b"synthetic-manifest-archive"
    artifact = {
        "artifact_id": 445566,
        "name": f"release-promotion-manifest-{release_sha}",
        "run_id": 987654,
        "expired": False,
        "size_in_bytes": len(archive),
        "digest": f"sha256:{hashlib.sha256(archive).hexdigest()}",
        "created_at": "2026-10-05T10:02:00Z",
    }
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    manifest_hash = hashlib.sha256(raw).hexdigest()
    comment["body"] = comment["body"].replace("b" * 64, manifest_hash)
    return contract, manifest, comment, run, manifest_hash, artifact, raw, archive


def test_valid_comment_and_candidate_timing(inputs):
    module = load_module()
    contract, manifest, comment, run, manifest_hash, artifact, raw, archive = inputs
    parsed = module.verify_release_go(
        contract=contract,
        manifest=manifest,
        manifest_raw=raw,
        manifest_sha256=manifest_hash,
        comment_id=12345,
        comment=comment,
        candidate_run=run,
        manifest_artifact=artifact,
        manifest_archive=archive,
    )
    assert parsed["source_sha"] == "a" * 40


def test_go_rejects_raw_manifest_that_does_not_decode_to_supplied_mapping(inputs):
    module = load_module()
    contract, manifest, comment, run, _, artifact, _, archive = inputs
    raw = b"{}\n"
    digest = hashlib.sha256(raw).hexdigest()
    comment["body"] = comment["body"].replace(
        comment["body"].split("promotion_manifest_sha256=", 1)[1].split("\n", 1)[0],
        digest,
    )
    with pytest.raises(module.ReleaseGoError, match="manifest content"):
        module.verify_release_go(
            contract=contract,
            manifest=manifest,
            manifest_raw=raw,
            manifest_sha256=digest,
            comment_id=12345,
            comment=comment,
            candidate_run=run,
            manifest_artifact=artifact,
            manifest_archive=archive,
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://api.github.com/repos/other/repository/issues/42",
        "https://api.github.com/repos/tatsuki-washimi/gwexpy/pulls/42",
        "https://api.github.com/repos/tatsuki-washimi/gwexpy/issues/43",
    ],
)
def test_go_rejects_issue_url_with_wrong_repository_path_or_number(inputs, url):
    module = load_module()
    contract, manifest, comment, run, digest, artifact, raw, archive = inputs
    comment["issue_url"] = url
    with pytest.raises(module.ReleaseGoError, match="tracking issue"):
        module.verify_release_go(
            contract=contract,
            manifest=manifest,
            manifest_raw=raw,
            manifest_sha256=digest,
            comment_id=12345,
            comment=comment,
            candidate_run=run,
            manifest_artifact=artifact,
            manifest_archive=archive,
        )


def test_go_rejects_contradictory_issue_number_and_url(inputs):
    module = load_module()
    contract, manifest, comment, run, digest, artifact, raw, archive = inputs
    comment["issue_number"] = 42
    comment["issue_url"] = (
        "https://api.github.com/repos/tatsuki-washimi/gwexpy/issues/43"
    )
    with pytest.raises(module.ReleaseGoError, match="tracking issue"):
        module.verify_release_go(
            contract=contract,
            manifest=manifest,
            manifest_raw=raw,
            manifest_sha256=digest,
            comment_id=12345,
            comment=comment,
            candidate_run=run,
            manifest_artifact=artifact,
            manifest_archive=archive,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "id",
        "issue",
        "author",
        "updated",
        "body",
        "run_status",
        "conclusion",
        "attempt",
        "attempt_float",
        "attempt_bool",
        "in_progress",
        "null_conclusion",
        "manifest_hash",
    ],
)
def test_invalid_go_or_candidate_is_rejected(inputs, mutation):
    module = load_module()
    contract, manifest, comment, run, manifest_hash, artifact, raw, archive = inputs
    if mutation == "id":
        comment_id = 12346
    else:
        comment_id = 12345
    if mutation == "issue":
        comment["issue_number"] = 43
    if mutation == "author":
        comment["user"]["login"] = "other"
    if mutation == "updated":
        comment["updated_at"] = "2026-10-05T10:04:00Z"
    if mutation == "body":
        comment["body"] += "\nextra=x"
    if mutation == "run_status":
        run["status"] = "in_progress"
    if mutation == "conclusion":
        run["conclusion"] = "failure"
    if mutation == "attempt":
        run["run_attempt"] = 2
    if mutation == "attempt_float":
        run["run_attempt"] = 1.0
    if mutation == "attempt_bool":
        run["run_attempt"] = True
    if mutation == "in_progress":
        run["status"] = "in_progress"
    if mutation == "null_conclusion":
        run["conclusion"] = None
    if mutation == "manifest_hash":
        manifest_hash = "e" * 64
    with pytest.raises(module.ReleaseGoError):
        module.verify_release_go(
            contract=contract,
            manifest=manifest,
            manifest_raw=raw,
            manifest_sha256=manifest_hash,
            comment_id=comment_id,
            comment=comment,
            candidate_run=run,
            manifest_artifact=artifact,
            manifest_archive=archive,
        )


@pytest.mark.parametrize(
    ("completed_at", "artifact_created", "comment_at", "error"),
    [
        (
            "2026-10-05T10:02:00Z",
            "2026-10-05T10:04:00Z",
            "2026-10-05T10:03:00Z",
            "later than manifest artifact upload",
        ),
        (
            "2026-10-05T10:04:00Z",
            "2026-10-05T10:02:00Z",
            "2026-10-05T10:03:00Z",
            "later than candidate completion",
        ),
        (
            "2026-10-05T10:03:00Z",
            "2026-10-05T10:02:00Z",
            "2026-10-05T10:03:00Z",
            "later than candidate completion",
        ),
        (
            "2026-10-05T10:02:00Z",
            "2026-10-05T10:03:00Z",
            "2026-10-05T10:03:00Z",
            "later than manifest artifact upload",
        ),
    ],
)
def test_go_timestamp_must_be_strictly_after_each_boundary(
    inputs, completed_at, artifact_created, comment_at, error
):
    module = load_module()
    contract, manifest, comment, run, digest, artifact, raw, archive = inputs
    run["completed_at"] = completed_at
    artifact["created_at"] = artifact_created
    comment["created_at"] = comment_at
    comment["updated_at"] = comment_at
    with pytest.raises(module.ReleaseGoError, match=error):
        module.verify_release_go(
            contract=contract,
            manifest=manifest,
            manifest_raw=raw,
            manifest_sha256=digest,
            comment_id=12345,
            comment=comment,
            candidate_run=run,
            manifest_artifact=artifact,
            manifest_archive=archive,
        )


@pytest.mark.parametrize("mutation", ["name", "run", "id", "expired", "digest", "size"])
def test_go_rejects_invalid_manifest_artifact_metadata(inputs, mutation):
    module = load_module()
    contract, manifest, comment, run, digest, artifact, raw, archive = inputs
    if mutation == "name":
        artifact["name"] = "unexpected-manifest"
    if mutation == "run":
        artifact["run_id"] = 987653
    if mutation == "id":
        artifact["artifact_id"] = 0
    if mutation == "expired":
        artifact["expired"] = True
    if mutation == "digest":
        artifact["digest"] = "sha256:" + "f" * 64
    if mutation == "size":
        artifact["size_in_bytes"] += 1
    with pytest.raises(module.ReleaseGoError):
        module.verify_release_go(
            contract=contract,
            manifest=manifest,
            manifest_raw=raw,
            manifest_sha256=digest,
            comment_id=12345,
            comment=comment,
            candidate_run=run,
            manifest_artifact=artifact,
            manifest_archive=archive,
        )


@pytest.mark.parametrize("kind", ["sdist", "wheel"])
def test_go_rejects_distribution_digest_mismatch(inputs, kind):
    module = load_module()
    contract, manifest, comment, run, manifest_hash, artifact, raw, archive = inputs
    payload_file = manifest["artifacts"][0]["files"][0 if kind == "sdist" else 1]
    payload_file["sha256"] = "e" * 64
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    manifest_hash = hashlib.sha256(raw).hexdigest()
    comment["body"] = comment["body"].replace(inputs[4], manifest_hash)
    with pytest.raises(module.ReleaseGoError, match="distribution hashes"):
        module.verify_release_go(
            contract=contract,
            manifest=manifest,
            manifest_raw=raw,
            manifest_sha256=manifest_hash,
            comment_id=12345,
            comment=comment,
            candidate_run=run,
            manifest_artifact=artifact,
            manifest_archive=archive,
        )
