#!/usr/bin/env python3
"""Collect a strict, privacy-safe aggregate from four smoke reports."""

from __future__ import annotations

import argparse
import json
import re
import stat
from pathlib import Path
from typing import Any

SOURCE_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PAYLOAD_SCHEMA = "gwexpy-v0113-release-payload-v1"
SMOKE_KEYS = {
    "python-3.11-wheel",
    "python-3.11-sdist",
    "python-3.12-wheel",
    "python-3.12-sdist",
}
SMOKE_FIELDS = {
    "source_sha",
    "python",
    "distribution",
    "repository_license_sha256",
    "embedded_license_sha256",
    "installed_version",
    "import_ok",
    "register_all_ok",
    "smoke_ok",
}
RELEASE_TAG = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class ReleaseEvidenceError(ValueError):
    """Raised when evidence is incomplete, unsafe, or from another release."""


class _DuplicateJSONKey(ValueError):
    """Raised by the JSON hook when an object has ambiguous keys."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _object(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    ):
        raise ReleaseEvidenceError(
            f"evidence input must be a regular file: {path.name}"
        )
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError, _DuplicateJSONKey) as exc:
        raise ReleaseEvidenceError(f"invalid JSON: {path.name}") from exc
    if not isinstance(data, dict):
        raise ReleaseEvidenceError(f"JSON must be an object: {path.name}")
    return data


def _keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    if set(data) != expected:
        raise ReleaseEvidenceError(f"{context} has missing or unknown keys")


def _payload(path: Path) -> dict[str, Any]:
    data = _object(path)
    _keys(data, {"schema", "source_sha", "version", "files"}, "payload manifest")
    if data["schema"] != PAYLOAD_SCHEMA or not isinstance(data["files"], dict):
        raise ReleaseEvidenceError("unsupported payload manifest")
    _keys(data["files"], {"wheel", "sdist"}, "payload files")
    for kind, suffix in (("wheel", ".whl"), ("sdist", ".tar.gz")):
        item = data["files"][kind]
        if not isinstance(item, dict) or set(item) != {"name", "sha256"}:
            raise ReleaseEvidenceError("invalid payload distribution entry")
        if (
            not isinstance(item["name"], str)
            or Path(item["name"]).name != item["name"]
            or not item["name"].endswith(suffix)
        ):
            raise ReleaseEvidenceError("invalid payload distribution name")
        if not isinstance(item["sha256"], str) or not SHA256.fullmatch(item["sha256"]):
            raise ReleaseEvidenceError("invalid payload distribution hash")
    return data


def _license(sidecars: Path, payload_manifest: Path) -> str:
    if sidecars.is_symlink() or not sidecars.is_dir():
        raise ReleaseEvidenceError("sidecars must be a real directory")
    if payload_manifest.parent != sidecars:
        raise ReleaseEvidenceError("payload manifest must be a detached sidecar")
    entries = {entry.name: entry for entry in sidecars.iterdir()}
    if set(entries) != {"LICENSE.sha256", "distribution-sha256.json"}:
        raise ReleaseEvidenceError("sidecars have missing or extra files")
    license_file = entries["LICENSE.sha256"]
    if license_file.is_symlink() or not license_file.is_file():
        raise ReleaseEvidenceError("sidecars must contain a regular LICENSE.sha256")
    value = license_file.read_text(encoding="ascii").strip()
    if not SHA256.fullmatch(value):
        raise ReleaseEvidenceError("LICENSE.sha256 must contain one lowercase SHA-256")
    return value


def _smoke_reports(
    directory: Path, payload: dict[str, Any], license_hash: str, source_sha: str
) -> dict[str, dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        raise ReleaseEvidenceError("smoke evidence must be a real directory")
    found: dict[str, dict[str, Any]] = {}
    for path in directory.iterdir():
        if (
            path.is_symlink()
            or not path.is_file()
            or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
        ):
            raise ReleaseEvidenceError("smoke evidence contains a non-regular file")
        match = re.fullmatch(r"python-(3\.11|3\.12)-(wheel|sdist)\.json", path.name)
        if not match:
            raise ReleaseEvidenceError(f"unexpected smoke evidence file: {path.name}")
        key = f"python-{match.group(1)}-{match.group(2)}"
        if key in found:
            raise ReleaseEvidenceError(f"duplicate smoke cell: {key}")
        report = _object(path)
        _keys(report, SMOKE_FIELDS, f"smoke {key}")
        distribution = report["distribution"]
        if not isinstance(distribution, dict) or set(distribution) != {
            "kind",
            "file",
            "sha256",
        }:
            raise ReleaseEvidenceError(f"invalid distribution in smoke {key}")
        expected = payload["files"][match.group(2)]
        if (
            report["source_sha"] != source_sha
            or report["python"] != match.group(1)
            or distribution
            != {
                "kind": match.group(2),
                "file": expected["name"],
                "sha256": expected["sha256"],
            }
            or report["repository_license_sha256"] != license_hash
            or report["embedded_license_sha256"] != license_hash
            or report["installed_version"] != payload["version"]
            or any(
                report[field] is not True
                for field in ("import_ok", "register_all_ok", "smoke_ok")
            )
        ):
            raise ReleaseEvidenceError(
                f"smoke report does not match release facts: {key}"
            )
        found[key] = report
    if set(found) != SMOKE_KEYS:
        raise ReleaseEvidenceError(
            "smoke evidence must contain exactly four fixed cells"
        )
    return found


def assemble_evidence(
    payload_manifest: Path | str,
    sidecars: Path | str,
    smoke_dir: Path | str,
    source_sha: str,
    repository: str,
    run_id: str,
    workflow_sha: str,
    workflow_ref: str,
    expected_tag: str,
) -> dict[str, Any]:
    """Return only the allowlisted aggregate for this workflow run."""
    if not SOURCE_SHA.fullmatch(source_sha) or not SOURCE_SHA.fullmatch(workflow_sha):
        raise ReleaseEvidenceError(
            "source and workflow SHA must be full lowercase SHAs"
        )
    if not repository or not run_id.isdecimal() or not workflow_ref:
        raise ReleaseEvidenceError("invalid workflow identity fields")
    tag_match = RELEASE_TAG.fullmatch(expected_tag)
    if tag_match is None:
        raise ReleaseEvidenceError("expected tag must be a final SemVer release tag")
    manifest_path = Path(payload_manifest)
    payload = _payload(manifest_path)
    if payload["source_sha"] != source_sha:
        raise ReleaseEvidenceError(
            "payload source SHA does not match workflow source SHA"
        )
    if payload["version"] != expected_tag.removeprefix("v"):
        raise ReleaseEvidenceError("payload version does not match expected tag")
    license_hash = _license(Path(sidecars), manifest_path)
    smoke = _smoke_reports(Path(smoke_dir), payload, license_hash, source_sha)
    return {
        "schema": "gwexpy-v0113-integration-evidence-v1",
        "artifact_name": f"v0113-integration-evidence-{source_sha}",
        "repository": repository,
        "run_id": run_id,
        "workflow_sha": workflow_sha,
        "workflow_ref": workflow_ref,
        "source_sha": source_sha,
        "expected_tag": expected_tag,
        "version": payload["version"],
        "payload": payload["files"],
        "license_sha256": license_hash,
        "smoke": smoke,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--workflow-ref", required=True)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = assemble_evidence(
            args.payload_manifest,
            args.sidecars,
            args.smoke_dir,
            args.source_sha,
            args.repository,
            args.run_id,
            args.workflow_sha,
            args.workflow_ref,
            args.expected_tag,
        )
        if args.output.exists() or args.output.is_symlink():
            raise ReleaseEvidenceError("aggregate output must not already exist")
        args.output.write_text(
            json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
        )
    except ReleaseEvidenceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
