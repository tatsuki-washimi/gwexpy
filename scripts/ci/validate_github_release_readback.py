#!/usr/bin/env python3
"""Fail closed unless GitHub Release readback matches verified local files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class ReleaseReadbackError(ValueError):
    """Raised when a tag, release, or public asset fails readback checks."""


def validate_no_conflicting_release(releases: Any, expected_tag: str) -> None:
    """Reject malformed API results and every pre-existing Release for the tag."""
    if not isinstance(releases, list):
        raise ReleaseReadbackError("GitHub Releases API readback is not a list")
    flattened: list[Any] = []
    for page in releases:
        if isinstance(page, list):
            flattened.extend(page)
        else:
            flattened.append(page)
    if any(
        not isinstance(release, dict) or not isinstance(release.get("tag_name"), str)
        for release in flattened
    ):
        raise ReleaseReadbackError("GitHub Releases API contains a malformed entry")
    if any(release["tag_name"] == expected_tag for release in flattened):
        raise ReleaseReadbackError("a Release already exists for the requested tag")


def validate_tag_identity(
    *,
    tag_object_sha: str,
    peeled_sha: str,
    expected_tag_object_sha: str,
    expected_source_sha: str,
) -> None:
    """Require the same annotated tag object and peeled source seen at preflight."""
    values = (tag_object_sha, peeled_sha, expected_tag_object_sha, expected_source_sha)
    if any(
        not isinstance(value, str) or not FULL_SHA.fullmatch(value) for value in values
    ):
        raise ReleaseReadbackError("remote tag identity is missing or malformed")
    if tag_object_sha != expected_tag_object_sha or peeled_sha != expected_source_sha:
        raise ReleaseReadbackError("remote annotated tag identity changed")


def _regular_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ReleaseReadbackError(
            f"asset directory is not a regular directory: {directory}"
        )
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.name in result:
            raise ReleaseReadbackError(
                "asset directory contains a non-regular or duplicate entry"
            )
        result[path.name] = path
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release_readback(
    *,
    release: Any,
    expected_tag: str,
    expected_source_sha: str,
    notes: str,
    payload_dir: Path,
    sidecars_dir: Path,
    downloaded_dir: Path,
    existing_release: bool = False,
) -> None:
    """Check final Release metadata and byte identity of all four public assets."""
    if existing_release:
        raise ReleaseReadbackError("a Release already exists for the requested tag")
    if not isinstance(release, dict):
        raise ReleaseReadbackError("GitHub Release API readback is not an object")
    release_id = release.get("id")
    if (
        not isinstance(release_id, int)
        or isinstance(release_id, bool)
        or release_id <= 0
    ):
        raise ReleaseReadbackError("GitHub Release readback has no valid release id")
    if (
        release.get("tag_name") != expected_tag
        or release.get("name") != expected_tag
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or release.get("body") != notes
    ):
        raise ReleaseReadbackError(
            "GitHub Release tag, notes, or final-release state mismatched"
        )

    payload = _regular_files(payload_dir)
    sidecars = _regular_files(sidecars_dir)
    expected_names = {
        *(name for name in payload if name.endswith(".whl")),
        *(name for name in payload if name.endswith(".tar.gz")),
        "distribution-sha256.json",
        "LICENSE.sha256",
    }
    if len(payload) != 2 or len(sidecars) != 2 or len(expected_names) != 4:
        raise ReleaseReadbackError("local same-run payload or sidecars are incomplete")
    if not FULL_SHA.fullmatch(expected_source_sha):
        raise ReleaseReadbackError("expected source SHA is malformed")

    assets = release.get("assets")
    if not isinstance(assets, list) or len(assets) != 4:
        raise ReleaseReadbackError("GitHub Release must contain exactly four assets")
    by_name: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            raise ReleaseReadbackError("GitHub Release asset entry is malformed")
        name = asset.get("name")
        if not isinstance(name, str) or name in by_name:
            raise ReleaseReadbackError(
                "GitHub Release asset names are missing or duplicated"
            )
        by_name[name] = asset
    if set(by_name) != expected_names:
        raise ReleaseReadbackError(
            "GitHub Release assets do not match the same-run file set"
        )

    local_files = {**payload, **sidecars}
    downloaded = _regular_files(downloaded_dir)
    if set(downloaded) != expected_names:
        raise ReleaseReadbackError(
            "downloaded GitHub Release asset set is incomplete or unexpected"
        )
    for name in sorted(expected_names):
        asset = by_name[name]
        local = local_files[name]
        uploaded_size = local.stat().st_size
        if (
            not isinstance(asset.get("id"), int)
            or isinstance(asset.get("id"), bool)
            or asset["id"] <= 0
            or asset.get("state") != "uploaded"
            or asset.get("size") != uploaded_size
        ):
            raise ReleaseReadbackError(
                f"GitHub Release asset is not fully uploaded: {name}"
            )
        expected_hash = _sha256(local)
        digest = asset.get("digest")
        if digest is not None and digest != f"sha256:{expected_hash}":
            raise ReleaseReadbackError(f"GitHub Release API digest mismatch: {name}")
        if _sha256(downloaded[name]) != expected_hash:
            raise ReleaseReadbackError(
                f"downloaded GitHub Release asset bytes mismatch: {name}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--sidecars-dir", type=Path, required=True)
    parser.add_argument("--downloaded-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        readback = json.loads(args.readback.read_text(encoding="utf-8"))
        validate_release_readback(
            release=readback,
            expected_tag=args.tag,
            expected_source_sha=args.source_sha,
            notes=args.notes.read_text(encoding="utf-8"),
            payload_dir=args.payload_dir,
            sidecars_dir=args.sidecars_dir,
            downloaded_dir=args.downloaded_dir,
        )
    except (OSError, json.JSONDecodeError, ReleaseReadbackError) as exc:
        print(f"release readback rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
