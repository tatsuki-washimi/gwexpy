#!/usr/bin/env python3
"""Validate the two-file PyPI payload without accepting sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "gwexpy-v0113-release-payload-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_SHA = re.compile(r"^[0-9a-f]{40}$")


class ReleasePayloadError(ValueError):
    """Raised when a payload is not exactly the approved distributions."""


class _DuplicateJSONKey(ValueError):
    """Raised by the JSON hook when an object has ambiguous keys."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


@dataclass(frozen=True)
class Payload:
    wheel: Path
    sdist: Path
    source_sha: str
    version: str


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleasePayloadError(f"manifest must be a regular file: {path}")
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError, _DuplicateJSONKey) as exc:
        raise ReleasePayloadError(f"invalid manifest: {path}") from exc
    if not isinstance(data, dict):
        raise ReleasePayloadError("manifest must be a JSON object")
    return data


def _require_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    if set(data) != expected:
        raise ReleasePayloadError(f"{context} has unknown or missing keys")


def _safe_filename(value: object, suffix: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ReleasePayloadError("distribution filename must be a plain basename")
    if "/" in value or "\\" in value or not value.endswith(suffix):
        raise ReleasePayloadError("distribution filename has an invalid suffix or path")
    return value


def _entry(data: object, suffix: str) -> tuple[str, str]:
    if not isinstance(data, dict):
        raise ReleasePayloadError("distribution manifest entry must be an object")
    _require_keys(data, {"name", "sha256"}, "distribution manifest entry")
    name = _safe_filename(data["name"], suffix)
    digest = data["sha256"]
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ReleasePayloadError("distribution hash must be a lowercase SHA-256")
    return name, digest


def _validate_distribution_names(wheel: str, sdist: str, version: str) -> None:
    """Require GWexpy's normalized project name and the requested version."""
    if sdist != f"gwexpy-{version}.tar.gz" or not re.fullmatch(
        rf"gwexpy-{re.escape(version)}-[^-]+-[^-]+-[^-]+\.whl", wheel
    ):
        raise ReleasePayloadError(
            "distribution names must use normalized gwexpy project and version"
        )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_entries(directory: Path) -> dict[str, Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ReleasePayloadError("payload directory must be a real directory")
    entries: dict[str, Path] = {}
    for entry in directory.iterdir():
        if entry.is_symlink() or not stat.S_ISREG(
            entry.stat(follow_symlinks=False).st_mode
        ):
            raise ReleasePayloadError(
                f"payload contains a non-regular file: {entry.name}"
            )
        entries[entry.name] = entry
    return entries


def _parse_manifest(
    path: Path, expected_version: str
) -> tuple[str, str, tuple[str, str], tuple[str, str]]:
    data = _load_object(path)
    _require_keys(
        data, {"schema", "source_sha", "version", "files"}, "payload manifest"
    )
    if data["schema"] != SCHEMA:
        raise ReleasePayloadError("unsupported payload manifest schema")
    source_sha = data["source_sha"]
    if not isinstance(source_sha, str) or not SOURCE_SHA.fullmatch(source_sha):
        raise ReleasePayloadError("payload manifest source_sha must be a full SHA")
    version = data["version"]
    if not isinstance(version, str) or version != expected_version:
        raise ReleasePayloadError(
            "payload manifest version does not match expected version"
        )
    files = data["files"]
    if not isinstance(files, dict):
        raise ReleasePayloadError("payload manifest files must be an object")
    _require_keys(files, {"wheel", "sdist"}, "payload manifest files")
    wheel = _entry(files["wheel"], ".whl")
    sdist = _entry(files["sdist"], ".tar.gz")
    _validate_distribution_names(wheel[0], sdist[0], version)
    return source_sha, version, wheel, sdist


def validate_payload(
    payload_dir: Path | str, manifest_path: Path | str, expected_version: str
) -> Payload:
    """Require exactly the manifest's one wheel and one source distribution."""
    payload = Path(payload_dir)
    source_sha, version, wheel_spec, sdist_spec = _parse_manifest(
        Path(manifest_path), expected_version
    )
    entries = _regular_entries(payload)
    expected = {wheel_spec[0], sdist_spec[0]}
    if set(entries) != expected:
        raise ReleasePayloadError(
            "payload must contain exactly the manifest wheel and sdist"
        )
    for name, digest in (wheel_spec, sdist_spec):
        if _hash(entries[name]) != digest:
            raise ReleasePayloadError(f"distribution hash mismatch: {name}")
    return Payload(entries[wheel_spec[0]], entries[sdist_spec[0]], source_sha, version)


def write_manifest(
    payload_dir: Path | str, output: Path | str, version: str, source_sha: str
) -> None:
    """Create a manifest only when a newly-built directory has exactly two files."""
    if not SOURCE_SHA.fullmatch(source_sha):
        raise ReleasePayloadError("source_sha must be a full SHA")
    entries = _regular_entries(Path(payload_dir))
    wheels = [path for name, path in entries.items() if name.endswith(".whl")]
    sdists = [path for name, path in entries.items() if name.endswith(".tar.gz")]
    if len(entries) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise ReleasePayloadError(
            "new payload must contain exactly one wheel and one sdist"
        )
    _validate_distribution_names(wheels[0].name, sdists[0].name, version)
    data = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "version": version,
        "files": {
            "wheel": {"name": wheels[0].name, "sha256": _hash(wheels[0])},
            "sdist": {"name": sdists[0].name, "sha256": _hash(sdists[0])},
        },
    }
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ReleasePayloadError("manifest output must not already exist")
    destination.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--source-sha")
    parser.add_argument("--kind", choices=("wheel", "sdist"))
    args = parser.parse_args(argv)
    try:
        if args.write_manifest:
            if args.source_sha is None:
                raise ReleasePayloadError(
                    "--source-sha is required with --write-manifest"
                )
            write_manifest(
                args.payload_dir, args.manifest, args.expected_version, args.source_sha
            )
        result = validate_payload(
            args.payload_dir, args.manifest, args.expected_version
        )
    except ReleasePayloadError as exc:
        parser.error(str(exc))
    if args.kind:
        print(getattr(result, args.kind).name)
    else:
        print(f"wheel={result.wheel.name}")
        print(f"sdist={result.sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
