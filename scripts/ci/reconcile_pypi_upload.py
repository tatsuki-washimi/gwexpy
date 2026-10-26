#!/usr/bin/env python3
"""Select only missing, exact candidate files for a Trusted Publishing retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

SHA256 = re.compile(r"^[0-9a-f]{64}$")
PROJECT = "gwexpy"


class PyPIReconciliationError(ValueError):
    """Raised when PyPI readback cannot safely continue an exact candidate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PyPIReconciliationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_files(directory: Path) -> dict[str, Path]:
    _require(
        directory.is_dir() and not directory.is_symlink(),
        "candidate payload is not a regular directory",
    )
    result = {}
    for path in directory.iterdir():
        _require(
            stat.S_ISREG(path.lstat().st_mode) and path.name not in result,
            "candidate payload contains a non-regular or duplicate file",
        )
        result[path.name] = path
    return result


def _candidate_files(
    version: str, manifest: dict[str, Any], payload_dir: Path
) -> dict[str, tuple[Path, str]]:
    _require(manifest.get("version") == version, "manifest version mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PyPIReconciliationError("manifest artifacts are missing")
    matches = [
        record
        for record in artifacts
        if isinstance(record, dict) and record.get("role") == "payload"
    ]
    _require(len(matches) == 1, "manifest payload record is missing or ambiguous")
    records = matches[0].get("files")
    if not isinstance(records, list) or len(records) != 2:
        raise PyPIReconciliationError("manifest payload is incomplete")

    expected = {}
    for record in records:
        _require(
            isinstance(record, dict)
            and set(record) == {"name", "size_in_bytes", "sha256"},
            "manifest payload file record is malformed",
        )
        name = record["name"]
        _require(
            isinstance(name, str)
            and name.startswith(f"{PROJECT}-{version}")
            and "/" not in name
            and "\\" not in name
            and "\n" not in name
            and "\r" not in name
            and name not in expected,
            "manifest payload filename is invalid or duplicated",
        )
        _require(
            isinstance(record["size_in_bytes"], int)
            and not isinstance(record["size_in_bytes"], bool)
            and record["size_in_bytes"] > 0
            and isinstance(record["sha256"], str)
            and SHA256.fullmatch(record["sha256"]) is not None,
            "manifest payload size or SHA-256 is malformed",
        )
        expected[name] = (record["size_in_bytes"], record["sha256"])

    names = list(expected)
    _require(
        sum(name.endswith(".whl") for name in names) == 1
        and sum(name.endswith(".tar.gz") for name in names) == 1,
        "manifest must bind one wheel and one source distribution",
    )
    files = _regular_files(payload_dir)
    _require(set(files) == set(expected), "candidate payload differs from its manifest")
    for name, (size, digest) in expected.items():
        path = files[name]
        _require(
            path.stat().st_size == size, f"candidate payload size mismatch: {name}"
        )
        _require(_sha256(path) == digest, f"candidate payload SHA-256 mismatch: {name}")
    return {name: (files[name], digest) for name, (_, digest) in expected.items()}


def _observed_pypi_files(metadata: dict[str, Any], version: str) -> dict[str, str]:
    info = metadata.get("info")
    urls = metadata.get("urls")
    if not isinstance(info, dict):
        raise PyPIReconciliationError("PyPI release metadata has no info object")
    project = info.get("name")
    _require(
        isinstance(project, str) and project.lower() == PROJECT,
        "PyPI project identity mismatch",
    )
    _require(info.get("version") == version, "PyPI version identity mismatch")
    if not isinstance(urls, list):
        raise PyPIReconciliationError("PyPI release metadata has no file list")
    observed = {}
    for record in urls:
        _require(isinstance(record, dict), "PyPI release file record is malformed")
        name = record.get("filename")
        digests = record.get("digests")
        _require(
            isinstance(name, str) and isinstance(digests, dict),
            "PyPI release file identity is malformed",
        )
        digest = digests.get("sha256")
        _require(
            isinstance(digest, str) and SHA256.fullmatch(digest) is not None,
            f"PyPI file has no valid SHA-256 digest: {name}",
        )
        _require(name not in observed, f"PyPI reports a duplicate filename: {name}")
        observed[name] = digest
    return observed


def _fetch_metadata(version: str) -> Optional[dict[str, Any]]:
    escaped_version = urllib.parse.quote(version, safe="")
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{PROJECT}/{escaped_version}/json",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise PyPIReconciliationError(
            f"PyPI release metadata request failed with HTTP {exc.code}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PyPIReconciliationError("PyPI release metadata request failed") from exc
    _require(isinstance(data, dict), "PyPI release metadata is malformed")
    return data


def reconcile(
    *,
    version: str,
    manifest: dict[str, Any],
    payload_dir: Path,
    metadata: Optional[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    """Copy only missing original candidate files after strict PyPI SHA checks."""
    candidate = _candidate_files(version, manifest, payload_dir)
    expected = {name: digest for name, (_, digest) in candidate.items()}
    observed = {} if metadata is None else _observed_pypi_files(metadata, version)
    _require(
        set(observed).issubset(expected),
        "PyPI has unexpected files for this candidate version",
    )
    for name, digest in observed.items():
        _require(
            digest == expected[name],
            f"PyPI SHA-256 conflicts with the candidate artifact: {name}",
        )
    missing = sorted(set(expected) - set(observed))
    already_published = sorted(observed)

    if output_dir.exists():
        _require(
            output_dir.is_dir()
            and not output_dir.is_symlink()
            and not any(output_dir.iterdir()),
            "PyPI recovery output directory must be empty",
        )
    else:
        output_dir.mkdir(parents=True)
    for name in missing:
        shutil.copyfile(candidate[name][0], output_dir / name)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as stream:
            stream.write(f"publish_required={str(bool(missing)).lower()}\n")
            stream.write(
                f"partial_upload={str(bool(already_published and missing)).lower()}\n"
            )
    return missing, already_published


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Use supplied PyPI JSON metadata for offline verification",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.metadata_json is None:
            metadata = _fetch_metadata(args.version)
        else:
            metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        _require(isinstance(manifest, dict), "promotion manifest is malformed")
        _require(
            metadata is None or isinstance(metadata, dict),
            "PyPI metadata is malformed",
        )
        missing, already_published = reconcile(
            version=args.version,
            manifest=manifest,
            payload_dir=args.payload_dir,
            metadata=metadata,
            output_dir=args.output_dir,
        )
    except (OSError, json.JSONDecodeError, PyPIReconciliationError) as exc:
        print(f"PyPI recovery rejected: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"already_published": already_published, "missing": missing},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
