"""Tests for safe continuation of a partial Trusted Publishing upload."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "reconcile_pypi_upload.py"


def candidate(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    version = "99.88.77"
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    payload = {
        f"gwexpy-{version}-py3-none-any.whl": b"original wheel bytes",
        f"gwexpy-{version}.tar.gz": b"original sdist bytes",
    }
    records = []
    for name, data in payload.items():
        (payload_dir / name).write_bytes(data)
        records.append(
            {
                "name": name,
                "size_in_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = tmp_path / "promotion-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gwexpy-release-promotion-manifest-v1",
                "version": version,
                "artifacts": [{"role": "payload", "files": records}],
            }
        ),
        encoding="utf-8",
    )
    return manifest, payload_dir, payload


def invoke(
    tmp_path: Path,
    manifest: Path,
    payload_dir: Path,
    metadata: dict[str, object],
) -> subprocess.CompletedProcess[str]:
    metadata_path = tmp_path / "pypi.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    env = os.environ.copy()
    env["GITHUB_OUTPUT"] = str(tmp_path / "github-output.txt")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--version",
            "99.88.77",
            "--manifest",
            str(manifest),
            "--payload-dir",
            str(payload_dir),
            "--output-dir",
            str(tmp_path / "missing"),
            "--metadata-json",
            str(metadata_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def pypi_metadata(version: str, files: dict[str, bytes]) -> dict[str, object]:
    return {
        "info": {"name": "gwexpy", "version": version},
        "urls": [
            {
                "filename": name,
                "digests": {"sha256": hashlib.sha256(data).hexdigest()},
            }
            for name, data in files.items()
        ],
    }


def test_partial_upload_copies_only_missing_exact_candidate_bytes(tmp_path: Path):
    manifest, payload_dir, payload = candidate(tmp_path)
    wheel, sdist = list(payload)
    result = invoke(
        tmp_path,
        manifest,
        payload_dir,
        pypi_metadata("99.88.77", {wheel: payload[wheel]}),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "already_published": [wheel],
        "missing": [sdist],
    }
    missing_dir = tmp_path / "missing"
    assert [path.name for path in missing_dir.iterdir()] == [sdist]
    assert (missing_dir / sdist).read_bytes() == payload[sdist]
    assert "publish_required=true" in (tmp_path / "github-output.txt").read_text()
    assert "partial_upload=true" in (tmp_path / "github-output.txt").read_text()


def test_existing_matching_full_version_requires_no_upload(tmp_path: Path):
    manifest, payload_dir, payload = candidate(tmp_path)
    result = invoke(
        tmp_path,
        manifest,
        payload_dir,
        pypi_metadata("99.88.77", payload),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "already_published": sorted(payload),
        "missing": [],
    }
    assert list((tmp_path / "missing").iterdir()) == []
    assert "publish_required=false" in (tmp_path / "github-output.txt").read_text()
    assert "partial_upload=false" in (tmp_path / "github-output.txt").read_text()


@pytest.mark.parametrize(
    "existing",
    [
        {"gwexpy-99.88.77-py3-none-any.whl": b"conflicting bytes"},
        {"unexpected-99.88.77.tar.gz": b"unexpected file"},
    ],
)
def test_conflicting_or_unexpected_pypi_files_fail_closed(
    tmp_path: Path, existing: dict[str, bytes]
):
    manifest, payload_dir, _ = candidate(tmp_path)
    result = invoke(
        tmp_path,
        manifest,
        payload_dir,
        pypi_metadata("99.88.77", existing),
    )

    assert result.returncode != 0
    assert not (tmp_path / "missing").exists()


def test_absent_version_republishes_both_original_candidate_files(tmp_path: Path):
    manifest, payload_dir, payload = candidate(tmp_path)
    result = invoke(
        tmp_path,
        manifest,
        payload_dir,
        {"info": {"name": "gwexpy", "version": "99.88.77"}, "urls": []},
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "already_published": [],
        "missing": sorted(payload),
    }
    assert {
        path.name: path.read_bytes() for path in (tmp_path / "missing").iterdir()
    } == payload
    assert "partial_upload=false" in (tmp_path / "github-output.txt").read_text()


def test_reconciliation_rejects_manifest_or_payload_hash_mismatch(tmp_path: Path):
    manifest, payload_dir, payload = candidate(tmp_path)
    (payload_dir / next(iter(payload))).write_bytes(b"rebuilt or corrupted bytes")
    result = invoke(
        tmp_path,
        manifest,
        payload_dir,
        pypi_metadata("99.88.77", {}),
    )

    assert result.returncode != 0
    assert not (tmp_path / "missing").exists()
