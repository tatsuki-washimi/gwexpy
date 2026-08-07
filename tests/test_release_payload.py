"""Fail-closed contracts for the uploadable release payload."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "validate_release_payload.py"
)


def load_payload_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_release_payload", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_payload(tmp_path: Path) -> tuple[Path, Path]:
    payload = tmp_path / "release-payload"
    payload.mkdir()
    wheel = payload / "gwexpy-0.1.13-py3-none-any.whl"
    sdist = payload / "gwexpy-0.1.13.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    manifest = tmp_path / "distribution-sha256.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gwexpy-v0113-release-payload-v1",
                "source_sha": "a" * 40,
                "version": "0.1.13",
                "files": {
                    "wheel": {
                        "name": wheel.name,
                        "sha256": hashlib.sha256(b"wheel").hexdigest(),
                    },
                    "sdist": {
                        "name": sdist.name,
                        "sha256": hashlib.sha256(b"sdist").hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return payload, manifest


def test_payload_accepts_exactly_one_wheel_and_one_sdist_named_by_manifest(
    tmp_path: Path,
):
    validator = load_payload_validator()
    payload, manifest = write_payload(tmp_path)

    result = validator.validate_payload(payload, manifest, "0.1.13")

    assert result.wheel.name.endswith(".whl")
    assert result.sdist.name.endswith(".tar.gz")


@pytest.mark.parametrize("kind", ["missing", "extra", "symlink", "hash"])
def test_payload_rejects_invalid_files_fail_closed(tmp_path: Path, kind: str):
    validator = load_payload_validator()
    payload, manifest = write_payload(tmp_path)
    if kind == "missing":
        (payload / "gwexpy-0.1.13.tar.gz").unlink()
    elif kind == "extra":
        (payload / "LICENSE.sha256").write_text("sidecar", encoding="utf-8")
    elif kind == "symlink":
        (payload / "gwexpy-0.1.13.tar.gz").unlink()
        (payload / "gwexpy-0.1.13.tar.gz").symlink_to("/dev/null")
    else:
        (payload / "gwexpy-0.1.13.tar.gz").write_bytes(b"substituted")

    with pytest.raises(validator.ReleasePayloadError):
        validator.validate_payload(payload, manifest, "0.1.13")


def test_payload_rejects_manifest_path_traversal_and_unknown_keys(tmp_path: Path):
    validator = load_payload_validator()
    payload, manifest = write_payload(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"]["wheel"]["name"] = "../gwexpy-0.1.13-py3-none-any.whl"
    data["unexpected"] = True
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleasePayloadError):
        validator.validate_payload(payload, manifest, "0.1.13")


def test_payload_rejects_a_distribution_name_for_another_project(tmp_path: Path):
    validator = load_payload_validator()
    payload, manifest = write_payload(tmp_path)
    wheel = payload / "gwexpy-0.1.13-py3-none-any.whl"
    wrong_name = payload / "otherproject-0.1.13-py3-none-any.whl"
    wheel.rename(wrong_name)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"]["wheel"]["name"] = wrong_name.name
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(validator.ReleasePayloadError):
        validator.validate_payload(payload, manifest, "0.1.13")


def test_payload_rejects_duplicate_json_keys(tmp_path: Path):
    validator = load_payload_validator()
    payload, manifest = write_payload(tmp_path)
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            '"version": "0.1.13",', '"version": "0.1.13", "version": "0.1.13",'
        ),
        encoding="utf-8",
    )

    with pytest.raises(validator.ReleasePayloadError):
        validator.validate_payload(payload, manifest, "0.1.13")
