"""Contracts for four-cell release smoke evidence aggregation."""

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
    / "assemble_release_evidence.py"
)
SOURCE_SHA = "a" * 40


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "assemble_release_evidence", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_evidence_inputs(
    tmp_path: Path,
    *,
    version: str = "0.1.13",
    payload_schema: str = "gwexpy-v0113-release-payload-v1",
) -> tuple[Path, Path, Path]:
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    payload_manifest = sidecars / "distribution-sha256.json"
    payload_manifest.write_text(
        json.dumps(
            {
                "schema": payload_schema,
                "source_sha": SOURCE_SHA,
                "version": version,
                "files": {
                    "wheel": {
                        "name": f"gwexpy-{version}-py3-none-any.whl",
                        "sha256": "b" * 64,
                    },
                    "sdist": {
                        "name": f"gwexpy-{version}.tar.gz",
                        "sha256": "c" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (sidecars / "LICENSE.sha256").write_text("d" * 64 + "\n", encoding="ascii")
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    for python in ("3.11", "3.12"):
        for kind, filename, digest in (
            ("wheel", f"gwexpy-{version}-py3-none-any.whl", "b" * 64),
            ("sdist", f"gwexpy-{version}.tar.gz", "c" * 64),
        ):
            (smoke / f"python-{python}-{kind}.json").write_text(
                json.dumps(
                    {
                        "source_sha": SOURCE_SHA,
                        "python": python,
                        "distribution": {
                            "kind": kind,
                            "file": filename,
                            "sha256": digest,
                        },
                        "repository_license_sha256": "d" * 64,
                        "embedded_license_sha256": "d" * 64,
                        "installed_version": version,
                        "import_ok": True,
                        "register_all_ok": True,
                        "smoke_ok": True,
                    }
                ),
                encoding="utf-8",
            )
    return payload_manifest, sidecars, smoke


@pytest.mark.parametrize(
    ("tag", "version", "payload_schema", "integration_schema", "prefix"),
    [
        (
            "v0.1.13",
            "0.1.13",
            "gwexpy-v0113-release-payload-v1",
            "gwexpy-v0113-integration-evidence-v1",
            "v0113-integration-evidence",
        ),
        (
            "v0.1.14",
            "0.1.14",
            "gwexpy-v0114-release-payload-v1",
            "gwexpy-v0114-integration-evidence-v1",
            "v0114-integration-evidence",
        ),
    ],
)
def test_collector_uses_versioned_schema_and_fixed_smoke_cells(
    tmp_path: Path,
    tag: str,
    version: str,
    payload_schema: str,
    integration_schema: str,
    prefix: str,
):
    collector = load_collector()
    payload, sidecars, smoke = write_evidence_inputs(
        tmp_path, version=version, payload_schema=payload_schema
    )

    evidence = collector.assemble_evidence(
        payload,
        sidecars,
        smoke,
        SOURCE_SHA,
        "owner/repo",
        "123",
        SOURCE_SHA,
        "refs/heads/main",
        tag,
    )

    assert evidence["schema"] == integration_schema
    assert evidence["artifact_name"] == f"{prefix}-{SOURCE_SHA}"
    assert set(evidence["smoke"]) == {
        "python-3.11-wheel",
        "python-3.11-sdist",
        "python-3.12-wheel",
        "python-3.12-sdist",
    }


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "unknown-key"])
def test_collector_rejects_noncanonical_smoke_evidence(tmp_path: Path, kind: str):
    collector = load_collector()
    payload, sidecars, smoke = write_evidence_inputs(tmp_path)
    target = smoke / "python-3.11-wheel.json"
    if kind == "missing":
        target.unlink()
    elif kind == "duplicate":
        (smoke / "python-3.11-wheel-copy.json").write_bytes(target.read_bytes())
    elif kind == "extra":
        (smoke / "notes.txt").write_text("no", encoding="utf-8")
    else:
        data = json.loads(target.read_text(encoding="utf-8"))
        data["url"] = "https://example.invalid/secret"
        target.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(collector.ReleaseEvidenceError):
        collector.assemble_evidence(
            payload,
            sidecars,
            smoke,
            SOURCE_SHA,
            "owner/repo",
            "123",
            SOURCE_SHA,
            "refs/heads/main",
            "v0.1.13",
        )


def test_collector_rejects_extra_or_symlinked_sidecar(tmp_path: Path):
    collector = load_collector()
    payload, sidecars, smoke = write_evidence_inputs(tmp_path)
    (sidecars / "unexpected.txt").write_text("no", encoding="utf-8")

    with pytest.raises(collector.ReleaseEvidenceError):
        collector.assemble_evidence(
            payload,
            sidecars,
            smoke,
            SOURCE_SHA,
            "owner/repo",
            "123",
            SOURCE_SHA,
            "refs/heads/main",
            "v0.1.13",
        )


def test_collector_rejects_malformed_expected_tag_and_duplicate_json_key(
    tmp_path: Path,
):
    collector = load_collector()
    payload, sidecars, smoke = write_evidence_inputs(tmp_path)
    target = smoke / "python-3.11-wheel.json"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            '"smoke_ok": true', '"smoke_ok": true, "smoke_ok": true'
        ),
        encoding="utf-8",
    )

    with pytest.raises(collector.ReleaseEvidenceError):
        collector.assemble_evidence(
            payload,
            sidecars,
            smoke,
            SOURCE_SHA,
            "owner/repo",
            "123",
            SOURCE_SHA,
            "refs/heads/main",
            "v-not-a-release",
        )


def test_collector_rejects_unconfigured_release_tag(tmp_path: Path):
    collector = load_collector()
    payload, sidecars, smoke = write_evidence_inputs(
        tmp_path,
        version="0.1.15",
        payload_schema="gwexpy-v0115-release-payload-v1",
    )

    with pytest.raises(collector.ReleaseEvidenceError, match="unsupported release tag"):
        collector.assemble_evidence(
            payload,
            sidecars,
            smoke,
            SOURCE_SHA,
            "owner/repo",
            "123",
            SOURCE_SHA,
            "refs/heads/main",
            "v0.1.15",
        )
