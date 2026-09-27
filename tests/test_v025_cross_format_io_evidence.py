"""Release I/O evidence must bind eight environment cells to one payload."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ci/v025_cross_format_io_evidence.py"
SOURCE_SHA = "a" * 40


def load_module():
    spec = importlib.util.spec_from_file_location("v025_io_evidence", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_optional_node_list_covers_cross_format_physics_regressions() -> None:
    evidence = load_module()
    nodes = "\n".join(evidence.OPTIONAL_TEST_NODES)
    assert "test_public_legacy_read_rejects_irregular_time" in nodes
    assert "test_public_matrix_read_handles_heterogeneous_numeric_cell_dtypes" in nodes
    assert "test_public_zarr_reads_preserve_int64_and_complex_values" in nodes
    assert "test_public_matrix_read_preserves_units_for_native_float_channels" in nodes
    assert (
        "test_public_tdms_readers_reject_missing_or_invalid_waveform_increment" in nodes
    )


def payload(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    artifacts = {
        "wheel": tmp_path / "gwexpy-0.2.5-py3-none-any.whl",
        "sdist": tmp_path / "gwexpy-0.2.5.tar.gz",
    }
    for kind, path in artifacts.items():
        path.write_bytes(kind.encode())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gwexpy-v025-release-payload-v1",
                "version": "0.2.5",
                "source_sha": SOURCE_SHA,
                "files": {
                    kind: {
                        "name": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for kind, path in artifacts.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, artifacts


def test_eight_reports_validate_one_payload_and_backend_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = load_module()
    manifest, artifacts = payload(tmp_path)
    monkeypatch.setattr(evidence, "_installed", lambda _path, _digest: True)
    monkeypatch.setattr(evidence, "_junit", lambda _path: 4)
    reports = tmp_path / "reports"
    for cell in evidence.CELLS:
        present = cell.startswith("optional-")
        monkeypatch.setattr(
            evidence,
            "backend_presence",
            lambda present=present: dict.fromkeys(evidence.BACKENDS, present),
        )
        kind = cell.rsplit("-", 1)[1]
        evidence.record(
            cell,
            SOURCE_SHA,
            manifest,
            artifacts[kind],
            tmp_path / "pytest.xml",
            reports / cell / "cross-format-io.json",
        )
    result = evidence.aggregate(
        SOURCE_SHA, manifest, reports, tmp_path / "aggregate.json"
    )
    assert len(result["cells"]) == 8
    assert {cell["cell"] for cell in result["cells"]} == set(evidence.CELLS)
    assert all(
        cell["backend_presence"]
        == dict.fromkeys(evidence.BACKENDS, cell["cell"].startswith("optional-"))
        for cell in result["cells"]
    )

    wrong = reports / "base-3.11-wheel/cross-format-io.json"
    data = json.loads(wrong.read_text(encoding="utf-8"))
    data["backend_presence"]["zarr"] = True
    wrong.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(evidence.CrossFormatEvidenceError, match="mismatched"):
        evidence.aggregate(SOURCE_SHA, manifest, reports, tmp_path / "rejected.json")


def test_record_rejects_substituted_artifact_and_wrong_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = load_module()
    manifest, artifacts = payload(tmp_path)
    monkeypatch.setattr(evidence, "_installed", lambda _path, _digest: True)
    monkeypatch.setattr(evidence, "_junit", lambda _path: 4)
    wheel = artifacts["wheel"]
    wheel.write_bytes(b"substituted")
    with pytest.raises(evidence.CrossFormatEvidenceError, match="digest"):
        evidence.record(
            "base-3.11-wheel",
            SOURCE_SHA,
            manifest,
            wheel,
            tmp_path / "pytest.xml",
            tmp_path / "report.json",
        )
    wheel.write_bytes(b"wheel")
    monkeypatch.setattr(
        evidence,
        "backend_presence",
        lambda: {"zarr": True, "xarray": False, "netCDF4": False},
    )
    with pytest.raises(evidence.CrossFormatEvidenceError, match="presence"):
        evidence.record(
            "base-3.11-wheel",
            SOURCE_SHA,
            manifest,
            wheel,
            tmp_path / "pytest.xml",
            tmp_path / "report.json",
        )
