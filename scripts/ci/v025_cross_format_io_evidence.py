#!/usr/bin/env python3
"""Record and aggregate eight candidate-bound cross-format I/O cells."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

VERSION = "0.2.5"
CELL_SCHEMA = "gwexpy-v025-cross-format-io-cell-v1"
AGGREGATE_SCHEMA = "gwexpy-v025-cross-format-io-evidence-v1"
PAYLOAD_SCHEMA = "gwexpy-v025-release-payload-v1"
BACKENDS = ("zarr", "xarray", "netCDF4")
CELLS = tuple(
    f"{mode}-{python}-{kind}"
    for mode in ("base", "optional")
    for python in ("3.11", "3.12")
    for kind in ("wheel", "sdist")
)
BASE_TEST_NODES = (
    "io/test_optional_deps.py::TestZarrImportGuard::test_public_auto_zarr_reads_preserve_missing_backend_importerror",
    "io/test_optional_deps.py::TestZarrImportGuard::test_public_matrix_auto_zarr_write_preserves_missing_backend_importerror",
    "io/test_hdf5_manifest_collection_integrity.py::test_timeseriesdict_auto_read_rejects_unreadable_manifest_entry",
    "io/test_gbd_gl500_header_validation.py::test_valid_gl500_public_readers_keep_timing_scaling_and_digital_values",
)
OPTIONAL_TEST_NODES = (
    "io/test_netcdf4_matrix_validation.py::test_public_legacy_read_rejects_irregular_time",
    "io/test_netcdf4_matrix_validation.py::test_public_matrix_roundtrip_preserves_unit",
    "io/test_netcdf4_matrix_validation.py::test_public_matrix_read_handles_heterogeneous_numeric_cell_dtypes",
    "io/test_zarr_reader.py::test_public_zarr_reads_preserve_int64_and_complex_values",
    "io/test_zarr_reader.py::test_public_matrix_read_preserves_multichannel_native_int64",
    "io/test_zarr_reader.py::test_public_matrix_read_preserves_units_for_native_float_channels",
    "io/test_zarr_reader.py::test_public_matrix_read_rejects_lossy_mixed_dtype_zarr_stores",
    "io/test_tdms_invalid_increment_contract.py::test_public_tdms_readers_reject_missing_or_invalid_waveform_increment",
)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CrossFormatEvidenceError(ValueError):
    """Raised when cross-format release evidence is incomplete or inconsistent."""


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CrossFormatEvidenceError("duplicate JSON key")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise CrossFormatEvidenceError("evidence input must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CrossFormatEvidenceError("invalid evidence JSON") from exc
    if not isinstance(value, dict):
        raise CrossFormatEvidenceError("evidence must be an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CrossFormatEvidenceError("evidence output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _payload(path: Path, source_sha: str) -> dict[str, dict[str, str]]:
    value = _load(path)
    if (
        set(value) != {"schema", "source_sha", "version", "files"}
        or value["schema"] != PAYLOAD_SCHEMA
        or value["source_sha"] != source_sha
        or value["version"] != VERSION
    ):
        raise CrossFormatEvidenceError("payload manifest is not bound to candidate")
    files = value["files"]
    if not isinstance(files, dict) or set(files) != {"wheel", "sdist"}:
        raise CrossFormatEvidenceError("payload must contain wheel and sdist")
    for kind, entry in files.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"name", "sha256"}
            or not isinstance(entry["name"], str)
            or Path(entry["name"]).name != entry["name"]
            or not isinstance(entry["sha256"], str)
            or SHA256.fullmatch(entry["sha256"]) is None
        ):
            raise CrossFormatEvidenceError(f"invalid {kind} payload entry")
        if (
            kind == "wheel"
            and re.fullmatch(r"gwexpy-0\.2\.5-[^-]+-[^-]+-[^-]+\.whl", entry["name"])
            is None
        ):
            raise CrossFormatEvidenceError("wrong wheel filename")
        if kind == "sdist" and entry["name"] != "gwexpy-0.2.5.tar.gz":
            raise CrossFormatEvidenceError("wrong sdist filename")
    return files


def backend_presence() -> dict[str, bool]:
    """Report actual optional-backend importability in the cell interpreter."""
    return {name: importlib.util.find_spec(name) is not None for name in BACKENDS}


def _installed(artifact: Path, digest: str) -> bool:
    import gwexpy

    if gwexpy.__version__ != VERSION or importlib.metadata.version("gwexpy") != VERSION:
        raise CrossFormatEvidenceError("installed version mismatch")
    if not any(
        part in {"site-packages", "dist-packages"}
        for part in Path(gwexpy.__file__).resolve().parts
    ):
        raise CrossFormatEvidenceError("gwexpy did not import from installed package")
    raw = importlib.metadata.distribution("gwexpy").read_text("direct_url.json")
    data = json.loads(raw) if raw else None
    if not isinstance(data, dict) or urlparse(data.get("url", "")).scheme != "file":
        raise CrossFormatEvidenceError(
            "installed package has no local artifact provenance"
        )
    if Path(unquote(urlparse(data["url"]).path)).resolve() != artifact.resolve():
        raise CrossFormatEvidenceError("installed package came from another artifact")
    info = data.get("archive_info", {})
    hashes = {info.get("hash"), f"sha256={info.get('hashes', {}).get('sha256')}"}
    if f"sha256={digest}" not in hashes:
        raise CrossFormatEvidenceError("installed package digest mismatch")
    return True


def _junit(path: Path) -> int:
    spec = importlib.util.spec_from_file_location(
        "qualification_evidence_for_cross_format",
        Path(__file__).with_name("qualification_evidence.py"),
    )
    if spec is None or spec.loader is None:
        raise CrossFormatEvidenceError("JUnit parser unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        count, skips = module._parse_junit(path)
    except module.QualificationEvidenceError as exc:
        raise CrossFormatEvidenceError(str(exc)) from exc
    if count <= 0 or skips:
        raise CrossFormatEvidenceError("I/O cell has no tests or unexpected skips")
    return count


def record(
    cell: str,
    source_sha: str,
    manifest: Path,
    artifact: Path,
    junit: Path,
    output: Path,
) -> dict[str, Any]:
    if cell not in CELLS or SHA40.fullmatch(source_sha) is None:
        raise CrossFormatEvidenceError("unknown cell or source SHA")
    files = _payload(manifest, source_sha)
    kind = cell.rsplit("-", 1)[1]
    entry = files[kind]
    if (
        artifact.is_symlink()
        or not artifact.is_file()
        or artifact.name != entry["name"]
    ):
        raise CrossFormatEvidenceError("wrong candidate artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise CrossFormatEvidenceError("candidate artifact digest mismatch")
    present = backend_presence()
    expected = cell.startswith("optional-")
    if present != dict.fromkeys(BACKENDS, expected):
        raise CrossFormatEvidenceError("optional backend presence mismatch")
    _installed(artifact, digest)
    report = {
        "schema": CELL_SCHEMA,
        "source_sha": source_sha,
        "version": VERSION,
        "cell": cell,
        "artifact": {"kind": kind, "filename": artifact.name, "sha256": digest},
        "backend_presence": present,
        "candidate_installed_from_payload": True,
        "test_status": "passed",
        "testcase_count": _junit(junit),
        "observed_skips": [],
    }
    _write(output, report)
    return report


def aggregate(
    source_sha: str, manifest: Path, reports_dir: Path, output: Path
) -> dict[str, Any]:
    if SHA40.fullmatch(source_sha) is None:
        raise CrossFormatEvidenceError("invalid source SHA")
    files = _payload(manifest, source_sha)
    if reports_dir.is_symlink() or not reports_dir.is_dir():
        raise CrossFormatEvidenceError("invalid reports directory")
    paths = sorted(reports_dir.rglob("*.json"))
    if len(paths) != len(CELLS) or any(
        path.name != "cross-format-io.json" for path in paths
    ):
        raise CrossFormatEvidenceError("expected exactly eight I/O cell reports")
    seen: set[str] = set()
    reports = []
    for path in paths:
        report = _load(path)
        cell = report.get("cell")
        if not isinstance(cell, str) or cell not in CELLS or cell in seen:
            raise CrossFormatEvidenceError("unknown or duplicate I/O cell")
        kind = cell.rsplit("-", 1)[1]
        expected = cell.startswith("optional-")
        if (
            set(report)
            != {
                "schema",
                "source_sha",
                "version",
                "cell",
                "artifact",
                "backend_presence",
                "candidate_installed_from_payload",
                "test_status",
                "testcase_count",
                "observed_skips",
            }
            or report["schema"] != CELL_SCHEMA
            or report["source_sha"] != source_sha
            or report["version"] != VERSION
            or report["artifact"]
            != {
                "kind": kind,
                "filename": files[kind]["name"],
                "sha256": files[kind]["sha256"],
            }
            or report["backend_presence"] != dict.fromkeys(BACKENDS, expected)
            or report["candidate_installed_from_payload"] is not True
            or report["test_status"] != "passed"
            or not isinstance(report["testcase_count"], int)
            or isinstance(report["testcase_count"], bool)
            or report["testcase_count"] <= 0
            or report["observed_skips"] != []
        ):
            raise CrossFormatEvidenceError(
                "I/O report has mismatched candidate, backend, or test facts"
            )
        seen.add(cell)
        reports.append(report)
    if seen != set(CELLS):
        raise CrossFormatEvidenceError("missing I/O cells")
    result = {
        "schema": AGGREGATE_SCHEMA,
        "source_sha": source_sha,
        "version": VERSION,
        "files": files,
        "cells": sorted(reports, key=lambda item: item["cell"]),
    }
    _write(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    nodes = commands.add_parser("test-nodes")
    nodes.add_argument("--mode", required=True, choices=("base", "optional"))
    rec = commands.add_parser("record")
    rec.add_argument("--cell", required=True)
    rec.add_argument("--source-sha", required=True)
    rec.add_argument("--payload-manifest", type=Path, required=True)
    rec.add_argument("--artifact", type=Path, required=True)
    rec.add_argument("--junit", type=Path, required=True)
    rec.add_argument("--report", type=Path, required=True)
    agg = commands.add_parser("aggregate")
    agg.add_argument("--source-sha", required=True)
    agg.add_argument("--payload-manifest", type=Path, required=True)
    agg.add_argument("--reports-dir", type=Path, required=True)
    agg.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "test-nodes":
            print(
                *(BASE_TEST_NODES if args.mode == "base" else OPTIONAL_TEST_NODES),
                sep="\n",
            )
        elif args.command == "record":
            record(
                args.cell,
                args.source_sha,
                args.payload_manifest,
                args.artifact,
                args.junit,
                args.report,
            )
        else:
            aggregate(
                args.source_sha, args.payload_manifest, args.reports_dir, args.output
            )
    except CrossFormatEvidenceError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
