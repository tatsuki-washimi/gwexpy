"""Deterministic, package-independent fixtures for the v0.2.5 I/O campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from .format_fixtures import make_sdb_fixtures, make_tdms_fixtures
except ImportError:  # Direct execution under python -I.
    from format_fixtures import make_sdb_fixtures, make_tdms_fixtures


def sha256(path: Path) -> str:
    """Hash a fixture or manifest as stored on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_fixtures(
    destination: Path, *, large_rows: int = 65536, include_formats: bool = False
) -> dict:
    """Create repeatable CSV fixtures without importing either candidate wheel."""
    if large_rows < 4096:
        raise ValueError("large_rows must be at least 4096")
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    path = destination / "frequency-nonuniform-4096.csv"
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for index in range(4096):
            frequency = index * 0.125 + (index % 7) * 0.000001
            value = ((index * 37) % 1009 - 504) / 128
            stream.write(f"{frequency:.9f},{value:.9f}\n")
    paths["frequency"] = path

    for key, rows, channels in (
        ("general", 8192, 8),
        ("large", large_rows, 16),
    ):
        path = destination / f"{key}-{rows}-{channels}.csv"
        with path.open("w", encoding="ascii", newline="\n") as stream:
            for index in range(rows):
                values = ",".join(
                    f"{((index * (channel + 7)) % 2003 - 1001) / 256:.9f}"
                    for channel in range(channels)
                )
                stream.write(f"{index / 4:.9f},{values}\n")
        paths[key] = path

    segment_paths = []
    for segment in range(12):
        path = destination / f"segment-{segment:02d}.csv"
        with path.open("w", encoding="ascii", newline="\n") as stream:
            for offset in range(2048):
                index = segment * 2048 + offset
                stream.write(f"{index / 4:.9f},{index % 997:.9f}\n")
        segment_paths.append(path)
    paths.update({f"segment_{i:02d}": path for i, path in enumerate(segment_paths)})

    faults = {
        "selected_invalid": "0,1,2\n1,not-a-number,3\n2,4,5\n",
        "unselected_invalid": "0,1,2\n1,3,not-a-number\n2,4,5\n",
        "timestamp_invalid": "0,1,2\nnot-a-number,3,4\n2,5,6\n",
        "timestamp_irregular": "0,1,2\n1,3,4\n3,5,6\n",
        "row_short": "0,1,2\n1,3\n2,5,6\n",
        "row_extra": "0,1,2\n1,3,4,5\n2,5,6\n",
        "early_unselected_late_selected": (
            "0,1,2\n1,3,not-a-number\n2,not-a-number,5\n"
        ),
        "early_selected_late_unselected": (
            "0,1,2\n1,not-a-number,4\n2,5,not-a-number\n"
        ),
    }
    for name, body in faults.items():
        path = destination / f"fault-{name}.csv"
        path.write_text(body, encoding="ascii", newline="\n")
        paths[f"fault_{name}"] = path

    manifest = {
        "schema": 1,
        "generator": "benchmarks/io/fixtures.py",
        "large_rows": large_rows,
        "files": {
            key: {
                "name": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in sorted(paths.items())
        },
        "csv_routes": {
            "f2_1a": "frequency",
            "f2_1b": "frequency",
            "f2_2": "general",
            "f2_3": "large",
        },
    }
    if include_formats:
        formats = {
            "sdb": make_sdb_fixtures(destination / "sdb"),
            "tdms": make_tdms_fixtures(destination / "tdms"),
        }
        manifest["formats"] = formats
        for format_name, format_manifest in formats.items():
            for case_name, case in format_manifest["cases"].items():
                manifest["files"][f"{format_name}_{case_name}"] = {
                    "name": f"{format_name}/{case['name']}",
                    "sha256": case["sha256"],
                    "bytes": case["bytes"],
                }
    manifest_path = destination / "fixtures.json"
    encoded = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("existing fixture manifest differs; use a new destination")
    manifest_path.write_text(encoded, encoding="utf-8")
    return manifest


def verify_fixtures(destination: Path) -> dict:
    """Reject missing or modified fixture bytes before any run."""
    manifest = json.loads((destination / "fixtures.json").read_text(encoding="utf-8"))
    for key, entry in manifest["files"].items():
        path = destination / entry["name"]
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"fixture changed: {key}")
    return manifest
