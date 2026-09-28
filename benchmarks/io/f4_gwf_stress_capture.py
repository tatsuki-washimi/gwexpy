"""Capture the old-R/B0 256-frame GWF public and retained-part contract.

Run one arm and one route per fresh process. This is a correctness/structure
probe: elapsed time from this script is never performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(result: Any) -> dict[str, Any]:
    channels = {}
    for key, series in result.items():
        values = np.ascontiguousarray(series.value)
        channels[str(key)] = {
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "unit": str(series.unit),
            "t0_gps": float(series.t0.value),
            "dt_s": float(series.dt.value),
            "span_gps": [float(value) for value in series.span],
            "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "first_values": values[:4].tolist(),
            "last_values": values[-4:].tolist(),
        }
    return {"channel_order": [str(key) for key in result], "channels": channels}


class PartSnapshot:
    """Observe unique parent part objects, ignoring list/dict aliases.

    Old R retains all worker payloads in ``completed_parts`` and creates new
    GWexpy part objects in ``parts``. Trace the actual reader frame after
    each line, not allocator traffic. Workers have shut down before ``parts``
    is built; the concurrent worker bound is audited from the executor path.
    """

    def __init__(self) -> None:
        self.peak = 0
        self.snapshots: list[dict[str, int]] = []

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Trace only the installed GWF merge reader's local part references."""
        if event != "call" or frame.f_code.co_name != "_read_gwf_dict":
            return None
        if not frame.f_code.co_filename.endswith("gwexpy/timeseries/_gwf_io.py"):
            return None

        def trace_reader(current: Any, action: str, unused: Any) -> Any:
            if action != "line":
                return trace_reader
            local = current.f_locals
            completed = local.get("completed_parts")
            parts = local.get("parts")
            ordered = local.get("ordered_parts")
            groups = (
                tuple(completed.values()) if isinstance(completed, dict) else (),
                tuple(parts) if isinstance(parts, list) else (),
                tuple(ordered) if isinstance(ordered, list) else (),
            )
            ids = {id(part) for group in groups for part in group}
            count = len(ids)
            if count > self.peak:
                self.peak = count
                self.snapshots.append(
                    {
                        "line": current.f_lineno,
                        "completed_parts": len(groups[0]),
                        "parts": len(groups[1]),
                        "ordered_aliases": len(groups[2]),
                        "unique_parent_part_objects": count,
                    }
                )
            return trace_reader

        return trace_reader


def capture(root: Path, *, route: str, mode: str) -> dict[str, Any]:
    """Verify fixtures and capture one public read in a fresh wheel process."""
    import gwexpy
    from gwexpy.timeseries import TimeSeriesDict

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "gwexpy-v025-b-f4-stress-fixtures-v1":
        raise ValueError("Unexpected F4 stress fixture schema")
    if manifest["parts"] != 256 or manifest["samples_per_part"] != 8192:
        raise ValueError("F4 stress workload changed")
    files = [root / item["name"] for item in manifest["files"]]
    if len(files) != 256 or [path.name for path in files] != manifest["source_order"]:
        raise ValueError("F4 source list changed")
    for path, item in zip(files, manifest["files"], strict=True):
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"F4 fixture changed: {path.name}")

    source = Path(gwexpy.__file__).resolve()
    if not source.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"GWexpy imported outside benchmark prefix: {source}")
    probe = PartSnapshot() if mode == "structure" else None
    if probe is not None:
        sys.settrace(probe)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = TimeSeriesDict.read(
                    files,
                    channels=[manifest["channel"]],
                    format="gwf",
                    parallel=(False if route == "serial" else 2),
                )
            except Exception as exc:
                outcome = {
                    "kind": "error",
                    "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                }
            else:
                outcome = {"kind": "normal-return", "fingerprint": _fingerprint(result)}
    finally:
        sys.settrace(None)

    fixture_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "gwexpy-v025-b-f4-stress-capture-v1",
        "mode": mode,
        "route": route,
        "parallel": False if route == "serial" else 2,
        "fixture_manifest_sha256": fixture_digest,
        "helper_sha256": _sha256(Path(__file__).resolve()),
        "runtime": {
            "executable": str(Path(sys.executable).resolve()),
            "prefix": str(Path(sys.prefix).resolve()),
            "gwexpy_path": str(source),
            "gwexpy_version": importlib.metadata.version("gwexpy"),
            "gwpy_version": importlib.metadata.version("gwpy"),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "outcome": outcome,
        "warnings": [
            {
                "category": f"{item.category.__module__}.{item.category.__qualname__}",
                "message": str(item.message),
            }
            for item in caught
        ],
        "parent_part_snapshots": [] if probe is None else probe.snapshots,
        "parent_unique_part_peak": None if probe is None else probe.peak,
        "concurrency_audit": (
            "Old R waits for executor shutdown before building parts from "
            "completed_parts. The parent 512-object peak therefore occurs "
            "after workers exit. Before shutdown the parent retains at most "
            "256 completed parts and two workers each execute one part task; "
            "the earlier concurrent count is at most 258. Aliases in "
            "ordered_parts do not add objects. This source bound is not a "
            "process-memory or PSS measurement."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--route", choices=("serial", "parallel"), required=True)
    parser.add_argument("--mode", choices=("correctness", "structure"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            capture(args.root, route=args.route, mode=args.mode),
            indent=2,
            sort_keys=True,
        )
    )
