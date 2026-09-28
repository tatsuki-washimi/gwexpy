"""Reproducible one-sample GWF boundary fault fixtures and public capture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import warnings
from pathlib import Path

import numpy as np

CHANNEL = "K1:V025-F4-BOUNDARY"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(root: Path) -> None:
    from gwpy.timeseries import TimeSeries, TimeSeriesDict

    root.mkdir(parents=True, exist_ok=False)
    for name, shift in (("overlap", -0.125), ("gap", 0.125)):
        case = root / name
        case.mkdir()
        files = []
        for index in range(16):
            path = case / f"part_{index:03d}.gwf"
            series = TimeSeries(
                np.arange(index * 8, (index + 1) * 8, dtype=np.float64),
                sample_rate=8,
                t0=1_000_000_000 + index + (shift if index == 8 else 0),
                unit="m",
                channel=CHANNEL,
                name=CHANNEL,
            )
            TimeSeriesDict({CHANNEL: series}).write(path, format="gwf")
            files.append({"name": path.name, "sha256": sha256(path)})
        (case / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "gwexpy-v025-f4-one-sample-boundary-v1",
                    "channel": CHANNEL,
                    "shift_seconds": shift,
                    "files": files,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def capture(case: Path) -> dict:
    from gwexpy.timeseries import TimeSeriesDict

    manifest = json.loads((case / "manifest.json").read_text())
    assert manifest["schema"] == "gwexpy-v025-f4-one-sample-boundary-v1"
    paths = [case / item["name"] for item in manifest["files"]]
    for path, item in zip(paths, manifest["files"], strict=True):
        assert sha256(path) == item["sha256"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = TimeSeriesDict.read(
                paths, [manifest["channel"]], format="gwf", parallel=False
            )
        except Exception as error:
            outcome = {
                "kind": "error",
                "type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
            }
        else:
            series = result[manifest["channel"]]
            values = np.ascontiguousarray(series.value)
            outcome = {
                "kind": "return",
                "channel_order": list(result),
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "unit": str(series.unit),
                "t0": float(series.t0.value),
                "dt": float(series.dt.value),
                "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            }
    return {
        "case": case.name,
        "fixture_manifest_sha256": sha256(case / "manifest.json"),
        "outcome": outcome,
        "warnings": [
            {
                "category": f"{item.category.__module__}.{item.category.__qualname__}",
                "message": str(item.message),
            }
            for item in caught
        ],
        "python": sys.version.split()[0],
        "gwexpy_version": importlib.metadata.version("gwexpy"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "capture"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.path)
    else:
        print(json.dumps(capture(args.path), indent=2, sort_keys=True))
