#!/usr/bin/env python3
"""Measure lazy bootstrap and explicit HDF5 I/O in isolated processes.

The benchmark records evidence only.  It deliberately has no pass/fail
performance threshold; a maintainer must review the generated comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

SCENARIOS = (
    "plain_import",
    "constructor_only",
    "first_hdf5_io",
    "repeated_hdf5_io",
)

CHILD = r"""
import json
import os
import platform
import resource
import tempfile
import time
from pathlib import Path

scenario = os.environ["GWEXPY_BENCHMARK_SCENARIO"]
target_root = Path(os.environ["GWEXPY_BENCHMARK_TARGET"]).resolve()

started = time.perf_counter()
if scenario == "plain_import":
    import gwexpy
elif scenario == "constructor_only":
    import numpy as np
    import gwexpy
    from gwexpy.timeseries import TimeSeries
    TimeSeries(np.ones(4096), sample_rate=4096)
elif scenario in {"first_hdf5_io", "repeated_hdf5_io"}:
    import numpy as np
    import gwexpy
    from gwexpy.timeseries import TimeSeries
    series = TimeSeries(np.ones(4096), sample_rate=4096)
    repeats = 1 if scenario == "first_hdf5_io" else 5
    with tempfile.TemporaryDirectory() as directory:
        for index in range(repeats):
            path = Path(directory) / f"series-{index}.hdf5"
            series.write(path, format="hdf5", path="data")
            restored = TimeSeries.read(path, format="hdf5", path="data")
            assert restored.shape == series.shape
else:
    raise AssertionError(scenario)
wall_seconds = time.perf_counter() - started

from gwpy.io.registry import default_registry
from gwexpy.timeseries import TimeSeries

imported = Path(gwexpy.__file__).resolve()
if not imported.is_relative_to(target_root):
    raise RuntimeError(f"imported {imported}, expected a module below {target_root}")

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if platform.system() != "Darwin":
    rss *= 1024
print(json.dumps({
    "wall_seconds": wall_seconds,
    "peak_rss_bytes": int(rss),
    "registry_counts": {
        "read": len(default_registry.get_formats(TimeSeries, "Read")),
        "write": len(default_registry.get_formats(TimeSeries, "Write")),
    },
    "timeseries_io_loaded": "gwexpy.timeseries.io" in __import__("sys").modules,
    "imported_from": str(imported),
}, sort_keys=True))
"""


def _target(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target must be LABEL=PATH") from exc
    path = Path(raw_path).expanduser().resolve()
    if not label or not (path / "gwexpy" / "__init__.py").is_file():
        raise argparse.ArgumentTypeError(f"not a gwexpy source tree: {value}")
    return label, path


def _percentile95(values: list[float | int]) -> float | int:
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def _source_identity(path: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        check=False,
    )
    dirty_bytes = bytearray(diff.stdout)
    for relative in untracked.stdout.decode("utf-8").splitlines():
        dirty_bytes.extend(relative.encode("utf-8") + b"\0")
        dirty_bytes.extend((path / relative).read_bytes())
    return {
        "head_sha": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(dirty_bytes),
        "working_tree_digest": hashlib.sha256(dirty_bytes).hexdigest(),
    }


def _run_one(target: Path, scenario: str, cwd: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    env["GWEXPY_BENCHMARK_SCENARIO"] = scenario
    env["GWEXPY_BENCHMARK_TARGET"] = str(target)
    result = subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{target} {scenario} failed ({result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [float(run["wall_seconds"]) for run in runs]
    rss = [int(run["peak_rss_bytes"]) for run in runs]
    return {
        "runs": len(runs),
        "wall_seconds": {"median": median(walls), "p95": _percentile95(walls)},
        "peak_rss_bytes": {"median": median(rss), "p95": _percentile95(rss)},
        "registry_counts": {
            operation: sorted({run["registry_counts"][operation] for run in runs})
            for operation in ("read", "write")
        },
        "timeseries_io_loaded": sorted({run["timeseries_io_loaded"] for run in runs}),
        "raw": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", type=_target, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    targets = dict(args.target)
    if len(targets) != len(args.target):
        parser.error("target labels must be unique")

    collected = {label: {scenario: [] for scenario in SCENARIOS} for label in targets}
    with tempfile.TemporaryDirectory(prefix="gwexpy-bootstrap-benchmark-") as tmp:
        cwd = Path(tmp)
        for repetition in range(args.repetitions):
            labels = list(targets)
            if repetition % 2:
                labels.reverse()
            for label in labels:
                for scenario in SCENARIOS:
                    collected[label][scenario].append(
                        _run_one(targets[label], scenario, cwd)
                    )

    report = {
        "schema": "gwexpy-bootstrap-io-benchmark-v1",
        "environment": {
            "executable": sys.executable,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "repetitions": args.repetitions,
        },
        "targets": {
            label: {
                "source_root": str(path),
                "source_identity": _source_identity(path),
                "scenarios": {
                    scenario: _summarize(collected[label][scenario])
                    for scenario in SCENARIOS
                },
            }
            for label, path in targets.items()
        },
        "decision": {"status": "requires-maintainer-review"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
