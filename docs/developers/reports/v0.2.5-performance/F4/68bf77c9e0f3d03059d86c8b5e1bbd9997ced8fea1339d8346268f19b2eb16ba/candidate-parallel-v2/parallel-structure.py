"""Diagnostic structural spy for the candidate parallel GWF route."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import weakref
from collections import deque
from pathlib import Path

import numpy as np

import gwexpy.timeseries._gwf_io as io
from gwexpy.timeseries import TimeSeriesDict


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(root: Path, count: int) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "gwexpy-v025-b-f4-stress-fixtures-v1"
    paths = [root / name for name in manifest["source_order"][:count]]
    expected = {item["name"]: item["sha256"] for item in manifest["files"]}
    for path in paths:
        assert sha(path) == expected[path.name]

    live = peak_live = created = peak_pending = reached = succeeded = 0
    original_coerce = io._coerce_gwf_timeseriesdict
    original_fast = io._try_bounded_gwf_parallel

    def counted_coerce(*args, **kwargs):
        nonlocal live, peak_live, created
        result = original_coerce(*args, **kwargs)
        live += 1
        created += 1
        peak_live = max(peak_live, live)

        def release() -> None:
            nonlocal live
            live -= 1

        weakref.finalize(result, release)
        return result

    def counted_fast(*args, **kwargs):
        nonlocal reached, succeeded
        reached += 1
        result = original_fast(*args, **kwargs)
        if result is not io._GWF_BOUNDED_FALLBACK:
            succeeded += 1
        return result

    def trace(frame, event, arg):
        nonlocal peak_pending
        if event != "call" or frame.f_code.co_name != "checked_parts":
            return None
        if not frame.f_code.co_filename.endswith("gwexpy/timeseries/_gwf_io.py"):
            return None

        def trace_parts(current, action, unused):
            nonlocal peak_pending
            if action == "line":
                pending = current.f_locals.get("pending")
                if isinstance(pending, deque):
                    peak_pending = max(peak_pending, len(pending))
            return trace_parts

        return trace_parts

    io._coerce_gwf_timeseriesdict = counted_coerce
    io._try_bounded_gwf_parallel = counted_fast
    sys.settrace(trace)
    try:
        result = TimeSeriesDict.read(
            paths, [manifest["channel"]], format="gwf", parallel=2
        )
    finally:
        sys.settrace(None)
        io._coerce_gwf_timeseriesdict = original_coerce
        io._try_bounded_gwf_parallel = original_fast
    values = np.ascontiguousarray(result[manifest["channel"]].value)
    return {
        "count": count,
        "fast_route_reached": reached,
        "fast_route_succeeded": succeeded,
        "peak_parent_live_coerced_parts": peak_live,
        "created_parent_coerced_parts": created,
        "live_parent_coerced_parts_after_read": live,
        "peak_pending_futures": peak_pending,
        "queue_limit": 4,
        "worker_limit": 2,
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "fixture_manifest_sha256": sha(root / "manifest.json"),
        "harness_sha256": sha(Path(__file__)),
        "interpretation": "Independent parent-part and future-window counters; not a PSS claim or a sum of process peaks.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.root, args.count), indent=2, sort_keys=True))
