"""Installed-wheel GWF parallel worker fault and diagnostic replay probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(source, channels, start, end, backend, read_kwargs):
    """Picklable spawned worker with one deterministic injected condition."""
    from gwpy.timeseries.io.gwf.core import read_timeseriesdict

    index = int(Path(source).stem.rsplit("_", 1)[-1])
    mode = os.environ.get("GWEXPY_F4_WORKER_PROBE")
    if mode == "fault" and index == 15:
        raise RuntimeError("injected late worker fault")
    if mode == "diagnostic" and index == 4:
        warnings.warn("injected worker warning", UserWarning, stacklevel=1)
        logging.getLogger("gwexpy.test.f4").warning("injected worker log")
        print("injected Python stdout")
        print("injected Python stderr", file=sys.stderr)
        os.write(1, b"injected native stdout\n")
        os.write(2, b"injected native stderr\n")
    return read_timeseriesdict(
        source,
        list(channels),
        start=start,
        end=end,
        backend=backend,
        **read_kwargs,
    )


def capture(root: Path, mode: str) -> dict:
    import gwexpy
    import gwexpy.timeseries._gwf_io as io
    from gwexpy.timeseries import TimeSeriesDict

    assert Path(gwexpy.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "gwexpy-v025-b-f4-stress-fixtures-v1"
    paths = [root / name for name in manifest["source_order"][:16]]
    expected = {item["name"]: item["sha256"] for item in manifest["files"]}
    for path in paths:
        assert sha(path) == expected[path.name]
    os.environ["GWEXPY_F4_WORKER_PROBE"] = mode
    io._read_gwf_timeseriesdict_worker = worker
    route_attempted = route_succeeded = 0
    original_fast = getattr(io, "_try_bounded_gwf_parallel", None)
    if original_fast is not None:

        def counted_fast(*args, **kwargs):
            nonlocal route_attempted, route_succeeded
            route_attempted += 1
            result = original_fast(*args, **kwargs)
            if result is not io._GWF_BOUNDED_FALLBACK:
                route_succeeded += 1
            return result

        io._try_bounded_gwf_parallel = counted_fast
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = TimeSeriesDict.read(
                paths, [manifest["channel"]], format="gwf", parallel=2
            )
        except Exception as error:
            outcome = {
                "kind": "error",
                "type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
            }
        else:
            series = result[manifest["channel"]]
            outcome = {
                "kind": "return",
                "dtype": series.dtype.str,
                "shape": list(series.shape),
                "unit": str(series.unit),
                "t0": float(series.t0.value),
                "dt": float(series.dt.value),
                "values_sha256": hashlib.sha256(series.value.tobytes()).hexdigest(),
            }
    return {
        "mode": mode,
        "outcome": outcome,
        "warnings": [
            {
                "category": f"{item.category.__module__}.{item.category.__qualname__}",
                "message": str(item.message),
            }
            for item in caught
        ],
        "candidate_fast_attempted": route_attempted,
        "candidate_fast_succeeded": route_succeeded,
        "fixture_manifest_sha256": sha(root / "manifest.json"),
        "harness_sha256": sha(Path(__file__)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("fault", "diagnostic"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(capture(args.root, args.mode), indent=2, sort_keys=True) + "\n")
