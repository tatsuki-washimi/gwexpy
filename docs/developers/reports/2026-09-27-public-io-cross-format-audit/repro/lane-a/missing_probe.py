"""Temporary public-route optional-missing behavior probe."""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The fixture directory contains a directory named "zarr".  Exclude the
# script directory so it cannot masquerade as a namespace package when the
# actual optional dependency is absent.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]

import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

ROOT = Path(__file__).parent


def versions():
    out = {}
    for name in ("numpy", "gwpy", "xarray", "netCDF4", "zarr"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def attempt(fmt, operation, cls, route, call):
    try:
        result = call()
        outcome = {"returned_type": type(result).__name__}
    except Exception as exc:
        outcome = {"error_type": type(exc).__name__, "error": str(exc)}
    print(json.dumps({"format": fmt, "operation": operation, "class": cls.__name__,
                      "route": route, **outcome}, sort_keys=True))


if __name__ == "__main__":
    print(json.dumps({"source_sha": "ae10234a1e37853508c54901bf7c9e80878f25aa",
                      "dependencies": versions()}, sort_keys=True))
    ts = TimeSeries(np.array([1.0, 2.0, 3.0]), t0=1_234_567_890.25, dt=0.25, name="signal", unit="V")
    objs = ((TimeSeries, ts), (TimeSeriesDict, TimeSeriesDict({"signal": ts})),
            (TimeSeriesMatrix, TimeSeriesMatrix(np.array([[[1.0, 2.0, 3.0]]]),
                                                t0=1_234_567_890.25, dt=0.25, unit="V")))
    for fmt, source in (("nc", ROOT / "netcdf" / "complete.nc"),
                        ("zarr", ROOT / "zarr" / "native_float64.zarr")):
        for cls, obj in objs:
            for route, kwargs in (("explicit", {"format": fmt}), ("auto", {})):
                attempt(fmt, "read", cls, route, lambda: cls.read(source, **kwargs))
                target = ROOT / "missing_outputs" / f"{fmt}_{cls.__name__}_{route}.{fmt}"
                target.parent.mkdir(exist_ok=True)
                attempt(fmt, "write", cls, route, lambda: obj.write(target, **kwargs))
