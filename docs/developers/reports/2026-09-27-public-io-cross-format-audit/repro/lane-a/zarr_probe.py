"""Temporary native-Zarr fixtures and public I/O dtype probe."""

from __future__ import annotations

import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import zarr
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

ROOT = Path(__file__).parent / "zarr"
ROOT.mkdir(exist_ok=True)
T0 = 1_234_567_890.25
SR = 16.0
VALUES = {
    "int16": [123, -456, 789],
    "int32": [2**30 + 1, -(2**30) - 3, 17],
    "int64": [2**53 + 1, -(2**53) - 3, 17],
    "float32": [1.25, -2.5, 3.75],
    "float64": [1.1, -2.2, 3.3],
    "complex64": [1 + 2j, -3 + 4j, 5 - 6j],
    "complex128": [1.1 + 2.2j, -3.3 + 4.4j, 5.5 - 6.6j],
}


def emit(**data):
    print(json.dumps(data, default=str, sort_keys=True))


def payload(values):
    arr = np.asarray(values)
    return {"dtype": str(arr.dtype), "shape": arr.shape,
            "values": [str(x) for x in arr.tolist()],
            "real": np.real(arr).tolist(), "imag": np.imag(arr).tolist(),
            "phase": np.angle(arr).tolist()}


def describe(obj):
    if isinstance(obj, TimeSeriesDict):
        return {"type": type(obj).__name__, "keys": list(obj), "series": {k: describe(v) for k, v in obj.items()}}
    arr = np.asarray(obj.value)
    unit = str(obj[0, 0].unit) if isinstance(obj, TimeSeriesMatrix) else str(obj.unit)
    out = {"type": type(obj).__name__, **payload(arr), "t0": float(obj.t0.value),
           "dt": float(obj.dt.value), "unit": unit}
    if isinstance(obj, TimeSeriesMatrix):
        out.update(rows=list(obj.row_keys()), cols=list(obj.col_keys()))
    else:
        out.update(name=obj.name, channel=str(obj.channel))
    return out


def attempt(case, route, call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = call()
            outcome = {"result": describe(result) if result is not None else None}
        except Exception as exc:
            outcome = {"error_type": type(exc).__name__, "error": str(exc)}
    emit(case=case, route=route, warnings=[str(w.message) for w in caught], **outcome)


def native_group(path):
    try:
        return zarr.open_group(str(path), mode="w", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="w")


def native_array(group, name, values):
    creator = getattr(group, "create_array", None) or group.create_dataset
    arr = creator(name, data=values)
    arr.attrs["sample_rate"] = SR
    arr.attrs["t0"] = T0
    arr.attrs["unit"] = "V"


def native_store(group):
    return {name: {**payload(np.asarray(group[name][:])), "attrs": dict(group[name].attrs)}
            for name in group.keys()}


def native_dtype_reads():
    for dtype, raw in VALUES.items():
        values = np.asarray(raw, dtype=dtype)
        path = ROOT / f"native_{dtype}.zarr"
        try:
            group = native_group(path)
            native_array(group, "signal", values)
            stored = np.asarray(zarr.open_group(str(path), mode="r")["signal"][:])
        except Exception as exc:
            emit(case=f"native_{dtype}", status="NO_VALID_FIXTURE", error_type=type(exc).__name__, error=str(exc))
            continue
        emit(case=f"native_{dtype}", oracle="zarr", path=str(path), source=payload(stored),
             intended=payload(values), equal=np.array_equal(stored, values))
        attempt(f"native_{dtype}", "single-explicit-zarr", lambda: TimeSeries.read(path, format="zarr"))
        attempt(f"native_{dtype}", "dict-explicit-zarr", lambda: TimeSeriesDict.read(path, format="zarr"))
        attempt(f"native_{dtype}", "matrix-explicit-zarr", lambda: TimeSeriesMatrix.read(path, format="zarr"))
        if dtype in {"int64", "complex128"}:
            attempt(f"native_{dtype}", "single-auto", lambda: TimeSeries.read(path))
            attempt(f"native_{dtype}", "dict-auto", lambda: TimeSeriesDict.read(path))
            attempt(f"native_{dtype}", "matrix-auto", lambda: TimeSeriesMatrix.read(path))


def public_dtype_writes():
    for dtype, raw in VALUES.items():
        values = np.asarray(raw, dtype=dtype)
        try:
            ts = TimeSeries(values, t0=T0, sample_rate=SR, name="signal", unit="V")
        except Exception as exc:
            emit(case=f"write_{dtype}", status="CONSTRUCTOR_ERROR", error_type=type(exc).__name__, error=str(exc))
            continue
        emit(case=f"write_{dtype}", oracle="numpy_and_constructor", source=payload(values),
             constructed=describe(ts))
        for route, writer in (
            ("single-explicit-zarr", lambda p: ts.write(p, format="zarr")),
            ("dict-explicit-zarr", lambda p: TimeSeriesDict({"signal": ts}).write(p, format="zarr")),
            ("matrix-explicit-zarr", lambda p: TimeSeriesMatrix(values.reshape(1, 1, -1), t0=T0, sample_rate=SR, unit="V").write(p, format="zarr")),
        ):
            path = ROOT / f"write_{dtype}_{route}.zarr"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    writer(path)
                    group = zarr.open_group(str(path), mode="r")
                    arrays = native_store(group)
                    outcome = {"native_store": arrays}
                except Exception as exc:
                    outcome = {"error_type": type(exc).__name__, "error": str(exc)}
            emit(case=f"write_{dtype}", route=route, path=str(path),
                 warnings=[str(w.message) for w in caught], **outcome)
        if dtype in {"int64", "complex128"}:
            for route, writer in (
                ("single-auto", lambda p: ts.write(p)),
                ("dict-auto", lambda p: TimeSeriesDict({"signal": ts}).write(p)),
                ("matrix-auto", lambda p: TimeSeriesMatrix(values.reshape(1, 1, -1), t0=T0, sample_rate=SR, unit="V").write(p)),
            ):
                path = ROOT / f"write_{dtype}_{route}.zarr"
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        writer(path)
                        group = zarr.open_group(str(path), mode="r")
                        outcome = {"native_store": native_store(group)}
                    except Exception as exc:
                        outcome = {"error_type": type(exc).__name__, "error": str(exc)}
                emit(case=f"write_{dtype}", route=route, path=str(path),
                     warnings=[str(w.message) for w in caught], **outcome)


if __name__ == "__main__":
    emit(source_sha="ae10234a1e37853508c54901bf7c9e80878f25aa",
         numpy=np.__version__, zarr=zarr.__version__, gwpy=version("gwpy"))
    native_dtype_reads()
    public_dtype_writes()
