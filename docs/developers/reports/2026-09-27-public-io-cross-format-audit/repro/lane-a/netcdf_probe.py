"""Temporary, independent NetCDF fixtures for the public I/O audit."""

from __future__ import annotations

import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

ROOT = Path(__file__).parent / "netcdf"
ROOT.mkdir(exist_ok=True)
T0 = 1_234_567_890.25
DT = 0.25


def emit(**data):
    print(json.dumps(data, default=str, sort_keys=True))


def describe(obj):
    if isinstance(obj, TimeSeriesDict):
        return {"type": type(obj).__name__, "keys": list(obj), "series": {k: describe(v) for k, v in obj.items()}}
    data = np.asarray(obj.value)
    unit = str(obj[0, 0].unit) if isinstance(obj, TimeSeriesMatrix) else str(obj.unit)
    out = {"type": type(obj).__name__, "shape": data.shape, "dtype": str(data.dtype),
           "values": data.tolist(), "t0": float(obj.t0.value), "dt": float(obj.dt.value),
           "unit": unit}
    try:
        out["times"] = np.asarray(obj.times.value).tolist()
    except AttributeError:
        out["times"] = None
    if isinstance(obj, TimeSeriesMatrix):
        out.update(rows=list(obj.row_keys()), cols=list(obj.col_keys()),
                   cell_units=[[str(obj[i, j].unit) for j in range(obj.shape[1])]
                               for i in range(obj.shape[0])])
    else:
        out.update(name=obj.name, channel=str(obj.channel))
    return out


def attempt(case, route, path, reader, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = reader.read(path, **kwargs)
            outcome = {"result": describe(result)}
        except Exception as exc:
            outcome = {"error_type": type(exc).__name__, "error": str(exc)}
    emit(case=case, route=route, path=str(path), warnings=[str(w.message) for w in caught], **outcome)


def schema():
    numerator, denominator = DT.as_integer_ratio()
    return {"gwexpy_netcdf_schema_version": 2, "gwexpy_t0_float_hex": T0.hex(),
            "gwexpy_t0_gps_seconds": 1_234_567_890,
            "gwexpy_t0_gps_nanoseconds": 250_000_000,
            "gwexpy_dt_numerator": str(numerator),
            "gwexpy_dt_denominator": str(denominator),
            "gwexpy_axis_encoding": "t(i)=t0+i*dt"}


def cell(name, row, col, ri, ci, value, unit="V"):
    attrs = {"gwexpy_row_key": json.dumps(row), "gwexpy_col_key": json.dumps(col),
             "gwexpy_key_format": "json", "gwexpy_row_index": ri,
             "gwexpy_col_index": ci, "units": unit}
    return name, xr.DataArray(np.full(3, value, dtype=np.float64), dims=["sample"], attrs=attrs)


def cells():
    return [cell("c00", "r0", "c0", 0, 0, 11),
            cell("c01", "r0", "c1", 0, 1, 12),
            cell("c10", "r1", "c0", 1, 0, 21),
            cell("c11", "r1", "c1", 1, 1, 22)]


def matrix_fixture(case, entries):
    path = ROOT / f"{case}.nc"
    xr.Dataset(dict(entries), coords={"sample": np.arange(3, dtype=np.int64)}, attrs=schema()).to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, engine="netcdf4") as ds:
        native = [{"var": name, "values": ds[name].values.tolist(), "attrs": dict(ds[name].attrs)} for name in ds.data_vars]
    emit(case=case, oracle="xarray", path=str(path), source_cells=native)
    attempt(case, "matrix-explicit-nc", path, TimeSeriesMatrix, format="nc")


def topology():
    base = cells()
    cases = {
        "complete": base,
        "missing_cell": base[:3],
        "duplicate_cell": base + [cell("duplicate", "r0", "c0", 0, 0, 99)],
        "same_key_different_index": base[:3] + [cell("c11", "r1", "c1", 2, 1, 22)],
        "same_col_key_different_index": base[:3] + [cell("c11", "r1", "c1", 1, 2, 22)],
        "different_key_same_index": base[:2] + [cell("c10", "r1", "c0", 0, 0, 21),
                                                   cell("c11", "r1", "c1", 0, 1, 22)],
        "different_col_key_same_index": [base[0], cell("c01", "r0", "c1", 0, 0, 12),
                                         base[2], cell("c11", "r1", "c1", 1, 0, 22)],
        "negative_index": base[:2] + [cell("c10", "r1", "c0", -1, 0, 21),
                                       cell("c11", "r1", "c1", -1, 1, 22)],
        "negative_col_index": [base[0], cell("c01", "r0", "c1", 0, -1, 12),
                               base[2], cell("c11", "r1", "c1", 1, -1, 22)],
        "sparse_index": base[:2] + [cell("c10", "r1", "c0", 3, 0, 21),
                                     cell("c11", "r1", "c1", 3, 1, 22)],
        "sparse_col_index": [base[0], cell("c01", "r0", "c1", 0, 3, 12),
                             base[2], cell("c11", "r1", "c1", 1, 3, 22)],
        "out_of_range_index": base[:2] + [cell("c10", "r1", "c0", 1000, 0, 21),
                                           cell("c11", "r1", "c1", 1000, 1, 22)],
        "out_of_range_col_index": [base[0], cell("c01", "r0", "c1", 0, 1000, 12),
                                   base[2], cell("c11", "r1", "c1", 1, 1000, 22)],
        "unit_mismatch": base[:3] + [cell("c11", "r1", "c1", 1, 1, 22, unit="m")],
        "cell_t0_conflict": base[:3] + [("c11", base[3][1].assign_attrs({**base[3][1].attrs, "t0": T0 + 1}))],
        "cell_dt_conflict": base[:3] + [("c11", base[3][1].assign_attrs({**base[3][1].attrs, "dt": DT * 2}))],
    }
    for case, entries in cases.items():
        matrix_fixture(case, entries)
    emit(case="cell_length_mismatch", status="NO_VALID_FIXTURE",
         reason="All variables sharing the sample dimension have its fixed length in a valid NetCDF dataset.")


def legacy():
    for case, coord in (
        ("legacy_irregular", np.array([0, 1, 2, 4, 5], dtype=np.float64)),
        ("legacy_datetime", np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01",
                                      "2020-01-01T00:00:02", "2020-01-01T00:00:04",
                                      "2020-01-01T00:00:05"], dtype="datetime64[s]")),
    ):
        path = ROOT / f"{case}.nc"
        xr.Dataset({"sig": xr.DataArray(np.array([10, 11, 12, 14, 15]), dims=["time"], attrs={"units": "V"})},
                   coords={"time": coord}).to_netcdf(path, engine="netcdf4")
        with xr.open_dataset(path, engine="netcdf4") as ds:
            emit(case=case, oracle="xarray", path=str(path), source_coord=ds.time.values.astype(str).tolist(),
                 source_data=ds.sig.values.tolist(), source_coord_dtype=str(ds.time.dtype))
        for name, cls, kwargs in (
            ("single-explicit-nc", TimeSeries, {"format": "nc"}),
            ("dict-explicit-netcdf4", TimeSeriesDict, {"format": "netcdf4"}),
            ("matrix-explicit-nc", TimeSeriesMatrix, {"format": "nc"}),
            ("single-auto", TimeSeries, {}),
            ("dict-auto", TimeSeriesDict, {}),
            ("matrix-auto", TimeSeriesMatrix, {}),
        ):
            attempt(case, name, path, cls, **kwargs)
        try:
            from gwpy.timeseries import TimeSeries as GwpyTimeSeries

            attempt(case, "gwpy-single-auto", path, GwpyTimeSeries)
        except Exception as exc:
            emit(case=case, route="gwpy-single-auto", error_type=type(exc).__name__, error=str(exc))


def public_writes():
    values = np.array([2**53 + 1, -(2**53) - 3, 17], dtype=np.int64)
    ts = TimeSeries(values, t0=T0, dt=DT, name="signal", unit="V")
    matrix = TimeSeriesMatrix(values.reshape(1, 1, -1), t0=T0, dt=DT, unit="V")
    for route, writer in (
        ("single-explicit-nc", lambda p: ts.write(p, format="nc")),
        ("dict-explicit-nc", lambda p: TimeSeriesDict({"signal": ts}).write(p, format="nc")),
        ("matrix-explicit-nc", lambda p: matrix.write(p, format="nc")),
        ("single-auto", lambda p: ts.write(p)),
        ("dict-auto", lambda p: TimeSeriesDict({"signal": ts}).write(p)),
        ("matrix-auto", lambda p: matrix.write(p)),
    ):
        path = ROOT / f"public_write_{route}.nc"
        try:
            writer(path)
            with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
                vars_out = {name: {"dtype": str(ds[name].dtype),
                                   "values": ds[name].values.tolist(), "attrs": dict(ds[name].attrs)}
                            for name in ds.data_vars}
                emit(case="public_write_int64", route=route, path=str(path),
                     oracle="xarray", native_vars=vars_out, source_values=values.tolist(),
                     source_matrix_cell_unit=str(matrix[0, 0].unit),
                     source_series_unit=str(ts.unit), global_attrs=dict(ds.attrs))
        except Exception as exc:
            emit(case="public_write_int64", route=route, path=str(path),
                 error_type=type(exc).__name__, error=str(exc))


if __name__ == "__main__":
    emit(source_sha="ae10234a1e37853508c54901bf7c9e80878f25aa", numpy=np.__version__,
         xarray=xr.__version__, netCDF4=version("netCDF4"), gwpy=version("gwpy"))
    topology()
    legacy()
    public_writes()
