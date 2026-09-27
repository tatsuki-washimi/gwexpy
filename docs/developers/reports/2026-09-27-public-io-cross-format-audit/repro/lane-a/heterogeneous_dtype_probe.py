"""Reproduce lossy mixed-dtype NetCDF matrix reads with an independent xarray fixture."""

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from gwexpy.timeseries import TimeSeriesMatrix


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if name == "gwexpy":
            return "source checkout (distribution metadata unavailable)"
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    timing = {
        "gwexpy_netcdf_schema_version": 2,
        "gwexpy_t0_float_hex": (0.0).hex(),
        "gwexpy_t0_gps_seconds": 0,
        "gwexpy_t0_gps_nanoseconds": 0,
        "gwexpy_dt_numerator": "1",
        "gwexpy_dt_denominator": "1",
        "gwexpy_axis_encoding": "t(i)=t0+i*dt",
    }

    def cell_attrs(column: int) -> dict[str, object]:
        return {
            "gwexpy_row_key": '\"r0\"',
            "gwexpy_col_key": f'\"c{column}\"',
            "gwexpy_key_format": "json",
            "gwexpy_row_index": 0,
            "gwexpy_col_index": column,
            "units": "V",
        }

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "heterogeneous-dtype.nc"
        xr.Dataset(
            {
                "first": xr.DataArray(
                    np.array([1, 2, 3], dtype=np.int32),
                    dims=["sample"],
                    attrs=cell_attrs(0),
                ),
                "second": xr.DataArray(
                    np.array([1.5, 2.25, 3.75], dtype=np.float64),
                    dims=["sample"],
                    attrs=cell_attrs(1),
                ),
            },
            coords={"sample": np.arange(3, dtype=np.int64)},
            attrs=timing,
        ).to_netcdf(path)

        with xr.open_dataset(path) as dataset:
            source_cells = {
                name: {
                    "dtype": str(dataset[name].dtype),
                    "values": dataset[name].values.tolist(),
                }
                for name in dataset.data_vars
            }

        try:
            matrix = TimeSeriesMatrix.read(path, format="nc")
            actual = {
                "dtype": str(matrix.dtype),
                "values": matrix.value.tolist(),
            }
        except Exception as exc:  # record reader behavior, including explicit failure
            actual = {"exception": type(exc).__name__, "message": str(exc)}

    record = {
        "finding_id": "NC-DTYPE-001",
        "source_sha": "ae10234a1e37853508c54901bf7c9e80878f25aa",
        "route": "TimeSeriesMatrix.read(path, format='nc')",
        "oracle": "xarray native data variable dtypes and values",
        "fixture": "1x2 matrix with int32 first cell and float64 second cell",
        "source_cells": source_cells,
        "expected": {"second_cell_values": [1.5, 2.25, 3.75]},
        "actual": actual,
        "versions": {
            name: package_version(name)
            for name in ("gwexpy", "gwpy", "numpy", "xarray", "netCDF4")
        },
    }
    line = json.dumps(record, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(line)
    else:
        print(line, end="")


if __name__ == "__main__":
    main()
