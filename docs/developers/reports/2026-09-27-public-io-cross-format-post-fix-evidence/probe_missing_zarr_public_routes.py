"""Probe public auto routes with Zarr and NetCDF optional packages absent."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform

import numpy as np
import sys
import tempfile
from pathlib import Path

optional_modules = ("zarr", "xarray", "netCDF4")
missing = {name: importlib.util.find_spec(name) is None for name in optional_modules}
if not all(missing.values()):
    raise RuntimeError(f"probe requires absent optional packages: {missing}")
if any(name in sys.modules for name in optional_modules):
    raise RuntimeError("optional package was imported before the absence check")

import gwexpy

gwexpy.register_all()
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

with tempfile.TemporaryDirectory(prefix="gwexpy-optional-missing-") as temporary:
    store = Path(temporary) / "missing-backend.zarr"
    store.mkdir()
    matrix_output = Path(temporary) / "matrix-output.zarr"
    netcdf_input = Path(temporary) / "missing-backend.nc"
    netcdf_input.touch()
    netcdf_output = Path(temporary) / "matrix-output.nc"
    matrix = TimeSeriesMatrix(
        np.ones((1, 1, 3)),
        t0=1_234_567_890.25,
        sample_rate=16.0,
    )
    operations = {
        "timeseries_auto_read": lambda: TimeSeries.read(store),
        "timeseriesdict_auto_read": lambda: TimeSeriesDict.read(store),
        "matrix_auto_read": lambda: TimeSeriesMatrix.read(store),
        "matrix_auto_write": lambda: matrix.write(matrix_output),
        "matrix_auto_netcdf_read": lambda: TimeSeriesMatrix.read(netcdf_input),
        "matrix_auto_netcdf_write": lambda: matrix.write(netcdf_output),
    }
    observations = {}
    for name, operation in operations.items():
        try:
            operation()
        except ImportError as error:
            observations[name] = {
                "exception": type(error).__name__,
                "message": str(error),
            }
        except Exception as error:  # record and fail unexpected route results
            observations[name] = {
                "exception": type(error).__name__,
                "message": str(error),
            }
        else:
            observations[name] = {"exception": None, "message": "returned"}

result = {
    "environment": {
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "gwexpy_source": gwexpy.__file__,
        "gwexpy_version": importlib.metadata.version("gwexpy"),
        "optional_package_absence": missing,
    },
    "public_routes": observations,
}
print(json.dumps(result, indent=2, sort_keys=True))
if any(
    record["exception"] != "ImportError"
    or (
        "xarray" if "netcdf" in name else "zarr"
    ) not in record["message"].lower()
    for name, record in observations.items()
):
    raise SystemExit("one or more public routes failed to preserve the backend ImportError")
