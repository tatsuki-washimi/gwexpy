"""Characterize Zarr timing, unit, and multi-store dtype behavior."""

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path

import numpy as np
import zarr
from gwexpy.timeseries import TimeSeriesMatrix


SOURCE_SHA = "ae10234a1e37853508c54901bf7c9e80878f25aa"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if name == "gwexpy":
            return "source checkout (distribution metadata unavailable)"
        raise


def create_matrix_cell(
    path: Path,
    array_name: str,
    data: np.ndarray,
    *,
    t0: float,
    col: int = 0,
    unit: str = "V",
) -> None:
    store = zarr.open_group(str(path), mode="a")
    creator = getattr(store, "create_array", None) or store.create_dataset
    array = creator(array_name, data=data)
    array.attrs["sample_rate"] = 16.0
    array.attrs["dt"] = 1.0 / 16.0
    array.attrs["t0"] = float(t0)
    array.attrs["unit"] = unit
    array.attrs["gwexpy_row_key"] = json.dumps("r0")
    array.attrs["gwexpy_col_key"] = json.dumps(f"c{col}")
    array.attrs["gwexpy_key_format"] = "json"
    array.attrs["gwexpy_row_index"] = 0
    array.attrs["gwexpy_col_index"] = int(col)


def characterize_axis(root: Path) -> dict[str, object]:
    path = root / "misaligned-axis.zarr"
    create_matrix_cell(
        path, "cell0", np.array([1.0, 2.0]), t0=1_234_567_890.25, col=0
    )
    create_matrix_cell(
        path, "cell1", np.array([3.0, 4.0]), t0=1_234_567_890.251, col=1
    )
    try:
        matrix = TimeSeriesMatrix.read(str(path), format="zarr")
        return {
            "result": "accepted",
            "shape": list(matrix.shape),
            "returned_t0": float(matrix.x0.value),
            "values": np.asarray(matrix.value).tolist(),
        }
    except Exception as exc:  # Record reader behavior, including explicit failure.
        return {"result": "raised", "exception": type(exc).__name__, "message": str(exc)}


def characterize_native_unit(root: Path) -> dict[str, object]:
    path = root / "native-channels.zarr"
    store = zarr.open_group(str(path), mode="a")
    creator = getattr(store, "create_array", None) or store.create_dataset
    for name in ("a", "b"):
        array = creator(name, data=np.array([1.0, 2.0], dtype=np.float64))
        array.attrs["sample_rate"] = 16.0
        array.attrs["t0"] = 1_234_567_890.25
        array.attrs["unit"] = "V"
    try:
        matrix = TimeSeriesMatrix.read(str(path), format="zarr")
        return {
            "result": "accepted",
            "source_unit": "V",
            "matrix_units": str(matrix.units),
            "matrix_unit_cells": [str(value) for value in matrix.units.flat],
        }
    except Exception as exc:  # Record reader behavior, including explicit failure.
        return {"result": "raised", "exception": type(exc).__name__, "message": str(exc)}


def characterize_multistore_dtype(root: Path) -> dict[str, object]:
    first = root / "int64.zarr"
    second = root / "float64.zarr"
    source_ints = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)
    source_floats = np.array([0.5, 1.5], dtype=np.float64)
    create_matrix_cell(first, "cell", source_ints, t0=1_234_567_890.25)
    create_matrix_cell(
        second,
        "cell",
        source_floats,
        t0=1_234_567_890.375,
    )
    try:
        matrix = TimeSeriesMatrix.read([str(first), str(second)], format="zarr")
        return {
            "result": "accepted",
            "source_int64": source_ints.tolist(),
            "source_float64": source_floats.tolist(),
            "actual_dtype": str(np.asarray(matrix.value).dtype),
            "actual_values": np.asarray(matrix.value).tolist(),
        }
    except Exception as exc:  # Record reader behavior, including explicit failure.
        return {"result": "raised", "exception": type(exc).__name__, "message": str(exc)}


def characterize_uint64(root: Path) -> dict[str, object]:
    path = root / "uint64.zarr"
    source_values = np.array([2**53 + 1, 2**53 + 3], dtype=np.uint64)
    create_matrix_cell(path, "cell", source_values, t0=1_234_567_890.25)
    try:
        matrix = TimeSeriesMatrix.read(str(path), format="zarr")
        return {
            "result": "accepted",
            "source_dtype": str(source_values.dtype),
            "source_values": source_values.tolist(),
            "actual_dtype": str(np.asarray(matrix.value).dtype),
            "actual_values": np.asarray(matrix.value).tolist(),
        }
    except Exception as exc:  # Record reader behavior, including explicit failure.
        return {"result": "raised", "exception": type(exc).__name__, "message": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        records = [
            {
                "finding_id": "ZARR-AXIS-001",
                "source_sha": SOURCE_SHA,
                "route": "TimeSeriesMatrix.read(path, format='zarr')",
                "oracle": "independent native Zarr cell timing metadata",
                "fixture": "matrix cells differ in GPS t0 by 1 ms at 16 Hz",
                "expected": "reject cells that cannot share one matrix time axis",
                "actual": characterize_axis(root),
            },
            {
                "finding_id": "ZARR-UNIT-001",
                "source_sha": SOURCE_SHA,
                "route": "TimeSeriesMatrix.read(path, format='zarr')",
                "oracle": "independent native Zarr per-channel unit metadata",
                "fixture": "two native channel arrays both declare unit V",
                "expected": "preserve V on both returned matrix cells",
                "actual": characterize_native_unit(root),
            },
            {
                "finding_id": "ZARR-MULTISTORE-DTYPE-001",
                "source_sha": SOURCE_SHA,
                "route": "TimeSeriesMatrix.read([int_store, float_store], format='zarr')",
                "oracle": "independent native Zarr source dtypes and values",
                "fixture": "adjacent int64 and float64 matrix stores; integer values exceed 2**53",
                "expected": "preserve integers exactly or reject an incompatible dtype merge",
                "actual": characterize_multistore_dtype(root),
            },
            {
                "finding_id": "ZARR-UINT64-001",
                "source_sha": SOURCE_SHA,
                "route": "TimeSeriesMatrix.read(path, format='zarr')",
                "oracle": "independent native Zarr source dtype and values",
                "fixture": "one matrix cell stores uint64 values above 2**53",
                "expected": "preserve unsigned integers exactly or reject unsupported conversion",
                "actual": characterize_uint64(root),
            },
        ]
    versions = {
        name: package_version(name)
        for name in ("gwexpy", "gwpy", "numpy", "zarr")
    }
    for record in records:
        record["versions"] = versions
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
