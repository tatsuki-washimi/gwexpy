"""Create deterministic ATS and homogeneous NetCDF fixtures for B-X/#586.

This generator is independent of GWexpy. Its output directory must not exist,
and the resulting file bytes are frozen before any B-X wheel capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

SCHEMA = "gwexpy-v025-bx-fixtures-v1"
ATS_SAMPLES = 4_194_304
MATRIX_ROWS = 4
MATRIX_COLUMNS = 4
MATRIX_SAMPLES = 262_144


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ats_values(bits: int) -> np.ndarray:
    index = np.arange(ATS_SAMPLES, dtype=np.int64)
    if bits == 32:
        return ((index * 17) % 65_536 - 32_768).astype("<i4")
    sign = np.where(index % 2 == 0, 1, -1)
    return ((2**45 + (index * 17) % 65_536) * sign).astype("<i8")


def _ats_header(bits: int, samples: int, lsb_millivolt: float) -> bytearray:
    header = bytearray(1024)
    struct.pack_into("<H", header, 0x00, len(header))
    struct.pack_into("<h", header, 0x02, 81 if bits == 64 else 80)
    struct.pack_into("<I", header, 0x04, samples)
    struct.pack_into("<f", header, 0x08, 256.0)
    struct.pack_into("<I", header, 0x0C, 1_600_000_000)
    struct.pack_into("<d", header, 0x10, lsb_millivolt)
    struct.pack_into("<H", header, 0x20, 7)
    header[0x26:0x28] = b"Ex"
    header[0x28:0x2E] = b"MFS06 "
    struct.pack_into("<h", header, 0x2E, 19)
    header[0x84:0x90] = b"ADU08E      "
    struct.pack_into("<h", header, 0xAA, 1 if bits == 64 else 0)
    return header


def _write_ats(path: Path, bits: int) -> dict[str, object]:
    header = _ats_header(bits, ATS_SAMPLES, 123.456)
    values = _ats_values(bits)
    with path.open("xb") as stream:
        stream.write(header)
        values.tofile(stream)
    return {
        "name": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "raw_dtype": values.dtype.str,
        "samples": ATS_SAMPLES,
        "raw_values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "lsb_millivolt_per_count": 123.456,
    }


def _write_ats_overflow(path: Path) -> dict[str, object]:
    values = np.array([0, 1, -1, np.iinfo(np.int64).max] * 4, dtype="<i8")
    header = _ats_header(64, values.size, 1e300)
    with path.open("xb") as stream:
        stream.write(header)
        values.tofile(stream)
    return {
        "name": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "raw_dtype": values.dtype.str,
        "samples": values.size,
        "raw_values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "lsb_millivolt_per_count": 1e300,
        "purpose": "Capture B1 NumPy overflow warning and seterr=raise behavior",
    }


def _matrix_values(row: int, column: int, variant: str) -> np.ndarray:
    if variant == "large":
        index = np.arange(MATRIX_SAMPLES, dtype=np.float64)
        return index / 8.0 + row * 100_000.0 + column * 10_000.0
    if variant == "nan_inf":
        return np.array([0.0, np.nan, np.inf, -np.inf, -0.0, 1.5, -2.5, 4.0])
    if variant == "int64_extrema":
        limits = np.iinfo(np.int64)
        return np.array(
            [limits.min, limits.max, 0, 1, -1, 2**53 + 1, -(2**53 + 1), row + column],
            dtype=np.int64,
        )
    if variant == "object_strings":
        return np.array(["a", "b", "", "unicode-λ", "c", "d", "e", "f"], dtype=object)
    raise ValueError(f"Unknown matrix variant: {variant}")


def _write_matrix(path: Path, variant: str) -> dict[str, object]:
    import xarray as xr

    t0 = 1_000_000_000.125
    dt = 0.125
    data_vars = {}
    cells = []
    rows = MATRIX_ROWS if variant == "large" else 2
    columns = MATRIX_COLUMNS if variant == "large" else 2
    samples = MATRIX_SAMPLES if variant == "large" else 8
    for row in range(rows):
        for column in range(columns):
            values = _matrix_values(row, column, variant)
            name = f"cell_r{row}_c{column}"
            data_vars[name] = xr.DataArray(
                values,
                dims=["sample"],
                attrs={
                    "units": "V",
                    "gwexpy_row_key": json.dumps(f"r{row}"),
                    "gwexpy_col_key": json.dumps(f"c{column}"),
                    "gwexpy_key_format": "json",
                    "gwexpy_row_index": row,
                    "gwexpy_col_index": column,
                },
            )
            cells.append(
                {
                    "name": name,
                    "row": row,
                    "column": column,
                    "values_sha256": hashlib.sha256(values.tobytes()).hexdigest()
                    if values.dtype != object
                    else None,
                }
            )
    numerator, denominator = dt.as_integer_ratio()
    ds = xr.Dataset(
        data_vars,
        coords={"sample": np.arange(samples, dtype=np.int64)},
        attrs={
            "gwexpy_netcdf_schema_version": 2,
            "gwexpy_t0_float_hex": t0.hex(),
            "gwexpy_t0_gps_seconds": 1_000_000_000,
            "gwexpy_t0_gps_nanoseconds": 125_000_000,
            "gwexpy_dt_numerator": str(numerator),
            "gwexpy_dt_denominator": str(denominator),
            "gwexpy_axis_encoding": "t(i)=t0+i*dt",
            "gwexpy_matrix_rows": rows,
            "gwexpy_matrix_columns": columns,
        },
    )
    ds.to_netcdf(path, engine="netcdf4", mode="w")
    return {
        "name": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "variant": variant,
        "dtype": next(iter(data_vars.values())).dtype.str,
        "shape": [rows, columns, samples],
        "t0_float_hex": t0.hex(),
        "dt_float_hex": dt.hex(),
        "unit": "V",
        "cells": cells,
    }


def generate(root: Path) -> None:
    """Write every B-X fixture and a manifest to a new directory."""
    root.mkdir(parents=True, exist_ok=False)
    files = [
        _write_ats(root / "ats_int32.ats", 32),
        _write_ats(root / "ats_int64.ats", 64),
        _write_ats_overflow(root / "ats_overflow.ats"),
        _write_matrix(root / "homogeneous_matrix.nc", "large"),
        _write_matrix(root / "matrix_nan_inf.nc", "nan_inf"),
        _write_matrix(root / "matrix_int64_extrema.nc", "int64_extrema"),
        _write_matrix(root / "matrix_object_strings.nc", "object_strings"),
    ]
    manifest = {
        "schema": SCHEMA,
        "generator_sha256": _sha256(Path(__file__)),
        "files": files,
        "notes": "Fixture bytes and manifest must be frozen before B-X wheel measurements.",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
