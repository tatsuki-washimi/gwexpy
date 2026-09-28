"""Independent, deterministic inputs for the #584 range-read baseline.

This module creates files with h5py, xarray/netCDF4, and Zarr, never with a
GWexpy reader or writer. Run it in the frozen B1 environment, save the returned
manifest, and run the public calls described by ``manifest["cases"]`` against
B0/B1. The B1 public outcome (including warnings and exceptions) is the oracle.
An unbounded read followed by ``crop`` is only a *secondary* oracle after that
particular B1 case has been shown to agree in values, dtype, axes, metadata,
and warning/error behavior. The #611 zero-length-entry exception applies only
to a completely disjoint plain HDF5 entry after the parent read succeeds;
single-channel disjoint requests may still raise a parent coverage error.

The manifest is JSON serializable. Float bounds are recorded both as Python
floats for invocation and as binary64 hex for exact review. ``fixture_sha256``
hashes paths and file bytes in sorted order; ``manifest_sha256`` hashes the
other manifest facts. Neither hash depends on the absolute output directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

N_SAMPLES = 4096
CHUNK_SAMPLES = 128
T0 = 12345.125
DT = 0.1
UNIT = "V"
CHANNEL = "signal"
FORMAT_PATHS = {
    "hdf5": "plain.h5",
    "hdf.ndscope": "ndscope.h5",
    "nc": "series.nc",
    "zarr": "series.zarr",
}


def _values(n_samples: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return an exact sample pattern with no random state."""
    indexes = np.arange(n_samples, dtype=np.int32)
    return (indexes % 251 - 125).astype(np.float32)


def _write_plain_hdf5(path: Path, values: np.ndarray[Any, Any], chunk: int) -> None:
    with h5py.File(path, "w") as file:
        _write_hdf5_series(file, CHANNEL, values, chunk, T0)


def _write_hdf5_series(
    file: h5py.File,
    name: str,
    values: np.ndarray[Any, Any],
    chunk: int,
    t0: float,
) -> None:
    dataset = file.create_dataset(
        name,
        data=values,
        chunks=(chunk,),
        compression="gzip",
        compression_opts=1,
    )
    # The native GWpy TimeSeries HDF5 schema, without GWexpy sidecars.
    dataset.attrs.update({"x0": t0, "dx": DT, "xunit": "s", "unit": UNIT, "name": name})


def _write_mixed_hdf5(path: Path, values: np.ndarray[Any, Any], chunk: int) -> None:
    """Make a covered channel and a later channel for #611 key retention."""
    with h5py.File(path, "w") as file:
        _write_hdf5_series(file, "covered", values, chunk, T0)
        _write_hdf5_series(file, "disjoint", values, chunk, T0 + (values.size + 1) * DT)


def _write_ndscope(path: Path, values: np.ndarray[Any, Any], chunk: int) -> None:
    with h5py.File(path, "w") as file:
        group = file.create_group(CHANNEL)
        group.attrs.update({"gps_start": T0, "rate_hz": 1.0 / DT, "unit": UNIT})
        group.create_dataset(
            "raw",
            data=values,
            chunks=(chunk,),
            compression="gzip",
            compression_opts=1,
        )


def _write_netcdf(path: Path, values: np.ndarray[Any, Any], chunk: int) -> None:
    import xarray as xr

    seconds = math.floor(T0)
    nanoseconds = round((T0 - seconds) * 1_000_000_000)
    numerator, denominator = DT.as_integer_ratio()
    dataset = xr.Dataset(
        {CHANNEL: xr.DataArray(values, dims=["sample"], attrs={"units": UNIT})},
        coords={"sample": np.arange(values.size, dtype=np.int64)},
        attrs={
            "gwexpy_netcdf_schema_version": 2,
            "gwexpy_t0_float_hex": T0.hex(),
            "gwexpy_t0_gps_seconds": np.int64(seconds),
            "gwexpy_t0_gps_nanoseconds": np.int32(nanoseconds),
            "gwexpy_dt_numerator": str(numerator),
            "gwexpy_dt_denominator": str(denominator),
            "gwexpy_axis_encoding": "t(i)=t0+i*dt",
        },
    )
    dataset.to_netcdf(
        path,
        engine="netcdf4",
        encoding={CHANNEL: {"zlib": True, "complevel": 1, "chunksizes": (chunk,)}},
    )


def _write_zarr(path: Path, values: np.ndarray[Any, Any], chunk: int) -> None:
    import zarr

    group = zarr.open_group(str(path), mode="w")
    creator = getattr(group, "create_array", None) or group.create_dataset
    array = creator(CHANNEL, data=values, chunks=(chunk,))
    array.attrs.update({"sample_rate": 1.0 / DT, "dt": DT, "t0": T0, "unit": UNIT})


def _corrupt_hdf5_chunk(source: Path, target: Path, dataset: str, index: int) -> None:
    """Damage one gzip chunk using HDF5's direct-chunk API, retaining metadata.

    The damaged payload remains on disk at a known in-range sample. This is
    supported by h5py/HDF5 for chunked datasets. It avoids changing unrelated
    HDF5 structure or relying on byte offsets in the file container.
    """
    shutil.copyfile(source, target)
    with h5py.File(target, "r+") as file:
        node = file[dataset]
        if not isinstance(node, h5py.Dataset):
            raise TypeError(f"Expected dataset at {dataset!r}")
        offset = (index // node.chunks[0] * node.chunks[0],)
        node.id.write_direct_chunk(offset, b"invalid gzip chunk", filter_mask=0)


def _corrupt_zarr_chunk(source: Path, target: Path, index: int, chunk: int) -> str:
    """Damage one on-disk Zarr chunk using its declared one-dimensional key.

    The fixture uses a local directory store. Zarr 3's default key is ``c/N``;
    Zarr 2 uses ``N``. Read metadata rather than guessing a filesystem path,
    and fail if a future layout cannot be identified. The other chunks remain
    untouched for an out-of-range bounded read to test.
    """
    shutil.copytree(source, target)
    array_dir = target / CHANNEL
    v3_metadata = array_dir / "zarr.json"
    v2_metadata = array_dir / ".zarray"
    if v3_metadata.exists():
        metadata = json.loads(v3_metadata.read_text())
        shape = metadata["chunk_grid"]["configuration"]["chunk_shape"]
        encoding = metadata.get("chunk_key_encoding", {"name": "default"})
        chunk_number = str(index // chunk)
        if encoding["name"] == "default":
            separator = encoding.get("configuration", {}).get("separator", "/")
            key = separator.join(("c", chunk_number))
        elif encoding["name"] == "v2":
            key = chunk_number
        else:
            raise ValueError(f"Unsupported Zarr chunk key encoding: {encoding!r}")
    elif v2_metadata.exists():
        metadata = json.loads(v2_metadata.read_text())
        shape = metadata["chunks"]
        key = str(index // chunk)
    else:
        raise ValueError("Zarr array metadata not found for corruption fixture")
    if shape != [chunk]:
        raise ValueError(f"Unexpected Zarr chunk shape: {shape!r}")
    chunk_path = array_dir / key
    if not chunk_path.is_file():
        raise FileNotFoundError(f"Zarr chunk is missing: {chunk_path}")
    chunk_path.write_bytes(b"invalid Zarr chunk payload")
    return chunk_path.relative_to(target).as_posix()


def _window_rows(n_samples: int) -> list[tuple[str, float | None, float | None]]:
    def at(index: int) -> float:
        return T0 + index * DT

    lo, hi = at(0), at(n_samples)
    left, right = at(512), at(516)
    return [
        ("short_intersecting", left, right),
        ("one_sample", at(512), at(513)),
        ("left_exact", left, at(514)),
        ("left_minus_ulp", math.nextafter(left, -math.inf), at(514)),
        ("left_plus_ulp", math.nextafter(left, math.inf), at(514)),
        ("right_exact", at(514), right),
        ("right_minus_ulp", at(514), math.nextafter(right, -math.inf)),
        ("right_plus_ulp", at(514), math.nextafter(right, math.inf)),
        ("half_sample", at(512) + DT / 2, at(514) + DT / 2),
        ("span_start_minus_ulp", math.nextafter(lo, -math.inf), at(2)),
        ("span_start_exact", lo, at(2)),
        ("span_start_plus_ulp", math.nextafter(lo, math.inf), at(2)),
        ("span_end_minus_ulp", at(n_samples - 2), math.nextafter(hi, -math.inf)),
        ("span_end_exact", at(n_samples - 2), hi),
        ("span_end_plus_ulp", at(n_samples - 2), math.nextafter(hi, math.inf)),
        ("end_at_span_start", lo - 2 * DT, lo),
        ("start_at_span_end", hi, hi + 2 * DT),
        ("start_only", at(512), None),
        ("end_only", None, at(514)),
        ("partial_before", lo - 2 * DT, at(2)),
        ("partial_after", at(n_samples - 2), hi + 2 * DT),
        ("disjoint_before", lo - 4 * DT, math.nextafter(lo, -math.inf)),
        ("disjoint_after", math.nextafter(hi, math.inf), hi + 4 * DT),
    ]


def _case(
    name: str,
    format_name: str,
    relative_path: str,
    start: float | None,
    end: float | None,
    *,
    channels: list[str] | None = None,
    pad: float | None = None,
    fault: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"format": format_name}
    if channels is not None:
        kwargs["channels"] = channels
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end
    if pad is not None:
        kwargs["pad"] = pad
    oracle = (
        "b1_public_result_issue611_zero_length"
        if format_name == "hdf5" and name == "mixed_disjoint_zero_length"
        else "b1_public_result"
    )
    return {
        "id": f"{format_name}:{name}",
        "source": relative_path,
        "public_call": "TimeSeriesDict.read(source, **kwargs)",
        "kwargs": kwargs,
        "start_hex": None if start is None else start.hex(),
        "end_hex": None if end is None else end.hex(),
        "oracle": oracle,
        "crop_secondary_oracle": "only_if_b1_characterization_proves_equivalent",
        "fault": fault,
    }


def _file_facts(root: Path) -> tuple[list[dict[str, Any]], str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    entries = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            {"path": relative, "size": path.stat().st_size, "sha256": digest}
        )
        aggregate.update(relative.encode("utf-8") + b"\0" + bytes.fromhex(digest))
    return entries, aggregate.hexdigest()


def write_fixture_set(
    root: str | Path,
    *,
    n_samples: int = N_SAMPLES,
    chunk_samples: int = CHUNK_SAMPLES,
) -> dict[str, Any]:
    """Create B-F1 files and return a hashable, public-call case manifest.

    The output directory must be empty. A case's ``source`` is relative to it.
    The harness should record each B1 public outcome before treating any
    ``full_read.crop`` result as equivalent. Fault cases damage HDF5 gzip
    chunks in plain HDF5, NDScope, and HDF5-backed NetCDF4 files, and one
    Zarr chunk selected by the array's on-disk metadata.
    """
    root = Path(root)
    if (
        n_samples < 1024
        or chunk_samples < 1
        or n_samples % chunk_samples
        or 512 // chunk_samples == 800 // chunk_samples
    ):
        raise ValueError(
            "Require >=1024 samples and a dividing chunk size that separates "
            "samples 512 and 800"
        )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Fixture directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    values = _values(n_samples)
    _write_plain_hdf5(root / FORMAT_PATHS["hdf5"], values, chunk_samples)
    _write_mixed_hdf5(root / "plain-mixed.h5", values, chunk_samples)
    _write_ndscope(root / FORMAT_PATHS["hdf.ndscope"], values, chunk_samples)
    _write_netcdf(root / FORMAT_PATHS["nc"], values, chunk_samples)
    _write_zarr(root / FORMAT_PATHS["zarr"], values, chunk_samples)

    cases = [
        _case(name, fmt, relative, start, end)
        for fmt, relative in FORMAT_PATHS.items()
        for name, start, end in _window_rows(n_samples)
    ]
    cases.append(
        _case(
            "mixed_disjoint_zero_length",
            "hdf5",
            "plain-mixed.h5",
            None,
            T0 + 2 * DT,
        )
    )
    cases.append(
        _case(
            "disjoint_before_with_pad",
            "hdf5",
            FORMAT_PATHS["hdf5"],
            T0 - 4 * DT,
            math.nextafter(T0, -math.inf),
            pad=0.0,
        )
    )
    fault_sources = {
        "hdf5": ("plain-corrupt-selected.h5", "signal"),
        "hdf.ndscope": ("ndscope-corrupt-selected.h5", "signal/raw"),
        "nc": ("netcdf-corrupt-selected.nc", "signal"),
    }
    for fmt, (relative, dataset) in fault_sources.items():
        _corrupt_hdf5_chunk(root / FORMAT_PATHS[fmt], root / relative, dataset, 512)
        cases.append(
            _case(
                "selected_corrupt_chunk",
                fmt,
                relative,
                T0 + 512 * DT,
                T0 + 514 * DT,
                fault="gzip chunk at sample 512 is invalid",
            )
        )
        cases.append(
            _case(
                "out_of_range_corrupt_chunk",
                fmt,
                relative,
                T0 + 800 * DT,
                T0 + 802 * DT,
                fault="gzip chunk at sample 512 is invalid, outside requested window",
            )
        )

    zarr_fault_source = "zarr-corrupt-selected.zarr"
    damaged_zarr_key = _corrupt_zarr_chunk(
        root / FORMAT_PATHS["zarr"], root / zarr_fault_source, 512, chunk_samples
    )
    for name, start_index in (
        ("selected_corrupt_chunk", 512),
        ("out_of_range_corrupt_chunk", 800),
    ):
        cases.append(
            _case(
                name,
                "zarr",
                zarr_fault_source,
                T0 + start_index * DT,
                T0 + (start_index + 2) * DT,
                fault=f"Zarr chunk {damaged_zarr_key!r} at sample 512 is invalid",
            )
        )

    files, fixture_hash = _file_facts(root)
    manifest: dict[str, Any] = {
        "schema": "f1-range-fixtures-v1",
        "source_revision": "1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c",
        "n_samples": n_samples,
        "chunk_samples": chunk_samples,
        "t0_hex": T0.hex(),
        "dt_hex": DT.hex(),
        "source_dtype": str(values.dtype),
        "unit": UNIT,
        "value_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "files": files,
        "fixture_sha256": fixture_hash,
        "cases": cases,
        "corruption_limits": {
            "hdf5": "gzip direct-chunk corruption in plain HDF5 and NDScope",
            "nc": "NetCDF4 engine gzip chunk; requires HDF5-backed netCDF4",
            "zarr": (
                "Local directory store; damaged chunk key is derived from Zarr "
                "2/3 array metadata; other stores/encodings are unsupported"
            ),
        },
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return manifest
