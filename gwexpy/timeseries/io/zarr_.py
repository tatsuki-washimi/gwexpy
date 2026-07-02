"""Zarr reader/writer for gwexpy.

Convention
----------
A Zarr store is a group where each array represents one channel.
Per-array attributes ``sample_rate`` (Hz) and ``t0`` (GPS seconds) are
used to reconstruct the time axis. If ``sample_rate`` is absent the
inverse of ``dt`` is tried. If both are absent, reading fails with a
clear error instead of silently assuming 1 Hz.

Directory stores, zip stores, and any other backend supported by the
``zarr`` library can be used as *source* / *target*.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict

import numpy as np

from gwexpy.io.utils import (
    apply_unit,
    ensure_dependency,
    filter_by_channels,
    set_provenance,
)

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from ._multi import expand_multi_source, read_multi_dict
from ._registration import register_timeseries_format

_MATRIX_ARRAY_PREFIX = "__gwexpy_matrix__"


def _import_zarr():
    try:
        zarr = ensure_dependency("zarr", extra="zarr")
    except ImportError as exc:
        raise ImportError(
            "zarr is required for reading/writing Zarr stores. "
            "Install with `pip install 'gwexpy[zarr]'`."
        ) from exc
    return zarr


def _to_json_native(value):
    """Convert scalar-like labels to JSON-serializable native values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return [_to_json_native(item) for item in value]
    if not isinstance(value, (bool, int, float, str, type(None))):
        return str(value)
    return value


def _encode_zarr_array_name(key) -> str:
    """Return a Zarr-safe array name for a TimeSeriesDict key."""
    if isinstance(key, tuple):
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
        return f"{_MATRIX_ARRAY_PREFIX}{digest}"
    return str(key)


def _decode_zarr_key(raw):
    """Deserialize a key stored in Zarr attrs; fall back to string labels."""
    if raw is None:
        return None

    def _normalize(decoded):
        if isinstance(decoded, list):
            return tuple(_normalize(item) for item in decoded)
        return decoded

    try:
        result = json.loads(raw)
        return _normalize(result)
    except (json.JSONDecodeError, TypeError, ValueError):
        return str(raw)


# -- Reader --------------------------------------------------------------------


def _coerce_sample_rate_from_attrs_or_override(
    attrs,
    *,
    channel_name,
    sample_rate_override=None,
    dt_override=None,
) -> float:
    """Resolve sample rate from metadata or explicit override.

    The recovery path is intentionally explicit: callers may provide either
    ``sample_rate_override`` or ``dt_override`` when reading legacy stores that
    lack timing metadata, but never both.
    """
    if sample_rate_override is not None and dt_override is not None:
        raise ValueError(
            "Pass only one of sample_rate_override or dt_override when reading "
            f"Zarr channel '{channel_name}'."
        )

    sr = attrs.get("sample_rate")
    if sr is not None:
        sample_rate = float(sr)
        if sample_rate <= 0:
            raise ValueError(
                f"Zarr channel '{channel_name}' has invalid sample_rate={sr!r}. "
                "Expected a positive value in Hz."
            )
        return sample_rate

    dt_val = attrs.get("dt")
    if dt_val is not None:
        dt = float(dt_val)
        if dt <= 0:
            raise ValueError(
                f"Zarr channel '{channel_name}' has invalid dt={dt_val!r}. "
                "Expected a positive time spacing in seconds."
            )
        return 1.0 / dt

    if sample_rate_override is not None:
        sample_rate = float(sample_rate_override)
        if sample_rate <= 0:
            raise ValueError(
                f"sample_rate_override must be positive; got {sample_rate_override!r}."
            )
        return sample_rate

    if dt_override is not None:
        dt = float(dt_override)
        if dt <= 0:
            raise ValueError(f"dt_override must be positive; got {dt_override!r}.")
        return 1.0 / dt

    raise ValueError(
        f"Zarr channel '{channel_name}' is missing timing metadata. "
        "Expected per-array 'sample_rate' (Hz) or 'dt' (seconds) attributes. "
        "To read legacy data intentionally, add that metadata to the store or "
        "pass sample_rate_override=... or dt_override=... to the Zarr reader."
    )


def _series_from_zarr_array(
    key,
    arr,
    *,
    unit=None,
    sample_rate_override=None,
    dt_override=None,
) -> TimeSeries:
    attrs = dict(arr.attrs)
    t0 = float(attrs.get("t0", 0.0))

    sample_rate = _coerce_sample_rate_from_attrs_or_override(
        attrs,
        channel_name=key,
        sample_rate_override=sample_rate_override,
        dt_override=dt_override,
    )

    arr_unit = unit or attrs.get("unit") or attrs.get("units")

    ts = TimeSeries(
        np.asarray(arr[:], dtype=np.float64),
        t0=t0,
        sample_rate=sample_rate,
        name=str(key),
        channel=str(key),
    )
    return apply_unit(ts, arr_unit) if arr_unit else ts


def read_timeseriesdict_zarr(
    source,
    *,
    channels=None,
    unit=None,
    sample_rate_override=None,
    dt_override=None,
    **kwargs,
) -> TimeSeriesDict:
    """Read a Zarr store into a TimeSeriesDict.

    Parameters
    ----------
    source : str, path-like, zarr store, or list thereof
        Path to the ``.zarr`` directory (or any zarr-compatible store),
        or a list of stores.  When a list is given, channels found in
        several stores are concatenated along the time axis and
        channels unique to one store are merged in.
    channels : iterable of str, optional
        Array names to read.  If *None*, all arrays in the root group
        are loaded.
    unit : str, optional
        Physical unit override applied to every channel.
    sample_rate_override : float, optional
        Explicit sample-rate recovery value in Hz for legacy stores that lack
        per-array ``sample_rate`` and ``dt`` metadata. Mutually exclusive with
        ``dt_override``.
    dt_override : float, optional
        Explicit sample-spacing recovery value in seconds for legacy stores
        that lack per-array timing metadata. Mutually exclusive with
        ``sample_rate_override``.
    **kwargs
        Additional keyword arguments forwarded to ``zarr.open_group``.

    """
    multi = expand_multi_source(source)
    if multi is not None:
        return read_multi_dict(
            read_timeseriesdict_zarr,
            multi,
            "zarr",
            channels=channels,
            unit=unit,
            sample_rate_override=sample_rate_override,
            dt_override=dt_override,
            **kwargs,
        )

    zarr = _import_zarr()

    # Only coerce to str for path-like objects; pass store objects through
    # directly so that in-memory / remote stores work unchanged.
    if isinstance(source, (str, os.PathLike)):
        source = str(source)
    store = zarr.open_group(source, mode="r", **kwargs)

    tsd = TimeSeriesDict()

    keys = list(channels) if channels else list(store.keys())

    for key in keys:
        if key not in store:
            continue
        arr = store[key]
        # Only load arrays (skip sub-groups)
        if not hasattr(arr, "shape"):
            continue

        ts = _series_from_zarr_array(
            key,
            arr,
            unit=unit,
            sample_rate_override=sample_rate_override,
            dt_override=dt_override,
        )
        tsd[key] = ts

    if channels:
        tsd = TimeSeriesDict(filter_by_channels(tsd, channels))

    set_provenance(
        tsd,
        {
            "format": "zarr",
            "channels": list(tsd.keys()),
            "unit_source": "override" if unit else "file",
        },
    )
    return tsd


def read_timeseries_zarr(source, **kwargs) -> TimeSeries:
    """Read a Zarr store and return the first channel."""
    tsd = read_timeseriesdict_zarr(source, **kwargs)
    if not tsd:
        raise ValueError("No arrays found in Zarr store")
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_zarr(
    source,
    *,
    channels=None,
    unit=None,
    sample_rate_override=None,
    dt_override=None,
    **kwargs,
) -> TimeSeriesMatrix:
    """Read a Zarr store and convert its channels to a matrix."""
    if isinstance(source, (list, tuple)):
        sources = list(source)
        if not sources:
            raise ValueError("no Zarr stores provided")
        matrices = [
            read_timeseriesmatrix_zarr(
                s,
                channels=channels,
                unit=unit,
                sample_rate_override=sample_rate_override,
                dt_override=dt_override,
                **kwargs,
            )
            for s in sources
        ]
        merged = matrices[0]
        for mat in matrices[1:]:
            merged = merged.append(mat, inplace=False, gap="pad", pad=np.nan)
        return merged

    zarr = _import_zarr()

    if isinstance(source, (str, os.PathLike)):
        source = str(source)
    store = zarr.open_group(source, mode="r", **kwargs)

    keys = list(channels) if channels else list(store.keys())
    matrix_entries = []
    saw_matrix_attrs = False
    required_attrs = {
        "gwexpy_row_key",
        "gwexpy_col_key",
        "gwexpy_key_format",
        "gwexpy_row_index",
        "gwexpy_col_index",
    }
    for key in keys:
        if key not in store:
            continue
        arr = store[key]
        if not hasattr(arr, "shape"):
            continue

        attrs = dict(arr.attrs)
        if any(name in attrs for name in required_attrs):
            saw_matrix_attrs = True
            missing = sorted(required_attrs.difference(attrs))
            if missing:
                raise ValueError(
                    f"Zarr matrix channel '{key}' is missing required "
                    f"matrix metadata attributes: {missing}."
                )

        row_raw = attrs.get("gwexpy_row_key")
        col_raw = attrs.get("gwexpy_col_key")
        if row_raw is None or col_raw is None:
            continue

        if attrs.get("gwexpy_key_format") == "json":
            row_key = _decode_zarr_key(row_raw)
            col_key = _decode_zarr_key(col_raw)
        else:
            row_key = str(row_raw)
            col_key = str(col_raw)

        ts = _series_from_zarr_array(
            key,
            arr,
            unit=unit,
            sample_rate_override=sample_rate_override,
            dt_override=dt_override,
        )
        row_index = attrs.get("gwexpy_row_index")
        col_index = attrs.get("gwexpy_col_index")
        if row_index is None or col_index is None:
            raise ValueError(
                f"Zarr matrix channel '{key}' is missing required matrix index "
                "metadata."
            )
        source_unit = attrs.get("unit") or attrs.get("units")
        matrix_entries.append(
            (row_key, col_key, int(row_index), int(col_index), source_unit, ts)
        )

    if not matrix_entries:
        if saw_matrix_attrs:
            raise ValueError("No complete Zarr matrix channels found in store.")
        tsd = read_timeseriesdict_zarr(
            source,
            channels=channels,
            unit=unit,
            sample_rate_override=sample_rate_override,
            dt_override=dt_override,
            **kwargs,
        )
        return tsd.to_matrix()

    seen_cells = set()
    row_positions: dict[object, int] = {}
    col_positions: dict[object, int] = {}
    for row_key, col_key, row_index, col_index, _, _ in matrix_entries:
        cell = (row_key, col_key)
        if cell in seen_cells:
            raise ValueError(f"duplicate Zarr matrix cell {cell!r}.")
        seen_cells.add(cell)

        existing_row_index = row_positions.setdefault(row_key, row_index)
        if existing_row_index != row_index:
            raise ValueError(f"Conflicting row index for Zarr matrix key {row_key!r}.")
        existing_col_index = col_positions.setdefault(col_key, col_index)
        if existing_col_index != col_index:
            raise ValueError(f"Conflicting column index for Zarr matrix key {col_key!r}.")

    row_index_values = sorted(row_positions.values())
    col_index_values = sorted(col_positions.values())
    if row_index_values != list(range(len(row_index_values))):
        raise ValueError("Zarr matrix row indices must be dense and zero-based.")
    if col_index_values != list(range(len(col_index_values))):
        raise ValueError("Zarr matrix column indices must be dense and zero-based.")

    row_keys = [
        row for row, _ in sorted(row_positions.items(), key=lambda item: item[1])
    ]
    col_keys = [
        col for col, _ in sorted(col_positions.items(), key=lambda item: item[1])
    ]

    expected_entries = len(row_keys) * len(col_keys)
    if len(matrix_entries) != expected_entries:
        raise ValueError(
            "Zarr matrix store is incomplete: expected "
            f"{expected_entries} cells from row/column metadata, found "
            f"{len(matrix_entries)}."
        )

    first_source_unit = matrix_entries[0][4]
    first = matrix_entries[0][5]
    n_samples = len(first)
    first_t0 = float(first.t0.value)
    first_sample_rate = float(first.sample_rate.value)
    first_unit = str(first_source_unit) if first_source_unit is not None else None

    data = np.full(
        (len(row_keys), len(col_keys), n_samples),
        np.nan,
        dtype=np.float64,
    )
    row_to_index = {row_key: i for i, row_key in enumerate(row_keys)}
    col_to_index = {col_key: j for j, col_key in enumerate(col_keys)}
    for row_key, col_key, _, _, source_unit, ts in matrix_entries:
        if len(ts) != n_samples:
            raise ValueError("Zarr matrix channels must have matching sample counts")
        if not np.isclose(float(ts.t0.value), first_t0, rtol=1e-12, atol=1e-12):
            raise ValueError("Zarr matrix channels must have matching t0 values")
        if not np.isclose(
            float(ts.sample_rate.value),
            first_sample_rate,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "Zarr matrix channels must have matching sample_rate values"
            )
        ts_unit = str(source_unit) if source_unit is not None else None
        if ts_unit != first_unit:
            raise ValueError("Zarr matrix channels must have matching units")
        i = row_to_index[row_key]
        j = col_to_index[col_key]
        data[i, j, :] = np.asarray(ts.value, dtype=np.float64)

    matrix = TimeSeriesMatrix(
        data,
        t0=first_t0,
        sample_rate=first_sample_rate,
        unit=first.unit,
    )
    if row_keys != list(matrix.row_keys()) or col_keys != list(matrix.col_keys()):
        from gwexpy.types.metadata import MetaData, MetaDataDict

        matrix.rows = MetaDataDict(
            OrderedDict((key, MetaData()) for key in row_keys),
            expected_size=len(row_keys),
            key_prefix="row",
        )
        matrix.cols = MetaDataDict(
            OrderedDict((key, MetaData()) for key in col_keys),
            expected_size=len(col_keys),
            key_prefix="col",
        )
    return matrix


# -- Writer --------------------------------------------------------------------


def write_timeseriesdict_zarr(tsd, target, **kwargs):
    """Write a TimeSeriesDict to a Zarr store.

    Each channel is written as an array in the root group with
    ``sample_rate``, ``t0``, and ``unit`` stored as attributes.
    """
    zarr = _import_zarr()

    if not tsd:
        raise ValueError("Cannot write empty TimeSeriesDict to Zarr")

    if isinstance(target, (str, os.PathLike)):
        target = str(target)
    store = zarr.open_group(target, mode="w", **kwargs)

    row_positions: OrderedDict[object, int] = OrderedDict()
    col_positions: OrderedDict[object, int] = OrderedDict()
    for key in tsd:
        if isinstance(key, tuple) and len(key) == 2:
            row_positions.setdefault(key[0], len(row_positions))
            col_positions.setdefault(key[1], len(col_positions))

    for key, ts in tsd.items():
        data = np.asarray(ts.value, dtype=np.float64)
        # zarr>=3 uses create_array(data=...) and removed create_dataset.
        # zarr<3 has create_dataset; fall back for compatibility.
        creator = getattr(store, "create_array", None) or store.create_dataset
        arr = creator(_encode_zarr_array_name(key), data=data, overwrite=True)
        arr.attrs["sample_rate"] = float(ts.sample_rate.value)
        arr.attrs["t0"] = float(ts.t0.value)
        arr.attrs["dt"] = float(ts.dt.value)
        if ts.unit is not None:
            arr.attrs["unit"] = str(ts.unit)
        if isinstance(key, tuple) and len(key) == 2:
            arr.attrs["gwexpy_row_key"] = json.dumps(_to_json_native(key[0]))
            arr.attrs["gwexpy_col_key"] = json.dumps(_to_json_native(key[1]))
            arr.attrs["gwexpy_key_format"] = "json"
            arr.attrs["gwexpy_row_index"] = row_positions[key[0]]
            arr.attrs["gwexpy_col_index"] = col_positions[key[1]]


def write_timeseries_zarr(ts, target, **kwargs):
    """Write one ``TimeSeries`` to a Zarr store."""
    write_timeseriesdict_zarr(
        TimeSeriesDict({ts.name or "channel_0": ts}), target, **kwargs
    )


def write_timeseriesmatrix_zarr(tsm, target, **kwargs):
    """Write a TimeSeriesMatrix to a Zarr store preserving row/column labels."""
    zarr = _import_zarr()

    if isinstance(target, (str, os.PathLike)):
        target = str(target)
    store = zarr.open_group(target, mode="w", **kwargs)

    row_keys = list(tsm.row_keys())
    col_keys = list(tsm.col_keys())
    if not row_keys or not col_keys:
        raise ValueError("Cannot write empty TimeSeriesMatrix to Zarr")

    creator = getattr(store, "create_array", None) or store.create_dataset
    first = tsm[0, 0]
    n_samples = len(first)
    first_t0 = float(first.t0.value)
    first_sample_rate = float(first.sample_rate.value)
    first_unit = str(first.unit) if first.unit is not None else None

    for i, row_key in enumerate(row_keys):
        for j, col_key in enumerate(col_keys):
            ts = tsm[i, j]
            if len(ts) != n_samples:
                raise ValueError("Zarr matrix channels must have matching sample counts")
            if not np.isclose(float(ts.t0.value), first_t0, rtol=1e-12, atol=1e-12):
                raise ValueError("Zarr matrix channels must have matching t0 values")
            if not np.isclose(
                float(ts.sample_rate.value),
                first_sample_rate,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    "Zarr matrix channels must have matching sample_rate values"
                )
            ts_unit = str(ts.unit) if ts.unit is not None else None
            if ts_unit != first_unit:
                raise ValueError("Zarr matrix channels must have matching units")
            data = np.asarray(ts.value, dtype=np.float64)
            array_key = (row_key, col_key)
            arr = creator(_encode_zarr_array_name(array_key), data=data, overwrite=True)
            arr.attrs["sample_rate"] = float(ts.sample_rate.value)
            arr.attrs["t0"] = float(ts.t0.value)
            arr.attrs["dt"] = float(ts.dt.value)
            if ts.unit is not None:
                arr.attrs["unit"] = str(ts.unit)
            arr.attrs["gwexpy_row_key"] = json.dumps(_to_json_native(row_key))
            arr.attrs["gwexpy_col_key"] = json.dumps(_to_json_native(col_key))
            arr.attrs["gwexpy_key_format"] = "json"
            arr.attrs["gwexpy_row_index"] = i
            arr.attrs["gwexpy_col_index"] = j


# -- Registration --------------------------------------------------------------

register_timeseries_format(
    "zarr",
    reader_dict=read_timeseriesdict_zarr,
    reader_single=read_timeseries_zarr,
    reader_matrix=read_timeseriesmatrix_zarr,
    writer_dict=write_timeseriesdict_zarr,
    writer_single=write_timeseries_zarr,
    writer_matrix=write_timeseriesmatrix_zarr,
    extension="zarr",
)
