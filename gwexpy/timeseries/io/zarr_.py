"""
Zarr reader/writer for gwexpy.

Convention
----------
A Zarr store is a group where each array represents one channel.
Per-array attributes ``sample_rate`` (Hz) and ``t0`` (GPS seconds) are
used to reconstruct the time axis.  If ``sample_rate`` is absent the
inverse of ``dt`` is tried; failing that, 1 Hz is assumed.

Directory stores, zip stores, and any other backend supported by the
``zarr`` library can be used as *source* / *target*.
"""

from __future__ import annotations

import numpy as np
from gwpy.io.registry import default_registry as io_registry

from gwexpy.io.utils import apply_unit, filter_by_channels, set_provenance

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _import_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "zarr is required for reading/writing Zarr stores. "
            "Install with `pip install zarr`."
        ) from exc
    return zarr


# -- Reader --------------------------------------------------------------------


def read_timeseriesdict_zarr(
    source,
    *,
    channels=None,
    unit=None,
    **kwargs,
) -> TimeSeriesDict:
    """Read a Zarr store into a TimeSeriesDict.

    Parameters
    ----------
    source : str, path-like, or zarr store
        Path to the ``.zarr`` directory (or any zarr-compatible store).
    channels : iterable of str, optional
        Array names to read.  If *None*, all arrays in the root group
        are loaded.
    unit : str, optional
        Physical unit override applied to every channel.
    """
    zarr = _import_zarr()

    store = zarr.open_group(str(source), mode="r", **kwargs)

    tsd = TimeSeriesDict()

    keys = list(channels) if channels else list(store.keys())

    for key in keys:
        if key not in store:
            continue
        arr = store[key]
        # Only load arrays (skip sub-groups)
        if not hasattr(arr, "shape"):
            continue

        attrs = dict(arr.attrs)
        t0 = float(attrs.get("t0", 0.0))

        sr = attrs.get("sample_rate")
        if sr is not None:
            sample_rate = float(sr)
        else:
            dt_val = attrs.get("dt")
            sample_rate = 1.0 / float(dt_val) if dt_val else 1.0

        arr_unit = unit or attrs.get("unit") or attrs.get("units")

        ts = TimeSeries(
            np.asarray(arr[:], dtype=np.float64),
            t0=t0,
            sample_rate=sample_rate,
            name=key,
            channel=key,
        )
        ts = apply_unit(ts, arr_unit) if arr_unit else ts
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
    tsd = read_timeseriesdict_zarr(source, **kwargs)
    if not tsd:
        raise ValueError("No arrays found in Zarr store")
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_zarr(source, **kwargs) -> TimeSeriesMatrix:
    tsd = read_timeseriesdict_zarr(source, **kwargs)
    return tsd.to_matrix()


# -- Writer --------------------------------------------------------------------


def write_timeseriesdict_zarr(tsd, target, **kwargs):
    """Write a TimeSeriesDict to a Zarr store.

    Each channel is written as an array in the root group with
    ``sample_rate``, ``t0``, and ``unit`` stored as attributes.
    """
    zarr = _import_zarr()

    if not tsd:
        raise ValueError("Cannot write empty TimeSeriesDict to Zarr")

    store = zarr.open_group(str(target), mode="w", **kwargs)

    for key, ts in tsd.items():
        data = np.asarray(ts.value, dtype=np.float64)
        arr = store.create_dataset(key, data=data, overwrite=True)
        arr.attrs["sample_rate"] = float(ts.sample_rate.value)
        arr.attrs["t0"] = float(ts.t0.value)
        arr.attrs["dt"] = float(ts.dt.value)
        if ts.unit is not None:
            arr.attrs["unit"] = str(ts.unit)


def write_timeseries_zarr(ts, target, **kwargs):
    write_timeseriesdict_zarr(
        TimeSeriesDict({ts.name or "channel_0": ts}), target, **kwargs
    )


# -- Registration --------------------------------------------------------------

io_registry.register_reader("zarr", TimeSeriesDict, read_timeseriesdict_zarr, force=True)
io_registry.register_reader("zarr", TimeSeries, read_timeseries_zarr, force=True)
io_registry.register_reader("zarr", TimeSeriesMatrix, read_timeseriesmatrix_zarr, force=True)

io_registry.register_writer("zarr", TimeSeriesDict, write_timeseriesdict_zarr, force=True)
io_registry.register_writer("zarr", TimeSeries, write_timeseries_zarr, force=True)

io_registry.register_identifier(
    "zarr",
    TimeSeriesDict,
    lambda *args, **kwargs: str(args[1]).lower().endswith(".zarr"),
)
io_registry.register_identifier(
    "zarr",
    TimeSeries,
    lambda *args, **kwargs: str(args[1]).lower().endswith(".zarr"),
)
