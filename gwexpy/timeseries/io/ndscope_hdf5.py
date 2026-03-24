"""
ndscope HDF5 format reader/writer for gwexpy.

ndscope (https://git.ligo.org/cds/software/ndscope) saves time-series data
in HDF5 files with a specific schema:

    File attrs: t0, window, ...
    ├── <channel_name> (Group)
    │   ├── attrs: rate_hz, gps_start, unit
    │   ├── "raw" (Dataset)            # full-rate data
    │   ├── "mean" (Dataset, optional)  # trend data
    │   ├── "min"  (Dataset, optional)
    │   └── "max"  (Dataset, optional)

This module registers the ``"ndscope-hdf5"`` format so that
``TimeSeries.read()`` / ``TimeSeriesDict.read()`` can auto-detect and
read these files, and ``.write(..., format="ndscope-hdf5")`` can produce
ndscope-compatible output.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .. import TimeSeries, TimeSeriesDict
from ._registration import register_timeseries_format

# Dataset names recognised as ndscope data fields.
_NDSCOPE_DATA_KEYS = frozenset({"raw", "mean", "min", "max"})


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------


def identify_ndscope_hdf5(
    origin: type,
    filepath: str | Path | None,
    fileobj: Any,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Identify an ndscope HDF5 file by its internal structure.

    Returns ``True`` when *filepath* points to an HDF5 file whose root
    contains at least one Group with ``rate_hz`` and ``gps_start``
    attributes and at least one dataset named ``raw``, ``mean``, ``min``,
    or ``max``.
    """
    if filepath is None:
        return False
    path = str(filepath)
    if not (path.lower().endswith(".hdf5") or path.lower().endswith(".h5")):
        return False
    try:
        with h5py.File(path, "r") as f:
            for key in f:
                item = f[key]
                if not isinstance(item, h5py.Group):
                    continue
                attrs = item.attrs
                if "rate_hz" not in attrs or "gps_start" not in attrs:
                    continue
                if any(ds_name in item for ds_name in _NDSCOPE_DATA_KEYS):
                    return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _resolve_source(source: Any) -> str:
    """Extract a file path string from *source*.

    gwpy's I/O registry may pass an open file object rather than a plain
    path string.  This helper normalises both cases.
    """
    if isinstance(source, (str, Path)):
        return str(source)
    # file-like object (e.g. _io.FileIO opened by gwpy)
    if hasattr(source, "name"):
        return str(source.name)
    return str(source)


def read_timeseriesdict_ndscope_hdf5(
    source: str | Path,
    *,
    channels: Iterable[str] | None = None,
    start: float | None = None,
    end: float | None = None,
    **kwargs: Any,
) -> TimeSeriesDict:
    """Read an ndscope HDF5 file into a `TimeSeriesDict`.

    Parameters
    ----------
    source : str or Path
        Path to the HDF5 file.
    channels : iterable of str, optional
        Channel names to read.  If ``None``, all channels are read.
    start : float, optional
        GPS start time for cropping.
    end : float, optional
        GPS end time for cropping.

    Returns
    -------
    TimeSeriesDict
    """
    wanted = set(channels) if channels is not None else None
    out = TimeSeriesDict()

    with h5py.File(_resolve_source(source), "r") as f:
        for grp_name in f:
            item = f[grp_name]
            if not isinstance(item, h5py.Group):
                continue
            attrs = item.attrs
            if "rate_hz" not in attrs or "gps_start" not in attrs:
                continue

            # Skip if not in the requested channel set.
            if wanted is not None and grp_name not in wanted:
                continue

            sample_rate = float(attrs["rate_hz"])
            gps_start = float(attrs["gps_start"])
            unit = str(attrs.get("unit", ""))

            ds_names = sorted(set(item.keys()) & _NDSCOPE_DATA_KEYS)
            if not ds_names:
                continue

            for ds_name in ds_names:
                data = np.asarray(item[ds_name])
                # Build the channel key: append .stat for trend datasets
                if ds_name == "raw" and len(ds_names) == 1:
                    ch_key = grp_name
                else:
                    ch_key = f"{grp_name}.{ds_name}"

                ts = TimeSeries(
                    data,
                    sample_rate=sample_rate,
                    t0=gps_start,
                    name=ch_key,
                    channel=ch_key,
                    unit=unit or None,
                )

                # Crop to [start, end] if requested.
                if start is not None or end is not None:
                    s = max(start, ts.span[0]) if start is not None else ts.span[0]
                    e = min(end, ts.span[1]) if end is not None else ts.span[1]
                    if s < e:
                        ts = ts.crop(s, e)
                    else:
                        continue

                out[ch_key] = ts

    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_timeseriesdict_ndscope_hdf5(
    tsdict: TimeSeriesDict,
    target: str | Path,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Write a `TimeSeriesDict` in ndscope-compatible HDF5 format.

    Parameters
    ----------
    tsdict : TimeSeriesDict
        Data to write.
    target : str or Path
        Output file path.
    overwrite : bool, optional
        If ``True``, overwrite an existing file.  Default: ``False``.
    """
    mode = "w" if overwrite else "w-"
    with h5py.File(_resolve_source(target), mode) as f:
        groups: dict[str, h5py.Group] = {}
        for key, ts in tsdict.items():
            name = str(key)
            # Determine group name and dataset name.
            # If the channel key contains a ".raw"/".mean"/".min"/".max"
            # suffix, split it to reconstruct the ndscope group structure.
            ds_name = "raw"
            grp_name = name
            for suffix in ("raw", "mean", "min", "max"):
                if name.endswith(f".{suffix}"):
                    grp_name = name[: -(len(suffix) + 1)]
                    ds_name = suffix
                    break

            if grp_name not in groups:
                grp = f.create_group(grp_name)
                groups[grp_name] = grp
            grp = groups[grp_name]

            grp.create_dataset(ds_name, data=ts.value)
            grp.attrs["rate_hz"] = float(ts.sample_rate.value)
            grp.attrs["gps_start"] = float(ts.t0.value)
            grp.attrs["unit"] = str(ts.unit) if ts.unit else ""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_timeseries_format(
    "ndscope-hdf5",
    reader_dict=read_timeseriesdict_ndscope_hdf5,
    writer_dict=write_timeseriesdict_ndscope_hdf5,
    magic_identifier=identify_ndscope_hdf5,
    extension="hdf5",
)
