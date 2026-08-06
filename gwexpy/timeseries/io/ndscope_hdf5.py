"""ndscope HDF5 format reader/writer for gwexpy.

ndscope (https://git.ligo.org/cds/software/ndscope) saves time-series data
in HDF5 files with a specific schema:

    File attrs: t0, window, ...
    ├── <channel_name> (Group)
    │   ├── attrs: rate_hz, gps_start, unit
    │   ├── "raw" (Dataset)            # full-rate data
    │   ├── "mean" (Dataset, optional)  # trend data
    │   ├── "min"  (Dataset, optional)
    │   └── "max"  (Dataset, optional)

This module registers the canonical ``"hdf.ndscope"`` format so that
``TimeSeries.read()`` / ``TimeSeriesDict.read()`` can auto-detect and
read these files, and ``.write(..., format="hdf.ndscope")`` can produce
ndscope-compatible output. Backward-compatible aliases are also registered
through the shared registration helper.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .. import TimeSeries, TimeSeriesDict
from ._multi import expand_multi_source, read_multi_dict
from ._registration import register_timeseries_format

# Dataset names recognised as ndscope data fields.
_NDSCOPE_DATA_KEYS = frozenset({"raw", "mean", "min", "max"})
_NDSCOPE_SAMPLE_RATE_KEYS = ("rate_hz", "sample_rate")


def _sample_rate_from_attrs(attrs: Any, *, group_name: str) -> float:
    """Return a validated NDScope sampling rate from group attributes.

    ``rate_hz`` is the canonical attribute written by gwexpy.  External
    NDScope files may instead use ``sample_rate``; accept it for reading while
    rejecting missing, invalid, or conflicting timing metadata explicitly.

    Raises
    ------
    ValueError
        If the group carries neither attribute, or if the value present is
        not a positive finite rate, or if both are present and disagree.
        Callers must only invoke this for data-bearing groups: a group with
        no NDScope dataset legitimately has no sampling rate to report.

    """
    values: dict[str, float] = {}
    for key in _NDSCOPE_SAMPLE_RATE_KEYS:
        if key not in attrs:
            continue
        try:
            value = float(attrs[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"NDScope group {group_name!r} has invalid {key}={attrs[key]!r}; "
                "expected a positive finite sampling rate in Hz"
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"NDScope group {group_name!r} has invalid {key}={value!r}; "
                "expected a positive finite sampling rate in Hz"
            )
        values[key] = value

    if not values:
        raise ValueError(
            f"NDScope group {group_name!r} contains data but has no "
            "sampling-rate metadata; expected one of: "
            f"{', '.join(_NDSCOPE_SAMPLE_RATE_KEYS)}"
        )

    if len(values) == 2 and not np.isclose(
        values["rate_hz"], values["sample_rate"], rtol=1e-12, atol=0.0
    ):
        raise ValueError(
            f"NDScope group {group_name!r} has conflicting sampling-rate "
            f"metadata: rate_hz={values['rate_hz']!r}, "
            f"sample_rate={values['sample_rate']!r}"
        )
    # `values` is non-empty here, so one of the two keys is always present.
    return values["rate_hz"] if "rate_hz" in values else values["sample_rate"]


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
    contains at least one Group with a ``gps_start`` attribute plus at least
    one dataset named ``raw``, ``mean``, ``min``, or ``max``.

    Sampling-rate metadata is deliberately *not* part of this test.  An
    NDScope file whose groups all lack ``rate_hz``/``sample_rate`` is still an
    NDScope file -- it is a malformed one.  Requiring the rate here would
    de-select this reader for exactly those files, so ``TimeSeriesDict.read()``
    would fall through to another format instead of surfacing the reader's
    explicit error, silently reintroducing the channel loss this identifier
    is meant to help catch.  Validity of the rate is the reader's contract;
    see :func:`_sample_rate_from_attrs`.
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
                if "gps_start" not in item.attrs:
                    continue
                if any(ds_name in item for ds_name in _NDSCOPE_DATA_KEYS):
                    return True
    except (OSError, KeyError, AttributeError, TypeError):
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
    source : str, Path, or list of str/Path
        Path to the HDF5 file, or a list of paths.  When a list is
        given, channels found in several files are concatenated along
        the time axis and channels unique to one file are merged in.
    channels : iterable of str, optional
        Channel names to read.  If ``None``, all channels are read.
    start : float, optional
        GPS start time for cropping.
    end : float, optional
        GPS end time for cropping.
    **kwargs
        No dataset creation options are supported in v0.1.13.  Unknown
        keywords raise before the target is opened.

    Returns
    -------
    TimeSeriesDict

    """
    multi = expand_multi_source(source)
    if multi is not None:
        return read_multi_dict(
            read_timeseriesdict_ndscope_hdf5,
            multi,
            "hdf.ndscope",
            channels=channels,
            start=start,
            end=end,
            **kwargs,
        )

    wanted = set(channels) if channels is not None else None
    out = TimeSeriesDict()

    with h5py.File(_resolve_source(source), "r") as f:
        for grp_name in f:
            item = f[grp_name]
            if not isinstance(item, h5py.Group):
                continue
            attrs = item.attrs
            if "gps_start" not in attrs:
                continue

            # Skip if not in the requested channel set.
            if wanted is not None and grp_name not in wanted:
                continue

            # Establish that the group is data-bearing *before* requiring
            # timing metadata: a group with no NDScope dataset carries no
            # samples, so a missing sampling rate on it is not an error.
            ds_names = sorted(set(item.keys()) & _NDSCOPE_DATA_KEYS)
            if not ds_names:
                continue

            # A data-bearing group with no sampling rate is an error, not a
            # silent skip: dropping it would return a TimeSeriesDict missing
            # a channel with no indication anything went wrong.
            sample_rate = _sample_rate_from_attrs(attrs, group_name=grp_name)
            gps_start = float(attrs["gps_start"])
            unit = str(attrs.get("unit", ""))

            for ds_name in ds_names:
                data = np.asarray(item[ds_name])
                # "raw" always maps to the bare channel name (grp_name).
                # Trend datasets (mean/min/max) are always suffixed so they
                # are unambiguous regardless of what other datasets are present.
                if ds_name == "raw":
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
    **kwargs
        Additional keyword arguments reserved for compatibility with I/O dispatch.

    """
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported NDScope writer keyword arguments: {names}")

    mode = "w" if overwrite else "w-"
    # group_meta tracks the first-seen metadata for each group so that
    # subsequent series in the same group can be validated for consistency.
    group_meta: dict[str, dict[str, float | str]] = {}
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

            rate_hz = float(ts.sample_rate.value)
            gps_start = float(ts.t0.value)
            unit = str(ts.unit) if ts.unit else ""

            if grp_name not in groups:
                grp = f.create_group(grp_name)
                groups[grp_name] = grp
                group_meta[grp_name] = {
                    "rate_hz": rate_hz,
                    "gps_start": gps_start,
                    "unit": unit,
                }
                grp.attrs["rate_hz"] = rate_hz
                grp.attrs["gps_start"] = gps_start
                grp.attrs["unit"] = unit
            else:
                # Validate consistency with the first series in this group.
                meta = group_meta[grp_name]
                if rate_hz != meta["rate_hz"]:
                    raise ValueError(
                        f"Inconsistent sample_rate for group {grp_name!r}: "
                        f"expected {meta['rate_hz']} Hz, got {rate_hz} Hz"
                    )
                if gps_start != meta["gps_start"]:
                    raise ValueError(
                        f"Inconsistent gps_start for group {grp_name!r}: "
                        f"expected {meta['gps_start']}, got {gps_start}"
                    )
                if unit != meta["unit"]:
                    raise ValueError(
                        f"Inconsistent unit for group {grp_name!r}: "
                        f"expected {meta['unit']!r}, got {unit!r}"
                    )

            grp = groups[grp_name]
            grp.create_dataset(ds_name, data=ts.value)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_timeseries_format(
    "hdf.ndscope",
    aliases=("ndscope-hdf5", "ndscope_hdf5", "ndscopehdf5"),
    reader_dict=read_timeseriesdict_ndscope_hdf5,
    writer_dict=write_timeseriesdict_ndscope_hdf5,
    magic_identifier=identify_ndscope_hdf5,
    extension="hdf5",
)
