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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from gwexpy.io.utils import _reject_timezone_reinterpretation

from .. import TimeSeries, TimeSeriesDict
from ._multi import expand_multi_source, read_multi_dict
from ._registration import register_timeseries_format

# Dataset names recognised as ndscope data fields.
_NDSCOPE_DATA_KEYS = frozenset({"raw", "mean", "min", "max"})
_NDSCOPE_SAMPLE_RATE_KEYS = ("rate_hz", "sample_rate")
_NDSCOPE_DATASET_OPTION_KEYS = frozenset(
    {"chunks", "compression", "compression_opts", "shuffle", "fletcher32"}
)


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
    timezone = kwargs.pop("timezone", None)
    _reject_timezone_reinterpretation("hdf.ndscope", timezone, None)

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


@dataclass(frozen=True)
class _NDScopeWriteEntry:
    """Validated output mapping used after the writer preflight."""

    group_name: str
    dataset_name: str
    data: np.ndarray
    rate_hz: float
    gps_start: float
    unit: str


def _normalise_ndscope_dataset_options(
    dataset_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy and validate dataset options that do not depend on data shape."""
    if dataset_options is None:
        return {}
    if not isinstance(dataset_options, Mapping):
        raise TypeError("dataset_options must be a Mapping or None")

    options = dict(dataset_options)
    unknown = [key for key in options if key not in _NDSCOPE_DATASET_OPTION_KEYS]
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise TypeError(f"Unsupported NDScope dataset option(s): {names}")

    if "chunks" in options:
        chunks = options["chunks"]
        if isinstance(chunks, bool):
            options["chunks"] = chunks if chunks else None
        elif chunks is not None and not isinstance(chunks, tuple):
            raise TypeError("NDScope chunks must be None, a bool, or a tuple")

    compression = options.get("compression")
    if "compression" in options:
        if compression is not None and not isinstance(compression, str):
            raise TypeError("NDScope compression must be None, 'gzip', or 'lzf'")
        if compression not in (None, "gzip", "lzf"):
            raise ValueError("NDScope compression must be None, 'gzip', or 'lzf'")

    if "compression_opts" in options:
        if compression is None:
            raise ValueError("compression_opts requires a compression")
        compression_opts = options["compression_opts"]
        if compression == "gzip":
            if compression_opts is not None and (
                isinstance(compression_opts, bool)
                or not isinstance(compression_opts, Integral)
            ):
                raise TypeError("gzip compression_opts must be an integral level")
            if compression_opts is not None and not 0 <= int(compression_opts) <= 9:
                raise ValueError("gzip compression_opts must be in the range 0..9")
            if compression_opts is not None:
                options["compression_opts"] = int(compression_opts)
        elif compression == "lzf" and compression_opts is not None:
            raise ValueError("lzf compression_opts must be None")

    for filter_name in ("shuffle", "fletcher32"):
        if filter_name in options and type(options[filter_name]) is not bool:
            raise TypeError(f"NDScope {filter_name} must be an exact bool")

    active_filter = (
        compression is not None
        or options.get("shuffle") is True
        or options.get("fletcher32") is True
    )
    if active_filter and "chunks" in options and options["chunks"] is None:
        raise ValueError("NDScope filters require chunking")

    if compression == "gzip" and not h5py.h5z.filter_avail(h5py.h5z.FILTER_DEFLATE):
        raise ValueError("NDScope compression codec 'gzip' is unavailable")
    if compression == "lzf" and not h5py.h5z.filter_avail(h5py.h5z.FILTER_LZF):
        raise ValueError("NDScope compression codec 'lzf' is unavailable")

    return options


def _validate_ndscope_dataset_options_for_shape(
    options: Mapping[str, Any], shape: tuple[int, ...]
) -> None:
    """Validate rank and size constraints for one output dataset."""
    chunks = options.get("chunks")
    if any(size == 0 for size in shape) and chunks is not True:
        raise ValueError("zero-length NDScope data requires chunks=True")
    if not isinstance(chunks, tuple):
        return
    if len(chunks) != len(shape):
        raise ValueError("NDScope chunks tuple rank must match data rank")
    for chunk, size in zip(chunks, shape):
        if isinstance(chunk, bool) or not isinstance(chunk, Integral):
            raise TypeError("NDScope chunk entries must be integral, not bool")
        chunk_size = int(chunk)
        if chunk_size <= 0:
            raise ValueError("NDScope chunk entries must be positive")
        if chunk_size > size:
            raise ValueError("NDScope chunk entries cannot exceed data shape")


def _ndscope_destination(name: str) -> tuple[str, str]:
    """Map a public channel key to its ndscope group and dataset names."""
    dataset_name = "raw"
    group_name = name
    for suffix in ("raw", "mean", "min", "max"):
        if name.endswith(f".{suffix}"):
            group_name = name[: -(len(suffix) + 1)]
            dataset_name = suffix
            break
    if not group_name:
        raise ValueError("NDScope channel name must not produce an empty group")
    _canonical_ndscope_path(group_name, object_kind="group")
    return group_name, dataset_name


def _canonical_ndscope_path(path: str, *, object_kind: str) -> tuple[str, ...]:
    """Return a structurally valid HDF5 path as its component tuple."""
    if not path or path.startswith("/") or path.endswith("/"):
        raise ValueError(
            f"invalid NDScope {object_kind} path {path!r}: "
            "leading/trailing slash is not allowed"
        )
    components = tuple(path.split("/"))
    if any(
        not component or component in {".", ".."} or "\x00" in component
        for component in components
    ):
        raise ValueError(
            f"invalid NDScope {object_kind} path {path!r}: "
            "empty, '.', '..', and NUL components are not allowed"
        )
    return components


def _validate_ndscope_object_paths(entries: Iterable[_NDScopeWriteEntry]) -> None:
    """Reject structural HDF5 group/dataset path collisions."""
    objects: dict[tuple[str, ...], str] = {}

    def record(path: tuple[str, ...], object_kind: str) -> None:
        previous = objects.get(path)
        if previous is None:
            objects[path] = object_kind
            return
        if previous == object_kind == "group":
            return
        if previous == object_kind:
            raise ValueError(
                f"duplicate NDScope {object_kind} path: {'/'.join(path)!r}"
            )
        raise ValueError(
            f"NDScope HDF5 path conflict at {'/'.join(path)!r}: "
            f"required as both {previous} and {object_kind}"
        )

    for entry in entries:
        group_path = _canonical_ndscope_path(entry.group_name, object_kind="group")
        for end in range(1, len(group_path) + 1):
            record(group_path[:end], "group")
        record(group_path + (entry.dataset_name,), "dataset")


def _preflight_ndscope_write(
    tsdict: TimeSeriesDict,
    dataset_options: Mapping[str, Any] | None,
) -> tuple[list[_NDScopeWriteEntry], dict[str, Any]]:
    """Validate the complete write before opening or truncating the target."""
    options = _normalise_ndscope_dataset_options(dataset_options)
    entries: list[_NDScopeWriteEntry] = []
    destinations: set[tuple[str, str]] = set()
    group_meta: dict[str, tuple[float, float, str]] = {}

    for key, ts in tsdict.items():
        name = str(key)
        group_name, dataset_name = _ndscope_destination(name)
        destination = (group_name, dataset_name)
        if destination in destinations:
            raise ValueError(
                "duplicate NDScope destination: "
                f"group={group_name!r}, dataset={dataset_name!r}"
            )
        destinations.add(destination)

        data = np.asarray(ts.value)
        if data.ndim != 1:
            raise ValueError(
                f"NDScope dataset {name!r} must contain suitable 1-D data; "
                f"got shape {data.shape}"
            )
        _validate_ndscope_dataset_options_for_shape(options, data.shape)

        try:
            rate_hz = float(ts.sample_rate.value)
            gps_start = float(ts.t0.value)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"NDScope metadata for {name!r} has invalid timing values"
            ) from exc
        if not np.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError(
                f"NDScope rate_hz for {name!r} must be positive and finite"
            )
        if not np.isfinite(gps_start):
            raise ValueError(f"NDScope gps_start for {name!r} must be finite")
        unit = str(ts.unit) if ts.unit else ""

        metadata = (rate_hz, gps_start, unit)
        if group_name in group_meta and group_meta[group_name] != metadata:
            expected = group_meta[group_name]
            expected_value: float | str
            actual_value: float | str
            if rate_hz != expected[0]:
                field = "sample_rate"
                expected_value, actual_value = expected[0], rate_hz
            elif gps_start != expected[1]:
                field = "gps_start"
                expected_value, actual_value = expected[1], gps_start
            else:
                field = "unit"
                expected_value, actual_value = expected[2], unit
            raise ValueError(
                f"Inconsistent {field} for group {group_name!r}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
        group_meta.setdefault(group_name, metadata)
        entries.append(
            _NDScopeWriteEntry(
                group_name=group_name,
                dataset_name=dataset_name,
                data=data,
                rate_hz=rate_hz,
                gps_start=gps_start,
                unit=unit,
            )
        )

    _validate_ndscope_object_paths(entries)
    return entries, options


def write_timeseriesdict_ndscope_hdf5(
    tsdict: TimeSeriesDict,
    target: str | Path,
    *,
    overwrite: bool = False,
    dataset_options: Mapping[str, Any] | None = None,
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
    dataset_options : Mapping, optional
        HDF5 dataset creation options.  Allowed keys are ``chunks``,
        ``compression``, ``compression_opts``, ``shuffle``, and ``fletcher32``.
        The mapping is copied and is never mutated.
    **kwargs
        Additional keyword arguments are rejected.  Dataset creation options
        must be supplied through ``dataset_options``.

    """
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported NDScope writer keyword arguments: {names}")

    entries, normalized_options = _preflight_ndscope_write(tsdict, dataset_options)
    mode = "w" if overwrite else "w-"
    with h5py.File(_resolve_source(target), mode) as f:
        groups: dict[str, h5py.Group] = {}
        for entry in entries:
            if entry.group_name not in groups:
                grp = f.require_group(entry.group_name)
                groups[entry.group_name] = grp
                grp.attrs["rate_hz"] = entry.rate_hz
                grp.attrs["gps_start"] = entry.gps_start
                grp.attrs["unit"] = entry.unit
            grp = groups[entry.group_name]
            grp.create_dataset(
                entry.dataset_name, data=entry.data, **dict(normalized_options)
            )


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
