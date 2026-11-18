"""GWDama HDF5 format reader/writer for gwexpy.

GWDama (https://gwnoisehunt.gitlab.io/gwdama/) stores time-series datasets in
HDF5 files without requiring the ``gwdama`` Python package to read or write
with ``h5py``:

    File attrs: gwexpy_schema="gwexpy.hdf.gwdama", gwexpy_schema_version=1, ...
    ├── <dts_key> (Dataset)            # dataset at root, or nested in groups
    │   └── attrs: sample_rate, t0, channel, unit, ...
    └── <group> (Group, optional, arbitrarily nested)
        ├── attrs: ...
        └── <dts_key> (Dataset)
            └── attrs: sample_rate, t0, channel, unit, ...

This module registers the canonical ``"hdf.gwdama"`` format (along with aliases
``"gwdama"`` and ``"gwdama-hdf5"``) using ``h5py`` only.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np

from gwexpy.io.time_selection import apply_time_selection
from gwexpy.io.utils import _reject_timezone_reinterpretation, set_provenance
from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
    get_time_reference,
    set_time_reference,
)

from .. import TimeSeries, TimeSeriesDict
from ._multi import expand_multi_source, read_multi_dict
from ._registration import register_timeseries_format

_SCHEMA_NAME = "gwexpy.hdf.gwdama"
_SCHEMA_VERSION = 1
_TIME_REFERENCE_ATTR = "gwexpy_time_reference"
_KNOWN_DATASET_ATTRS = frozenset(
    {"sample_rate", "dt", "t0", "channel", "unit", "units"}
)
_SUPPORTED_SUFFIXES = (".hdf5", ".h5", ".hdf")
_NDSCOPE_DATA_KEYS = frozenset({"raw", "mean", "min", "max"})
_GWPY_READ_KWARGS = frozenset({"pad", "gap", "nproc", "scaled"})


def _decode_attr_text(value: Any) -> Any:
    """Decode HDF5 byte-string scalars into Python ``str``."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.str_):
        return str(value)
    return value


def _copy_attr_value(value: Any) -> Any:
    """Detach an HDF5 attribute value into a plain Python or NumPy object."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_unknown_attrs(
    attrs: Any,
    *,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return non-gwexpy, non-schema attributes from an HDF5 attribute map."""
    preserved: dict[str, Any] = {}
    for key, value in attrs.items():
        name = str(key)
        if name in exclude or name.startswith("gwexpy_"):
            continue
        preserved[name] = _copy_attr_value(value)
    return preserved


def _normalize_hdf5_subpath(raw: str, *, allow_empty: bool = False) -> str:
    """Validate and normalize a relative POSIX HDF5 path."""
    cleaned = str(raw).strip("/")
    if not cleaned:
        if allow_empty:
            return ""
        raise ValueError(f"Invalid empty GWDama HDF5 path: {raw!r}")
    if "\x00" in cleaned:
        raise ValueError(f"Invalid GWDama HDF5 path containing NUL: {raw!r}")
    parts = cleaned.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid GWDama HDF5 path component in {raw!r}")
    return PurePosixPath(*parts).as_posix()


def _resolve_source(source: Any) -> str:
    """Extract a file path string from *source*."""
    if isinstance(source, (str, Path)):
        return str(source)
    if isinstance(source, h5py.File):
        return str(source.filename)
    if hasattr(source, "name"):
        return str(source.name)
    return str(source)


def _has_hdf5_suffix(path: Any) -> bool:
    """Return whether *path* has a supported HDF5 filename suffix."""
    if not isinstance(path, (str, Path)):
        return False
    return str(path).lower().endswith(_SUPPORTED_SUFFIXES)


def _is_ndscope_group(group: h5py.Group) -> bool:
    """Return whether *group* follows the NDScope group layout."""
    if "gps_start" not in group.attrs:
        return False
    return any(ds_name in group for ds_name in _NDSCOPE_DATA_KEYS)


def _validate_root_schema(root: h5py.File | h5py.Group) -> None:
    """Validate ``gwexpy_schema`` and ``gwexpy_schema_version`` when present."""
    if "gwexpy_schema" in root.attrs:
        schema = _decode_attr_text(root.attrs["gwexpy_schema"])
        if schema != _SCHEMA_NAME:
            raise ValueError(
                f"Unsupported gwexpy_schema {schema!r} for format 'hdf.gwdama'; "
                f"expected {_SCHEMA_NAME!r}"
            )
    if "gwexpy_schema_version" in root.attrs:
        raw_version = root.attrs["gwexpy_schema_version"]
        if isinstance(raw_version, np.ndarray) and raw_version.ndim == 0:
            raw_version = raw_version.item()
        if isinstance(raw_version, (bool, np.bool_)) or not isinstance(
            raw_version, (int, np.integer)
        ):
            raise ValueError(
                f"Unsupported gwexpy_schema_version {raw_version!r} for format "
                f"'hdf.gwdama'; expected integer {_SCHEMA_VERSION}"
            )
        if int(raw_version) != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported gwexpy_schema_version {int(raw_version)!r} for "
                f"format 'hdf.gwdama'; expected {_SCHEMA_VERSION}"
            )


def _coerce_finite_float(value: Any, *, field: str, dataset_name: str) -> float:
    """Coerce a scalar attribute to a finite float, rejecting booleans."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(
            f"GWDama dataset {dataset_name!r} has invalid {field}={value!r}"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"GWDama dataset {dataset_name!r} has invalid {field}={value!r}"
        ) from exc
    if not np.isfinite(numeric):
        raise ValueError(
            f"GWDama dataset {dataset_name!r} has non-finite {field}={value!r}"
        )
    return numeric


def _sample_rate_from_attrs(attrs: Any, *, dataset_name: str) -> float:
    """Return a validated sampling rate in Hz from GWDama dataset attrs."""
    sr: float | None = None
    dt_rate: float | None = None

    if "sample_rate" in attrs:
        sr = _coerce_finite_float(
            attrs["sample_rate"], field="sample_rate", dataset_name=dataset_name
        )
        if sr <= 0:
            raise ValueError(
                f"GWDama dataset {dataset_name!r} has invalid sample_rate={sr!r}; "
                "expected a positive finite sampling rate in Hz"
            )

    if "dt" in attrs:
        dt = _coerce_finite_float(attrs["dt"], field="dt", dataset_name=dataset_name)
        if dt <= 0:
            raise ValueError(
                f"GWDama dataset {dataset_name!r} has invalid dt={dt!r}; "
                "expected a positive finite time step in seconds"
            )
        dt_rate = 1.0 / dt

    if sr is None and dt_rate is None:
        raise ValueError(
            f"GWDama dataset {dataset_name!r} is missing timing metadata; "
            "expected 'sample_rate' (Hz) or 'dt' (s)"
        )

    if (
        sr is not None
        and dt_rate is not None
        and not np.isclose(sr, dt_rate, rtol=1e-12, atol=0.0)
    ):
        raise ValueError(
            f"GWDama dataset {dataset_name!r} has conflicting sampling-rate "
            f"metadata: sample_rate={sr!r}, 1/dt={dt_rate!r}"
        )

    if sr is not None:
        return sr
    assert dt_rate is not None
    return float(dt_rate)


def _t0_from_attrs(
    attrs: Any,
    *,
    dataset_name: str,
    t0_override: float | None = None,
) -> tuple[float | None, str]:
    """Resolve ``(t0, time_reference)`` from GWDama dataset attrs."""
    raw_ref = attrs.get(_TIME_REFERENCE_ATTR)
    explicit_ref: str | None = None
    if raw_ref is not None:
        decoded_ref = _decode_attr_text(raw_ref)
        if decoded_ref not in (TIME_REFERENCE_ABSOLUTE, TIME_REFERENCE_RELATIVE):
            raise ValueError(
                f"GWDama dataset {dataset_name!r} has invalid "
                f"{_TIME_REFERENCE_ATTR}={decoded_ref!r}"
            )
        explicit_ref = str(decoded_ref)

    if "t0" in attrs:
        raw_t0 = attrs["t0"]
        if isinstance(raw_t0, np.ndarray) and raw_t0.ndim == 0:
            raw_t0 = raw_t0.item()
        if isinstance(raw_t0, (bool, np.bool_, complex, np.complexfloating)):
            raise ValueError(
                f"GWDama dataset {dataset_name!r} has invalid t0={raw_t0!r}"
            )
        try:
            t0_val = float(raw_t0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"GWDama dataset {dataset_name!r} has invalid t0={raw_t0!r}"
            ) from exc
        if np.isnan(t0_val):
            if t0_override is not None:
                return _coerce_finite_float(
                    t0_override, field="t0_override", dataset_name=dataset_name
                ), TIME_REFERENCE_ABSOLUTE
            if explicit_ref != TIME_REFERENCE_RELATIVE:
                warnings.warn(
                    f"GWDama dataset {dataset_name!r} has NaN 't0' attribute; "
                    "assuming t0=0.0 (relative time reference).",
                    UserWarning,
                    stacklevel=3,
                )
            return None, TIME_REFERENCE_RELATIVE
        if not np.isfinite(t0_val):
            raise ValueError(
                f"GWDama dataset {dataset_name!r} has non-finite t0={raw_t0!r}"
            )
        return t0_val, (explicit_ref or TIME_REFERENCE_ABSOLUTE)

    if t0_override is not None:
        return _coerce_finite_float(
            t0_override, field="t0_override", dataset_name=dataset_name
        ), TIME_REFERENCE_ABSOLUTE

    if explicit_ref == TIME_REFERENCE_RELATIVE:
        return None, TIME_REFERENCE_RELATIVE

    warnings.warn(
        f"GWDama dataset {dataset_name!r} has no 't0' attribute; "
        "assuming t0=0.0 (relative time reference). Pass t0_override=... "
        "to set an explicit GPS epoch.",
        UserWarning,
        stacklevel=3,
    )
    return None, TIME_REFERENCE_RELATIVE


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------


def _has_gwdama_structure(root: h5py.File | h5py.Group) -> bool:
    """Return whether *root* has GWDama HDF5 schema or dataset structure."""
    if "gwexpy_schema" in root.attrs:
        return _decode_attr_text(root.attrs["gwexpy_schema"]) == _SCHEMA_NAME
    if "dama_name" in root.attrs:
        return True

    stack: list[h5py.Group] = [root]
    while stack:
        current = stack.pop()
        if current is not root and _is_ndscope_group(current):
            continue
        for key in current:
            item = current[key]
            if isinstance(item, h5py.Group):
                stack.append(item)
            elif isinstance(item, h5py.Dataset):
                ds_attrs = item.attrs
                if "sample_rate" in ds_attrs:
                    return True
                if "t0" in ds_attrs and "x0" not in ds_attrs and "dx" not in ds_attrs:
                    return True
    return False


def identify_gwdama_hdf5(
    origin: type,
    filepath: str | Path | None,
    fileobj: Any,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Identify a GWDama HDF5 file by its schema and dataset metadata.

    Do not reject broken GWDama files lacking ``sample_rate`` when they carry
    GWDama root metadata (``gwexpy_schema`` / ``dama_name``) or ``t0`` dataset
    attributes, so explicit reader errors surface rather than falling back.
    """
    try:
        h5file = fileobj
        if not isinstance(h5file, h5py.File) and args:
            h5file = args[0]
        if isinstance(h5file, h5py.File):
            if h5file.filename and not _has_hdf5_suffix(h5file.filename):
                # In-memory driver='core' handles still use a filename; only
                # reject if an explicit non-HDF5 suffix is present.
                return False
            return _has_gwdama_structure(h5file)

        path = filepath
        if path is None:
            path = getattr(fileobj, "name", None)
        if not _has_hdf5_suffix(path):
            return False
        with h5py.File(path, "r") as f:
            return _has_gwdama_structure(f)
    except (OSError, KeyError, AttributeError, TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _iter_gwdama_datasets(
    scan_root: h5py.File | h5py.Group,
) -> list[tuple[str, h5py.Dataset, dict[str, dict[str, Any]]]]:
    """Collect ``(rel_path, dataset, ancestor_group_attrs)`` under *scan_root*."""
    collected: list[tuple[str, h5py.Dataset, dict[str, dict[str, Any]]]] = []

    def _walk(
        grp: h5py.File | h5py.Group,
        prefix: str,
        ancestors: dict[str, dict[str, Any]],
    ) -> None:
        for key in grp:
            item = grp[key]
            rel_path = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(item, h5py.Group):
                grp_attrs = _extract_unknown_attrs(item.attrs)
                next_ancestors = dict(ancestors)
                if grp_attrs:
                    next_ancestors[rel_path] = grp_attrs
                _walk(item, rel_path, next_ancestors)
            elif isinstance(item, h5py.Dataset):
                collected.append((rel_path, item, ancestors))

    root_group_attrs = (
        _extract_unknown_attrs(scan_root.attrs) if prefix_needed(scan_root) else {}
    )
    initial_ancestors = {"": root_group_attrs} if root_group_attrs else {}
    _walk(scan_root, "", initial_ancestors)
    return collected


def prefix_needed(scan_root: h5py.File | h5py.Group) -> bool:
    """Return True when *scan_root* is a non-root subgroup."""
    return isinstance(scan_root, h5py.Group) and scan_root.name not in ("", "/")


def read_timeseriesdict_gwdama_hdf5(
    source: str | Path | h5py.File | list[str | Path] | tuple[str | Path, ...],
    *,
    channels: Iterable[str] | str | None = None,
    group: str | None = None,
    start: float | None = None,
    end: float | None = None,
    t0_override: float | None = None,
    **kwargs: Any,
) -> TimeSeriesDict:
    """Read a GWDama HDF5 file into a :class:`TimeSeriesDict`.

    Parameters
    ----------
    source : str, Path, h5py.File, or sequence of paths
        GWDama HDF5 file path, caller-owned open ``h5py.File``, or list/tuple
        of paths to concatenate across time / merge across channels.
    channels : str or iterable of str, optional
        Dataset key(s) (relative to *group* when given) to read.
    group : str, optional
        Subgroup path inside the HDF5 file to restrict reading to.
    start, end : float, Quantity, datetime, or str, optional
        GPS bounds for cropping each channel.
    t0_override : float, optional
        Explicit GPS epoch to use when a dataset lacks a ``t0`` attribute.
    **kwargs
        Registry-forwarded options (``timezone`` is rejected per contract).

    Returns
    -------
    TimeSeriesDict

    """
    timezone = kwargs.pop("timezone", None)
    _reject_timezone_reinterpretation("hdf.gwdama", timezone, None)
    for gwpy_key in _GWPY_READ_KWARGS:
        kwargs.pop(gwpy_key, None)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported GWDama reader keyword arguments: {names}")

    multi = expand_multi_source(source)
    if multi is not None:
        return read_multi_dict(
            read_timeseriesdict_gwdama_hdf5,
            multi,
            "hdf.gwdama",
            channels=channels,
            group=group,
            start=start,
            end=end,
            t0_override=t0_override,
        )

    if isinstance(channels, str):
        wanted: set[str] | None = {channels}
    elif channels is not None:
        wanted = {str(ch) for ch in channels}
    else:
        wanted = None

    out = TimeSeriesDict()
    source_ctx = (
        nullcontext(source)
        if isinstance(source, h5py.File)
        else h5py.File(_resolve_source(source), "r")
    )
    with source_ctx as f:
        _validate_root_schema(f)

        if group is not None:
            group_norm = _normalize_hdf5_subpath(group, allow_empty=True)
            if group_norm:
                if group_norm not in f:
                    raise ValueError(f"Group {group!r} not found in GWDama HDF5 file")
                target_obj = f[group_norm]
                if not isinstance(target_obj, h5py.Group):
                    raise ValueError(
                        f"Path {group!r} in GWDama HDF5 file is not a group"
                    )
                scan_root: h5py.File | h5py.Group = target_obj
            else:
                scan_root = f
        else:
            scan_root = f

        entries = _iter_gwdama_datasets(scan_root)
        group_attrs_map: dict[str, dict[str, Any]] = {}

        for rel_path, ds, ancestor_attrs in entries:
            if wanted is not None and rel_path not in wanted:
                continue

            data = np.asarray(ds[()])
            if data.ndim != 1:
                raise ValueError(
                    f"GWDama dataset {rel_path!r} must be 1D, got shape {data.shape}"
                )

            sample_rate = _sample_rate_from_attrs(ds.attrs, dataset_name=rel_path)
            t0_val, time_ref = _t0_from_attrs(
                ds.attrs, dataset_name=rel_path, t0_override=t0_override
            )

            channel_name = (
                str(_decode_attr_text(ds.attrs["channel"]))
                if "channel" in ds.attrs
                else rel_path
            )
            if "unit" in ds.attrs:
                unit_str = str(_decode_attr_text(ds.attrs["unit"]))
            elif "units" in ds.attrs:
                unit_str = str(_decode_attr_text(ds.attrs["units"]))
            else:
                unit_str = ""

            ds_unknown = _extract_unknown_attrs(ds.attrs, exclude=_KNOWN_DATASET_ATTRS)

            ts = TimeSeries(
                data,
                sample_rate=sample_rate,
                t0=t0_val,
                name=rel_path,
                channel=channel_name,
                unit=unit_str or None,
            )
            if time_ref == TIME_REFERENCE_RELATIVE:
                set_time_reference(ts, TIME_REFERENCE_RELATIVE)

            if start is not None or end is not None:
                ts = apply_time_selection(ts, start, end)

            prov_info: dict[str, Any] = {
                "format": "hdf.gwdama",
                "attrs": ds_unknown,
            }
            if ancestor_attrs:
                prov_info["group_attrs"] = {
                    k: dict(v) for k, v in ancestor_attrs.items()
                }
                for grp_key, grp_vals in ancestor_attrs.items():
                    group_attrs_map[grp_key] = dict(grp_vals)
            set_provenance(ts, prov_info)

            out[rel_path] = ts

        root_unknown = _extract_unknown_attrs(f.attrs)
        dict_prov: dict[str, Any] = {
            "format": "hdf.gwdama",
            "channels": list(out.keys()),
            "gwdama_root_attrs": root_unknown,
            "gwdama_group_attrs": group_attrs_map,
        }
        if group is not None:
            dict_prov["group"] = group
        set_provenance(out, dict_prov)

    return out


def read_timeseries_gwdama_hdf5(
    source: str | Path | h5py.File | list[str | Path] | tuple[str | Path, ...],
    *,
    channel: str | None = None,
    channels: Iterable[str] | str | None = None,
    group: str | None = None,
    **kwargs: Any,
) -> TimeSeries:
    """Read a single :class:`TimeSeries` from a GWDama HDF5 file."""
    if channel is not None and channels is not None:
        raise ValueError("Pass only one of 'channel' or 'channels'")
    selected = [channel] if channel is not None else channels
    tsd = read_timeseriesdict_gwdama_hdf5(
        source, channels=selected, group=group, **kwargs
    )
    if not tsd:
        raise ValueError("No data found in GWDama HDF5 file")
    return tsd[next(iter(tsd.keys()))]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _write_attrs_preserving(
    target_attrs: Any,
    attrs_dict: dict[str, Any] | None,
    *,
    reserved: frozenset[str],
) -> None:
    """Write preserved user attributes without touching reserved keys."""
    if not isinstance(attrs_dict, dict):
        return
    for key, value in attrs_dict.items():
        name = str(key)
        if name in reserved or name.startswith("gwexpy_"):
            continue
        target_attrs[name] = value


def write_timeseriesdict_gwdama_hdf5(
    tsdict: TimeSeriesDict,
    target: str | Path | h5py.File | h5py.Group,
    *,
    group: str | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Write a :class:`TimeSeriesDict` in GWDama-compatible HDF5 format.

    Parameters
    ----------
    tsdict : TimeSeriesDict
        Data to write. Keys may contain ``/`` to create nested HDF5 groups.
    target : str, Path, h5py.File, or h5py.Group
        Output file path or open HDF5 container.
    group : str, optional
        Subgroup inside the target HDF5 file to write datasets under.
    overwrite : bool, optional
        If ``True``, overwrite an existing file (or existing datasets on an
        open HDF5 container). Default: ``False``.
    **kwargs
        Unsupported keyword arguments raise ``TypeError`` before touching disk.

    """
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported GWDama writer keyword arguments: {names}")

    normalized_entries: list[tuple[str, Any]] = []
    seen_paths: set[str] = set()
    for key, ts in tsdict.items():
        rel_path = _normalize_hdf5_subpath(str(key), allow_empty=False)
        if rel_path in seen_paths:
            raise ValueError(f"Duplicate normalized GWDama dataset path: {rel_path!r}")
        seen_paths.add(rel_path)
        normalized_entries.append((rel_path, ts))

    group_norm = (
        _normalize_hdf5_subpath(group, allow_empty=True) if group is not None else ""
    )

    if isinstance(target, (h5py.File, h5py.Group)):
        _write_gwdama_to_container(
            tsdict,
            target,
            normalized_entries=normalized_entries,
            group_norm=group_norm,
            overwrite=overwrite,
        )
        return

    mode = "w" if overwrite else "w-"
    with h5py.File(_resolve_source(target), mode) as handle:
        _write_gwdama_to_container(
            tsdict,
            handle,
            normalized_entries=normalized_entries,
            group_norm=group_norm,
            overwrite=overwrite,
        )


def _write_gwdama_to_container(
    tsdict: TimeSeriesDict,
    handle: h5py.File | h5py.Group,
    *,
    normalized_entries: list[tuple[str, Any]],
    group_norm: str,
    overwrite: bool,
) -> None:
    """Write normalized GWDama datasets and preserved attrs into *handle*."""
    root_file = handle.file
    root_file.attrs["gwexpy_schema"] = _SCHEMA_NAME
    root_file.attrs["gwexpy_schema_version"] = _SCHEMA_VERSION

    dict_prov = getattr(tsdict, "_gwexpy_io", None)
    if isinstance(dict_prov, dict):
        _write_attrs_preserving(
            root_file.attrs,
            dict_prov.get("gwdama_root_attrs"),
            reserved=frozenset(),
        )

    base_group: h5py.File | h5py.Group
    if group_norm:
        base_group = handle.require_group(group_norm)
    else:
        base_group = handle

    if isinstance(dict_prov, dict):
        grp_map = dict_prov.get("gwdama_group_attrs")
        if isinstance(grp_map, dict):
            for grp_rel, grp_attrs in grp_map.items():
                if not grp_rel:
                    if group_norm:
                        _write_attrs_preserving(
                            base_group.attrs, grp_attrs, reserved=frozenset()
                        )
                else:
                    sub = base_group.require_group(_normalize_hdf5_subpath(grp_rel))
                    _write_attrs_preserving(sub.attrs, grp_attrs, reserved=frozenset())

    for rel_path, ts in normalized_entries:
        if "/" in rel_path:
            parent_path, leaf = rel_path.rsplit("/", 1)
            parent_grp = base_group.require_group(parent_path)
        else:
            parent_grp = base_group
            leaf = rel_path

        if leaf in parent_grp:
            if overwrite:
                del parent_grp[leaf]
            else:
                raise OSError(f"GWDama HDF5 dataset {rel_path!r} already exists")

        ds = parent_grp.create_dataset(leaf, data=np.asarray(ts.value))
        ds.attrs["sample_rate"] = float(ts.sample_rate.value)
        ds.attrs["t0"] = float(ts.t0.value)
        ds.attrs["channel"] = str(ts.channel) if ts.channel is not None else rel_path
        ds.attrs["unit"] = str(ts.unit) if ts.unit is not None else ""
        if get_time_reference(ts) == TIME_REFERENCE_RELATIVE:
            ds.attrs[_TIME_REFERENCE_ATTR] = TIME_REFERENCE_RELATIVE

        ts_prov = getattr(ts, "_gwexpy_io", None)
        if isinstance(ts_prov, dict):
            ts_grp_attrs = ts_prov.get("group_attrs")
            if isinstance(ts_grp_attrs, dict):
                for grp_rel, grp_attrs in ts_grp_attrs.items():
                    if not grp_rel:
                        if group_norm:
                            _write_attrs_preserving(
                                base_group.attrs,
                                grp_attrs,
                                reserved=frozenset(),
                            )
                    else:
                        sub = base_group.require_group(_normalize_hdf5_subpath(grp_rel))
                        _write_attrs_preserving(
                            sub.attrs, grp_attrs, reserved=frozenset()
                        )
            ds_attrs = ts_prov.get("attrs") or ts_prov.get("gwdama_attrs")
            _write_attrs_preserving(ds.attrs, ds_attrs, reserved=_KNOWN_DATASET_ATTRS)

        explicit_attrs = getattr(ts, "attrs", None)
        if isinstance(explicit_attrs, dict):
            _write_attrs_preserving(
                ds.attrs, explicit_attrs, reserved=_KNOWN_DATASET_ATTRS
            )


def write_timeseries_gwdama_hdf5(
    ts: TimeSeries,
    target: str | Path | h5py.File | h5py.Group,
    *,
    group: str | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Write a single :class:`TimeSeries` in GWDama-compatible HDF5 format."""
    key = ts.name or (str(ts.channel) if ts.channel else "channel_0")
    write_timeseriesdict_gwdama_hdf5(
        TimeSeriesDict({key: ts}),
        target,
        group=group,
        overwrite=overwrite,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_timeseries_format(
    "hdf.gwdama",
    aliases=("gwdama", "gwdama-hdf5"),
    reader_dict=read_timeseriesdict_gwdama_hdf5,
    reader_single=read_timeseries_gwdama_hdf5,
    writer_dict=write_timeseriesdict_gwdama_hdf5,
    writer_single=write_timeseries_gwdama_hdf5,
    magic_identifier=identify_gwdama_hdf5,
)
