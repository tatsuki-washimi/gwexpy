"""Strict GWexpy metadata sidecars for the native HDF5 handlers.

The sidecar is deliberately a single root attribute.  Native GWpy datasets,
groups, and attributes remain the source of truth for payload reconstruction;
this module only preserves GWexpy state that has no native representation.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import h5py
from gwpy.io import registry as _io_registry

from gwexpy.provenance import copy_provenance, dumps_json, loads_json, normalize_json

SIDECAR_ATTRIBUTE = "_gwexpy_sidecar_json_v1"
SIDECAR_SCHEMA = "gwexpy.hdf5.sidecar"
SIDECAR_VERSION = 1
TIME_STATE_KEY = "_gwexpy_t0_gps_state"
TIME_STATE_NS_KEY = "_gwex_t0_gps_ns"
TIME_STATE_PRECISION_KEY = "precision"
_SIDECAR_READ_MARKER = "_gwexpy_hdf5_sidecar_read"

_MISSING = object()
_DQLIST_WRITE_ACTIVE: ContextVar[bool] = ContextVar(
    "gwexpy_hdf5_dqflag_write_active", default=False
)
_REGISTERED = False
_REGISTRATION_LOCK_ATTRIBUTE = "_gwexpy_hdf5_sidecar_registration_lock"
_REGISTRATION_LOCK: Any = getattr(_io_registry, _REGISTRATION_LOCK_ATTRIBUTE, None)
if _REGISTRATION_LOCK is None:
    _REGISTRATION_LOCK = threading.RLock()
    setattr(_io_registry, _REGISTRATION_LOCK_ATTRIBUTE, _REGISTRATION_LOCK)


def _relative_path(value: Any, *, label: str = "HDF5 path") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not value or value.startswith("/"):
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{label} contains an invalid path component")
    if any("\x00" in component for component in components):
        raise ValueError(f"{label} contains NUL")
    return "/".join(components)


def _group_prefix(group: h5py.Group) -> str:
    name = group.name
    if name in {"", "/"}:
        return ""
    return _relative_path(name.lstrip("/"), label="HDF5 containing group")


def _filesystem_path(value: Any) -> Path:
    return Path(os.fsdecode(os.fspath(value)))


def _object_path(group: h5py.Group | None, path: Any, obj: Any) -> str:
    if path is None:
        path = getattr(obj, "name", None)
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"Cannot determine HDF5 path for {type(obj).__name__}; "
                "pass path= explicitly"
            )
    relative = _relative_path(path)
    prefix = "" if group is None else _group_prefix(group)
    return relative if not prefix else f"{prefix}/{relative}"


def _object_path_from_hdf5(
    source: h5py.Group,
    path: Any,
    *,
    flag: bool = False,
) -> str:
    prefix = _group_prefix(source)
    if path is not None:
        relative = _relative_path(path)
        return relative if not prefix else f"{prefix}/{relative}"

    if flag and not isinstance(source, h5py.Dataset):
        from gwpy.segments.io.hdf5 import _get_flag_group

        obj = _get_flag_group(source, None)
    elif isinstance(source, h5py.Dataset):
        obj = source
    else:
        from gwpy.io.hdf5 import find_dataset

        obj = find_dataset(source, path=None)
    name = obj.name
    if not isinstance(name, str) or not name.startswith("/"):
        raise ValueError("HDF5 object has no absolute root path")
    return _relative_path(name.lstrip("/"))


def _normalize_time_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        TIME_STATE_NS_KEY,
        TIME_STATE_PRECISION_KEY,
    }:
        raise ValueError("invalid reserved TimeSeries epoch state")
    from gwexpy.timeseries.utils import _validate_t0_gps_ns

    try:
        state_ns = _validate_t0_gps_ns(value[TIME_STATE_NS_KEY])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid reserved TimeSeries epoch nanoseconds") from exc
    precision = value[TIME_STATE_PRECISION_KEY]
    if not isinstance(precision, str) or precision not in {"exact", "quantized"}:
        raise ValueError("TimeSeries epoch precision must be exact or quantized")
    return {
        TIME_STATE_NS_KEY: state_ns,
        TIME_STATE_PRECISION_KEY: precision,
    }


def _time_state(obj: Any) -> dict[str, Any] | None:
    state_ns = getattr(obj, "_gwex_t0_gps_ns", None)
    precision = getattr(obj, "_gwex_t0_gps_precision", None)
    if state_ns is None and precision is None:
        return None
    if state_ns is None or precision is None:
        raise ValueError("TimeSeries epoch state must contain ns and precision")
    return _normalize_time_state(
        {TIME_STATE_NS_KEY: state_ns, TIME_STATE_PRECISION_KEY: precision}
    )


def _metadata_for_write(obj: Any, value: Any) -> dict[str, Any]:
    if value is _MISSING:
        value = getattr(obj, "metadata", {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    if TIME_STATE_KEY in value:
        raise ValueError(f"metadata key {TIME_STATE_KEY!r} is reserved")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):  # pragma: no cover
        raise TypeError("metadata must normalize to a mapping")
    state = _time_state(obj)
    if state is not None:
        normalized[TIME_STATE_KEY] = state
    return normalized


def _provenance_for_write(obj: Any, value: Any) -> dict[str, Any]:
    if value is _MISSING:
        value = getattr(obj, "provenance", {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("provenance must be a mapping")
    return copy_provenance(value)


def _validate_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "version", "objects"}:
        raise ValueError("invalid GWexpy HDF5 sidecar schema")
    if not isinstance(value["schema"], str) or value["schema"] != SIDECAR_SCHEMA:
        raise ValueError("unknown GWexpy HDF5 sidecar schema or version")
    if (
        isinstance(value["version"], bool)
        or not isinstance(value["version"], int)
        or value["version"] != SIDECAR_VERSION
    ):
        raise ValueError("unknown GWexpy HDF5 sidecar schema or version")
    objects = value["objects"]
    if not isinstance(objects, dict):
        raise ValueError("sidecar objects must be a mapping")
    validated: dict[str, Any] = {}
    for path, entry in objects.items():
        normalized_path = _relative_path(path, label="sidecar object path")
        if normalized_path in validated:
            raise ValueError("duplicate normalized sidecar object path")
        if not isinstance(entry, dict) or set(entry) != {"metadata", "provenance"}:
            raise ValueError("sidecar object entry has invalid keys")
        metadata = entry["metadata"]
        provenance = entry["provenance"]
        if not isinstance(metadata, dict):
            raise ValueError("sidecar metadata must be a mapping")
        if not isinstance(provenance, dict):
            raise ValueError("sidecar provenance must be a mapping")
        normalized_metadata = normalize_json(metadata)
        if not isinstance(normalized_metadata, dict):  # pragma: no cover
            raise ValueError("sidecar metadata must normalize to a mapping")
        if TIME_STATE_KEY in normalized_metadata:
            normalized_state = _normalize_time_state(
                normalized_metadata[TIME_STATE_KEY]
            )
            metadata = copy.deepcopy(metadata)
            metadata[TIME_STATE_KEY] = normalized_state
        normalized_provenance = copy_provenance(provenance)
        validated[normalized_path] = {
            "metadata": copy.deepcopy(metadata),
            "provenance": normalized_provenance,
        }
    return {
        "schema": SIDECAR_SCHEMA,
        "version": SIDECAR_VERSION,
        "objects": validated,
    }


def _empty_document() -> dict[str, Any]:
    return {"schema": SIDECAR_SCHEMA, "version": 1, "objects": {}}


def _read_document(root: h5py.Group) -> dict[str, Any]:
    if SIDECAR_ATTRIBUTE not in root.attrs:
        return _empty_document()
    raw = root.attrs[SIDECAR_ATTRIBUTE]
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("sidecar attribute is not UTF-8") from exc
    if not isinstance(raw, str):
        raise ValueError("sidecar attribute must be a UTF-8 JSON string")
    try:
        value = loads_json(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid GWexpy HDF5 sidecar JSON") from exc
    try:
        return _validate_document(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid GWexpy HDF5 sidecar document") from exc


@contextmanager
def _open_root(source: Any, mode: str = "r") -> Iterator[h5py.Group]:
    if isinstance(source, h5py.HLObject):
        yield source.file
        return
    with h5py.File(source, mode) as root:
        yield root


def _existing_document(target: Any) -> dict[str, Any]:
    if isinstance(target, h5py.HLObject):
        return _read_document(target.file)
    try:
        exists = _filesystem_path(target).exists()
    except TypeError as exc:
        raise TypeError("HDF5 target must be a filesystem path or h5py object") from exc
    if not exists:
        return _empty_document()
    with h5py.File(target, "r") as root:
        return _read_document(root)


def _write_document(target: Any, document: dict[str, Any]) -> None:
    payload = dumps_json(document)
    if isinstance(target, h5py.HLObject):
        target.file.attrs[SIDECAR_ATTRIBUTE] = payload
        return
    with h5py.File(target, "r+") as root:
        root.attrs[SIDECAR_ATTRIBUTE] = payload


def _entry_for_write(obj: Any, metadata: Any, provenance: Any) -> dict[str, Any]:
    return {
        "metadata": _metadata_for_write(obj, metadata),
        "provenance": _provenance_for_write(obj, provenance),
    }


def _preflight_native_write(
    base_writer: Callable[..., Any],
    obj: Any,
    target: Any,
    path: Any,
    kwargs: dict[str, Any],
) -> None:
    """Run the captured native HDF5 writer against a disposable topology.

    This is intentionally bounded to the native writer call.  It validates
    h5py options and native attribute writes without invoking the sidecar
    wrapper or touching a caller-owned file/group; arbitrary application
    callbacks are outside this preflight boundary.
    """
    native_path = None if path is None else _relative_path(path)
    containing_prefix = _group_prefix(target) if isinstance(target, h5py.Group) else ""

    with tempfile.TemporaryDirectory(prefix="gwexpy-hdf5-preflight-") as directory:
        isolated_path = Path(directory) / "native.h5"
        with h5py.File(isolated_path, "w") as isolated_root:
            isolated_group: h5py.Group = isolated_root
            if containing_prefix:
                isolated_group = isolated_root.require_group(containing_prefix)
            base_writer(obj, isolated_group, path=native_path, **kwargs)


def _write_wrapper(
    base_writer: Callable[..., Any],
    *,
    flag: bool = False,
) -> Callable[..., Any]:
    @functools.wraps(base_writer)
    def write(
        obj: Any,
        target: Any,
        path: Any = None,
        *,
        metadata: Any = _MISSING,
        provenance: Any = _MISSING,
        **kwargs: Any,
    ) -> Any:
        group = target if isinstance(target, h5py.Group) else None
        object_path = _object_path(group, path, obj)
        entry = _entry_for_write(obj, metadata, provenance)

        target_exists = isinstance(target, h5py.HLObject) or (
            isinstance(target, (str, bytes, os.PathLike))
            and _filesystem_path(target).exists()
        )
        existing = _empty_document()
        if target_exists and (
            isinstance(target, h5py.HLObject)
            or kwargs.get("append")
            or kwargs.get("overwrite")
        ):
            existing = _existing_document(target)

        if _DQLIST_WRITE_ACTIVE.get() and not flag:
            return base_writer(obj, target, path=path, **kwargs)

        token = _DQLIST_WRITE_ACTIVE.set(True) if flag else None
        try:
            _preflight_native_write(base_writer, obj, target, path, kwargs)
            result = base_writer(obj, target, path=path, **kwargs)
        finally:
            if token is not None:
                _DQLIST_WRITE_ACTIVE.reset(token)

        if (
            isinstance(target, (str, bytes, os.PathLike))
            and kwargs.get("overwrite", False)
            and not kwargs.get("append", False)
        ):
            existing = _empty_document()
        objects = copy.deepcopy(existing["objects"])
        objects[object_path] = entry
        _write_document(
            target,
            {"schema": SIDECAR_SCHEMA, "version": 1, "objects": objects},
        )
        return result

    setattr(write, "_gwexpy_hdf5_sidecar", True)
    return write


def _rebuild_array(result: Any, target_class: type[Any]) -> Any:
    if isinstance(result, target_class):
        return result
    if hasattr(result, "dt"):
        return target_class(
            result.value,
            dt=result.dt,
            t0=getattr(result, "t0", None),
            unit=result.unit,
            name=getattr(result, "name", None),
            channel=getattr(result, "channel", None),
        )
    if hasattr(result, "frequencies"):
        return target_class(
            result.value,
            frequencies=result.frequencies,
            unit=result.unit,
            name=getattr(result, "name", None),
            channel=getattr(result, "channel", None),
            epoch=getattr(result, "epoch", None),
        )
    return target_class(
        result.value,
        times=getattr(result, "times", None),
        unit=result.unit,
        name=getattr(result, "name", None),
        channel=getattr(result, "channel", None),
    )


def _call_base_reader(
    base_reader: Callable[..., Any],
    source: Any,
    path: Any,
    kwargs: dict[str, Any],
    target_class: type[Any],
) -> Any:
    if isinstance(base_reader, functools.partial) and "array_type" in (
        base_reader.keywords or {}
    ):
        call_kwargs = dict(base_reader.keywords or {})
        call_kwargs.update(kwargs)
        call_kwargs["array_type"] = target_class
        return base_reader.func(source, path=path, **call_kwargs)
    result = base_reader(source, path=path, **kwargs)
    return _rebuild_array(result, target_class)


def _attach_sidecar(result: Any, document: dict[str, Any], object_path: str) -> Any:
    entry = document["objects"].get(object_path)
    if entry is None:
        metadata: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
    else:
        metadata = copy.deepcopy(entry["metadata"])
        state = metadata.pop(TIME_STATE_KEY, None)
        if state is not None:
            if not hasattr(type(result), "t0_gps_ns"):
                raise ValueError("reserved TimeSeries epoch state on a non-TimeSeries")
            result._gwex_t0_gps_ns = state[TIME_STATE_NS_KEY]
            result._gwex_t0_gps_precision = state[TIME_STATE_PRECISION_KEY]
        provenance = copy.deepcopy(entry["provenance"])
    setattr(result, "metadata", metadata)
    setattr(result, "provenance", provenance)
    setattr(result, _SIDECAR_READ_MARKER, True)
    return result


def _read_wrapper(
    base_reader: Callable[..., Any],
    target_class: type[Any],
    *,
    flag: bool = False,
) -> Callable[..., Any]:
    @functools.wraps(base_reader)
    def read(source: Any, path: Any = None, **kwargs: Any) -> Any:
        if path is not None:
            _relative_path(path)

        document: dict[str, Any] | None = None
        object_path: str | None = None
        with _open_root(source) as root:
            if SIDECAR_ATTRIBUTE in root.attrs:
                lookup = source if isinstance(source, h5py.HLObject) else root
                object_path = _object_path_from_hdf5(lookup, path, flag=flag)
                document = _read_document(root)

        result = _call_base_reader(base_reader, source, path, kwargs, target_class)
        if document is None:
            if target_class.__module__.startswith("gwexpy."):
                result = _rebuild_array(result, target_class)
                result.metadata = {}
                result.provenance = {}
            return result
        if object_path is None:  # pragma: no cover - guarded with document
            return result
        return _attach_sidecar(result, document, object_path)

    setattr(read, "_gwexpy_hdf5_sidecar", True)
    return read


def _patch_segmentlist_merge() -> None:
    """Keep sidecar state when GWpy's single-file reader merges one result."""
    from gwpy.segments.connect import SegmentListRead

    if getattr(SegmentListRead, "_gwexpy_sidecar_merge", False):
        return
    original_merge = SegmentListRead.merge

    @functools.wraps(original_merge)
    def merge(self: Any, items: Any, **kwargs: Any) -> Any:
        marked = [item for item in items if getattr(item, _SIDECAR_READ_MARKER, False)]
        state: tuple[dict[str, Any], dict[str, Any]] | None = None
        if marked:
            first_metadata = copy.deepcopy(getattr(marked[0], "metadata", {}))
            first_provenance = copy.deepcopy(getattr(marked[0], "provenance", {}))
            for item in marked[1:]:
                if (
                    getattr(item, "metadata", {}) != first_metadata
                    or getattr(item, "provenance", {}) != first_provenance
                ):
                    raise ValueError(
                        "conflicting sidecar metadata or provenance in merged "
                        "SegmentList inputs"
                    )
            state = (first_metadata, first_provenance)

        result = original_merge(self, items, **kwargs)
        if state is not None:
            result.metadata = copy.deepcopy(state[0])
            result.provenance = copy.deepcopy(state[1])
            setattr(result, _SIDECAR_READ_MARKER, True)
        return result

    SegmentListRead.merge = merge
    SegmentListRead._gwexpy_sidecar_merge = True


def register_hdf5_sidecars() -> None:
    """Replace only the six canonical native HDF5 handlers, once."""
    global _REGISTERED
    from gwpy.io import registry
    from gwpy.segments import DataQualityFlag, SegmentList
    from gwpy.timeseries import StateVector

    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.spectrogram.spectrogram import Spectrogram
    from gwexpy.timeseries import TimeSeries

    targets = [
        (TimeSeries, False),
        (FrequencySeries, False),
        (Spectrogram, False),
        (StateVector, False),
        (SegmentList, False),
        (DataQualityFlag, True),
    ]
    with _REGISTRATION_LOCK:
        captured = [
            (
                cls,
                is_flag,
                registry.default_registry.get_reader("hdf5", cls),
                registry.default_registry.get_writer("hdf5", cls),
            )
            for cls, is_flag in targets
        ]
        if all(
            getattr(reader, "_gwexpy_hdf5_sidecar", False)
            and getattr(writer, "_gwexpy_hdf5_sidecar", False)
            for _, _, reader, writer in captured
        ):
            _patch_segmentlist_merge()
            _REGISTERED = True
            return

        for cls, is_flag, reader, writer in captured:
            if not getattr(reader, "_gwexpy_hdf5_sidecar", False):
                registry.default_registry.register_reader(
                    "hdf5", cls, _read_wrapper(reader, cls, flag=is_flag), force=True
                )
            if not getattr(writer, "_gwexpy_hdf5_sidecar", False):
                registry.default_registry.register_writer(
                    "hdf5", cls, _write_wrapper(writer, flag=is_flag), force=True
                )
        _patch_segmentlist_merge()
        _REGISTERED = True


__all__ = [
    "SIDECAR_ATTRIBUTE",
    "SIDECAR_SCHEMA",
    "SIDECAR_VERSION",
    "TIME_STATE_KEY",
    "register_hdf5_sidecars",
]
