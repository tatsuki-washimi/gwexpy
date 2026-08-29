from __future__ import annotations

import functools
import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
from gwpy.io import hdf5 as _gwpy_io_hdf5
from gwpy.io import registry as _io_registry
from gwpy.timeseries.io.hdf5 import (
    SEC_UNIT,
    StateVector,
    StateVectorDict,
    TimeSeries,
    TimeSeriesDict,
    dict_class,
    identify_hdf5,
    read_hdf5_array,
    read_hdf5_dict,
    read_hdf5_factory,
    read_hdf5_timeseries,
    reader,
    registry,
    series_class,
    units,
    with_read_hdf5,
    with_write_hdf5,
    write_hdf5_dict,
    write_hdf5_series,
)

SIDECAR_ATTRIBUTE = "_gwexpy_sidecar_json_v1"
SIDECAR_SCHEMA = "gwexpy.hdf5.sidecar"
SIDECAR_VERSION = 1
TIME_STATE_KEY = "_gwexpy_t0_gps_state"
TIME_STATE_NS_KEY = "_gwex_t0_gps_ns"
TIME_STATE_PRECISION_KEY = "precision"

_MISSING = object()
_WRAPPER_MARKER = "_gwexpy_exact_t0_hdf5"
_ROLLBACK_PREFIX = "__gwexpy_t0_rollback_"
_BASE_READER: Callable[..., Any] | None = None
_BASE_WRITER: Callable[..., h5py.Dataset] | None = None


class _RollbackError(RuntimeError):
    """Report an incomplete handle rollback while retaining recovery state."""

    def __init__(
        self,
        operation_error: BaseException,
        rollback_errors: tuple[BaseException, ...],
        recovery_path: str | None,
    ) -> None:
        self.operation_error = operation_error
        self.rollback_errors = rollback_errors
        self.recovery_path = recovery_path
        self.errors = (operation_error, *rollback_errors)
        message = "TimeSeries HDF5 write failed and rollback was incomplete"
        if recovery_path is not None:
            message += f"; recovery retained at {recovery_path}"
        super().__init__(message)


def _relative_path(value: Any, *, label: str = "HDF5 path") -> str:
    """Return one canonical, relative POSIX HDF5 object path."""
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    if "\x00" in value:
        raise ValueError(f"{label} contains NUL")
    pure = PurePosixPath(value)
    components = value.split("/")
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(component in {"", ".", ".."} for component in components)
    ):
        raise ValueError(f"{label} contains an invalid path component")
    return value


def _group_prefix(group: h5py.Group | h5py.File) -> str:
    name = group.name
    if name in {"", "/"}:
        return ""
    return _relative_path(name.lstrip("/"), label="HDF5 containing group")


def _write_path(array: Any, container: h5py.Group | h5py.File, path: Any) -> str:
    candidate = path if path is not None else getattr(array, "name", None)
    if candidate is None:
        raise ValueError(
            f"Cannot determine HDF5 path for {type(array).__name__}; "
            "set name or pass path explicitly"
        )
    relative = _relative_path(candidate)
    prefix = _group_prefix(container)
    return relative if not prefix else f"{prefix}/{relative}"


def _dataset_path(dataset: h5py.Dataset) -> str:
    name = dataset.name
    if not isinstance(name, str) or not name.startswith("/"):
        raise ValueError("HDF5 dataset has no canonical root path")
    return _relative_path(name.lstrip("/"), label="HDF5 dataset path")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def _validate_time_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        TIME_STATE_NS_KEY,
        TIME_STATE_PRECISION_KEY,
    }:
        raise ValueError("invalid TimeSeries exact epoch state in sidecar")
    epoch = value[TIME_STATE_NS_KEY]
    if type(epoch) is not int:
        raise ValueError("TimeSeries sidecar epoch must be an integer")
    precision = value[TIME_STATE_PRECISION_KEY]
    if precision not in {"exact", "quantized"}:
        raise ValueError("TimeSeries sidecar epoch precision is invalid")
    return {
        TIME_STATE_NS_KEY: epoch,
        TIME_STATE_PRECISION_KEY: precision,
    }


def _validate_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "version", "objects"}:
        raise ValueError("invalid GWexpy HDF5 sidecar schema")
    if value["schema"] != SIDECAR_SCHEMA or type(value["schema"]) is not str:
        raise ValueError("unknown GWexpy HDF5 sidecar schema")
    if type(value["version"]) is not int or value["version"] != SIDECAR_VERSION:
        raise ValueError("unknown GWexpy HDF5 sidecar version")
    objects = value["objects"]
    if not isinstance(objects, dict):
        raise ValueError("GWexpy HDF5 sidecar objects must be a mapping")

    validated_objects: dict[str, Any] = {}
    for object_path, entry in objects.items():
        normalized = _relative_path(object_path, label="sidecar object path")
        if normalized in validated_objects:
            raise ValueError("duplicate normalized sidecar object path")
        if not isinstance(entry, dict) or set(entry) != {"metadata", "provenance"}:
            raise ValueError("GWexpy HDF5 sidecar entry has invalid keys")
        metadata = entry["metadata"]
        provenance = entry["provenance"]
        if not isinstance(metadata, dict) or not isinstance(provenance, dict):
            raise ValueError("GWexpy HDF5 sidecar state must use mappings")
        if TIME_STATE_KEY in metadata:
            metadata = dict(metadata)
            metadata[TIME_STATE_KEY] = _validate_time_state(metadata[TIME_STATE_KEY])
        validated_objects[normalized] = {
            "metadata": metadata,
            "provenance": provenance,
        }
    return {
        "schema": SIDECAR_SCHEMA,
        "version": SIDECAR_VERSION,
        "objects": validated_objects,
    }


def _empty_document() -> dict[str, Any]:
    return {"schema": SIDECAR_SCHEMA, "version": SIDECAR_VERSION, "objects": {}}


def _read_sidecar(h5file: h5py.File) -> dict[str, Any]:
    if SIDECAR_ATTRIBUTE not in h5file.attrs:
        return _empty_document()
    raw = h5file.attrs[SIDECAR_ATTRIBUTE]
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid GWexpy HDF5 sidecar UTF-8") from exc
    if not isinstance(raw, str):
        raise ValueError("GWexpy HDF5 sidecar must be a UTF-8 JSON string")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        return _validate_document(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid GWexpy HDF5 sidecar document") from exc


def _write_sidecar(h5file: h5py.File, document: dict[str, Any]) -> None:
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    h5file.attrs[SIDECAR_ATTRIBUTE] = payload


def _exact_epoch(array: Any) -> int | None:
    value = getattr(array, TIME_STATE_NS_KEY, None)
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError("authoritative TimeSeries epoch must be an integer")
    return value


def _commit_sidecar(
    h5file: h5py.File,
    document: dict[str, Any],
    dataset: h5py.Dataset,
    exact_epoch: int | None,
) -> None:
    objects = dict(document["objects"])
    object_path = _dataset_path(dataset)
    if exact_epoch is None:
        objects.pop(object_path, None)
    else:
        objects[object_path] = {
            "metadata": {
                TIME_STATE_KEY: {
                    TIME_STATE_NS_KEY: exact_epoch,
                    TIME_STATE_PRECISION_KEY: "exact",
                }
            },
            "provenance": {},
        }
    if objects:
        _write_sidecar(
            h5file,
            {
                "schema": SIDECAR_SCHEMA,
                "version": SIDECAR_VERSION,
                "objects": objects,
            },
        )
    elif SIDECAR_ATTRIBUTE in h5file.attrs:
        del h5file.attrs[SIDECAR_ATTRIBUTE]


def _sidecar_snapshot(h5file: h5py.File) -> tuple[bool, Any]:
    if SIDECAR_ATTRIBUTE in h5file.attrs:
        return True, h5file.attrs[SIDECAR_ATTRIBUTE]
    return False, _MISSING


def _restore_sidecar(h5file: h5py.File, snapshot: tuple[bool, Any]) -> None:
    exists, raw = snapshot
    if exists:
        h5file.attrs[SIDECAR_ATTRIBUTE] = raw
    elif SIDECAR_ATTRIBUTE in h5file.attrs:
        del h5file.attrs[SIDECAR_ATTRIBUTE]


def _native_reader() -> Callable[..., Any]:
    if _BASE_READER is None:  # pragma: no cover - registration invariant
        raise RuntimeError("TimeSeries HDF5 exact reader is not registered")
    return _BASE_READER


def _native_writer() -> Callable[..., h5py.Dataset]:
    if _BASE_WRITER is None:  # pragma: no cover - registration invariant
        raise RuntimeError("TimeSeries HDF5 exact writer is not registered")
    return _BASE_WRITER


def _write_core(
    array: Any,
    container: h5py.Group | h5py.File,
    path: str | None,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    return _native_writer()(array, container, path=path, **kwargs)


def _preflight_core_write(
    array: Any,
    container: h5py.Group | h5py.File,
    path: str | None,
    kwargs: dict[str, Any],
) -> None:
    """Exercise the native writer without touching a caller-owned handle."""
    prefix = _group_prefix(container)
    name = f"gwexpy-t0-preflight-{uuid.uuid4().hex}.hdf5"
    with h5py.File(name, "w", driver="core", backing_store=False) as isolated:
        isolated_container: h5py.Group | h5py.File = isolated
        if prefix:
            isolated_container = isolated.require_group(prefix)
        _native_writer()(array, isolated_container, path=path, **kwargs)


def _existing_dataset(
    container: h5py.Group | h5py.File,
    path: str,
) -> h5py.Dataset | None:
    link = container.get(path, getlink=True)
    if link is None:
        return None
    if isinstance(link, h5py.SoftLink):
        raise ValueError(f"cannot overwrite HDF5 soft link {path!r}")
    if isinstance(link, h5py.ExternalLink):
        raise ValueError(f"cannot overwrite HDF5 external link {path!r}")
    if not isinstance(link, h5py.HardLink):
        raise ValueError(f"unsupported existing HDF5 link {path!r}")
    target = container[path]
    if not isinstance(target, h5py.Dataset):
        raise ValueError(f"cannot write TimeSeries to existing HDF5 group {path!r}")
    return target


def _rollback_link(
    h5file: h5py.File,
    dataset: h5py.Dataset,
    sidecar_snapshot: tuple[bool, Any],
) -> h5py.Group:
    while True:
        name = f"{_ROLLBACK_PREFIX}{uuid.uuid4().hex}"
        if name not in h5file:
            break
    rollback = h5file.create_group(name)
    try:
        rollback["dataset"] = dataset
        exists, raw = sidecar_snapshot
        rollback.attrs["sidecar_snapshot_present"] = exists
        if exists:
            rollback.attrs["sidecar_snapshot"] = raw
    except BaseException:
        del h5file[name]
        raise
    return rollback


def _delete_rollback(h5file: h5py.File, rollback: h5py.Group | None) -> None:
    if rollback is not None and rollback.name in h5file:
        del h5file[rollback.name]


def _restore_dataset(
    container: h5py.Group | h5py.File,
    candidate_path: str,
    rollback: h5py.Group | None,
    created_parent_paths: tuple[str, ...],
) -> None:
    if candidate_path in container:
        del container[candidate_path]
    if rollback is not None:
        container[candidate_path] = rollback["dataset"]
    for parent_path in reversed(created_parent_paths):
        parent = container.get(parent_path)
        if isinstance(parent, h5py.Group) and len(parent) == 0:
            del container[parent_path]


def _missing_parent_paths(
    container: h5py.Group | h5py.File,
    candidate_path: str,
) -> tuple[str, ...]:
    components = candidate_path.split("/")[:-1]
    prefixes = ["/".join(components[: index + 1]) for index in range(len(components))]
    return tuple(path for path in prefixes if container.get(path, getlink=True) is None)


def _write_open_container(
    array: Any,
    container: h5py.Group | h5py.File,
    path: str | None,
    exact_epoch: int | None,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    h5file = container.file
    document = _read_sidecar(h5file)
    object_path = _write_path(array, container, path)
    relative_path = _relative_path(
        path if path is not None else getattr(array, "name", None)
    )
    existing = _existing_dataset(container, relative_path)
    if existing is not None and not kwargs.get("overwrite", False):
        return _write_core(array, container, path, kwargs)

    _preflight_core_write(array, container, path, kwargs)
    snapshot = _sidecar_snapshot(h5file)
    created_parent_paths = _missing_parent_paths(container, relative_path)
    rollback = (
        _rollback_link(h5file, existing, snapshot) if existing is not None else None
    )
    try:
        dataset = _write_core(array, container, path, kwargs)
        if _dataset_path(dataset) != object_path:  # pragma: no cover - native invariant
            raise RuntimeError("native HDF5 writer returned an unexpected dataset path")
        _commit_sidecar(h5file, document, dataset, exact_epoch)
    except BaseException as operation_error:
        rollback_errors: list[BaseException] = []
        try:
            _restore_dataset(
                container,
                relative_path,
                rollback,
                created_parent_paths,
            )
        except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
            rollback_errors.append(exc)
        try:
            _restore_sidecar(h5file, snapshot)
        except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
            rollback_errors.append(exc)
        if not rollback_errors:
            try:
                _delete_rollback(h5file, rollback)
            except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
                rollback_errors.append(exc)
        if rollback_errors:
            recovery_path = None
            if rollback is not None and rollback.name in h5file:
                recovery_path = rollback.name
            raise _RollbackError(
                operation_error,
                tuple(rollback_errors),
                recovery_path,
            ) from rollback_errors[0]
        raise
    _delete_rollback(h5file, rollback)
    return dataset


def _path_status(path: Path) -> tuple[bool, int | None]:
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return False, None
    if stat.S_ISLNK(status.st_mode):
        raise OSError(f"refusing to overwrite symbolic link: {path}")
    mode = stat.S_IMODE(status.st_mode) if stat.S_ISREG(status.st_mode) else None
    return True, mode


def _filesystem_path(value: Any) -> Path:
    try:
        return Path(os.fsdecode(os.fspath(value)))
    except TypeError as exc:
        raise TypeError("HDF5 target must be a path or h5py File/Group") from exc


def _write_path_transaction(
    array: Any,
    target: Any,
    path: str | None,
    exact_epoch: int | None,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    filepath = _filesystem_path(target)
    existed, target_mode = _path_status(filepath)
    append = bool(kwargs.get("append", False))
    overwrite = bool(kwargs.get("overwrite", False))
    if existed and not (append or overwrite):
        raise OSError(f"File exists: {filepath}")
    if existed and h5py.is_hdf5(filepath):
        with h5py.File(filepath, "r") as existing_file:
            _read_sidecar(existing_file)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=f".{filepath.name}.gwexpy-",
        suffix=".hdf5",
    )
    os.close(descriptor)
    try:
        if existed and append:
            shutil.copy2(filepath, temporary_name)
            mode = "r+"
        else:
            mode = "w"
        with h5py.File(temporary_name, mode) as temporary_file:
            result = _write_open_container(
                array,
                temporary_file,
                path,
                exact_epoch,
                kwargs,
            )
        if target_mode is not None:
            os.chmod(temporary_name, target_mode)
        os.replace(temporary_name, filepath)
        return result
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_core(
    dataset: h5py.Dataset,
    target_class: type[Any],
    kwargs: dict[str, Any],
) -> Any:
    kwargs = dict(kwargs)
    kwargs["array_type"] = target_class
    return _native_reader()(dataset, path=None, **kwargs)


def _read_open_container(
    source: h5py.HLObject,
    path: str | None,
    target_class: type[Any],
    kwargs: dict[str, Any],
) -> Any:
    document = _read_sidecar(source.file)
    if path is not None:
        _relative_path(path)
    dataset = _gwpy_io_hdf5.find_dataset(source, path=path)
    object_path = _dataset_path(dataset)
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    result = _read_core(dataset, target_class, kwargs)
    if not isinstance(result, target_class):  # pragma: no cover - reader invariant
        result = result.view(target_class)

    entry = document["objects"].get(object_path)
    if entry is not None:
        state = entry["metadata"].get(TIME_STATE_KEY)
        if state is not None and state[TIME_STATE_PRECISION_KEY] == "exact":
            result._gwex_t0_gps_ns = state[TIME_STATE_NS_KEY]

    if start is not None:
        start = max(start, result.span[0])
    if end is not None:
        end = min(end, result.span[1])
    if start is not None or end is not None:
        result = result.crop(start, end)
    return result


def register_hdf5_exact_t0_io() -> None:
    """Register TimeSeries-only exact-epoch wrappers for native HDF5."""
    global _BASE_READER, _BASE_WRITER
    from gwexpy.timeseries.timeseries import TimeSeries as GwexTimeSeries

    registry = _io_registry.default_registry
    current_reader = registry.get_reader("hdf5", GwexTimeSeries)
    current_writer = registry.get_writer("hdf5", GwexTimeSeries)
    if getattr(current_reader, _WRAPPER_MARKER, False) and getattr(
        current_writer, _WRAPPER_MARKER, False
    ):
        return

    _BASE_READER = registry.get_reader("hdf5", TimeSeries)
    _BASE_WRITER = registry.get_writer("hdf5", TimeSeries)

    @functools.wraps(_BASE_READER)
    def read_exact(
        source: Any,
        path: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(source, h5py.HLObject):
            return _read_open_container(
                source,
                path,
                GwexTimeSeries,
                dict(kwargs),
            )
        with h5py.File(source, "r") as h5file:
            return _read_open_container(
                h5file,
                path,
                GwexTimeSeries,
                dict(kwargs),
            )

    @functools.wraps(_BASE_WRITER)
    def write_exact(
        array: Any,
        target: Any,
        path: str | None = None,
        **kwargs: Any,
    ) -> h5py.Dataset:
        exact_epoch = _exact_epoch(array)
        if path is not None:
            _relative_path(path)
        if isinstance(target, (h5py.File, h5py.Group)):
            return _write_open_container(
                array,
                target,
                path,
                exact_epoch,
                dict(kwargs),
            )
        return _write_path_transaction(
            array,
            target,
            path,
            exact_epoch,
            dict(kwargs),
        )

    setattr(read_exact, _WRAPPER_MARKER, True)
    setattr(write_exact, _WRAPPER_MARKER, True)
    registry.register_reader("hdf5", GwexTimeSeries, read_exact, force=True)
    registry.register_writer("hdf5", GwexTimeSeries, write_exact, force=True)


register_hdf5_exact_t0_io()


__all__ = [
    "SEC_UNIT",
    "StateVector",
    "StateVectorDict",
    "TimeSeries",
    "TimeSeriesDict",
    "dict_class",
    "identify_hdf5",
    "read_hdf5_array",
    "read_hdf5_dict",
    "read_hdf5_factory",
    "read_hdf5_timeseries",
    "reader",
    "registry",
    "series_class",
    "units",
    "with_read_hdf5",
    "with_write_hdf5",
    "write_hdf5_dict",
    "write_hdf5_series",
]
