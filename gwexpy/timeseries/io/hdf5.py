from __future__ import annotations

import functools
import io
import math
import os
import shutil
import stat
import struct
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np
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

from ._hdf5_exact_epoch import (
    EpochMarker,
    SidecarDocument,
    decode_epoch_marker,
    encode_epoch_marker,
    parse_v2_sidecar,
    record_from_marker,
    serialize_v2_sidecar,
    validate_marker_record,
)

SIDECAR_ATTRIBUTE_V1 = "_gwexpy_sidecar_json_v1"
SIDECAR_ATTRIBUTE_V2 = "_gwexpy_sidecar_json_v2"
TIME_STATE_NS_KEY = "_gwex_t0_gps_ns"

_MISSING = object()
_WRAPPER_MARKER = "_gwexpy_exact_t0_hdf5"
_ROLLBACK_PREFIX = "__gwexpy_t0_rollback_"
_MAX_V2_RECORDS = 10_000
_MAX_V2_BYTES = 8 * 1024 * 1024
_BASE_READER: Callable[..., Any] | None = None
_BASE_WRITER: Callable[..., h5py.Dataset] | None = None

_SidecarSnapshot = tuple[tuple[str, bool, Any], ...]


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


def _exact_epoch(array: Any) -> int | None:
    value = getattr(array, TIME_STATE_NS_KEY, None)
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError("authoritative TimeSeries epoch must be an integer")
    return value


def _external_storage_requested(kwargs: dict[str, Any]) -> bool:
    external = kwargs.get("external")
    if external is None:
        return False
    try:
        return len(external) != 0
    except TypeError:
        return True


def _array_axis_metadata(array: Any) -> tuple[Any, Any]:
    x0 = getattr(array, "x0", None)
    raw_x0 = getattr(x0, "value", x0)
    return raw_x0, getattr(array, "xunit", None)


def _caller_binary64_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{label} must be a finite binary64 scalar")
    if isinstance(value, np.ndarray):
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} must be a finite binary64 scalar")
    try:
        projected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite binary64 scalar") from exc
    if not math.isfinite(projected):
        raise ValueError(f"{label} must be a finite binary64 scalar")
    return projected


def _validate_caller_write_metadata(
    array: Any,
    exact_epoch: int | None,
    kwargs: dict[str, Any],
) -> EpochMarker | None:
    attrs = kwargs.get("attrs")
    if exact_epoch is None:
        if attrs and "epoch" in attrs:
            raw_x0, xunit = _array_axis_metadata(array)
            supplied_epoch = attrs["epoch"]
            if isinstance(supplied_epoch, bytes):
                try:
                    supplied_epoch = supplied_epoch.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("caller HDF5 epoch must use UTF-8") from exc
            supplied_marker = decode_epoch_marker(
                supplied_epoch,
                raw_x0=raw_x0,
                xunit=xunit,
            )
            if supplied_marker is not None:
                raise ValueError(
                    "caller HDF5 epoch claims exact authority for a non-exact array"
                )
        return None
    raw_x0, xunit = _array_axis_metadata(array)
    expected_marker = encode_epoch_marker(
        epoch_ns=exact_epoch,
        raw_x0=raw_x0,
        xunit=xunit,
        token=b"\x00" * 16,
    )
    output_marker = encode_epoch_marker(
        epoch_ns=exact_epoch,
        raw_x0=raw_x0,
        xunit=xunit,
    )
    if not attrs:
        return output_marker
    if "x0" in attrs:
        supplied = struct.pack(
            ">d",
            _caller_binary64_scalar(attrs["x0"], label="caller HDF5 x0"),
        )
        if supplied.hex() != expected_marker.x0_bits:
            raise ValueError("caller HDF5 x0 does not match the TimeSeries axis")
    if "xunit" in attrs:
        supplied_axis = encode_epoch_marker(
            epoch_ns=exact_epoch,
            raw_x0=raw_x0,
            xunit=attrs["xunit"],
            token=b"\x00" * 16,
        ).axis
        if supplied_axis != expected_marker.axis:
            raise ValueError("caller HDF5 xunit does not match the TimeSeries axis")
    if "epoch" in attrs:
        supplied_epoch = attrs["epoch"]
        if isinstance(supplied_epoch, bytes):
            try:
                supplied_epoch = supplied_epoch.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("caller HDF5 epoch must use UTF-8") from exc
        supplied_marker = decode_epoch_marker(
            supplied_epoch,
            raw_x0=raw_x0,
            xunit=xunit,
        )
        if supplied_marker is not None:
            if supplied_marker.epoch_ns != exact_epoch:
                raise ValueError(
                    "caller HDF5 epoch conflicts with the exact TimeSeries epoch"
                )
            output_marker = supplied_marker
        else:
            projected = _caller_binary64_scalar(
                supplied_epoch, label="caller HDF5 epoch"
            )
            if struct.pack(">d", projected).hex() != expected_marker.x0_bits:
                raise ValueError(
                    "caller HDF5 epoch does not match the TimeSeries x0 bits"
                )
    return output_marker


def _native_path_components(
    array: Any,
    path: Any,
) -> tuple[bool, tuple[str, ...]]:
    candidate = path if path is not None else getattr(array, "name", None)
    if isinstance(candidate, bytes):
        try:
            candidate = candidate.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("external HDF5 dataset path must use UTF-8") from exc
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("external HDF5 dataset path must be a non-empty string")
    try:
        candidate.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("HDF5 dataset path must use UTF-8") from exc
    if "\x00" in candidate:
        raise ValueError("external HDF5 dataset path contains NUL")
    absolute = candidate.startswith("/")
    raw_components = candidate.split("/")[1:] if absolute else candidate.split("/")
    if any(component in {"", ".", ".."} for component in raw_components):
        raise ValueError("external HDF5 dataset path has an invalid path component")
    return absolute, tuple(raw_components)


def _native_object_path(
    array: Any,
    container: h5py.Group | h5py.File,
    path: Any,
) -> str:
    absolute, components = _native_path_components(array, path)
    relative = "/".join(components)
    if absolute:
        return relative
    prefix = _group_prefix(container)
    return relative if not prefix else f"{prefix}/{relative}"


def _reject_external_link_traversal(
    array: Any,
    container: h5py.Group | h5py.File,
    path: Any,
) -> None:
    object_path = _native_object_path(array, container, path)
    h5file = container.file
    current: h5py.File | h5py.Group = h5file
    for component in object_path.split("/")[:-1]:
        link = current.get(component, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            raise ValueError("cannot write through an HDF5 external link")
        child = current.get(component)
        if not isinstance(child, (h5py.File, h5py.Group)):
            return
        if child.file.id != h5file.id:
            raise ValueError("cannot write through an HDF5 external link")
        current = child


def _reject_stale_external_sidecar(
    array: Any,
    container: h5py.Group | h5py.File,
    path: str | None,
) -> None:
    _reject_external_link_traversal(array, container, path)
    object_path = _native_object_path(array, container, path)
    document = _read_v2_sidecar(container.file)
    target = container.file.get(object_path)
    if isinstance(target, h5py.Dataset):
        raw_epoch = _text_attribute(target.attrs.get("epoch"), label="epoch")
        raw_xunit = _text_attribute(target.attrs.get("xunit"), label="xunit")
        marker = decode_epoch_marker(
            raw_epoch,
            raw_x0=target.attrs.get("x0"),
            xunit=raw_xunit,
        )
        if marker is not None:
            validate_marker_record(marker, document)
            raise ValueError(
                "external HDF5 storage cannot replace a canonically marked dataset"
            )
    if document is not None and any(
        object_path in record.paths for record in document.records.values()
    ):
        raise ValueError(
            "external HDF5 storage cannot replace a sidecar-managed dataset"
        )


def _local_object_identity(value: h5py.Group | h5py.Dataset) -> int:
    return int(h5py.h5o.get_info(value.id).addr)


def _serialize_marker_observations(
    observations: Iterable[tuple[str, EpochMarker]],
) -> str | None:
    """Merge local marker observations into one canonical v2 document."""
    markers: dict[str, EpochMarker] = {}
    paths: dict[str, list[str]] = {}
    empty_size = len(serialize_v2_sidecar([]).encode("utf-8"))
    payload_size = empty_size
    entry_sizes: dict[str, int] = {}
    for object_path, marker in observations:
        previous = markers.get(marker.lineage_token)
        if previous is None and len(markers) >= _MAX_V2_RECORDS:
            raise ValueError("sidecar exceeds 10000 records")
        if previous is not None and previous != marker:
            raise ValueError("conflicting local HDF5 markers share one lineage token")
        markers[marker.lineage_token] = marker
        representatives = paths.setdefault(marker.lineage_token, [])
        path_added = object_path not in representatives and len(representatives) < 16
        if path_added:
            representatives.append(object_path)
        if previous is None or path_added:
            record = record_from_marker(marker, representatives)
            single_size = len(serialize_v2_sidecar([record]).encode("utf-8"))
            new_entry_size = single_size - empty_size
            old_entry_size = entry_sizes.get(marker.lineage_token)
            proposed_size = payload_size + new_entry_size
            if old_entry_size is None:
                proposed_size += int(bool(entry_sizes))
            else:
                proposed_size -= old_entry_size
            if proposed_size > _MAX_V2_BYTES:
                raise ValueError("sidecar JSON exceeds 8 MiB")
            entry_sizes[marker.lineage_token] = new_entry_size
            payload_size = proposed_size
    if not markers:
        return None
    records = [
        record_from_marker(markers[token], paths[token]) for token in sorted(markers)
    ]
    return serialize_v2_sidecar(records)


def _build_v2_sidecar(h5file: h5py.File) -> str | None:
    """Rebuild v2 records from cycle-safe local hard-link observations."""
    visited_groups: set[int] = set()
    visited_datasets: set[int] = set()

    def visit(
        group: h5py.File | h5py.Group,
        prefix: tuple[str, ...],
    ) -> Iterator[tuple[str, EpochMarker]]:
        group_identity = _local_object_identity(group)
        if group_identity in visited_groups:
            return
        visited_groups.add(group_identity)
        for name in sorted(group.keys()):
            if not prefix and name.startswith(_ROLLBACK_PREFIX):
                continue
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                continue
            child = group.get(name)
            if isinstance(child, h5py.Group):
                yield from visit(child, (*prefix, name))
                continue
            if not isinstance(child, h5py.Dataset):
                continue
            dataset_identity = _local_object_identity(child)
            if dataset_identity in visited_datasets:
                continue
            visited_datasets.add(dataset_identity)
            raw_epoch = _text_attribute(child.attrs.get("epoch"), label="epoch")
            raw_xunit = _text_attribute(child.attrs.get("xunit"), label="xunit")
            marker = decode_epoch_marker(
                raw_epoch,
                raw_x0=child.attrs.get("x0"),
                xunit=raw_xunit,
            )
            if marker is None:
                continue
            object_path = "/".join((*prefix, name))
            yield object_path, marker

    return _serialize_marker_observations(visit(h5file, ()))


def _apply_sidecar_payload(h5file: h5py.File, payload: str | None) -> None:
    """Apply one compacted v2 payload and remove unpublished v1 state."""
    if payload is None:
        if SIDECAR_ATTRIBUTE_V2 in h5file.attrs:
            del h5file.attrs[SIDECAR_ATTRIBUTE_V2]
    else:
        h5file.attrs[SIDECAR_ATTRIBUTE_V2] = payload
    if SIDECAR_ATTRIBUTE_V1 in h5file.attrs:
        del h5file.attrs[SIDECAR_ATTRIBUTE_V1]


def _write_epoch_marker(dataset: h5py.Dataset, marker: EpochMarker) -> None:
    dataset.attrs["epoch"] = marker.text


def _reset_dataset_axis(dataset: h5py.Dataset, marker: EpochMarker) -> None:
    dataset.attrs["x0"] = float(marker.text)
    dataset.attrs["xunit"] = marker.axis.xunit


def _commit_sidecar(
    h5file: h5py.File,
    dataset: h5py.Dataset,
    marker: EpochMarker | None,
) -> None:
    if marker is not None:
        _reset_dataset_axis(dataset, marker)
        validated_marker = decode_epoch_marker(
            marker.text,
            raw_x0=dataset.attrs["x0"],
            xunit=dataset.attrs["xunit"],
        )
        if validated_marker != marker:  # pragma: no cover - codec invariant
            raise RuntimeError("prepared exact-epoch marker changed before commit")
        _write_epoch_marker(dataset, marker)
    payload = _build_v2_sidecar(h5file)
    _apply_sidecar_payload(h5file, payload)


def _sidecar_snapshot(h5file: h5py.File) -> _SidecarSnapshot:
    return tuple(
        (
            name,
            name in h5file.attrs,
            h5file.attrs[name] if name in h5file.attrs else _MISSING,
        )
        for name in (SIDECAR_ATTRIBUTE_V1, SIDECAR_ATTRIBUTE_V2)
    )


def _restore_sidecar(h5file: h5py.File, snapshot: _SidecarSnapshot) -> None:
    for name, exists, raw in snapshot:
        if exists:
            h5file.attrs[name] = raw
        elif name in h5file.attrs:
            del h5file.attrs[name]


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
    sidecar_snapshot: _SidecarSnapshot,
) -> h5py.Group:
    while True:
        name = f"{_ROLLBACK_PREFIX}{uuid.uuid4().hex}"
        if name not in h5file:
            break
    rollback = h5file.create_group(name)
    try:
        rollback["dataset"] = dataset
        for version, (_, exists, raw) in zip(
            ("v1", "v2"), sidecar_snapshot, strict=True
        ):
            rollback.attrs[f"sidecar_{version}_snapshot_present"] = exists
            if exists:
                rollback.attrs[f"sidecar_{version}_snapshot"] = raw
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
    marker: EpochMarker | None,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    h5file = container.file
    _reject_external_link_traversal(array, container, path)
    _read_v2_sidecar(h5file)
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
        _commit_sidecar(h5file, dataset, marker)
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
        raise TypeError(
            "HDF5 target must be a path, file-like object, or h5py File/Group"
        ) from exc


def _is_seekable_filelike(value: Any) -> bool:
    return all(
        callable(getattr(value, method, None))
        for method in ("read", "write", "seek", "tell", "truncate")
    )


def _filelike_snapshot(target: Any) -> tuple[bytes, Any]:
    position = target.tell()
    try:
        target.seek(0)
        payload = target.read()
    finally:
        target.seek(position)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("HDF5 file-like target must use binary I/O")
    return bytes(payload), position


def _replace_filelike_bytes(target: Any, payload: bytes) -> None:
    target.seek(0)
    written = target.write(payload)
    if written is not None and written != len(payload):
        raise OSError("short write while committing HDF5 file-like target")
    target.truncate()
    flush = getattr(target, "flush", None)
    if callable(flush):
        flush()


def _preflight_native_external_write(
    array: Any,
    target: Any,
    path: str | None,
    kwargs: dict[str, Any],
) -> None:
    _native_path_components(array, path)
    if isinstance(target, (h5py.File, h5py.Group)):
        _reject_stale_external_sidecar(array, target, path)
        return

    if _is_seekable_filelike(target):
        position = target.tell()
        try:
            try:
                with h5py.File(target, "r") as h5file:
                    _reject_stale_external_sidecar(array, h5file, path)
            except OSError:
                pass
        finally:
            target.seek(position)
        return

    filepath = _filesystem_path(target)
    append = bool(kwargs.get("append", False))
    write_existing = append or bool(kwargs.get("overwrite", False))
    if write_existing and filepath.exists() and h5py.is_hdf5(filepath):
        with h5py.File(filepath, "r") as h5file:
            if append:
                _reject_stale_external_sidecar(array, h5file, path)
            else:
                _read_v2_sidecar(h5file)


def _create_sibling_transaction_file(filepath: Path) -> Path:
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
    while True:
        temporary_path = filepath.with_name(
            f".{filepath.name}.gwexpy-{uuid.uuid4().hex}.hdf5"
        )
        try:
            descriptor = os.open(temporary_path, flags, 0o666)
        except FileExistsError:  # pragma: no cover - UUID collision
            continue
        os.close(descriptor)
        return temporary_path


def _write_path_transaction(
    array: Any,
    target: Any,
    path: str | None,
    marker: EpochMarker | None,
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
            _read_v2_sidecar(existing_file)

    temporary_path = _create_sibling_transaction_file(filepath)
    try:
        if existed and append:
            shutil.copy2(filepath, temporary_path)
            mode = "r+"
        else:
            mode = "w"
        with h5py.File(temporary_path, mode) as temporary_file:
            result = _write_open_container(
                array,
                temporary_file,
                path,
                marker,
                kwargs,
            )
        if target_mode is not None:
            os.chmod(temporary_path, target_mode)
        os.replace(temporary_path, filepath)
        return result
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_filelike_transaction(
    array: Any,
    target: Any,
    path: str | None,
    marker: EpochMarker | None,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    snapshot, original_position = _filelike_snapshot(target)
    working = io.BytesIO(snapshot)
    mode = "a" if kwargs.get("append", False) else "w"
    with h5py.File(working, mode) as working_file:
        result = _write_open_container(
            array,
            working_file,
            path,
            marker,
            kwargs,
        )
    payload = working.getvalue()
    try:
        _replace_filelike_bytes(target, payload)
    except BaseException as operation_error:
        try:
            _replace_filelike_bytes(target, snapshot)
            target.seek(original_position)
        except BaseException as rollback_error:  # pragma: no cover - broken file object
            raise _RollbackError(
                operation_error,
                (rollback_error,),
                None,
            ) from rollback_error
        raise
    return result


def _read_core(
    dataset: h5py.Dataset,
    kwargs: dict[str, Any],
) -> Any:
    kwargs = dict(kwargs)
    kwargs["array_type"] = TimeSeries
    return _native_reader()(dataset, path=None, **kwargs)


def _read_v2_sidecar(h5file: h5py.File) -> SidecarDocument | None:
    if SIDECAR_ATTRIBUTE_V2 not in h5file.attrs:
        return None
    try:
        return parse_v2_sidecar(h5file.attrs[SIDECAR_ATTRIBUTE_V2])
    except ValueError as exc:
        raise ValueError("invalid exact-epoch sidecar v2") from exc


def _text_attribute(value: Any, *, label: str) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 in HDF5 {label} attribute") from exc
    return value


def _read_open_container(
    source: h5py.HLObject,
    path: str | None,
    target_class: type[Any],
    kwargs: dict[str, Any],
) -> Any:
    if path is not None:
        _relative_path(path)
    dataset = _gwpy_io_hdf5.find_dataset(source, path=path)
    document = _read_v2_sidecar(dataset.file)
    raw_x0 = dataset.attrs.get("x0")
    raw_xunit = _text_attribute(dataset.attrs.get("xunit"), label="xunit")
    raw_epoch = _text_attribute(dataset.attrs.get("epoch"), label="epoch")
    marker = decode_epoch_marker(raw_epoch, raw_x0=raw_x0, xunit=raw_xunit)
    if marker is not None:
        validate_marker_record(marker, document)
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    result = _read_core(dataset, kwargs)
    if not isinstance(result, target_class):  # pragma: no cover - reader invariant
        result = result.view(target_class)

    if marker is not None:
        result._gwex_t0_gps_ns = marker.epoch_ns

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
        write_kwargs = dict(kwargs)
        marker = _validate_caller_write_metadata(array, exact_epoch, write_kwargs)
        if _external_storage_requested(write_kwargs):
            if exact_epoch is not None:
                raise ValueError(
                    "external HDF5 storage is incompatible with exact TimeSeries "
                    "epoch transactions"
                )
            _preflight_native_external_write(
                array,
                target,
                path,
                write_kwargs,
            )
            return _native_writer()(array, target, path=path, **write_kwargs)
        if path is not None:
            _relative_path(path)
        if isinstance(target, (h5py.File, h5py.Group)):
            return _write_open_container(
                array,
                target,
                path,
                marker,
                write_kwargs,
            )
        if _is_seekable_filelike(target):
            return _write_filelike_transaction(
                array,
                target,
                path,
                marker,
                write_kwargs,
            )
        return _write_path_transaction(
            array,
            target,
            path,
            marker,
            write_kwargs,
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
