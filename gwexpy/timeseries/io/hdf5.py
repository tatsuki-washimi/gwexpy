from __future__ import annotations

import functools
import math
import os
import shutil
import stat
import struct
import tempfile
import uuid
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

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
_NATIVE_HANDLER_ATTR = "_gwexpy_exact_t0_native_handler"
_ROLLBACK_PREFIX = "__gwexpy_t0_rollback_"
_MAX_V2_RECORDS = 10_000
_MAX_V2_BYTES = 8 * 1024 * 1024
FILELIKE_COPY_CHUNK = 1024 * 1024
_BASE_READER: Callable[..., Any] | None = None
_BASE_WRITER: Callable[..., h5py.Dataset] | None = None

_SidecarSnapshot = tuple[tuple[str, bool, Any], ...]


@dataclass
class _HandleRecovery:
    """Verified recovery state for one caller-owned HDF5 dataset."""

    h5file: h5py.File
    group: h5py.Group
    path: str
    dataset: h5py.Dataset
    sidecar_snapshot: _SidecarSnapshot


class _RollbackError(RuntimeError):
    """Report an incomplete handle rollback while retaining recovery state."""

    def __init__(
        self,
        operation_error: BaseException,
        rollback_errors: tuple[BaseException, ...],
        recovery_path: str | None,
        *,
        state: str = "indeterminate",
        byte_state: str | None = None,
        position_state: str | None = None,
    ) -> None:
        self.operation_error = operation_error
        self.rollback_errors = rollback_errors
        self.recovery_path = recovery_path
        self.state = state
        self.byte_state = byte_state
        self.position_state = position_state
        self.errors = (operation_error, *rollback_errors)
        message = (
            f"TimeSeries HDF5 write failed and rollback was incomplete; state={state}"
        )
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
    rejected_types = (bool, np.bool_, complex, np.complexfloating)
    if isinstance(value, rejected_types) or np.ndim(value) != 0:
        raise ValueError(f"{label} must be a finite binary64 scalar")
    if isinstance(value, np.ndarray):
        value = value.item()
        if isinstance(value, rejected_types):
            raise ValueError(f"{label} must be a finite binary64 scalar")
    try:
        projected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite binary64 scalar") from exc
    if not math.isfinite(projected):
        raise ValueError(f"{label} must be a finite binary64 scalar")
    return projected


def _caller_text_scalar(value: Any) -> str | bytes | None:
    """Losslessly unbox caller values that HDF5 can store as scalar text."""
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        scalar = value
    else:
        try:
            candidate = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if candidate.ndim != 0:
            return None
        scalar = candidate.item()
    if isinstance(scalar, (str, np.str_)):
        return str(scalar)
    if isinstance(scalar, (bytes, np.bytes_)):
        return bytes(scalar)
    return None


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
            text_scalar = _caller_text_scalar(supplied_epoch)
            if text_scalar is not None:
                supplied_epoch = text_scalar
            if isinstance(supplied_epoch, bytes):
                supplied_epoch = supplied_epoch.decode("latin-1")
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
        text_scalar = _caller_text_scalar(supplied_epoch)
        if text_scalar is not None:
            supplied_epoch = text_scalar
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


def _transaction_coordinate(
    array: Any,
    container: h5py.Group | h5py.File,
    path: Any,
) -> tuple[h5py.Group | h5py.File, str]:
    """Return the rollback container and its decoded relative object path."""
    absolute, components = _native_path_components(array, path)
    relative = "/".join(components)
    if absolute:
        return container.file, relative
    return container, relative


def _reject_private_namespace(
    array: Any,
    target: Any,
    path: Any,
    *,
    preserve_existing: bool,
) -> None:
    _, components = _native_path_components(array, path)
    root_components = components
    if isinstance(target, (h5py.File, h5py.Group)):
        object_path = _native_object_path(array, target, path)
        root_components = tuple(object_path.split("/"))
    if root_components and root_components[0].startswith(_ROLLBACK_PREFIX):
        raise ValueError("the root HDF5 rollback namespace is private")
    if isinstance(target, (h5py.File, h5py.Group)):
        _reject_private_resolution(array, target, path)
        return
    if not preserve_existing:
        return
    if _is_seekable_filelike(target):
        position = target.tell()
        try:
            try:
                with h5py.File(target, "r") as h5file:
                    _reject_private_resolution(array, h5file, path)
            except OSError:
                pass
        finally:
            target.seek(position)
        return
    filepath = _filesystem_path(target)
    if filepath.exists() and h5py.is_hdf5(filepath):
        with h5py.File(filepath, "r") as h5file:
            _reject_private_resolution(array, h5file, path)


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
    path: Any,
) -> None:
    _reject_external_link_traversal(array, container, path)
    object_path = _native_object_path(array, container, path)
    document = _read_v2_sidecar(container.file)
    target = container.file.get(object_path)
    if isinstance(target, h5py.Dataset):
        marker = _decode_dataset_marker(target)
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


def _reject_external_document_replacement(h5file: h5py.File) -> None:
    """Reject a whole-file external write that would discard exact state."""
    document = _read_v2_sidecar(h5file)
    if _build_v2_sidecar(h5file) is not None:
        raise ValueError(
            "external HDF5 storage cannot replace a canonically marked dataset"
        )
    if document is not None and document.records:
        raise ValueError(
            "external HDF5 storage cannot replace a sidecar-managed dataset"
        )


def _local_object_identity(value: h5py.Group | h5py.Dataset) -> int:
    return int(h5py.h5o.get_info(value.id).addr)


def _serialize_marker_observations(
    observations: Iterable[tuple[str | None, EpochMarker]],
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
        path_added = (
            object_path is not None
            and object_path not in representatives
            and len(representatives) < 16
        )
        if path_added:
            assert object_path is not None
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


def _iter_raw_links(group: h5py.File | h5py.Group) -> Iterator[tuple[bytes, int]]:
    """Yield link names in raw-byte order without materializing group width."""
    index = 0
    link_count = len(group)

    def capture(name: bytes, info: Any) -> tuple[bytes, int]:
        return name, int(info.type)

    while index < link_count:
        captured, index = group.id.links.iterate(
            capture,
            info=True,
            idx=index,
            idx_type=h5py.h5.INDEX_NAME,
            order=h5py.h5.ITER_INC,
        )
        if captured is None:
            raise RuntimeError("HDF5 link iteration ended before declared group width")
        name, link_type = captured
        yield name, link_type


def _public_hard_object_reachable(
    h5file: h5py.File,
    target: h5py.File | h5py.Group | h5py.Dataset,
) -> bool:
    """Return whether a non-private root hard-link path reaches ``target``."""
    if target.file.id != h5file.id:
        return False
    target_identity = _local_object_identity(target)
    root_identity = _local_object_identity(h5file)
    if target_identity == root_identity:
        return True
    visited = {root_identity}
    stack: list[tuple[h5py.File | h5py.Group, Iterator[tuple[bytes, int]]]] = [
        (h5file, _iter_raw_links(h5file))
    ]
    while stack:
        group, links = stack[-1]
        try:
            name, link_type = next(links)
        except StopIteration:
            stack.pop()
            continue
        if len(stack) == 1 and name.startswith(_ROLLBACK_PREFIX.encode("ascii")):
            continue
        if link_type != h5py.h5l.TYPE_HARD:
            continue
        child = group[name]
        if not isinstance(child, (h5py.Group, h5py.Dataset)):
            continue
        identity = _local_object_identity(child)
        if identity == target_identity:
            return True
        if not isinstance(child, h5py.Group) or identity in visited:
            continue
        visited.add(identity)
        stack.append((child, _iter_raw_links(child)))
    return False


def _reject_private_resolution(
    array: Any,
    container: h5py.File | h5py.Group,
    path: Any,
) -> None:
    """Reject a parent that resolves only through the private root namespace."""
    object_path = _native_object_path(array, container, path)
    h5file = container.file
    current: h5py.File | h5py.Group = h5file
    for component in object_path.split("/")[:-1]:
        link = current.get(component, getlink=True)
        if link is None:
            break
        if isinstance(link, h5py.ExternalLink):
            raise ValueError("cannot write through an HDF5 external link")
        child = current.get(component)
        if not isinstance(child, h5py.Group):
            break
        if child.file.id != h5file.id:
            raise ValueError("cannot write through an HDF5 external link")
        current = child
    if not _public_hard_object_reachable(h5file, current):
        raise ValueError(
            "the resolved HDF5 parent is in the private rollback namespace"
        )


def _diagnostic_path(components: tuple[bytes, ...]) -> str | None:
    """Return a bounded UTF-8 path, or omit a non-authoritative spelling."""
    try:
        decoded = tuple(component.decode("utf-8") for component in components)
        if any(component in {"", ".", ".."} for component in decoded):
            return None
        path = "/".join(decoded)
        encoded = path.encode("utf-8")
    except UnicodeError:
        return None
    if len(encoded) > 4096:
        return None
    return path


def _build_v2_sidecar(h5file: h5py.File) -> str | None:
    """Rebuild v2 records from cycle-safe local hard-link observations."""
    visited_groups: set[int] = set()
    visited_datasets: set[int] = set()

    def observations() -> Iterator[tuple[str | None, EpochMarker]]:
        visited_groups.add(_local_object_identity(h5file))
        stack: list[tuple[h5py.File | h5py.Group, Iterator[tuple[bytes, int]]]] = [
            (h5file, _iter_raw_links(h5file))
        ]
        path_components: list[bytes] = []
        while stack:
            group, links = stack[-1]
            try:
                name, link_type = next(links)
            except StopIteration:
                stack.pop()
                if stack:
                    path_components.pop()
                continue
            if len(stack) == 1 and name.startswith(_ROLLBACK_PREFIX.encode("ascii")):
                continue
            if link_type != h5py.h5l.TYPE_HARD:
                continue
            child = group[name]
            if isinstance(child, h5py.Group):
                group_identity = _local_object_identity(child)
                if group_identity in visited_groups:
                    continue
                visited_groups.add(group_identity)
                path_components.append(name)
                stack.append((child, _iter_raw_links(child)))
                continue
            if not isinstance(child, h5py.Dataset):
                continue
            dataset_identity = _local_object_identity(child)
            if dataset_identity in visited_datasets:
                continue
            visited_datasets.add(dataset_identity)
            marker = _decode_dataset_marker(child)
            if marker is None:
                continue
            yield _diagnostic_path((*path_components, name)), marker

    return _serialize_marker_observations(observations())


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
    if not _public_hard_object_reachable(h5file, dataset):
        raise RuntimeError(
            "HDF5 output dataset is not reachable through public hard links"
        )
    if marker is None and _decode_dataset_marker(dataset) is not None:
        raise RuntimeError(
            "non-exact HDF5 output unexpectedly contains an exact epoch marker"
        )
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
    if marker is not None:
        if payload is None:
            raise RuntimeError("exact marker lineage token is missing from sidecar")
        try:
            document = parse_v2_sidecar(payload)
            record = validate_marker_record(marker, document)
        except ValueError as exc:  # pragma: no cover - builder invariant
            raise RuntimeError("exact marker sidecar postcondition failed") from exc
        if record is None:
            raise RuntimeError("exact marker lineage token is missing from sidecar")
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


def _restore_sidecar_attribute(
    h5file: h5py.File,
    snapshot: tuple[str, bool, Any],
) -> None:
    name, exists, raw = snapshot
    if exists:
        h5file.attrs[name] = raw
    elif name in h5file.attrs:
        del h5file.attrs[name]


def _restore_root_sidecars(
    h5file: h5py.File,
    snapshot: _SidecarSnapshot,
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    for attribute_snapshot in snapshot:
        try:
            _restore_sidecar_attribute(h5file, attribute_snapshot)
        except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
            errors.append(exc)
    return tuple(errors)


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
    path: Any,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    return _native_writer()(array, container, path=path, **kwargs)


def _write_dataset_once(
    array: Any,
    container: h5py.Group | h5py.File,
    path: Any,
    marker: EpochMarker | None,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    """Write one native dataset and commit its exact-time metadata."""
    object_path = _native_object_path(array, container, path)
    dataset = _write_core(array, container, path, kwargs)
    if _dataset_path(dataset) != object_path:  # pragma: no cover - native invariant
        raise RuntimeError("native HDF5 writer returned an unexpected dataset path")
    _commit_sidecar(container.file, dataset, marker)
    return dataset


def _write_disposable_stage(
    array: Any,
    stage: BinaryIO,
    path: Any,
    marker: EpochMarker | None,
    kwargs: dict[str, Any],
    *,
    mode: str,
) -> h5py.Dataset:
    """Write a disposable HDF5 image without in-file recovery objects."""
    with h5py.File(stage, mode) as h5file:
        _reject_private_resolution(array, h5file, path)
        _reject_external_link_traversal(array, h5file, path)
        _read_v2_sidecar(h5file)
        coordinate, relative_path = _transaction_coordinate(array, h5file, path)
        _existing_dataset(coordinate, relative_path)
        return _write_dataset_once(array, h5file, path, marker, kwargs)


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


def _new_handle_recovery_path(h5file: h5py.File) -> str:
    while True:
        name = f"{_ROLLBACK_PREFIX}{uuid.uuid4().hex}"
        if name not in h5file:
            return f"/{name}"


def _create_handle_recovery_group(h5file: h5py.File, path: str) -> h5py.Group:
    return h5file.create_group(path)


def _unlink_partial_handle_recovery(h5file: h5py.File, path: str) -> None:
    if path in h5file:
        del h5file[path]


def _link_handle_recovery_dataset(
    rollback: h5py.Group,
    dataset: h5py.Dataset,
) -> None:
    rollback["dataset"] = dataset


def _store_handle_sidecar_snapshot(
    rollback: h5py.Group,
    version: str,
    snapshot: tuple[str, bool, Any],
) -> None:
    _, exists, raw = snapshot
    rollback.attrs[f"sidecar_{version}_snapshot_present"] = exists
    if exists:
        rollback.attrs[f"sidecar_{version}_snapshot"] = raw


def _sidecar_values_equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.array_equal(left, right))
    except (TypeError, ValueError):
        return False


def _verify_handle_recovery(recovery: _HandleRecovery) -> None:
    recovery.h5file.flush()
    if recovery.path not in recovery.h5file:
        raise RuntimeError("HDF5 recovery group is not durably linked")
    linked = recovery.group.get("dataset")
    if not isinstance(linked, h5py.Dataset):
        raise RuntimeError("HDF5 recovery dataset link is missing")
    if h5py.h5o.get_info(linked.id).addr != h5py.h5o.get_info(recovery.dataset.id).addr:
        raise RuntimeError("HDF5 recovery dataset link changed object identity")
    for version, (_, exists, raw) in zip(
        ("v1", "v2"), recovery.sidecar_snapshot, strict=True
    ):
        present_name = f"sidecar_{version}_snapshot_present"
        if present_name not in recovery.group.attrs:
            raise RuntimeError(f"HDF5 recovery {version} presence flag is missing")
        if bool(recovery.group.attrs[present_name]) != exists:
            raise RuntimeError(f"HDF5 recovery {version} presence flag changed")
        snapshot_name = f"sidecar_{version}_snapshot"
        if exists:
            if snapshot_name not in recovery.group.attrs or not _sidecar_values_equal(
                recovery.group.attrs[snapshot_name], raw
            ):
                raise RuntimeError(f"HDF5 recovery {version} snapshot changed")
        elif snapshot_name in recovery.group.attrs:
            raise RuntimeError(
                f"HDF5 recovery {version} snapshot is unexpectedly present"
            )


def _linked_handle_recovery_path(recovery: _HandleRecovery) -> str | None:
    return recovery.path if recovery.path in recovery.h5file else None


def _unlink_handle_recovery(recovery: _HandleRecovery) -> None:
    _unlink_partial_handle_recovery(recovery.h5file, recovery.path)
    if recovery.group.id.valid:
        recovery.group.id.close()


def _close_unlinked_handle_recovery(
    recovery: _HandleRecovery,
) -> tuple[BaseException, ...]:
    if (
        _linked_handle_recovery_path(recovery) is not None
        or not recovery.group.id.valid
    ):
        return ()
    try:
        recovery.group.id.close()
    except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
        return (exc,)
    return ()


def _prepare_handle_recovery(
    h5file: h5py.File,
    dataset: h5py.Dataset,
    sidecar_snapshot: _SidecarSnapshot,
) -> _HandleRecovery:
    path = _new_handle_recovery_path(h5file)
    try:
        rollback = _create_handle_recovery_group(h5file, path)
    except BaseException as operation_error:
        try:
            _unlink_partial_handle_recovery(h5file, path)
        except BaseException as cleanup_error:
            cleanup_errors = [cleanup_error]
            try:
                recovery_path = path if path in h5file else None
            except BaseException as inspection_error:
                cleanup_errors.append(inspection_error)
                recovery_path = path
            raise _RollbackError(
                operation_error,
                tuple(cleanup_errors),
                recovery_path,
                state="old",
            ) from cleanup_error
        raise
    recovery = _HandleRecovery(
        h5file=h5file,
        group=rollback,
        path=path,
        dataset=dataset,
        sidecar_snapshot=sidecar_snapshot,
    )
    try:
        _link_handle_recovery_dataset(rollback, dataset)
        for version, snapshot in zip(("v1", "v2"), sidecar_snapshot, strict=True):
            _store_handle_sidecar_snapshot(
                rollback,
                version,
                snapshot,
            )
        _verify_handle_recovery(recovery)
    except BaseException as operation_error:
        try:
            _unlink_handle_recovery(recovery)
        except BaseException as cleanup_error:
            cleanup_errors = [cleanup_error]
            cleanup_errors.extend(_close_unlinked_handle_recovery(recovery))
            raise _RollbackError(
                operation_error,
                tuple(cleanup_errors),
                _linked_handle_recovery_path(recovery),
                state="old",
            ) from cleanup_error
        raise
    return recovery


def _remove_or_recreate_recovery(
    recovery: _HandleRecovery,
) -> tuple[
    _HandleRecovery | None,
    tuple[BaseException, ...],
    str | None,
]:
    linked_path = _linked_handle_recovery_path(recovery)
    if linked_path is not None:
        return recovery, (), linked_path

    errors = list(_close_unlinked_handle_recovery(recovery))
    try:
        recreated = _prepare_handle_recovery(
            recovery.h5file,
            recovery.dataset,
            recovery.sidecar_snapshot,
        )
    except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
        errors.append(exc)
        retained_path = exc.recovery_path if isinstance(exc, _RollbackError) else None
        return None, tuple(errors), retained_path
    return recreated, tuple(errors), recreated.path


def _restore_dataset(
    container: h5py.Group | h5py.File,
    candidate_path: str,
    original_dataset: h5py.Dataset | None,
    created_parent_paths: tuple[str, ...],
) -> None:
    current = container.get(candidate_path)
    current_is_original = (
        isinstance(current, h5py.Dataset)
        and original_dataset is not None
        and h5py.h5o.get_info(current.id).addr
        == h5py.h5o.get_info(original_dataset.id).addr
    )
    if not current_is_original:
        if container.get(candidate_path, getlink=True) is not None:
            del container[candidate_path]
        if original_dataset is not None:
            container[candidate_path] = original_dataset
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


def _public_dataset_is_restored(
    container: h5py.Group | h5py.File,
    candidate_path: str,
    original_dataset: h5py.Dataset | None,
    created_parent_paths: tuple[str, ...],
) -> bool:
    if original_dataset is None:
        if container.get(candidate_path, getlink=True) is not None:
            return False
    else:
        candidate = container.get(candidate_path)
        if not isinstance(candidate, h5py.Dataset):
            return False
        if (
            h5py.h5o.get_info(candidate.id).addr
            != h5py.h5o.get_info(original_dataset.id).addr
        ):
            return False
    return all(
        container.get(path, getlink=True) is None for path in created_parent_paths
    )


def _root_sidecars_are_restored(
    h5file: h5py.File,
    snapshot: _SidecarSnapshot,
) -> bool:
    for name, exists, raw in snapshot:
        if (name in h5file.attrs) != exists:
            return False
        if exists and not _sidecar_values_equal(h5file.attrs[name], raw):
            return False
    return True


def _restore_public_dataset(
    container: h5py.Group | h5py.File,
    candidate_path: str,
    original_dataset: h5py.Dataset | None,
    created_parent_paths: tuple[str, ...],
) -> tuple[tuple[BaseException, ...], bool]:
    errors: list[BaseException] = []
    try:
        _restore_dataset(
            container,
            candidate_path,
            original_dataset,
            created_parent_paths,
        )
    except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
        errors.append(exc)
    try:
        restored = _public_dataset_is_restored(
            container,
            candidate_path,
            original_dataset,
            created_parent_paths,
        )
    except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
        errors.append(exc)
        restored = False
    if not restored:
        errors.append(RuntimeError("HDF5 public dataset rollback postcondition failed"))
    return tuple(errors), restored


def _write_open_container(
    array: Any,
    container: h5py.Group | h5py.File,
    path: Any,
    marker: EpochMarker | None,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    h5file = container.file
    _reject_private_resolution(array, container, path)
    _reject_external_link_traversal(array, container, path)
    _read_v2_sidecar(h5file)
    coordinate, relative_path = _transaction_coordinate(array, container, path)
    existing = _existing_dataset(coordinate, relative_path)
    if existing is not None and not kwargs.get("overwrite", False):
        return _write_core(array, container, path, kwargs)

    snapshot = _sidecar_snapshot(h5file)
    created_parent_paths = _missing_parent_paths(coordinate, relative_path)
    recovery = (
        _prepare_handle_recovery(h5file, existing, snapshot)
        if existing is not None
        else None
    )
    try:
        dataset = _write_dataset_once(array, container, path, marker, kwargs)
        if recovery is not None:
            _unlink_handle_recovery(recovery)
    except BaseException as operation_error:
        rollback_errors: list[BaseException] = []
        recovery_path: str | None = None
        if recovery is not None:
            recovery, recovery_errors, recovery_path = _remove_or_recreate_recovery(
                recovery
            )
            rollback_errors.extend(recovery_errors)
        public_restore_errors, public_restored = _restore_public_dataset(
            coordinate,
            relative_path,
            existing,
            created_parent_paths,
        )
        sidecar_restore_errors = list(_restore_root_sidecars(h5file, snapshot))
        try:
            sidecars_restored = _root_sidecars_are_restored(h5file, snapshot)
        except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
            sidecar_restore_errors.append(exc)
            sidecars_restored = False
        if not sidecars_restored:
            sidecar_restore_errors.append(
                RuntimeError("HDF5 root sidecar rollback postcondition failed")
            )
        rollback_errors.extend(public_restore_errors)
        rollback_errors.extend(sidecar_restore_errors)
        state = "old" if public_restored and sidecars_restored else "indeterminate"
        if state == "old" and recovery is not None:
            try:
                _unlink_handle_recovery(recovery)
            except BaseException as exc:  # pragma: no cover - catastrophic HDF5 failure
                rollback_errors.append(exc)
                recovery_path = _linked_handle_recovery_path(recovery)
                rollback_errors.extend(_close_unlinked_handle_recovery(recovery))
            else:
                recovery_path = None
        if rollback_errors:
            if recovery is not None and recovery_path is None:
                recovery_path = _linked_handle_recovery_path(recovery)
            raise _RollbackError(
                operation_error,
                tuple(rollback_errors),
                recovery_path,
                state=state,
            ) from rollback_errors[0]
        raise
    return dataset


def _path_status(path: Path) -> tuple[bool, int | None]:
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return False, None
    if not stat.S_ISREG(status.st_mode):
        raise OSError(f"refusing to overwrite non-regular file: {path}")
    if status.st_nlink > 1:
        raise OSError(f"refusing to replace file with multiple hard links: {path}")
    return True, stat.S_IMODE(status.st_mode)


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


def _copy_filelike(
    source: Any,
    destination: Any,
    *,
    chunk_size: int = FILELIKE_COPY_CHUNK,
) -> int:
    """Copy binary data from current positions using bounded reads."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("HDF5 file-like copy chunk size must be positive")
    copied = 0
    while True:
        chunk = source.read(chunk_size)
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("HDF5 file-like read must return bytes-like data")
        if len(chunk) > chunk_size:
            raise OSError("oversized read while copying HDF5 file-like target")
        if not chunk:
            return copied
        remaining = memoryview(chunk)
        while remaining:
            written = destination.write(remaining)
            if type(written) is not int or written <= 0 or written > len(remaining):
                raise OSError("invalid write count while copying HDF5 file-like target")
            copied += written
            remaining = remaining[written:]


def _copy_filelike_image(source: Any, destination: Any) -> int:
    source.seek(0)
    destination.seek(0)
    size = _copy_filelike(source, destination)
    destination.truncate(size)
    return size


def _flush_filelike(target: Any, *, sync: bool = False) -> None:
    flush = getattr(target, "flush", None)
    if callable(flush):
        flush()
    if sync:
        os.fsync(target.fileno())


def _create_filelike_backup() -> tuple[BinaryIO, Path]:
    descriptor, name = tempfile.mkstemp(
        prefix="gwexpy-hdf5-backup-",
        suffix=".hdf5",
    )
    path = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        backup = os.fdopen(descriptor, "w+b")
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return cast(BinaryIO, backup), path


def _create_filelike_working() -> BinaryIO:
    return cast(BinaryIO, tempfile.TemporaryFile(mode="w+b"))


def _cleanup_filelike_temporaries(
    working: BinaryIO | None,
    backup: BinaryIO | None,
    backup_path: Path | None,
    *,
    retain_backup: bool,
) -> tuple[tuple[BaseException, ...], str | None]:
    errors: list[BaseException] = []
    for resource in (working, backup):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            errors.append(error)
    if backup_path is not None and not retain_backup:
        try:
            backup_path.unlink()
        except FileNotFoundError:
            pass
        except BaseException as error:
            errors.append(error)
    recovery_path = None
    if backup_path is not None:
        try:
            if backup_path.exists():
                recovery_path = str(backup_path)
        except BaseException as error:
            errors.append(error)
    return tuple(errors), recovery_path


def _warn_filelike_cleanup(
    errors: tuple[BaseException, ...],
    recovery_path: str | None,
) -> None:
    details = "; ".join(str(error) for error in errors)
    warnings.warn(
        "TimeSeries HDF5 write committed; state=new; "
        f"temporary cleanup failed: {details}; recovery_path={recovery_path!r}",
        ResourceWarning,
        stacklevel=2,
    )


def _preflight_native_external_write(
    array: Any,
    target: Any,
    path: Any,
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
                _reject_external_document_replacement(h5file)


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
    path: Any,
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
        python_mode = "r+b" if mode == "r+" else "w+b"
        with temporary_path.open(python_mode) as temporary_file:
            result = _write_disposable_stage(
                array,
                cast(BinaryIO, temporary_file),
                path,
                marker,
                kwargs,
                mode=mode,
            )
        if target_mode is not None:
            os.chmod(temporary_path, target_mode)
        os.replace(temporary_path, filepath)
    except BaseException as operation_error:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except BaseException as cleanup_error:
                raise _RollbackError(
                    operation_error,
                    (cleanup_error,),
                    str(temporary_path),
                    state="old",
                ) from cleanup_error
        raise
    return result


def _write_filelike_transaction(
    array: Any,
    target: Any,
    path: Any,
    marker: EpochMarker | None,
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    original_position = target.tell()
    backup: BinaryIO | None = None
    backup_path: Path | None = None
    working: BinaryIO | None = None
    try:
        backup, backup_path = _create_filelike_backup()
        working = _create_filelike_working()
        _copy_filelike_image(target, backup)
        _flush_filelike(backup, sync=True)
        _copy_filelike_image(target, working)
        working.seek(original_position)
        mode = "a" if kwargs.get("append", False) else "w"
        result = _write_disposable_stage(
            array,
            working,
            path,
            marker,
            kwargs,
            mode=mode,
        )
        committed_position = working.tell()
    except BaseException as operation_error:
        rollback_errors: list[BaseException] = []
        position_state = "old"
        try:
            target.seek(original_position)
        except BaseException as rollback_error:
            position_state = "indeterminate"
            rollback_errors.append(rollback_error)
        cleanup_errors, recovery_path = _cleanup_filelike_temporaries(
            working,
            backup,
            backup_path,
            retain_backup=bool(rollback_errors),
        )
        rollback_errors.extend(cleanup_errors)
        if rollback_errors:
            raise _RollbackError(
                operation_error,
                tuple(rollback_errors),
                recovery_path,
                state="old" if position_state == "old" else "indeterminate",
                byte_state="old",
                position_state=position_state,
            ) from rollback_errors[0]
        raise

    try:
        _copy_filelike_image(working, target)
        _flush_filelike(target)
        target.seek(committed_position)
    except BaseException as operation_error:
        rollback_errors = []
        byte_state = "indeterminate"
        position_state = "indeterminate"
        try:
            _copy_filelike_image(backup, target)
            _flush_filelike(target)
            byte_state = "old"
        except BaseException as rollback_error:
            rollback_errors.append(rollback_error)
        try:
            target.seek(original_position)
            position_state = "old"
        except BaseException as rollback_error:
            rollback_errors.append(rollback_error)
        cleanup_errors, recovery_path = _cleanup_filelike_temporaries(
            working,
            backup,
            backup_path,
            retain_backup=bool(rollback_errors),
        )
        rollback_errors.extend(cleanup_errors)
        if rollback_errors:
            raise _RollbackError(
                operation_error,
                tuple(rollback_errors),
                recovery_path,
                state=(
                    "old" if byte_state == position_state == "old" else "indeterminate"
                ),
                byte_state=byte_state,
                position_state=position_state,
            ) from rollback_errors[0]
        raise

    cleanup_errors, recovery_path = _cleanup_filelike_temporaries(
        working,
        backup,
        backup_path,
        retain_backup=False,
    )
    if cleanup_errors:
        _warn_filelike_cleanup(cleanup_errors, recovery_path)
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
        bounded = _read_bounded_text_attribute(
            h5file,
            SIDECAR_ATTRIBUTE_V2,
            limit=_MAX_V2_BYTES,
            reject_invalid=True,
        )
        if bounded is None:  # pragma: no cover - existence checked above
            raise ValueError("sidecar v2 attribute disappeared during read")
        raw, truncated = bounded
        if truncated:
            raise ValueError("sidecar JSON exceeds 8 MiB")
        return parse_v2_sidecar(raw)
    except ValueError as exc:
        raise ValueError("invalid exact-epoch sidecar v2") from exc


def _logical_fixed_text(raw: bytes, padding: int) -> bytes:
    if padding == h5py.h5t.STR_NULLPAD:
        return raw.rstrip(b"\0")
    if padding == h5py.h5t.STR_NULLTERM:
        return raw.partition(b"\0")[0]
    if padding == h5py.h5t.STR_SPACEPAD:
        return raw.rstrip(b" ")
    raise ValueError("unsupported HDF5 fixed-string padding")


def _read_bounded_text_attribute(
    owner: h5py.File | h5py.Group | h5py.Dataset,
    name: str,
    *,
    limit: int,
    reject_invalid: bool = False,
) -> tuple[bytes, bool] | None:
    """Read one scalar string attribute into at most ``limit + 1`` Python bytes.

    HDF5 may still allocate the source vlen value internally while converting it;
    this bound applies to the Python-owned destination buffer.
    """
    try:
        attribute = owner.attrs.get_id(name)
    except KeyError:
        return None
    try:
        type_id = attribute.get_type()
        try:
            if type_id.get_class() != h5py.h5t.STRING:
                if reject_invalid:
                    raise ValueError(f"HDF5 {name} attribute must be text")
                return None
            if attribute.shape != ():
                if reject_invalid:
                    raise ValueError(f"HDF5 {name} attribute must be scalar")
                return None
            variable = bool(type_id.is_variable_str())
            source_size = int(type_id.get_size())
            complete_fixed = not variable and source_size <= limit + 1
            read_size = (
                limit + 1 if variable else source_size if complete_fixed else limit + 1
            )
            read_size = max(read_size, 1)
            destination = np.empty((), dtype=f"S{read_size}")
            try:
                if complete_fixed:
                    attribute.read(destination, mtype=type_id)
                else:
                    attribute.read(destination)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"cannot read HDF5 {name} attribute safely") from exc
            if variable:
                raw = bytes(destination[()])
                return raw, len(raw) > limit
            if complete_fixed:
                raw = _logical_fixed_text(
                    destination.tobytes(),
                    int(type_id.get_strpad()),
                )
                return raw, len(raw) > limit
            raw = destination.tobytes().rstrip(b"\0")
            return raw, True
        finally:
            type_id.close()
    finally:
        attribute.close()


def _marker_epoch_candidate(dataset: h5py.Dataset) -> str | None:
    bounded = _read_bounded_text_attribute(dataset, "epoch", limit=4_096)
    if bounded is None:
        return None
    raw_epoch, truncated = bounded
    text = raw_epoch.decode("latin-1")
    try:
        provisional = decode_epoch_marker(text, raw_x0=0.0, xunit="s")
    except ValueError:
        if truncated:
            raise ValueError("HDF5 marker epoch exceeds 4096 bytes") from None
        return text
    if provisional is None:
        return None
    if truncated:
        raise ValueError("HDF5 marker epoch exceeds 4096 bytes")
    return text


def _marker_xunit(dataset: h5py.Dataset) -> str:
    bounded = _read_bounded_text_attribute(dataset, "xunit", limit=255)
    if bounded is None:
        raise ValueError("HDF5 marker xunit must be a scalar string")
    raw_xunit, truncated = bounded
    if truncated:
        raise ValueError("HDF5 marker xunit exceeds 255 UTF-8 bytes")
    try:
        return raw_xunit.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid UTF-8 in HDF5 marker xunit attribute") from exc


def _marker_x0(dataset: h5py.Dataset) -> float:
    try:
        attribute = dataset.attrs.get_id("x0")
    except KeyError as exc:
        raise ValueError("HDF5 marker x0 must be a finite binary64 scalar") from exc
    try:
        if attribute.shape != ():
            raise ValueError("HDF5 marker x0 must be a finite binary64 scalar")
        type_id = attribute.get_type()
        try:
            type_class = type_id.get_class()
        finally:
            type_id.close()
        if type_class not in (h5py.h5t.INTEGER, h5py.h5t.FLOAT):
            raise ValueError("HDF5 marker x0 must be a finite binary64 scalar")
        destination = np.empty((), dtype=np.float64)
        try:
            attribute.read(destination)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("HDF5 marker x0 must be a finite binary64 scalar") from exc
        projected = float(destination[()])
        if not math.isfinite(projected):
            raise ValueError("HDF5 marker x0 must be a finite binary64 scalar")
        return projected
    finally:
        attribute.close()


def _decode_dataset_marker(dataset: h5py.Dataset) -> EpochMarker | None:
    raw_epoch = _marker_epoch_candidate(dataset)
    if raw_epoch is None:
        return None
    return decode_epoch_marker(
        raw_epoch,
        raw_x0=_marker_x0(dataset),
        xunit=_marker_xunit(dataset),
    )


def _read_open_container(
    source: h5py.HLObject,
    path: Any,
    target_class: type[Any],
    kwargs: dict[str, Any],
) -> Any:
    if path is not None:
        _native_path_components(source, path)
    dataset = _gwpy_io_hdf5.find_dataset(source, path=path)
    document = _read_v2_sidecar(dataset.file)
    marker = _decode_dataset_marker(dataset)
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

    reader_wrapped = bool(getattr(current_reader, _WRAPPER_MARKER, False))
    writer_wrapped = bool(getattr(current_writer, _WRAPPER_MARKER, False))
    if reader_wrapped != writer_wrapped:
        raise RuntimeError("incomplete TimeSeries HDF5 wrapper registry pair")
    if reader_wrapped:
        recovered: list[Callable[..., Any]] = []
        for label, wrapper in (
            ("reader", current_reader),
            ("writer", current_writer),
        ):
            native = getattr(wrapper, _NATIVE_HANDLER_ATTR, None)
            if (
                not callable(native)
                or native is wrapper
                or bool(getattr(native, _WRAPPER_MARKER, False))
            ):
                raise RuntimeError(
                    f"invalid saved native HDF5 {label} handler on wrapper"
                )
            recovered.append(native)
        _BASE_READER, _BASE_WRITER = recovered
        return

    _BASE_READER = registry.get_reader("hdf5", TimeSeries)
    _BASE_WRITER = registry.get_writer("hdf5", TimeSeries)
    if (
        not callable(_BASE_READER)
        or not callable(_BASE_WRITER)
        or bool(getattr(_BASE_READER, _WRAPPER_MARKER, False))
        or bool(getattr(_BASE_WRITER, _WRAPPER_MARKER, False))
    ):
        raise RuntimeError("invalid native TimeSeries HDF5 registry handlers")

    @functools.wraps(_BASE_READER)
    def read_exact(
        source: Any,
        path: str | bytes | None = None,
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
        path: str | bytes | None = None,
        **kwargs: Any,
    ) -> h5py.Dataset:
        exact_epoch = _exact_epoch(array)
        write_kwargs = dict(kwargs)
        marker = _validate_caller_write_metadata(array, exact_epoch, write_kwargs)
        _reject_private_namespace(
            array,
            target,
            path,
            preserve_existing=bool(write_kwargs.get("append", False)),
        )
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
            _native_path_components(array, path)
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
    setattr(read_exact, _NATIVE_HANDLER_ATTR, _BASE_READER)
    setattr(write_exact, _NATIVE_HANDLER_ATTR, _BASE_WRITER)
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
