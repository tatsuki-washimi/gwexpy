"""Registry HDF5 hooks for durable Spectrogram provenance sidecars.

Provenance-aware reads and sidecar updates are serialized within this Python
process.  It provides rollback for a single path or open HDF5 handle, but does not claim a
cross-process transaction; that broader HDF5 atomicity contract belongs to
the later Wave3 work.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
from gwpy.io import hdf5 as gwpy_hdf5
from gwpy.io.hdf5 import identify_hdf5
from gwpy.io.registry import default_registry as io_registry
from gwpy.spectrogram import Spectrogram as BaseSpectrogram

from ..provenance import (
    HDF5_PROVENANCE_ATTRIBUTE,
    MAX_HDF5_PROVENANCE_SIDECAR_BYTES,
    ProvenanceSidecarError,
    validated_provenance,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# Locks are keyed by the HDF5 file identity and held only while a write
# transaction updates data plus its root sidecar. Weak values drop idle locks,
# so the registry does not retain an unbounded history of paths.
_FILE_LOCKS: weakref.WeakValueDictionary[str, threading.RLock] = (
    weakref.WeakValueDictionary()
)
_FILE_LOCKS_GUARD = threading.Lock()


class ProvenanceRollbackError(RuntimeError):
    """Report a failed rollback while retaining the original recovery link."""

    def __init__(
        self,
        operation_error: BaseException,
        rollback_error: BaseException,
        recovery_path: str,
    ) -> None:
        self.operation_error = operation_error
        self.rollback_error = rollback_error
        self.recovery_path = recovery_path
        super().__init__(
            "HDF5 provenance write failed "
            f"({operation_error!s}); rollback restoration failed "
            f"({rollback_error!s}); recovery artifact retained at "
            f"{recovery_path!r}"
        )


def _decode_sidecar(raw: Any) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar type")
    if len(raw.encode("utf-8")) > MAX_HDF5_PROVENANCE_SIDECAR_BYTES:
        raise ProvenanceSidecarError("HDF5 provenance sidecar is too large")
    return raw


def _read_sidecar(h5file: h5py.File) -> dict[str, dict[str, Any]]:
    """Read and validate the complete root sidecar before mutating data."""
    raw = h5file.attrs.get(HDF5_PROVENANCE_ATTRIBUTE)
    if raw is None:
        return {}
    try:
        decoded = _decode_sidecar(raw)
        sidecar = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar JSON") from exc
    if not isinstance(sidecar, dict):
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar: expected object")
    normalized: dict[str, dict[str, Any]] = {}
    for path, provenance in sidecar.items():
        if not isinstance(path, str):
            raise ProvenanceSidecarError("invalid HDF5 provenance sidecar path")
        try:
            normalized[path] = validated_provenance(provenance)
        except (TypeError, ValueError) as exc:
            raise ProvenanceSidecarError(
                f"invalid HDF5 provenance sidecar entry for {path!r}"
            ) from exc
    return normalized


def _write_sidecar(h5file: h5py.File, sidecar: dict[str, dict[str, Any]]) -> None:
    encoded = json.dumps(
        sidecar,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > MAX_HDF5_PROVENANCE_SIDECAR_BYTES:
        raise ProvenanceSidecarError("HDF5 provenance sidecar is too large")
    h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = encoded


def _commit_sidecar(h5file: h5py.File, dataset: h5py.Dataset, provenance: Any) -> None:
    sidecar = _read_sidecar(h5file)
    if provenance is None:
        sidecar.pop(dataset.name, None)
    else:
        sidecar[dataset.name] = validated_provenance(provenance)
    _write_sidecar(h5file, sidecar)


def _dataset(container: h5py.HLObject, path: str | None) -> h5py.Dataset:
    return gwpy_hdf5.find_dataset(container, path=path)


def _sidecar_provenance(
    container: h5py.HLObject,
    path: str | None,
) -> dict[str, Any] | None:
    dataset = _dataset(container, path)
    sidecar = _read_sidecar(dataset.file)
    provenance = sidecar.get(dataset.name)
    return None if provenance is None else validated_provenance(provenance)


def _sidecar_attr_snapshot(h5file: h5py.File) -> tuple[bool, Any]:
    key = HDF5_PROVENANCE_ATTRIBUTE
    return key in h5file.attrs, h5file.attrs.get(key)


def _restore_sidecar_attr(h5file: h5py.File, snapshot: tuple[bool, Any]) -> None:
    exists, raw = snapshot
    if exists:
        h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = raw
    elif HDF5_PROVENANCE_ATTRIBUTE in h5file.attrs:
        del h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]


def _rollback_group(h5file: h5py.File) -> h5py.Group:
    while True:
        path = f"__gwexpy_provenance_rollback_{uuid.uuid4().hex}"
        if path not in h5file:
            return h5file.create_group(path)


def _recovery_path(h5file: h5py.File) -> str:
    while True:
        path = f"__gwexpy_provenance_recovery_{uuid.uuid4().hex}"
        if path not in h5file:
            return path


def _path_lock_key(path: str | Path) -> str:
    """Return the filesystem identity, or a stable path before creation."""
    try:
        stat = os.stat(path)
    except OSError:
        return f"path:{Path(path).resolve(strict=False)}"
    return f"file:{stat.st_dev}:{stat.st_ino}"


def _file_lock_key(target: str | Path | h5py.HLObject) -> str:
    """Canonicalize a path or handle to its underlying file identity."""
    if isinstance(target, (str, Path)):
        return _path_lock_key(target)

    h5file = target.file
    try:
        descriptor = h5file.id.get_vfd_handle()
        if isinstance(descriptor, int):
            stat = os.fstat(descriptor)
            return f"file:{stat.st_dev}:{stat.st_ino}"
    except (AttributeError, OSError, RuntimeError, TypeError):
        pass
    if h5file.filename:
        return _path_lock_key(h5file.filename)
    return f"h5-object:{h5file.id.id}"


def _file_lock(target: str | Path | h5py.HLObject) -> threading.RLock:
    """Return the transient in-process lock for one HDF5 file/container."""
    key = _file_lock_key(target)
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _FILE_LOCKS[key] = lock
        return lock


def _target_object(
    container: h5py.Group | h5py.File,
    candidate_path: str | None,
) -> h5py.Dataset | None:
    """Preflight an existing target without ever treating a group as data."""
    if candidate_path is None:
        return None
    link = container.get(candidate_path, getlink=True)
    if link is None:
        return None
    if isinstance(link, h5py.SoftLink):
        raise ValueError(
            f"cannot overwrite HDF5 soft link {candidate_path!r}; "
            "its link topology must be preserved"
        )
    if isinstance(link, h5py.ExternalLink):
        raise ValueError(
            f"cannot overwrite HDF5 external link {candidate_path!r}; "
            "its link topology must be preserved"
        )
    if not isinstance(link, h5py.HardLink):  # pragma: no cover - h5py link API
        raise ValueError(f"unsupported existing HDF5 link {candidate_path!r}")
    target = container[candidate_path]
    if isinstance(target, h5py.Group):
        raise ValueError(
            f"cannot write Spectrogram to existing HDF5 group {target.name!r}"
        )
    if not isinstance(target, h5py.Dataset):
        raise ValueError(f"unsupported existing HDF5 target {target.name!r}")
    return target


def _create_rollback_hard_link(h5file: h5py.File, dataset: h5py.Dataset) -> h5py.Group:
    """Keep the original object alive without copying data or link topology."""
    rollback = _rollback_group(h5file)
    try:
        rollback["dataset"] = dataset
    except BaseException:
        del h5file[rollback.name]
        raise
    return rollback


def _move_rollback_dataset(
    h5file: h5py.File,
    rollback: h5py.Group,
    prior_path: str,
) -> None:
    """Move the preserved object back only after replacement deletion."""
    h5file.move(rollback["dataset"].name, prior_path)


def _retain_recovery_artifact(h5file: h5py.File, rollback: h5py.Group) -> str:
    """Keep the sole original object reachable if restoration itself fails."""
    original_path = rollback.name
    if original_path not in h5file:
        return original_path
    recovery_path = _recovery_path(h5file)
    try:
        h5file.move(original_path, recovery_path)
    except BaseException:
        # The rollback group still owns the only saved hard link.  Never try
        # to remove it merely because cosmetic recovery naming failed.
        return original_path
    return f"/{recovery_path}"


def _restore_dataset_link(
    container: h5py.Group | h5py.File,
    h5file: h5py.File,
    candidate_path: str | None,
    prior_path: str | None,
    rollback: h5py.Group | None,
) -> None:
    if candidate_path is not None and candidate_path in container:
        candidate = container[candidate_path]
        if isinstance(candidate, h5py.Dataset):
            del container[candidate_path]
    if prior_path is None:
        return
    if rollback is None:  # pragma: no cover - internal rollback invariant
        raise RuntimeError("missing HDF5 provenance rollback snapshot")
    _move_rollback_dataset(h5file, rollback, prior_path)


def _write_to_open_container(
    array: Any,
    container: h5py.Group | h5py.File,
    path: str | None,
    writer: Callable[..., h5py.Dataset],
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    """Write core data then atomically update its sidecar for one handle."""
    h5file = container.file
    _read_sidecar(h5file)  # fail before the core writer changes anything
    candidate_path = path if path is not None else getattr(array, "name", None)
    existing = _target_object(container, candidate_path)
    if existing is not None and not kwargs.get("overwrite", False):
        # Match GWpy's normal collision error without creating a temporary
        # hard link or touching the sidecar.
        return writer(array, container, path=path, **kwargs)

    prior_path: str | None = None
    rollback: h5py.Group | None = None
    discard_rollback = True
    sidecar_snapshot = _sidecar_attr_snapshot(h5file)
    if existing is not None:
        prior_path = existing.name
        rollback = _create_rollback_hard_link(h5file, existing)
    try:
        dataset = writer(array, container, path=path, **kwargs)
        _commit_sidecar(h5file, dataset, array.provenance)
        return dataset
    except BaseException as operation_error:
        try:
            _restore_dataset_link(
                container, h5file, candidate_path, prior_path, rollback
            )
        except BaseException as rollback_error:
            # Preserve the original object even if the failed restoration has
            # already removed the replacement at its public path.
            discard_rollback = False
            recovery_path = (
                _retain_recovery_artifact(h5file, rollback)
                if rollback is not None
                else "<unavailable>"
            )
            try:
                _restore_sidecar_attr(h5file, sidecar_snapshot)
            except BaseException as sidecar_error:
                rollback_error = RuntimeError(
                    f"{rollback_error!s}; sidecar restoration also failed: "
                    f"{sidecar_error!s}"
                )
            raise ProvenanceRollbackError(
                operation_error, rollback_error, recovery_path
            ) from rollback_error
        _restore_sidecar_attr(h5file, sidecar_snapshot)
        raise
    finally:
        if discard_rollback and rollback is not None and rollback.name in h5file:
            del h5file[rollback.name]


def _write_path_transaction(
    array: Any,
    target: str | Path,
    path: str | None,
    writer: Callable[..., h5py.Dataset],
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    """Write replacement paths to a complete sibling temp file then replace."""
    filepath = Path(target)
    existed = filepath.exists()
    append = bool(kwargs.get("append", False))
    overwrite = bool(kwargs.get("overwrite", False))
    if existed and not (append or overwrite):
        raise OSError(f"File exists: {filepath}")
    if existed and append:
        with h5py.File(filepath, "r+") as h5file:
            return _write_to_open_container(array, h5file, path, writer, kwargs)
    if existed and h5py.is_hdf5(filepath):
        with h5py.File(filepath, "r") as h5file:
            _read_sidecar(h5file)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=f".{filepath.name}.gwexpy-",
        suffix=".hdf5",
    )
    os.close(descriptor)
    try:
        with h5py.File(temporary_name, "w") as h5file:
            result = _write_to_open_container(array, h5file, path, writer, kwargs)
        os.replace(temporary_name, filepath)
        return result
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def register_hdf5_provenance_io(cls: type[Any]) -> None:
    """Register HDF5 provenance hooks while retaining GWpy descriptors."""
    base_reader = io_registry.get_reader("hdf5", BaseSpectrogram)
    base_writer = io_registry.get_writer("hdf5", BaseSpectrogram)

    def read_hdf5_spectrogram(
        source: Any, path: str | None = None, **kwargs: Any
    ) -> Any:
        if isinstance(source, (str, Path)):
            with _file_lock(source):
                with h5py.File(source, "r") as h5file:
                    provenance = _sidecar_provenance(h5file, path)
                    result = base_reader(h5file, path=path, **kwargs)
        elif isinstance(source, h5py.HLObject):
            with _file_lock(source):
                provenance = _sidecar_provenance(source, path)
                result = base_reader(source, path=path, **kwargs)
        else:
            with h5py.File(source, "r") as h5file:
                with _file_lock(h5file):
                    provenance = _sidecar_provenance(h5file, path)
                    result = base_reader(h5file, path=path, **kwargs)
        result = result.view(cls)
        if provenance is not None:
            result.provenance = provenance
        return result

    def write_hdf5_spectrogram(
        array: Any, target: Any, path: str | None = None, **kwargs: Any
    ) -> h5py.Dataset:
        if isinstance(target, (str, Path)):
            with _file_lock(target):
                return _write_path_transaction(array, target, path, base_writer, kwargs)
        if isinstance(target, (h5py.File, h5py.Group)):
            with _file_lock(target):
                return _write_to_open_container(
                    array, target, path, base_writer, kwargs
                )
        return base_writer(array, target, path=path, **kwargs)

    io_registry.register_reader("hdf5", cls, read_hdf5_spectrogram, force=True)
    io_registry.register_writer("hdf5", cls, write_hdf5_spectrogram, force=True)
    io_registry.register_identifier("hdf5", cls, identify_hdf5, force=True)
