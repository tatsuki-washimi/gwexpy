"""Registry HDF5 hooks for durable Spectrogram provenance sidecars.

The sidecar update is serialized within this Python process.  It provides
rollback for a single path or open HDF5 handle, but does not claim a
cross-process transaction; that broader HDF5 atomicity contract belongs to
the later Wave3 work.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
from gwpy.io import hdf5 as gwpy_hdf5
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


# A single re-entrant lock is deliberately bounded: it serializes every
# in-process sidecar read-modify-write, including callers that mix paths and
# open handles for the same file, without retaining a growing lock registry.
_SIDECAR_LOCK = threading.RLock()


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


def _restore_dataset(
    container: h5py.Group | h5py.File,
    h5file: h5py.File,
    candidate_path: str | None,
    prior_path: str | None,
    rollback: h5py.Group | None,
) -> None:
    if candidate_path is not None and candidate_path in container:
        del container[candidate_path]
    if prior_path is None:
        return
    if rollback is None:  # pragma: no cover - internal rollback invariant
        raise RuntimeError("missing HDF5 provenance rollback snapshot")
    parent_path, _, leaf = prior_path.rpartition("/")
    parent = h5file[parent_path or "/"]
    if leaf in parent:
        del parent[leaf]
    parent.copy(rollback["dataset"], leaf)


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
    prior_path: str | None = None
    rollback: h5py.Group | None = None
    if candidate_path is not None and candidate_path in container:
        old = container[candidate_path]
        if isinstance(old, h5py.Dataset):
            prior_path = old.name
            rollback = _rollback_group(h5file)
            h5file.copy(old, rollback, name="dataset")
    sidecar_snapshot = _sidecar_attr_snapshot(h5file)
    try:
        dataset = writer(array, container, path=path, **kwargs)
        _commit_sidecar(h5file, dataset, array.provenance)
        return dataset
    except BaseException:
        _restore_dataset(container, h5file, candidate_path, prior_path, rollback)
        _restore_sidecar_attr(h5file, sidecar_snapshot)
        raise
    finally:
        if rollback is not None and rollback.name in h5file:
            del h5file[rollback.name]


def _write_path_transaction(
    array: Any,
    target: str | Path,
    path: str | None,
    writer: Callable[..., h5py.Dataset],
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    """Use a full-file backup to roll back a pathname write on failure."""
    filepath = Path(target)
    existed = filepath.exists()
    backup_name: str | None = None
    if existed:
        with h5py.File(filepath, "r") as h5file:
            _read_sidecar(h5file)
        descriptor, backup_name = tempfile.mkstemp(
            prefix="gwexpy-provenance-", suffix=".hdf5"
        )
        os.close(descriptor)
        shutil.copy2(filepath, backup_name)
    append = bool(kwargs.get("append", False))
    overwrite = bool(kwargs.get("overwrite", False))
    if existed and not (append or overwrite):
        raise OSError(f"File exists: {filepath}")
    try:
        with h5py.File(filepath, "a" if append else "w") as h5file:
            return _write_to_open_container(array, h5file, path, writer, kwargs)
    except BaseException:
        if existed and backup_name is not None:
            shutil.copy2(backup_name, filepath)
        elif filepath.exists():
            filepath.unlink()
        raise
    finally:
        if backup_name is not None:
            os.unlink(backup_name)


def register_hdf5_provenance_io(cls: type[Any]) -> None:
    """Register HDF5 provenance hooks while retaining GWpy descriptors."""
    base_reader = io_registry.get_reader("hdf5", BaseSpectrogram)
    base_writer = io_registry.get_writer("hdf5", BaseSpectrogram)

    def read_hdf5_spectrogram(
        source: Any, path: str | None = None, **kwargs: Any
    ) -> Any:
        with _SIDECAR_LOCK:
            if isinstance(source, (str, Path)):
                with h5py.File(source, "r") as h5file:
                    provenance = _sidecar_provenance(h5file, path)
                    result = base_reader(h5file, path=path, **kwargs)
            elif isinstance(source, h5py.HLObject):
                provenance = _sidecar_provenance(source, path)
                result = base_reader(source, path=path, **kwargs)
            else:
                with h5py.File(source, "r") as h5file:
                    provenance = _sidecar_provenance(h5file, path)
                    result = base_reader(h5file, path=path, **kwargs)
        result = result.view(cls)
        if provenance is not None:
            result.provenance = provenance
        return result

    def write_hdf5_spectrogram(
        array: Any, target: Any, path: str | None = None, **kwargs: Any
    ) -> h5py.Dataset:
        with _SIDECAR_LOCK:
            if isinstance(target, (str, Path)):
                return _write_path_transaction(array, target, path, base_writer, kwargs)
            if isinstance(target, (h5py.File, h5py.Group)):
                return _write_to_open_container(
                    array, target, path, base_writer, kwargs
                )
        return base_writer(array, target, path=path, **kwargs)

    io_registry.register_reader("hdf5", cls, read_hdf5_spectrogram, force=True)
    io_registry.register_writer("hdf5", cls, write_hdf5_spectrogram, force=True)
    identifier = io_registry._identifiers[("hdf5", BaseSpectrogram)]
    io_registry.register_identifier("hdf5", cls, identifier, force=True)
