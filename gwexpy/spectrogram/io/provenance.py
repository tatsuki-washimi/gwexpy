"""Registry HDF5 hooks for durable Spectrogram provenance sidecars.

Provenance-aware reads and sidecar updates are serialized within this Python
process.  It provides rollback for a single path or open HDF5 handle, but does not claim a
cross-process transaction; that broader HDF5 atomicity contract belongs to
the later Wave3 work.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import uuid
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
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
_MAX_EXCEPTION_DESCRIPTION_CHARS = 512
_MAX_EXCEPTION_GROUP_NODES = 64
_MAX_EXCEPTION_GROUP_DEPTH = 8
_MAX_EXCEPTION_GROUP_CHILDREN = 16
_EXCEPTION_TRUNCATION_MARKER = "<truncated>"


class _ProvenanceRollbackInvariantError(RuntimeError):
    """Record invalid internal rollback-error construction without losing state."""


def _bounded_exception_text(
    text: str,
    limit: int = _MAX_EXCEPTION_DESCRIPTION_CHARS,
) -> str:
    """Bound untrusted exception text before storing it in an error message."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_EXCEPTION_TRUNCATION_MARKER):
        return _EXCEPTION_TRUNCATION_MARKER[:limit]
    return f"{text[: limit - len(_EXCEPTION_TRUNCATION_MARKER)]}{_EXCEPTION_TRUNCATION_MARKER}"


def _exception_type_name(error: BaseException) -> str:
    """Return an exception type name without consulting its instance state."""
    try:
        return type(error).__name__
    except BaseException:
        return "BaseException"


def _safe_exception_descriptor(
    error: BaseException,
    cache: dict[int, str],
    active: set[int],
    *,
    depth: int = 0,
    visited: list[int] | None = None,
    max_chars: int = _MAX_EXCEPTION_DESCRIPTION_CHARS,
) -> str:
    """Describe one exception with bounded group traversal and formatting."""
    if max_chars <= 0:
        return ""
    key = id(error)
    if key in cache:
        return _bounded_exception_text(cache[key], max_chars)
    if visited is None:
        visited = [0]
    if visited[0] >= _MAX_EXCEPTION_GROUP_NODES:
        return _bounded_exception_text(_EXCEPTION_TRUNCATION_MARKER, max_chars)
    visited[0] += 1
    type_name = _exception_type_name(error)
    if key in active:
        return _bounded_exception_text(
            f"{type_name}: <recursive exception group>", max_chars
        )
    active.add(key)
    try:
        if isinstance(error, BaseExceptionGroup):
            if depth >= _MAX_EXCEPTION_GROUP_DEPTH:
                descriptor = f"{type_name}: {_EXCEPTION_TRUNCATION_MARKER}"
            else:
                try:
                    children = error.exceptions
                    prefix = f"{type_name}: ["
                    parts: list[str] = []
                    used = len(prefix) + 1  # Closing bracket.
                    truncated = False
                    for index, child in enumerate(children):
                        if (
                            index >= _MAX_EXCEPTION_GROUP_CHILDREN
                            or visited[0] >= _MAX_EXCEPTION_GROUP_NODES
                        ):
                            truncated = True
                            break
                        separator = "; " if parts else ""
                        available = max_chars - used - len(separator)
                        if available <= len(_EXCEPTION_TRUNCATION_MARKER):
                            truncated = True
                            break
                        child_text = _safe_exception_descriptor(
                            child,
                            cache,
                            active,
                            depth=depth + 1,
                            visited=visited,
                            max_chars=available,
                        )
                        parts.append(child_text)
                        used += len(separator) + len(child_text)
                    if truncated:
                        separator = "; " if parts else ""
                        available = max_chars - used - len(separator)
                        if available > 0:
                            marker = _bounded_exception_text(
                                _EXCEPTION_TRUNCATION_MARKER, available
                            )
                            parts.append(marker)
                    descriptor = f"{prefix}{'; '.join(parts)}]"
                except BaseException:
                    descriptor = f"{type_name}: <unprintable exception group>"
        else:
            if type(error).__str__ is not BaseException.__str__:
                descriptor = f"{type_name}: <untrusted formatting omitted>"
            else:
                try:
                    rendered = str(error)
                except BaseException:
                    descriptor = f"{type_name}: <unprintable>"
                else:
                    descriptor = f"{type_name}: {_bounded_exception_text(rendered)}"
    finally:
        active.discard(key)
    descriptor = _bounded_exception_text(descriptor, max_chars)
    cache[key] = descriptor
    return descriptor


def _missing_causal_errors(
    grouped_errors: tuple[BaseException, ...],
    event_errors: tuple[BaseException, ...],
) -> tuple[BaseException, ...]:
    """Return grouped errors absent from an explicit causal event sequence."""
    remaining = {id(error): 0 for error in grouped_errors}
    for error in grouped_errors:
        remaining[id(error)] += 1
    for error in event_errors:
        key = id(error)
        if remaining.get(key, 0):
            remaining[key] -= 1
    missing: list[BaseException] = []
    for error in grouped_errors:
        key = id(error)
        if remaining[key]:
            missing.append(error)
            remaining[key] -= 1
    return tuple(missing)


def _causal_error_state(
    grouped_errors: tuple[BaseException, ...],
    event_errors: tuple[BaseException, ...] | None,
) -> tuple[tuple[BaseException, ...], tuple[BaseException, ...]]:
    """Keep all captured errors and synthesize an invariant when needed."""
    invariant_errors: list[BaseException] = []
    if event_errors is None:
        causal_errors = grouped_errors
    else:
        causal_errors = tuple(
            error for error in event_errors if isinstance(error, BaseException)
        )
        if len(causal_errors) != len(event_errors):
            invariant_errors.append(
                _ProvenanceRollbackInvariantError(
                    "rollback event sequence contains a non-exception value"
                )
            )
        missing = _missing_causal_errors(grouped_errors, causal_errors)
        if missing:
            causal_errors = (*causal_errors, *missing)
            invariant_errors.append(
                _ProvenanceRollbackInvariantError(
                    "rollback event sequence omitted captured phase errors"
                )
            )
    if not causal_errors and not invariant_errors:
        invariant_errors.append(
            _ProvenanceRollbackInvariantError(
                "rollback construction has no causal error"
            )
        )
    return (*causal_errors, *invariant_errors), tuple(invariant_errors)


class ProvenanceRollbackError(RuntimeError):
    """Report a failed rollback while retaining a recovery artifact."""

    def __init__(
        self,
        operation_error: BaseException | None,
        restoration_errors: tuple[BaseException, ...],
        recovery_path: str | None,
        *,
        preservation_errors: tuple[BaseException, ...] = (),
        cleanup_errors: tuple[BaseException, ...] = (),
        operation_committed: bool = False,
        event_errors: tuple[BaseException, ...] | None = None,
    ) -> None:
        self.operation_error = operation_error
        self.operation_committed = operation_committed
        self.restoration_errors = restoration_errors
        self.preservation_errors = preservation_errors
        self.cleanup_errors = cleanup_errors
        grouped_errors = (
            *(() if operation_error is None else (operation_error,)),
            *restoration_errors,
            *preservation_errors,
            *cleanup_errors,
        )
        self.errors, self.invariant_errors = _causal_error_state(
            grouped_errors,
            event_errors,
        )
        descriptions: dict[int, str] = {}
        self.error_descriptions = tuple(
            _safe_exception_descriptor(error, descriptions, set())
            for error in self.errors
        )
        # Retain this attribute for callers that caught the first version of
        # this internal exception before it reported every restoration error.
        rollback_errors = self.errors
        if operation_error is not None and rollback_errors[0] is operation_error:
            rollback_errors = rollback_errors[1:]
        self.rollback_error = rollback_errors[0] if rollback_errors else operation_error
        self.recovery_path = recovery_path
        self.recovery_available = recovery_path is not None
        rollback_descriptions = (
            self.error_descriptions[1:]
            if operation_error is not None and self.errors[0] is operation_error
            else self.error_descriptions
        )
        rollback_message = "; ".join(rollback_descriptions)
        if operation_committed:
            message = (
                "HDF5 provenance write committed, but rollback cleanup failed "
                f"({rollback_message})"
            )
        else:
            operation_message = (
                descriptions.get(id(operation_error), "<no operation error>")
                if operation_error is not None
                else "<no operation error>"
            )
            message = (
                "HDF5 provenance write failed "
                f"({operation_message}); rollback handling failed "
                f"({rollback_message})"
            )
        if recovery_path is None:
            message += "; recovery unavailable"
        else:
            message += f"; recovery artifact retained at {recovery_path!r}"
        super().__init__(message)


def _decode_sidecar(raw: Any) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar type")
    if len(raw.encode("utf-8")) > MAX_HDF5_PROVENANCE_SIDECAR_BYTES:
        raise ProvenanceSidecarError("HDF5 provenance sidecar is too large")
    return raw


def _validated_sidecar(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate one serialized sidecar without reading or mutating HDF5."""
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


def _read_sidecar(h5file: h5py.File) -> dict[str, dict[str, Any]]:
    """Read and validate the complete root sidecar before mutating data."""
    raw = h5file.attrs.get(HDF5_PROVENANCE_ATTRIBUTE)
    if raw is None:
        return {}
    return _validated_sidecar(raw)


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


def _apply_sidecar_snapshot(h5file: h5py.File, snapshot: tuple[bool, Any]) -> None:
    exists, raw = snapshot
    if exists:
        h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE] = raw
    elif HDF5_PROVENANCE_ATTRIBUTE in h5file.attrs:
        del h5file.attrs[HDF5_PROVENANCE_ATTRIBUTE]


def _restore_sidecar_attr(h5file: h5py.File, snapshot: tuple[bool, Any]) -> None:
    _apply_sidecar_snapshot(h5file, snapshot)


def _restore_sidecar_with_fallback(
    h5file: h5py.File,
    snapshot: tuple[bool, Any],
) -> tuple[BaseException, ...]:
    """Restore the sidecar and retry through the direct snapshot primitive."""
    try:
        _restore_sidecar_attr(h5file, snapshot)
    except BaseException as error:
        errors: list[BaseException] = [error]
        try:
            _apply_sidecar_snapshot(h5file, snapshot)
        except BaseException as fallback_error:
            errors.append(fallback_error)
        return tuple(errors)
    return ()


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


def _record_sidecar_snapshot(
    rollback: h5py.Group,
    snapshot: tuple[bool, Any],
) -> None:
    """Persist the prior root-sidecar attribute for manual recovery."""
    exists, raw = snapshot
    rollback.attrs["sidecar_snapshot_present"] = exists
    if exists:
        rollback.attrs["sidecar_snapshot"] = raw


def _ensure_recovery_hard_link(
    h5file: h5py.File,
    rollback: h5py.Group,
    prior_path: str | None,
) -> None:
    """Give a recovered original a second, explicitly named hard link."""
    if "dataset" in rollback:
        return
    if prior_path is None or prior_path not in h5file:
        raise RuntimeError("original HDF5 dataset is unavailable for recovery")
    original = h5file[prior_path]
    if not isinstance(original, h5py.Dataset):
        raise RuntimeError("original HDF5 recovery target is not a dataset")
    rollback["dataset"] = original


def _rename_recovery_artifact(
    h5file: h5py.File,
    original_path: str,
    recovery_path: str,
) -> None:
    """Rename the retained rollback group to its explicit recovery name."""
    h5file.move(original_path, recovery_path)


def _safe_artifact_path(
    h5file: h5py.File,
    path: str | None,
    errors: list[BaseException],
) -> str | None:
    """Return a reachable root path without leaking HDF5 handle failures."""
    if not isinstance(path, str):
        errors.append(RuntimeError("HDF5 recovery artifact path is unavailable"))
        return None
    try:
        if path in h5file:
            return path if path.startswith("/") else f"/{path}"
    except BaseException as error:
        errors.append(error)
        return None
    return None


def _safe_rollback_path(
    rollback: h5py.Group,
    errors: list[BaseException],
) -> str | None:
    """Inspect a rollback handle without allowing an invalid ID to escape."""
    try:
        path = rollback.name
    except BaseException as error:
        errors.append(error)
        return None
    if not isinstance(path, str):
        errors.append(RuntimeError("HDF5 recovery artifact path is unavailable"))
        return None
    return path if path.startswith("/") else f"/{path}"


def _valid_recovery_sidecar_snapshot(
    artifact: h5py.Group,
    errors: list[BaseException],
) -> bool:
    """Validate an actionable prior-sidecar snapshot without mutating it."""
    missing = object()
    try:
        marker = artifact.attrs.get("sidecar_snapshot_present", missing)
        snapshot = artifact.attrs.get("sidecar_snapshot", missing)
    except BaseException as error:
        errors.append(error)
        return False

    if marker is missing:
        if snapshot is not missing:
            errors.append(
                ProvenanceSidecarError(
                    "HDF5 recovery sidecar snapshot is missing its boolean marker"
                )
            )
        return False
    if not isinstance(marker, (bool, np.bool_)):
        errors.append(
            ProvenanceSidecarError(
                "HDF5 recovery sidecar marker must be an exact boolean"
            )
        )
        return False
    if not bool(marker):
        if snapshot is not missing:
            errors.append(
                ProvenanceSidecarError(
                    "HDF5 recovery sidecar absence marker has a snapshot"
                )
            )
            return False
        return True
    if snapshot is missing:
        errors.append(
            ProvenanceSidecarError("HDF5 recovery sidecar snapshot is missing")
        )
        return False
    try:
        _validated_sidecar(snapshot)
    except BaseException as error:
        errors.append(error)
        return False
    return True


def _artifact_has_recovery_content(
    h5file: h5py.File,
    path: str,
    errors: list[BaseException],
) -> bool:
    """Whether an artifact contains a dataset or usable sidecar snapshot."""
    try:
        artifact = h5file[path]
    except BaseException as error:
        errors.append(error)
        return False
    if not isinstance(artifact, h5py.Group):
        errors.append(RuntimeError("HDF5 recovery artifact is not a group"))
        return False

    actionable = False
    try:
        if "dataset" in artifact:
            dataset = artifact["dataset"]
            if isinstance(dataset, h5py.Dataset):
                actionable = True
            else:
                errors.append(RuntimeError("HDF5 recovery dataset is not a dataset"))
    except BaseException as error:
        errors.append(error)

    if _valid_recovery_sidecar_snapshot(artifact, errors):
        actionable = True
    return actionable


def _select_recovery_artifact(
    h5file: h5py.File,
    candidates: tuple[str | None, ...],
    errors: list[BaseException],
) -> str | None:
    """Safely inspect all candidate paths and retain the first usable one."""
    selected: str | None = None
    reachable = False
    errors_before_probes = len(errors)
    for candidate in candidates:
        try:
            path = _safe_artifact_path(h5file, candidate, errors)
        except BaseException as error:
            errors.append(error)
            continue
        if path is None:
            continue
        reachable = True
        try:
            actionable = _artifact_has_recovery_content(h5file, path, errors)
        except BaseException as error:
            errors.append(error)
            continue
        if actionable and selected is None:
            selected = path
    if selected is not None:
        return selected
    if reachable:
        errors.append(RuntimeError("HDF5 recovery artifact has no actionable content"))
    elif len(errors) == errors_before_probes:
        errors.append(RuntimeError("HDF5 recovery artifact is no longer reachable"))
    return None


def _capture_recovery_state(
    h5file: h5py.File,
    rollback: h5py.Group | None,
    prior_path: str | None,
    sidecar_snapshot: tuple[bool, Any],
    *,
    allow_public_dataset_link: bool = False,
) -> tuple[str | None, tuple[BaseException, ...]]:
    """Capture recovery state without masking any artifact-operation error."""
    preservation_errors: list[BaseException] = []
    if rollback is None:
        try:
            rollback = _rollback_group(h5file)
        except BaseException as error:
            return None, (error,)
    if allow_public_dataset_link:
        try:
            _ensure_recovery_hard_link(h5file, rollback, prior_path)
        except BaseException as error:
            preservation_errors.append(error)
    try:
        _record_sidecar_snapshot(rollback, sidecar_snapshot)
    except BaseException as error:
        preservation_errors.append(error)
    original_path = _safe_rollback_path(rollback, preservation_errors)
    if original_path is None:
        return None, tuple(preservation_errors)
    try:
        recovery_path = _recovery_path(h5file)
    except BaseException as error:
        preservation_errors.append(error)
        return (
            _select_recovery_artifact(h5file, (original_path,), preservation_errors),
            tuple(preservation_errors),
        )
    try:
        _rename_recovery_artifact(h5file, original_path, recovery_path)
    except BaseException as error:
        # HDF5 moves may succeed before a backend reports an error.  Probe the
        # intended destination and source independently before claiming loss.
        preservation_errors.append(error)
        return (
            _select_recovery_artifact(
                h5file, (recovery_path, original_path), preservation_errors
            ),
            tuple(preservation_errors),
        )
    return (
        _select_recovery_artifact(h5file, (recovery_path,), preservation_errors),
        tuple(preservation_errors),
    )


def _cleanup_rollback_group(
    h5file: h5py.File,
    rollback: h5py.Group | None,
) -> None:
    """Remove a no-longer-needed rollback group after a complete outcome."""
    if rollback is not None and rollback.name in h5file:
        del h5file[rollback.name]


def _capture_cleanup_state(
    h5file: h5py.File,
    rollback: h5py.Group | None,
    prior_path: str | None,
    sidecar_snapshot: tuple[bool, Any],
    *,
    allow_public_dataset_link: bool,
) -> tuple[tuple[BaseException, ...], str | None, tuple[BaseException, ...]]:
    """Run cleanup and capture its recoverability without allowing raw errors."""
    try:
        _cleanup_rollback_group(h5file, rollback)
    except BaseException as cleanup_error:
        recovery_path, preservation_errors = _capture_recovery_state(
            h5file,
            rollback,
            prior_path,
            sidecar_snapshot,
            allow_public_dataset_link=allow_public_dataset_link,
        )
        return (cleanup_error,), recovery_path, preservation_errors
    return (), None, ()


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
    sidecar_snapshot = _sidecar_attr_snapshot(h5file)
    if existing is not None:
        prior_path = existing.name
        rollback = _create_rollback_hard_link(h5file, existing)
    try:
        dataset = writer(array, container, path=path, **kwargs)
        _commit_sidecar(h5file, dataset, array.provenance)
    except BaseException as operation_error:
        event_errors: list[BaseException] = [operation_error]
        restoration_errors: list[BaseException] = []
        try:
            _restore_dataset_link(
                container, h5file, candidate_path, prior_path, rollback
            )
        except BaseException as error:
            restoration_errors.append(error)
            event_errors.append(error)
        sidecar_restoration_errors = _restore_sidecar_with_fallback(
            h5file, sidecar_snapshot
        )
        restoration_errors.extend(sidecar_restoration_errors)
        event_errors.extend(sidecar_restoration_errors)
        if restoration_errors:
            # Preserve the original object even if the failed restoration has
            # already removed the replacement at its public path, or if the
            # original link was restored but its matching sidecar was not.
            recovery_path, preservation_errors = _capture_recovery_state(
                h5file,
                rollback,
                prior_path,
                sidecar_snapshot,
                allow_public_dataset_link=True,
            )
            event_errors.extend(preservation_errors)
            raise ProvenanceRollbackError(
                operation_error,
                tuple(restoration_errors),
                recovery_path,
                preservation_errors=preservation_errors,
                event_errors=tuple(event_errors),
            ) from restoration_errors[0]
        cleanup_errors, recovery_path, preservation_errors = _capture_cleanup_state(
            h5file,
            rollback,
            prior_path,
            sidecar_snapshot,
            allow_public_dataset_link=True,
        )
        if cleanup_errors:
            event_errors.extend(cleanup_errors)
            event_errors.extend(preservation_errors)
            raise ProvenanceRollbackError(
                operation_error,
                (),
                recovery_path,
                preservation_errors=preservation_errors,
                cleanup_errors=cleanup_errors,
                event_errors=tuple(event_errors),
            ) from cleanup_errors[0]
        raise
    cleanup_errors, recovery_path, preservation_errors = _capture_cleanup_state(
        h5file,
        rollback,
        prior_path,
        sidecar_snapshot,
        allow_public_dataset_link=False,
    )
    if cleanup_errors:
        raise ProvenanceRollbackError(
            None,
            (),
            recovery_path,
            preservation_errors=preservation_errors,
            cleanup_errors=cleanup_errors,
            operation_committed=True,
            event_errors=(*cleanup_errors, *preservation_errors),
        ) from cleanup_errors[0]
    return dataset


def _path_replacement_preflight(filepath: Path) -> tuple[bool, int | None]:
    """Reject links and retain regular-file mode bits for ``os.replace``."""
    try:
        status = os.lstat(filepath)
    except FileNotFoundError:
        return False, None
    if stat.S_ISLNK(status.st_mode):
        raise OSError(f"refusing to overwrite symbolic link: {filepath}")
    if stat.S_ISREG(status.st_mode):
        return True, stat.S_IMODE(status.st_mode)
    return True, None


def _write_path_transaction(
    array: Any,
    target: str | Path,
    path: str | None,
    writer: Callable[..., h5py.Dataset],
    kwargs: dict[str, Any],
) -> h5py.Dataset:
    """Write replacement paths to a complete sibling temp file then replace."""
    filepath = Path(target)
    existed, target_mode = _path_replacement_preflight(filepath)
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
        if target_mode is not None:
            os.chmod(temporary_name, target_mode)
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
            _path_replacement_preflight(Path(target))
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
