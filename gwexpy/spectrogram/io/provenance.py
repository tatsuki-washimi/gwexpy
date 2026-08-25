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
from pathlib import Path, PurePosixPath
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
_MAX_ROLLBACK_PHASE_ERRORS = 64
_MAX_ROLLBACK_MESSAGE_CHARS = 2_048


class _ProvenanceRollbackInvariantError(RuntimeError):
    """Record invalid internal rollback-error construction without losing state."""


def _bounded_text(text: str, limit: int) -> str:
    """Bound trusted, internally generated text before storing it."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_EXCEPTION_TRUNCATION_MARKER):
        return _EXCEPTION_TRUNCATION_MARKER[:limit]
    return f"{text[: limit - len(_EXCEPTION_TRUNCATION_MARKER)]}{_EXCEPTION_TRUNCATION_MARKER}"


def _safe_exception_descriptor(
    error: BaseException,
    cache: dict[int, str],
    active: set[int],
    *,
    depth: int = 0,
    visited: list[int] | None = None,
    max_chars: int = _MAX_EXCEPTION_DESCRIPTION_CHARS,
) -> str:
    """Describe an exception without inspecting untrusted exception text.

    Original exception objects remain available through ``errors``.  The
    message deliberately uses fixed internal labels: exception classes,
    arguments, ``str``/``repr``, and metaclass attributes can run hostile code.
    """
    if max_chars <= 0:
        return ""
    key = id(error)
    if key in cache:
        return _bounded_text(cache[key], max_chars)
    if visited is None:
        visited = [0]
    if visited[0] >= _MAX_EXCEPTION_GROUP_NODES:
        return _bounded_text(_EXCEPTION_TRUNCATION_MARKER, max_chars)
    visited[0] += 1
    if key in active:
        return _bounded_text("BaseExceptionGroup: <recursive>", max_chars)
    active.add(key)
    try:
        if isinstance(error, BaseExceptionGroup):
            if depth >= _MAX_EXCEPTION_GROUP_DEPTH:
                descriptor = f"BaseExceptionGroup: {_EXCEPTION_TRUNCATION_MARKER}"
            else:
                try:
                    # Bypass user-defined ``__getattribute__``.  This is the
                    # only group state we inspect and it is bounded below.
                    children = object.__getattribute__(error, "exceptions")
                    if not isinstance(children, tuple):
                        raise TypeError
                    prefix = "BaseExceptionGroup: ["
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
                            marker = _bounded_text(
                                _EXCEPTION_TRUNCATION_MARKER, available
                            )
                            parts.append(marker)
                    descriptor = f"{prefix}{'; '.join(parts)}]"
                except BaseException:
                    descriptor = "BaseExceptionGroup: <unavailable>"
        else:
            descriptor = "BaseException"
    finally:
        active.discard(key)
    descriptor = _bounded_text(descriptor, max_chars)
    cache[key] = descriptor
    return descriptor


def _bounded_exception_tuple(value: object) -> tuple[BaseException, ...] | None:
    """Accept only a small, concrete tuple of exception objects."""
    if not isinstance(value, tuple) or len(value) > _MAX_ROLLBACK_PHASE_ERRORS:
        return None
    if not all(isinstance(error, BaseException) for error in value):
        return None
    return value


def _expected_rollback_events(
    operation_error: BaseException | None,
    restoration_errors: tuple[BaseException, ...],
    preservation_errors: tuple[BaseException, ...],
    cleanup_errors: tuple[BaseException, ...],
    operation_committed: bool,
) -> tuple[BaseException, ...] | None:
    """Return the only causal sequence allowed for a transaction state.

    A failed operation records the operation first, then either restoration
    failures or rollback-cleanup failures, followed by preservation failures.
    A committed operation has no operation/restoration failure and records
    cleanup plus preservation failures.  Thus every valid state has an actual
    rollback-phase error, and ``rollback_error`` can never be the operation.
    """
    if operation_committed:
        if operation_error is not None or restoration_errors or not cleanup_errors:
            return None
        return (*cleanup_errors, *preservation_errors)
    if operation_error is None:
        return None
    if restoration_errors:
        if cleanup_errors:
            return None
        return (operation_error, *restoration_errors, *preservation_errors)
    if cleanup_errors:
        return (operation_error, *cleanup_errors, *preservation_errors)
    return None


def _valid_rollback_state(
    operation_error: object,
    restoration_errors: object,
    preservation_errors: object,
    cleanup_errors: object,
    operation_committed: object,
    event_errors: object,
) -> (
    tuple[
        BaseException | None,
        tuple[BaseException, ...],
        tuple[BaseException, ...],
        tuple[BaseException, ...],
        bool,
        tuple[BaseException, ...],
    ]
    | None
):
    """Validate bounded phase data and its exact identity-preserving order."""
    if type(operation_committed) is not bool:
        return None
    if operation_error is not None and not isinstance(operation_error, BaseException):
        return None
    restoration = _bounded_exception_tuple(restoration_errors)
    preservation = _bounded_exception_tuple(preservation_errors)
    cleanup = _bounded_exception_tuple(cleanup_errors)
    if restoration is None or preservation is None or cleanup is None:
        return None
    expected = _expected_rollback_events(
        operation_error,
        restoration,
        preservation,
        cleanup,
        operation_committed,
    )
    if expected is None:
        return None
    # ``event_errors`` predates the structured phase fields and remains
    # optional for direct callers.  Its absence deterministically means the
    # exact phase order above; an explicitly supplied tuple remains a strict
    # identity- and multiplicity-preserving assertion of that order.
    events = (
        expected if event_errors is None else _bounded_exception_tuple(event_errors)
    )
    if events is None or len(expected) != len(events):
        return None
    if any(
        actual is not expected_error for actual, expected_error in zip(events, expected)
    ):
        return None
    return (
        operation_error,
        restoration,
        preservation,
        cleanup,
        operation_committed,
        events,
    )


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
        state = _valid_rollback_state(
            operation_error,
            restoration_errors,
            preservation_errors,
            cleanup_errors,
            operation_committed,
            event_errors,
        )
        if state is None:
            invariant = _ProvenanceRollbackInvariantError(
                "invalid bounded rollback transaction state"
            )
            self.operation_error: BaseException | None = None
            self.operation_committed: bool = False
            self.restoration_errors: tuple[BaseException, ...] = ()
            self.preservation_errors: tuple[BaseException, ...] = ()
            self.cleanup_errors: tuple[BaseException, ...] = ()
            self.errors: tuple[BaseException, ...] = (invariant,)
            self.invariant_errors: tuple[BaseException, ...] = self.errors
            self.rollback_error: BaseException = invariant
            self.recovery_path: str | None = None
            self.recovery_available: bool = False
            self.error_descriptions: tuple[str, ...] = ("error[0]: BaseException",)
            super().__init__(
                "HDF5 provenance rollback state invariant failed; recovery unavailable"
            )
            return

        (
            self.operation_error,
            self.restoration_errors,
            self.preservation_errors,
            self.cleanup_errors,
            self.operation_committed,
            self.errors,
        ) = state
        self.invariant_errors = ()
        descriptions: dict[int, str] = {}
        self.error_descriptions = tuple(
            f"error[{index}]: {_safe_exception_descriptor(error, descriptions, set())}"
            for index, error in enumerate(self.errors)
        )
        # A valid sequence always has a rollback phase following a failed
        # operation, or begins with cleanup after a committed operation.
        self.rollback_error = self.errors[0 if self.operation_committed else 1]
        self.recovery_path = recovery_path
        self.recovery_available = recovery_path is not None
        rollback_descriptions = self.error_descriptions[
            0 if self.operation_committed else 1 :
        ]
        rollback_message = _bounded_text(
            "; ".join(rollback_descriptions),
            _MAX_ROLLBACK_MESSAGE_CHARS,
        )
        if self.operation_committed:
            message = (
                "HDF5 provenance write committed, but rollback cleanup failed "
                f"({rollback_message})"
            )
        else:
            message = (
                "HDF5 provenance write failed; rollback handling failed "
                f"({rollback_message})"
            )
        if self.recovery_path is None:
            message += "; recovery unavailable"
        else:
            message += "; recovery artifact retained"
        super().__init__(message)


def _decode_sidecar(raw: Any) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar type")
    if len(raw.encode("utf-8")) > MAX_HDF5_PROVENANCE_SIDECAR_BYTES:
        raise ProvenanceSidecarError("HDF5 provenance sidecar is too large")
    return raw


def _canonical_hdf5_dataset_path(path: object) -> str:
    """Validate one canonical absolute HDF5 dataset name.

    Sidecar entries are keyed by ``h5py.Dataset.name``.  Accepting alternate
    spellings would make a valid provenance entry undiscoverable, so the
    serialized format permits only canonical absolute names below the root.
    """
    if not isinstance(path, str):
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar path")
    if (
        not path.startswith("/")
        or path == "/"
        or "\x00" in path
        or "//" in path
        or path.endswith("/")
    ):
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar path")
    pure_path = PurePosixPath(path)
    if not pure_path.is_absolute() or pure_path.as_posix() != path:
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar path")
    components = path.split("/")[1:]
    if not components or any(component in {"", ".", ".."} for component in components):
        raise ProvenanceSidecarError("invalid HDF5 provenance sidecar path")
    return path


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
        canonical_path = _canonical_hdf5_dataset_path(path)
        try:
            normalized[canonical_path] = validated_provenance(provenance)
        except (TypeError, ValueError) as exc:
            raise ProvenanceSidecarError(
                "invalid HDF5 provenance sidecar entry"
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
