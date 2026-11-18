"""Root-level time-reference metadata for GWpy-compatible HDF5 datasets.

The path-indexed records are not identity-bound. An external writer that
replaces a dataset without updating this attribute can leave a stale record.
"""

from __future__ import annotations

import json
from typing import Any

import h5py

ATTRIBUTE = "_gwexpy_time_reference_v1"
MAX_BYTES = 1_000_000


def _decode(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_BYTES:
        raise ValueError("invalid HDF5 time-reference sidecar")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid HDF5 time-reference sidecar") from exc
    if not isinstance(data, dict) or any(
        not isinstance(k, str)
        or not k.startswith("/")
        or v not in ("relative", "absolute")
        for k, v in data.items()
    ):
        raise ValueError("invalid HDF5 time-reference sidecar")
    return data


def _read(file: h5py.File) -> dict[str, str]:
    return _decode(file.attrs.get(ATTRIBUTE))


def read_reference(dataset: h5py.Dataset) -> str | None:
    """Return the recorded state, if present, for this dataset path."""
    return _read(dataset.file).get(dataset.name)


def prepare_reference(
    file: h5py.File, path: str, state: str, *, fresh: bool = False
) -> str:
    """Validate the current sidecar and prepare an update without writing.

    When ``fresh`` is true, validate the existing document but prepare a new
    one-record document for a full-file overwrite.
    """
    if state not in ("relative", "absolute"):
        raise ValueError("invalid time-reference state")
    data = _read(file)
    if fresh:
        data = {}
    data[path] = state
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True)
    if len(raw.encode("utf-8")) > MAX_BYTES:
        raise ValueError("HDF5 time-reference sidecar too large")
    return raw


def write_reference(dataset: h5py.Dataset, state: str) -> None:
    """Update the root-sidecar record for this dataset path."""
    raw = prepare_reference(dataset.file, dataset.name, state)
    dataset.file.attrs[ATTRIBUTE] = raw


def snapshot(file: h5py.File) -> tuple[bool, Any]:
    """Capture the sidecar before a transaction."""
    return ATTRIBUTE in file.attrs, file.attrs.get(ATTRIBUTE)


def restore(file: h5py.File, saved: tuple[bool, Any]) -> None:
    """Restore a previously captured sidecar."""
    exists, raw = saved
    if exists:
        file.attrs[ATTRIBUTE] = raw
    elif ATTRIBUTE in file.attrs:
        del file.attrs[ATTRIBUTE]
