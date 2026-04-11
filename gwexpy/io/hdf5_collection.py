from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import h5py

KEYMAP_ATTR = "gwexpy_keymap"
ORDER_ATTR = "gwexpy_order"
KIND_ATTR = "gwexpy_kind"
LAYOUT_ATTR = "gwexpy_layout"
VERSION_ATTR = "gwexpy_layout_version"
LAYOUT_DATASET = "dataset-per-entry"
LAYOUT_GROUP = "group-per-entry"

_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_hdf5_key(text: str, *, default: str = "item") -> str:
    """Return an HDF5-safe key derived from arbitrary text."""
    raw = (text or "").strip()
    raw = raw.replace("/", "_").replace("\\", "_")
    key = _SAFE_KEY_RE.sub("_", raw).strip("._-")
    return key or default


def unique_hdf5_key(key: str, *, used: set[str]) -> str:
    """Return a unique HDF5 key name within the provided used-set."""
    for i in range(10_000):  # pragma: no cover - defensive upper bound
        suffix = "" if i == 0 else f"__{i}"
        candidate = f"{key}{suffix}"
        if candidate in used:
            continue
        used.add(candidate)
        return candidate
    raise RuntimeError("Could not generate unique HDF5 key")


def detect_hdf5_layout(h5f: h5py.File) -> str | None:
    """Detect whether an HDF5 file stores entries as datasets or groups."""
    if not h5f.keys():
        return None
    kinds = {type(h5f[k]).__name__ for k in h5f.keys()}
    if kinds == {"Dataset"}:
        return LAYOUT_DATASET
    if kinds == {"Group"}:
        return LAYOUT_GROUP
    return None


def write_hdf5_manifest(
    h5f: h5py.File,
    *,
    kind: str,
    layout: str,
    keymap: dict[str, str],
    order: Iterable[str],
) -> None:
    """Write GWexpy collection layout metadata into HDF5 attributes."""
    h5f.attrs[KIND_ATTR] = kind
    h5f.attrs[LAYOUT_ATTR] = layout
    h5f.attrs[VERSION_ATTR] = 1
    h5f.attrs[KEYMAP_ATTR] = json.dumps(keymap, sort_keys=True)
    h5f.attrs[ORDER_ATTR] = json.dumps(list(order))


def read_hdf5_keymap(h5f: h5py.File) -> dict[str, str]:
    """Read the logical-to-physical HDF5 key map from file attributes."""
    raw = h5f.attrs.get(KEYMAP_ATTR)
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (TypeError, ValueError):
        pass
    return {}


def read_hdf5_order(h5f: h5py.File) -> list[str]:
    """Read the stored iteration order for HDF5 collection entries."""
    raw = h5f.attrs.get(ORDER_ATTR)
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
    except (TypeError, ValueError):
        pass
    return []


def ensure_hdf5_file(
    target: str | Path, *, mode: str | None = None, overwrite: bool = False
) -> h5py.File:
    """Open an HDF5 file using overwrite-aware GWexpy defaults."""
    if mode is None:
        mode = "w"
    if overwrite:
        mode = "w"
    return h5py.File(target, mode)


def normalize_layout(layout: str | None) -> str:
    """Normalize a user-facing HDF5 layout name to an internal constant."""
    if layout is None:
        return LAYOUT_DATASET
    key = str(layout).lower()
    if key in {"gwpy", "dataset", "dataset-per-entry"}:
        return LAYOUT_DATASET
    if key in {"group", "group-per-entry", "legacy"}:
        return LAYOUT_GROUP
    raise ValueError(f"Unknown HDF5 layout: {layout!r}")
