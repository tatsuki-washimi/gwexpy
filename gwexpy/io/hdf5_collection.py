from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py

KEYMAP_ATTR = "gwexpy_keymap"
ORDER_ATTR = "gwexpy_order"
KIND_ATTR = "gwexpy_kind"
LAYOUT_ATTR = "gwexpy_layout"
VERSION_ATTR = "gwexpy_layout_version"
COLLECTION_MANIFEST_ATTRS = frozenset(
    {KEYMAP_ATTR, ORDER_ATTR, KIND_ATTR, LAYOUT_ATTR, VERSION_ATTR}
)
LAYOUT_DATASET = "dataset-per-entry"
LAYOUT_GROUP = "group-per-entry"

_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Number of per-entry failures spelled out in a diagnostic message.
_MAX_REPORTED_FAILURES = 10


class SkippedHDF5Entries:
    """Per-entry read failures of one manifest-free HDF5 collection read.

    A manifest-free file is read by discovery: every top-level entry is a
    candidate, and a candidate that cannot be decoded is skipped.  This
    records those skips for one ``read()`` call so that the omission is
    reported instead of being silent.

    Parameters
    ----------
    kind : str
        Name of the collection class being read.
    source : object
        The file being read, used in messages only.
    total : int
        Number of candidate entries discovered by this call.

    """

    def __init__(self, kind: str, source: Any, total: int) -> None:
        self.kind = kind
        self.source = source
        self.total = total
        self.failures: list[tuple[str, BaseException, BaseException | None]] = []

    def record(
        self,
        name: str,
        exc: BaseException,
        first: BaseException | None = None,
    ) -> None:
        """Record that candidate entry ``name`` failed to read with ``exc``.

        ``first`` is the failure of an earlier read attempt on the same entry
        (the group layout tries its ``data`` member before the group itself).
        """
        self.failures.append((str(name), exc, first))

    @staticmethod
    def _describe_one(
        name: str, exc: BaseException, first: BaseException | None
    ) -> str:
        text = f"{name!r}: {type(exc).__name__}: {exc}"
        if first is not None and str(first) != str(exc):
            text += f" (first attempt: {type(first).__name__}: {first})"
        return text

    def _describe(self) -> str:
        shown = self.failures[:_MAX_REPORTED_FAILURES]
        detail = "; ".join(self._describe_one(*failure) for failure in shown)
        hidden = len(self.failures) - len(shown)
        if hidden > 0:
            detail += f"; and {hidden} more"
        return detail

    def report(self, *, stacklevel: int = 3) -> None:
        """Warn about a partial read, or raise if no candidate could be read.

        Does nothing when every candidate was read, including the case of a
        file with no candidate entries (a genuinely empty collection).

        Raises
        ------
        ValueError
            If candidate entries exist and none of them could be read.  The
            first underlying failure is attached as ``__cause__``.

        """
        if not self.failures:
            return
        if len(self.failures) >= self.total:
            msg = (
                f"{self.kind}.read: none of the {self.total} HDF5 entries in "
                f"{self.source!r} could be read: {self._describe()}"
            )
            raise ValueError(msg) from self.failures[0][1]
        warnings.warn(
            f"{self.kind}.read: {len(self.failures)} of {self.total} HDF5 "
            f"entries in {self.source!r} could not be read and were omitted "
            f"from the result: {self._describe()}",
            UserWarning,
            stacklevel=stacklevel,
        )


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


def has_hdf5_collection_manifest(h5f: h5py.Group | h5py.File) -> bool:
    """Return whether an HDF5 container has collection manifest metadata."""
    return any(attr in h5f.attrs for attr in COLLECTION_MANIFEST_ATTRS)


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
