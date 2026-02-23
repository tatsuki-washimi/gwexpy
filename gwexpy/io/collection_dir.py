from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

MANIFEST_NAME = "_gwexpy_collection.json"
MANIFEST_VERSION = 1

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(text: str, *, default: str) -> str:
    raw = (text or "").strip()
    raw = raw.replace("/", "_").replace("\\", "_")
    stem = _SAFE_STEM_RE.sub("_", raw).strip("._-")
    return stem or default


def _unique_path(
    dirpath: Path, stem: str, ext: str, *, used: set[str]
) -> tuple[str, Path]:
    for i in range(10_000):  # pragma: no cover - defensive upper bound
        suffix = "" if i == 0 else f"__{i}"
        filename = f"{stem}{suffix}{ext}"
        if filename in used:
            continue
        used.add(filename)
        return filename, dirpath / filename
    raise RuntimeError("Could not generate a unique filename")


def write_collection_dir(
    dirpath: str | Path,
    *,
    kind: str,
    entry_format: str,
    entries: Iterable[tuple[str, Any]],
    writer: Callable[[Any, Path, str], None],
    meta_getter: Callable[[Any], dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a dict/list-like collection as a directory of per-entry files.

    Parameters
    ----------
    dirpath
        Directory path to create/write into.
    kind
        A short identifier like "TimeSeriesDict" or "FrequencySeriesList".
    entry_format
        Per-entry file format (e.g. "csv" or "txt").
    entries
        Iterable of (key, value) pairs.
    writer
        Callback: writer(value, filepath, entry_format).
    overwrite
        If False and the directory exists and is non-empty, raise.
    """
    dp = Path(dirpath)
    if dp.exists() and dp.is_file():
        raise NotADirectoryError(str(dp))
    if dp.exists() and any(dp.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Directory {dp} is not empty; pass overwrite=True to replace contents"
        )
    dp.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    ext = f".{entry_format.lower()}"
    manifest_entries: list[dict[str, Any]] = []

    for key, value in entries:
        key_str = str(key)
        stem = _safe_stem(key_str, default="item")
        filename, filepath = _unique_path(dp, stem, ext, used=used)
        writer(value, filepath, entry_format)
        entry: dict[str, Any] = {"key": key_str, "filename": filename}
        if meta_getter is not None:
            meta = meta_getter(value) or {}
            if meta:
                entry["meta"] = meta
        manifest_entries.append(entry)

    manifest = {
        "version": MANIFEST_VERSION,
        "kind": kind,
        "entry_format": entry_format,
        "entries": manifest_entries,
    }
    (dp / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return dp


def iter_collection_dir_entries(
    dirpath: str | Path,
    *,
    expected_kind: str | None = None,
    entry_format: str | None = None,
) -> tuple[str, list[tuple[str, Path, dict[str, Any]]]]:
    """Return (entry_format, [(key, filepath), ...]) from a collection directory."""
    dp = Path(dirpath)
    if not dp.is_dir():
        raise NotADirectoryError(str(dp))

    manifest_path = dp / MANIFEST_NAME
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_kind is not None and data.get("kind") != expected_kind:
            raise ValueError(
                f"Manifest kind mismatch: expected {expected_kind!r}, got {data.get('kind')!r}"
            )
        fmt = str(data.get("entry_format") or "").lower()
        if entry_format is not None and fmt != entry_format.lower():
            raise ValueError(
                f"Manifest format mismatch: expected {entry_format!r}, got {fmt!r}"
            )
        pairs: list[tuple[str, Path, dict[str, Any]]] = []
        for ent in data.get("entries", []):
            key = str(ent.get("key"))
            fn = str(ent.get("filename"))
            meta = ent.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            pairs.append((key, dp / fn, meta))
        return fmt, pairs

    # No manifest: infer from csv/txt only
    if entry_format is None:
        candidates = sorted(dp.glob("*.csv")) or sorted(dp.glob("*.txt"))
        if not candidates:
            raise FileNotFoundError(f"No .csv/.txt files found in {dp}")
        fmt = candidates[0].suffix.lstrip(".").lower()
    else:
        fmt = entry_format.lower()
        candidates = sorted(dp.glob(f"*.{fmt}"))
        if not candidates:
            raise FileNotFoundError(f"No *.{fmt} files found in {dp}")

    pairs = [(p.stem, p, {}) for p in candidates]
    return fmt, pairs


def read_collection_dir(
    dirpath: str | Path,
    *,
    expected_kind: str | None,
    entry_format: str | None,
    reader: Callable[[Path, str], Any],
) -> tuple[str, list[tuple[str, Any, dict[str, Any]]]]:
    fmt, pairs = iter_collection_dir_entries(
        dirpath, expected_kind=expected_kind, entry_format=entry_format
    )
    out: list[tuple[str, Any, dict[str, Any]]] = [
        (key, reader(path, fmt), meta) for key, path, meta in pairs
    ]
    return fmt, out
