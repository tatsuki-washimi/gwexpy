"""Project-wide on-disk HDF5 schema versioning and unknown-field policy (#658).

This module implements the U-4 schema policy:

- Schema identity and version live in root HDF5 attributes
  ``gwexpy_schema`` (``str``) and ``gwexpy_schema_version`` (``int``), rather
  than dataset attributes, so GWpy-compatible readers can still inspect
  datasets.
- Missing schema version is treated as a legacy unversioned file (returning
  ``None`` and optionally emitting a ``UserWarning``).
- Integer versions greater than ``max_supported_version``, schema mismatches,
  and non-integer/boolean/non-positive versions fail closed with ``ValueError``.
- Unknown attributes are partitioned by namespace: unknown ``gwexpy_*`` /
  ``_gwexpy_*`` attributes trigger a warning or ``ValueError`` (when
  ``strict=True``), while external/non-gwexpy attributes are preserved in a
  ``dict`` for lossless round-trip.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

SCHEMA_ATTR = "gwexpy_schema"
SCHEMA_VERSION_ATTR = "gwexpy_schema_version"
SCHEMA_ATTRS = frozenset({SCHEMA_ATTR, SCHEMA_VERSION_ATTR})
DEFAULT_GWEXPY_PREFIXES: tuple[str, ...] = ("gwexpy_", "_gwexpy_")

__all__ = [
    "DEFAULT_GWEXPY_PREFIXES",
    "SCHEMA_ATTR",
    "SCHEMA_ATTRS",
    "SCHEMA_VERSION_ATTR",
    "classify_unknown_attrs",
    "read_and_validate_schema",
    "write_schema_attrs",
]


def _decode_attr_str(value: Any) -> Any:
    """Decode bytes/numpy byte scalars to ``str`` when inspecting HDF5 attrs."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return value


def _coerce_strict_int(value: Any) -> int:
    """Coerce an integer attribute strictly, rejecting booleans and floats."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"Invalid {SCHEMA_VERSION_ATTR!r} value {value!r}: expected a positive integer"
        )
    version = int(value)
    if version < 1:
        raise ValueError(
            f"Invalid {SCHEMA_VERSION_ATTR!r} value {version!r}: expected >= 1"
        )
    return version


def _resolve_attrs(h5group: Any) -> Mapping[str, Any]:
    return h5group.attrs if hasattr(h5group, "attrs") else h5group


def write_schema_attrs(h5group: Any, schema_name: str, version: int) -> None:
    """Write ``gwexpy_schema`` and ``gwexpy_schema_version`` onto an HDF5 group.

    Parameters
    ----------
    h5group : h5py.Group or h5py.File
        Target HDF5 root group (or container with ``.attrs``).
    schema_name : str
        Non-empty canonical schema identifier (e.g. ``"gwexpy.segment_table"``).
    version : int
        Positive integer schema version (>= 1).

    """
    if not isinstance(schema_name, str) or not schema_name.strip():
        raise ValueError(f"Invalid schema_name {schema_name!r}: expected non-empty str")
    validated_version = _coerce_strict_int(version)
    attrs = h5group.attrs if hasattr(h5group, "attrs") else h5group
    attrs[SCHEMA_ATTR] = schema_name
    attrs[SCHEMA_VERSION_ATTR] = validated_version


def read_and_validate_schema(
    h5group: Any,
    expected_schema: str,
    max_supported_version: int,
    *,
    warn_on_legacy: bool = True,
) -> int | None:
    """Read and validate root HDF5 schema attributes.

    Parameters
    ----------
    h5group : h5py.Group or h5py.File
        HDF5 container whose ``attrs`` hold schema metadata.
    expected_schema : str
        Expected value of ``gwexpy_schema``.
    max_supported_version : int
        Maximum schema version supported by the caller (>= 1).
    warn_on_legacy : bool, optional
        If True (default), emit a ``UserWarning`` when ``gwexpy_schema_version``
        is missing (legacy unversioned file).

    Returns
    -------
    int or None
        Validated integer schema version, or ``None`` for legacy files lacking
        ``gwexpy_schema_version``.

    Raises
    ------
    ValueError
        If ``gwexpy_schema`` mismatches ``expected_schema``, if
        ``gwexpy_schema_version`` is not a positive integer (including ``bool``),
        or if the file version exceeds ``max_supported_version``.

    """
    if not isinstance(expected_schema, str) or not expected_schema.strip():
        raise ValueError(
            f"Invalid expected_schema {expected_schema!r}: expected non-empty str"
        )
    max_version = _coerce_strict_int(max_supported_version)
    attrs = _resolve_attrs(h5group)

    has_schema = SCHEMA_ATTR in attrs
    has_version = SCHEMA_VERSION_ATTR in attrs

    if has_schema:
        raw_schema = _decode_attr_str(attrs[SCHEMA_ATTR])
        if raw_schema != expected_schema:
            raise ValueError(
                f"Schema mismatch: expected {expected_schema!r}, got {raw_schema!r}"
            )

    if not has_version:
        if warn_on_legacy:
            warnings.warn(
                f"Missing {SCHEMA_VERSION_ATTR!r} for schema {expected_schema!r}; "
                "reading as legacy format",
                UserWarning,
                stacklevel=2,
            )
        return None

    if not has_schema:
        raise ValueError(
            f"Missing {SCHEMA_ATTR!r} (expected {expected_schema!r}) while "
            f"{SCHEMA_VERSION_ATTR!r} is present"
        )

    version = _coerce_strict_int(attrs[SCHEMA_VERSION_ATTR])
    if version > max_version:
        raise ValueError(
            f"Unsupported schema {expected_schema!r} version {version}: "
            f"maximum supported version is {max_version}"
        )
    return version


def classify_unknown_attrs(
    attrs: Mapping[str, Any],
    known_keys: Iterable[str],
    *,
    gwexpy_prefixes: tuple[str, ...] = DEFAULT_GWEXPY_PREFIXES,
    strict: bool = False,
) -> dict[str, Any]:
    """Partition unknown attributes by namespace and preserve external ones.

    Unknown attributes starting with any prefix in ``gwexpy_prefixes`` either
    raise ``ValueError`` (when ``strict=True``) or emit a ``UserWarning`` and
    are discarded (when ``strict=False``). Unknown attributes outside those
    prefixes are preserved in the returned dictionary for lossless round-trip.

    Parameters
    ----------
    attrs : mapping
        Attribute mapping (e.g. ``dict(h5group.attrs)``).
    known_keys : iterable of str
        Attribute names recognized by the caller.
    gwexpy_prefixes : tuple of str, optional
        Prefixes identifying GWexpy-owned attributes. Default:
        ``("gwexpy_", "_gwexpy_")``.
    strict : bool, optional
        If True, raise ``ValueError`` on unknown GWexpy-namespaced attributes.
        If False (default), emit a ``UserWarning`` and drop them.

    Returns
    -------
    dict[str, Any]
        External (non-GWexpy) unknown attributes preserved for round-trip.

    """
    known = set(known_keys)
    unknown_gwexpy: list[str] = []
    external: dict[str, Any] = {}

    for key, value in attrs.items():
        key_str = str(key)
        if key_str in known:
            continue
        if any(key_str.startswith(prefix) for prefix in gwexpy_prefixes):
            unknown_gwexpy.append(key_str)
        else:
            external[key_str] = value

    if unknown_gwexpy:
        names = ", ".join(repr(k) for k in sorted(unknown_gwexpy))
        if strict:
            raise ValueError(f"Unknown GWexpy attributes: {names}")
        warnings.warn(
            f"Ignoring unknown GWexpy attributes: {names}",
            UserWarning,
            stacklevel=2,
        )

    return external
