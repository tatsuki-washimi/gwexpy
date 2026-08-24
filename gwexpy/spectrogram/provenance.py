"""Validated, versioned provenance records for :class:`Spectrogram`."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any

PROVENANCE_SCHEMA = "gwexpy.spectrogram.provenance"
PROVENANCE_SCHEMA_VERSION = 1
HDF5_PROVENANCE_ATTRIBUTE = "gwexpy_provenance"


def validated_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached, strictly JSON-safe provenance mapping.

    The mapping is deliberately restricted to the JSON data model: mappings
    with string keys, lists, strings, booleans, integers, finite floats, and
    ``None``.  This rejects live RNGs, NumPy scalar coercions, NaNs, tuples,
    and arbitrary objects rather than converting them ambiguously.
    """
    if not isinstance(value, Mapping):
        raise TypeError("provenance must be a mapping")
    if value.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(f"provenance schema must be {PROVENANCE_SCHEMA!r}")
    if value.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"provenance schema_version must be {PROVENANCE_SCHEMA_VERSION!r}"
        )
    normalized = _json_value(value, path="provenance")
    # json.dumps is both a final validation and a guard against future changes
    # to _json_value accidentally widening the accepted value domain.
    json.dumps(normalized, allow_nan=False, sort_keys=True)
    return copy.deepcopy(normalized)


def analysis_provenance(
    method: str,
    parameters: Mapping[str, Any],
    *,
    seed: int | None | object = ...,
    rng_provided: bool | None = None,
    seed_unused: bool = False,
) -> dict[str, Any]:
    """Build the common versioned analysis record used by statistics APIs."""
    analysis: dict[str, Any] = {
        "method": method,
        "parameters": dict(parameters),
    }
    if seed is not ...:
        analysis["random"] = {
            "seed": seed,
            "rng_provided": rng_provided,
            "seed_unused": seed_unused,
        }
    return validated_provenance(
        {
            "schema": PROVENANCE_SCHEMA,
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "analysis": analysis,
        }
    )


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            normalized[key] = _json_value(child, path=f"{path}.{key}")
        return normalized
    if isinstance(value, list):
        return [_json_value(child, path=f"{path}[]") for child in value]
    raise TypeError(
        f"{path} contains {type(value).__name__}, which is not a JSON-safe "
        "provenance value"
    )
