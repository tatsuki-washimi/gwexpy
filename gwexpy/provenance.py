"""JSON-safe provenance records for reproducible GWexpy analyses."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
from astropy import units as u
from astropy.units import Quantity, UnitBase

from ._version import __version__


def normalize_json(value: Any, *, _active: set[int] | None = None) -> Any:
    """Return a deterministic, JSON-safe copy of ``value``.

    Astropy units use the canonical tagged form
    ``{"__gwexpy_type__": "astropy.unit", "value": "m"}``.
    Scalar Astropy quantities use the same Unit tag nested in
    ``{"__gwexpy_type__": "astropy.quantity", "unit": ..., "value": ...}``.
    NumPy arrays are intentionally rejected so provenance cannot contain a
    hidden bulk-data payload.
    """
    active = set() if _active is None else _active

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Quantity):
        if value.ndim != 0:
            raise TypeError("array-valued Quantity payloads are not valid provenance")
        return {
            "__gwexpy_type__": "astropy.quantity",
            "unit": normalize_json(value.unit, _active=active),
            "value": normalize_json(value.value.item(), _active=active),
        }
    if isinstance(value, UnitBase):
        return {
            "__gwexpy_type__": "astropy.unit",
            "value": value.to_string(format="generic"),
        }
    if isinstance(value, np.ndarray):
        raise TypeError("ndarray values are not valid provenance payloads")
    if isinstance(value, np.generic):
        return normalize_json(value.item(), _active=active)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        finite_value = float(value)
        if not math.isfinite(finite_value):
            raise ValueError("non-finite provenance numbers are not allowed")
        return finite_value
    if isinstance(value, Mapping):
        _enter_container(value, active)
        try:
            if any(not isinstance(key, str) for key in value):
                raise TypeError("provenance mapping keys must be strings")
            if "__gwexpy_type__" in value:
                return _normalize_tagged_mapping(value, active)
            mapping_result: dict[str, Any] = {}
            for key in sorted(value):
                mapping_result[key] = normalize_json(value[key], _active=active)
            return mapping_result
        finally:
            active.remove(id(value))
    if isinstance(value, (list, tuple)):
        _enter_container(value, active)
        try:
            return [normalize_json(item, _active=active) for item in value]
        finally:
            active.remove(id(value))
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def _normalize_tagged_mapping(value: Mapping[str, Any], active: set[int]) -> Any:
    tag = value["__gwexpy_type__"]
    if tag == "astropy.unit":
        if set(value) != {"__gwexpy_type__", "value"}:
            raise ValueError("invalid astropy.unit provenance tag")
        unit_value = value["value"]
        if not isinstance(unit_value, str):
            raise ValueError("astropy.unit tag value must be a string")
        try:
            unit = u.Unit(unit_value)
        except Exception as exc:  # noqa: BLE001 - astropy parser boundary
            raise ValueError(f"invalid astropy.unit value: {unit_value!r}") from exc
        return normalize_json(unit, _active=active)
    if tag == "astropy.quantity":
        if set(value) != {"__gwexpy_type__", "unit", "value"}:
            raise ValueError("invalid astropy.quantity provenance tag")
        normalized_unit = normalize_json(value["unit"], _active=active)
        if not isinstance(normalized_unit, dict):
            raise ValueError("astropy.quantity unit must be an Astropy Unit")
        if set(normalized_unit) != {"__gwexpy_type__", "value"}:
            raise ValueError("astropy.quantity unit must be an Astropy Unit")
        if normalized_unit["__gwexpy_type__"] != "astropy.unit":
            raise ValueError("astropy.quantity unit must be an Astropy Unit")
        magnitude = normalize_json(value["value"], _active=active)
        try:
            quantity = u.Quantity(
                magnitude,
                unit=u.Unit(normalized_unit["value"]),
            )
        except Exception as exc:  # noqa: BLE001 - astropy parser boundary
            raise ValueError("invalid astropy.quantity value") from exc
        return normalize_json(quantity, _active=active)
    raise ValueError(f"invalid provenance type tag: {tag!r}")


def copy_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated deep copy of a provenance mapping."""
    result = normalize_json(provenance)
    if not isinstance(result, dict):  # pragma: no cover - Mapping guarantees it
        raise TypeError("provenance must normalize to a mapping")
    return result


def build_provenance(
    algorithm: str,
    parameters: Mapping[str, Any],
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Build the versioned provenance record used by public analyses."""
    if deterministic and (rng is not None or seed is not None):
        raise ValueError("deterministic provenance cannot have rng or seed")
    if not isinstance(algorithm, str):
        raise TypeError("provenance algorithm must be a string")
    if not isinstance(parameters, Mapping):
        raise TypeError("provenance parameters must be a mapping")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise TypeError("provenance seed must be an integer")

    if deterministic:
        rng_info = {
            "method": "none",
            "bit_generator": None,
            "seed": None,
        }
    else:
        rng_info = _rng_record(rng, seed)

    return {
        "schema": "gwexpy.provenance",
        "version": 1,
        "algorithm": algorithm,
        "parameters": normalize_json(parameters),
        "rng": normalize_json(rng_info),
        "software": {"gwexpy": __version__},
    }


def build_operation_provenance(
    algorithm: str,
    *,
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build deterministic provenance for a binary NumPy operation."""
    result = build_provenance(algorithm, {}, deterministic=True)
    result["inputs"] = {
        "left": None if left is None else copy_provenance(left),
        "right": None if right is None else copy_provenance(right),
    }
    return result


def dumps_json(value: Any) -> str:
    """Serialize a normalized value deterministically for sidecar storage."""
    return json.dumps(
        normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def loads_json(payload: str) -> Any:
    """Load strict JSON and decode the canonical Astropy Unit tag."""
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_object_members,
            parse_constant=_reject_non_finite_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid provenance JSON") from exc
    decoded = _decode_json(value)
    normalize_json(decoded)
    return decoded


def _rng_record(
    rng: np.random.Generator | None,
    seed: int | None,
) -> dict[str, Any]:
    if rng is not None:
        try:
            bit_generator = rng.bit_generator
        except AttributeError as exc:
            raise TypeError("rng must expose a bit_generator") from exc
        return {
            "method": "caller_managed",
            "bit_generator": type(bit_generator).__name__,
            "seed": None,
        }
    if seed is not None:
        return {
            "method": "seeded_generator",
            "bit_generator": type(np.random.default_rng(seed).bit_generator).__name__,
            "seed": int(seed),
        }
    return {
        "method": "legacy_global",
        "bit_generator": "MT19937",
        "seed": None,
    }


def _enter_container(value: object, active: set[int]) -> None:
    value_id = id(value)
    if value_id in active:
        raise ValueError("cycle detected in provenance payload")
    active.add(value_id)


def _decode_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_json(item) for item in value]
    if isinstance(value, dict):
        if "__gwexpy_type__" in value:
            tag = value["__gwexpy_type__"]
            if tag == "astropy.unit":
                if set(value) != {"__gwexpy_type__", "value"}:
                    raise ValueError("invalid astropy.unit provenance tag")
                unit = value["value"]
                if not isinstance(unit, str):
                    raise ValueError("astropy.unit tag value must be a string")
                try:
                    return u.Unit(unit)
                except Exception as exc:  # noqa: BLE001 - astropy parser boundary
                    raise ValueError(f"invalid astropy.unit value: {unit!r}") from exc
            if tag == "astropy.quantity":
                if set(value) != {"__gwexpy_type__", "unit", "value"}:
                    raise ValueError("invalid astropy.quantity provenance tag")
                unit = _decode_json(value["unit"])
                if not isinstance(unit, UnitBase):
                    raise ValueError("astropy.quantity unit must be an Astropy Unit")
                magnitude = _decode_json(value["value"])
                try:
                    return u.Quantity(magnitude, unit=unit)
                except Exception as exc:  # noqa: BLE001 - astropy parser boundary
                    raise ValueError("invalid astropy.quantity value") from exc
            raise ValueError(f"invalid provenance type tag: {tag!r}")
        return {key: _decode_json(item) for key, item in value.items()}
    return value


def _reject_duplicate_object_members(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object member: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")
