"""Versioned tabular schema for coupling-factor segments.

The public functions in this module intentionally operate on table-like
objects instead of introducing another table class. Pandas DataFrames are the
primary surface; Astropy Tables work through the same ``columns`` and
``__getitem__`` protocol when Astropy is available.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table

SCHEMA_NAME = "gwexpy.coupling.segment.v1"

__all__ = [
    "SCHEMA_NAME",
    "from_json_envelope",
    "from_result",
    "from_results",
    "to_astropy",
    "to_json_envelope",
    "to_pandas",
    "validate",
]

_REQUIRED_COLUMNS = (
    "start_gps_ns",
    "duration_ns",
    "source_channel",
    "response_channel",
    "frequency_hz",
    "coupling_factor",
    "coupling_factor_unit",
)
_OPTIONAL_COLUMNS = (
    "estimate_kind",
    "limit_method",
    "confidence_level",
)
_ROW_OPTIONAL_COLUMNS = ("limit_method", "confidence_level")
_KNOWN_COLUMNS = set(_REQUIRED_COLUMNS) | set(_OPTIONAL_COLUMNS)
_INT64_MAX = 2**63 - 1
_FREQUENCY_GRID_ULPS = 32
_CANONICAL_COLUMN_UNITS = {
    "start_gps_ns": u.ns,
    "duration_ns": u.ns,
    "frequency_hz": u.Hz,
}
_JSON_INTEGER_COLUMNS = {"start_gps_ns", "duration_ns"}
_JSON_FLOAT_COLUMNS = {"frequency_hz", "coupling_factor", "confidence_level"}
_JSON_STRING_COLUMNS = {
    "source_channel",
    "response_channel",
    "coupling_factor_unit",
    "estimate_kind",
    "limit_method",
}
_ADAPTER_METADATA_KEY = "gwexpy.coupling.segment.v1.astropy_metadata"
_ADAPTER_CARRIER_FIELDS = {"schema", "table_meta", "columns"}
_ADAPTER_COLUMN_FIELDS = {"meta", "description", "format"}


def _column_names(table: Any) -> list[str]:
    try:
        columns = table.columns
        names = list(columns.keys()) if hasattr(columns, "keys") else list(columns)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "table must provide a columns collection and __getitem__"
        ) from exc
    if not all(isinstance(name, str) for name in names):
        raise TypeError("table column names must be strings")
    try:
        for name in names:
            table[name]
    except (KeyError, TypeError, AttributeError) as exc:
        raise TypeError("table must support item access by column name") from exc
    return names


def _column_values(table: Any, name: str) -> list[Any]:
    column = table[name]
    try:
        return list(column)
    except TypeError:
        pass
    if hasattr(column, "tolist"):
        values = column.tolist()
        return list(values) if isinstance(values, Iterable) else [values]
    raise TypeError(f"column {name!r} must be iterable")


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NA or np.ma.is_masked(value):
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _is_absent_optional(value: Any) -> bool:
    """Return whether optional metadata is explicitly absent, never NaN."""
    return value is None or value is pd.NA or np.ma.is_masked(value)


def _is_missing_optional(value: Any, name: str, estimate_kind: str) -> bool:
    """Return whether a row's optional value is an allowed absence.

    Explicit nulls are accepted for either optional field on measurement rows.
    The legacy empty string is also accepted only for the two row-optional
    fields on measurement rows; upper limits must carry concrete metadata.
    """
    return _is_absent_optional(value) or (
        name in _ROW_OPTIONAL_COLUMNS
        and estimate_kind == "measurement"
        and isinstance(value, str)
        and value == ""
    )


def _canonical_optional_value(value: Any, name: str, estimate_kind: str) -> Any:
    """Map an allowed optional absence to its canonical ``None`` form."""
    if _is_missing_optional(value, name, estimate_kind):
        return None
    return value


def _validate_int(value: Any, name: str, *, positive: bool) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a signed int64")
    integer = int(value)
    if integer < (1 if positive else 0) or integer > _INT64_MAX:
        condition = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {condition} and fit signed int64")
    return integer


def _validate_interval(start_gps_ns: Any, duration_ns: Any) -> tuple[int, int]:
    """Validate a nanosecond interval and reject an unrepresentable endpoint."""
    start = _validate_int(start_gps_ns, "start_gps_ns", positive=False)
    duration = _validate_int(duration_ns, "duration_ns", positive=True)
    if duration > _INT64_MAX - start:
        raise ValueError("start_gps_ns + duration_ns endpoint must fit signed int64")
    return start, duration


def _normalize_binary64(value: Any, name: str) -> float:
    """Return a finite binary64 value without losing nonzero semantics."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must fit finite binary64") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must fit finite binary64")
    try:
        source_is_nonzero = bool(value != 0)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if source_is_nonzero and converted == 0.0:
        raise ValueError(f"{name} must not underflow binary64 to zero")
    return converted


def _validate_nonnegative_float(value: Any, name: str) -> float:
    converted = _normalize_binary64(value, name)
    if not np.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return converted


def _canonical_unit(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("coupling_factor_unit must be a nonempty unit string")
    if not value.strip():
        raise ValueError("coupling_factor_unit must be nonempty")
    try:
        unit = u.Unit(value)
        canonical = unit.to_string()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid coupling_factor_unit {value!r}") from exc
    if not canonical:
        if unit == u.dimensionless_unscaled:
            return "1"
        raise ValueError("coupling_factor_unit must be nonempty")
    return canonical


def _validate_canonical_column_unit(table: Any, name: str, unit: u.UnitBase) -> None:
    """Accept unitless schema columns or an explicitly canonical Astropy unit."""
    column = table[name]
    missing_unit = object()
    column_unit = getattr(column, "unit", missing_unit)
    if column_unit is missing_unit or column_unit is None:
        return
    try:
        if u.Unit(column_unit) != unit:
            raise ValueError
    except (TypeError, ValueError, u.UnitsError) as exc:
        raise ValueError(
            f"{name} must use the canonical {unit.to_string()} unit"
        ) from exc


def _validate_frequency_column(table: Any, values: list[Any]) -> None:
    """Validate the declared frequency unit without changing the table."""
    _validate_canonical_column_unit(table, "frequency_hz", u.Hz)
    for value in values:
        _validate_nonnegative_float(value, "frequency_hz")


def _validate_coupling_factor_units(table: Any, unit_strings: list[str]) -> None:
    """Require Astropy's optional column unit to agree with row unit strings.

    ``coupling_factor_unit`` is the v1 interchange authority.  A pandas frame
    has no separate column-unit channel.  Astropy's column unit is therefore
    accepted only as matching metadata; it is never used to relabel values.
    """
    column_unit = getattr(table["coupling_factor"], "unit", None)
    if column_unit is None:
        return
    try:
        declared_unit = u.Unit(column_unit)
    except (TypeError, ValueError, u.UnitsError) as exc:
        raise ValueError(
            "coupling_factor_unit conflicts with coupling_factor unit"
        ) from exc
    for unit_string in unit_strings:
        if declared_unit != u.Unit(unit_string):
            raise ValueError("coupling_factor_unit conflicts with coupling_factor unit")


def validate(table: Any) -> Any:
    """Validate a v1 coupling-segment table without mutating it.

    Parameters
    ----------
    table : pandas.DataFrame or astropy.table.Table
        Table with the required columns and optional estimate metadata.

    Returns
    -------
    same table type
        The original table object. Unit strings are parsed and canonicalized
        only in local validation temporaries.

    """
    names = _column_names(table)
    unknown = sorted(set(names) - _KNOWN_COLUMNS)
    if unknown:
        raise ValueError(f"unknown columns: {unknown}")
    missing = [name for name in _REQUIRED_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    columns = {name: _column_values(table, name) for name in names}
    row_count = len(columns[_REQUIRED_COLUMNS[0]])
    if any(len(values) != row_count for values in columns.values()):
        raise ValueError("all table columns must have the same length")

    for name in _REQUIRED_COLUMNS:
        if any(_is_null(value) for value in columns[name]):
            raise ValueError(f"required column {name!r} must not contain nulls")

    for name, unit in _CANONICAL_COLUMN_UNITS.items():
        if name != "frequency_hz":
            _validate_canonical_column_unit(table, name, unit)
    for start, duration in zip(columns["start_gps_ns"], columns["duration_ns"]):
        _validate_interval(start, duration)
    for name in ("source_channel", "response_channel"):
        for value in columns[name]:
            if not isinstance(value, str):
                raise TypeError(f"{name} must contain strings")
            if not value.strip():
                raise ValueError(f"{name} must contain nonempty strings")
    _validate_frequency_column(table, columns["frequency_hz"])
    for value in columns["coupling_factor"]:
        _validate_nonnegative_float(value, "coupling_factor")

    unit_strings = [_canonical_unit(value) for value in columns["coupling_factor_unit"]]
    _validate_coupling_factor_units(table, unit_strings)
    estimate_kind = ["measurement"] * row_count
    if "estimate_kind" in names:
        estimate_kind = []
        for value in columns["estimate_kind"]:
            if _is_null(value) or value not in {"measurement", "upper_limit"}:
                raise ValueError("estimate_kind must be measurement or upper_limit")
            estimate_kind.append(value)

    if "limit_method" in names:
        limit_methods = columns["limit_method"]
        for kind, value in zip(estimate_kind, limit_methods):
            missing_value = _is_missing_optional(value, "limit_method", kind)
            if kind == "upper_limit":
                if missing_value or not isinstance(value, str) or not value.strip():
                    raise ValueError("limit_method is required for upper_limit")
            elif not missing_value:
                raise ValueError("limit_method is forbidden for measurement")
    elif "upper_limit" in estimate_kind:
        raise ValueError("limit_method is required for upper_limit")
    if "confidence_level" in names:
        for kind, value in zip(estimate_kind, columns["confidence_level"]):
            missing_value = _is_missing_optional(value, "confidence_level", kind)
            if kind == "upper_limit":
                if missing_value:
                    raise ValueError("confidence_level is required for upper_limit")
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise TypeError("confidence_level must be a real number")
                q = _normalize_binary64(value, "confidence_level")
                if not np.isfinite(q) or not 0 < q < 1:
                    raise ValueError("confidence_level must satisfy 0 < q < 1")
            elif not missing_value:
                raise ValueError("confidence_level is only valid for upper_limit")
    return table


def _series_values(series: Any, name: str) -> np.ndarray:
    try:
        values = np.asarray(series.value, dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"result.{name} must provide numeric values") from exc
    return values


def _frequency_values(series: Any, name: str) -> np.ndarray:
    try:
        axis = series.xindex
        if axis is None:
            raise ValueError
        axis_unit = getattr(axis, "unit", None)
        if axis_unit is None or not u.Unit(axis_unit).is_equivalent(u.Hz):
            raise ValueError
        return np.asarray(axis.to_value(u.Hz), dtype=float)
    except (AttributeError, TypeError, ValueError, u.UnitsError) as exc:
        raise TypeError(
            f"{name} must provide a frequency axis with units convertible to Hz"
        ) from exc


def _series_unit(series: Any, name: str) -> u.UnitBase:
    value = getattr(series, "unit", None)
    if value is None:
        raise TypeError(f"{name} must provide a coupling-factor unit")
    try:
        return u.Unit(value)
    except (TypeError, ValueError, u.UnitsError) as exc:
        raise TypeError(f"{name} must provide a valid coupling-factor unit") from exc


def _boolean_mask(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise TypeError("result.valid_mask must contain only boolean values")
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError("result.valid_mask must align with result.cf")
    if mask.dtype == np.dtype(bool):
        return mask
    if mask.dtype == np.dtype(object) and all(
        isinstance(item, (bool, np.bool_)) for item in mask.flat
    ):
        return np.asarray(mask, dtype=bool)
    raise TypeError("result.valid_mask must contain only boolean values")


def _frequency_grids_match(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Compare grids using at most 32 binary64 nextafter steps per direction."""
    if reference.shape != candidate.shape:
        return False
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        return False
    direction = np.where(candidate >= reference, np.inf, -np.inf)
    boundary = reference.copy()
    for _ in range(_FREQUENCY_GRID_ULPS):
        boundary = np.nextafter(boundary, direction)
    within = np.where(
        candidate >= reference, candidate <= boundary, candidate >= boundary
    )
    return bool(np.all(within))


def _requested_optional_columns(
    limit_method: str | None, confidence_level: float | None
) -> list[str]:
    """Return the deterministic v1 optional shape requested by a factory call."""
    columns: list[str] = []
    if limit_method is not None:
        columns.append("limit_method")
    if confidence_level is not None:
        columns.append("confidence_level")
    return columns


def _empty_frame(optional_columns: Iterable[str] = ()) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[*_REQUIRED_COLUMNS, "estimate_kind", *optional_columns]
    )


def _validate_result_options(
    start_gps_ns: int,
    duration_ns: int,
    limit_method: str | None,
    confidence_level: float | None,
) -> tuple[int, int]:
    """Validate arguments shared by single- and mapping-result factories."""
    start, duration = _validate_interval(start_gps_ns, duration_ns)
    if limit_method is not None and (
        not isinstance(limit_method, str) or not limit_method.strip()
    ):
        raise TypeError("limit_method must be a nonempty string")
    if confidence_level is not None:
        if limit_method is None:
            raise ValueError("confidence_level requires limit_method")
        if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
            confidence_level, Real
        ):
            raise TypeError("confidence_level must be a real number")
        q = _normalize_binary64(confidence_level, "confidence_level")
        if not np.isfinite(q) or not 0 < q < 1:
            raise ValueError("confidence_level must satisfy 0 < q < 1")
    return start, duration


def from_result(
    result: Any,
    *,
    start_gps_ns: int,
    duration_ns: int,
    limit_method: str | None = None,
    confidence_level: float | None = None,
) -> pd.DataFrame:
    """Convert a coupling result into a validated v1 pandas DataFrame.

    Valid finite nonnegative measurements are emitted first. A bin that is not
    a valid measurement is emitted as an upper limit only when ``limit_method``
    is supplied and ``result.cf_ul`` contains a finite nonnegative value.
    Invalid bins are otherwise omitted. Supplying ``limit_method`` requests its
    column for every emitted row, with ``None`` for measurements; supplying
    ``confidence_level`` similarly requests that column. Without either
    argument, the frame uses the established minimal v1 shape.
    """
    if isinstance(result, Mapping):
        raise TypeError("result mappings must be passed to from_results")
    if not hasattr(result, "cf"):
        raise TypeError("result must provide cf; use from_results for mappings")

    start, duration = _validate_result_options(
        start_gps_ns, duration_ns, limit_method, confidence_level
    )
    optional_columns = _requested_optional_columns(limit_method, confidence_level)

    cf = result.cf
    cf_values = _series_values(cf, "cf")
    frequencies = _frequency_values(cf, "result.cf")
    if cf_values.ndim != 1 or frequencies.shape != cf_values.shape:
        raise ValueError(
            "result.cf values and frequency axis must be one-dimensional and aligned"
        )
    valid_mask = _boolean_mask(result.valid_mask, cf_values.shape)

    cf_ul = getattr(result, "cf_ul", None)
    ul_values = np.full(cf_values.shape, np.nan)
    cf_unit = _series_unit(cf, "result.cf")
    if cf_ul is not None:
        ul_frequencies = _frequency_values(cf_ul, "result.cf_ul")
        ul_values = _series_values(cf_ul, "cf_ul")
        if ul_values.shape != cf_values.shape:
            raise ValueError("result.cf_ul must align with result.cf")
        if not _frequency_grids_match(frequencies, ul_frequencies):
            raise ValueError("result.cf_ul frequency grid must match result.cf")
        ul_unit = _series_unit(cf_ul, "result.cf_ul")
        try:
            ul_values = ul_values * ul_unit.to(cf_unit)
        except (TypeError, ValueError, u.UnitsError) as exc:
            raise ValueError(
                "result.cf_ul coupling-factor unit must be compatible with result.cf"
            ) from exc

    unit_text = cf_unit.to_string() or "1"
    source = getattr(result, "witness_name", None)
    response = getattr(result, "target_name", None)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("result.witness_name must be a nonempty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("result.target_name must be a nonempty string")

    rows: list[dict[str, Any]] = []
    for index, frequency in enumerate(frequencies):
        common = {
            "start_gps_ns": start,
            "duration_ns": duration,
            "source_channel": source,
            "response_channel": response,
            "frequency_hz": frequency,
            "coupling_factor_unit": unit_text,
        }
        if (
            valid_mask[index]
            and np.isfinite(frequency)
            and frequency >= 0
            and np.isfinite(cf_values[index])
            and cf_values[index] >= 0
        ):
            rows.append(
                {
                    **common,
                    "coupling_factor": float(cf_values[index]),
                    "estimate_kind": "measurement",
                }
            )
        elif (
            limit_method is not None
            and np.isfinite(frequency)
            and frequency >= 0
            and np.isfinite(ul_values[index])
            and ul_values[index] >= 0
        ):
            rows.append(
                {
                    **common,
                    "coupling_factor": float(ul_values[index]),
                    "estimate_kind": "upper_limit",
                }
            )

    frame = pd.DataFrame(
        rows, columns=[*_REQUIRED_COLUMNS, "estimate_kind", *optional_columns]
    )
    kinds = frame["estimate_kind"].tolist()
    if "limit_method" in optional_columns:
        frame["limit_method"] = pd.Series(
            [limit_method if kind == "upper_limit" else None for kind in kinds],
            dtype=object,
        )
    if "confidence_level" in optional_columns:
        frame["confidence_level"] = pd.Series(
            [confidence_level if kind == "upper_limit" else None for kind in kinds],
            dtype=object,
        )

    validate(frame)
    return frame


def from_results(
    results: Mapping[str, Any],
    *,
    start_gps_ns: int,
    duration_ns: int,
    limit_method: str | None = None,
    confidence_level: float | None = None,
) -> pd.DataFrame:
    """Convert an ``estimate_coupling`` result mapping into a v1 DataFrame.

    ``estimate_coupling`` returns one result directly for a single target and a
    mapping for zero or multiple targets.  Use :func:`from_result` for the
    former and this adapter for the latter.  Empty mappings produce an empty,
    validated v1 frame with the same requested optional shape as a populated
    factory call.
    """
    if not isinstance(results, Mapping):
        raise TypeError("results must be an estimate_coupling result mapping")
    _validate_result_options(start_gps_ns, duration_ns, limit_method, confidence_level)
    optional_columns = _requested_optional_columns(limit_method, confidence_level)
    if not all(isinstance(key, str) for key in results):
        raise TypeError("results mapping keys must be strings")
    frames = [
        from_result(
            result,
            start_gps_ns=start_gps_ns,
            duration_ns=duration_ns,
            limit_method=limit_method,
            confidence_level=confidence_level,
        )
        for _, result in sorted(results.items())
    ]
    if not frames:
        frame = _empty_frame(optional_columns)
    else:
        normalized_frames = [
            _normalize_optional_columns(item, optional_columns) for item in frames
        ]
        frame = pd.concat(normalized_frames, ignore_index=True)
    validate(frame)
    return frame


def _normalize_optional_columns(
    frame: pd.DataFrame, optional_columns: list[str]
) -> pd.DataFrame:
    """Copy a factory frame with explicit nulls for absent optional metadata.

    Factory arguments determine the canonical optional shape. Before
    heterogeneous factory frames are concatenated, add each requested optional
    column and use ``None`` for non-applicable measurement cells. Normalizing
    before concat prevents pandas from manufacturing floating ``NaN`` values.
    """
    if not optional_columns:
        return frame
    normalized = frame.copy(deep=True)
    kinds = (
        normalized["estimate_kind"].tolist()
        if "estimate_kind" in normalized
        else ["measurement"] * len(normalized)
    )
    for name in optional_columns:
        if name not in normalized:
            normalized[name] = pd.Series([None] * len(normalized), dtype=object)
        else:
            normalized[name] = pd.Series(
                [
                    _canonical_optional_value(value, name, kind)
                    for kind, value in zip(kinds, normalized[name])
                ],
                dtype=object,
            )
    return normalized


def _copy_adapter_metadata(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception as exc:
        raise ValueError("adapter metadata must be deep-copyable") from exc


def _build_adapter_carrier(table: Table, columns: list[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "table_meta": _copy_adapter_metadata(dict(table.meta)),
        "columns": {
            name: {
                "meta": _copy_adapter_metadata(dict(table[name].meta)),
                "description": table[name].description,
                "format": table[name].format,
            }
            for name in columns
        },
    }


def _adapter_carrier(
    attrs: Mapping[Any, Any], columns: list[str]
) -> dict[str, Any] | None:
    if _ADAPTER_METADATA_KEY not in attrs:
        return None
    carrier = attrs[_ADAPTER_METADATA_KEY]
    if not isinstance(carrier, Mapping) or set(carrier) != _ADAPTER_CARRIER_FIELDS:
        raise ValueError("invalid coupling segment adapter metadata")
    if carrier["schema"] != SCHEMA_NAME:
        raise ValueError("invalid coupling segment adapter metadata schema")
    if not isinstance(carrier["table_meta"], Mapping):
        raise ValueError("invalid coupling segment adapter metadata table_meta")
    column_metadata = carrier["columns"]
    if not isinstance(column_metadata, Mapping) or set(column_metadata) != set(columns):
        raise ValueError("invalid coupling segment adapter metadata columns")
    validated_columns: dict[str, dict[str, Any]] = {}
    for name in columns:
        metadata = column_metadata[name]
        if not isinstance(metadata, Mapping) or set(metadata) != _ADAPTER_COLUMN_FIELDS:
            raise ValueError("invalid coupling segment adapter metadata column")
        if not isinstance(metadata["meta"], Mapping):
            raise ValueError("invalid coupling segment adapter metadata column meta")
        if metadata["description"] is not None and not isinstance(
            metadata["description"], str
        ):
            raise ValueError("invalid coupling segment adapter metadata description")
        if metadata["format"] is not None and not isinstance(metadata["format"], str):
            raise ValueError("invalid coupling segment adapter metadata format")
        validated_columns[name] = {
            "meta": _copy_adapter_metadata(dict(metadata["meta"])),
            "description": metadata["description"],
            "format": metadata["format"],
        }
    return {
        "schema": SCHEMA_NAME,
        "table_meta": _copy_adapter_metadata(dict(carrier["table_meta"])),
        "columns": validated_columns,
    }


def to_pandas(table: Any) -> pd.DataFrame:
    """Return a schema-aware pandas copy with explicit optional nulls.

    Unlike :meth:`astropy.table.Table.to_pandas`, this adapter maps Astropy
    masked optional metadata and a permitted legacy measurement empty string to
    ``None`` rather than floating ``NaN``. It also records Astropy table/column
    metadata needed by :func:`to_astropy`.
    """
    validate(table)
    columns = _column_names(table)
    values = {name: _column_values(table, name) for name in columns}
    estimate_kind = values.get(
        "estimate_kind", ["measurement"] * len(next(iter(values.values()), []))
    )
    frame = pd.DataFrame(
        {
            name: [
                _canonical_optional_value(value, name, estimate_kind[index])
                if name in _ROW_OPTIONAL_COLUMNS
                else (
                    None
                    if name in _OPTIONAL_COLUMNS and _is_absent_optional(value)
                    else value
                )
                for index, value in enumerate(column_values)
            ]
            for name, column_values in values.items()
        },
        dtype=object,
    )
    if isinstance(table, pd.DataFrame):
        frame.attrs = deepcopy(table.attrs)
        _adapter_carrier(frame.attrs, columns)
        return frame
    if not isinstance(table, Table):
        raise TypeError("table must be a pandas DataFrame or Astropy Table")
    frame.attrs[_ADAPTER_METADATA_KEY] = _build_adapter_carrier(table, columns)
    return frame


def to_astropy(table: Any) -> Table:
    """Return a schema-aware Astropy copy with canonical schema units.

    The adapter attaches ``ns`` to the two time columns and ``Hz`` to the
    frequency column. It restores metadata captured by :func:`to_pandas` and
    masks canonical optional nulls without changing the input object.
    """
    frame = to_pandas(table)
    columns = list(frame.columns)
    data: dict[str, list[Any]] = {}
    masks: dict[str, list[bool]] = {}
    for name in columns:
        values = frame[name].tolist()
        mask = [
            name in _OPTIONAL_COLUMNS and _is_absent_optional(value) for value in values
        ]
        if any(mask):
            placeholder = "" if name == "limit_method" else 0.0
            values = [
                placeholder if is_missing else value
                for value, is_missing in zip(values, mask)
            ]
            masks[name] = mask
        data[name] = values

    result = Table(data, masked=True)
    for name, mask in masks.items():
        result[name].mask = mask
    for name, unit in _CANONICAL_COLUMN_UNITS.items():
        result[name].unit = unit
    unit_strings = [_canonical_unit(value) for value in frame["coupling_factor_unit"]]
    if unit_strings and all(value == unit_strings[0] for value in unit_strings):
        result["coupling_factor"].unit = u.Unit(unit_strings[0])

    adapter_metadata = _adapter_carrier(frame.attrs, columns)
    if adapter_metadata is not None:
        result.meta.update(adapter_metadata["table_meta"])
        for name, metadata in adapter_metadata["columns"].items():
            result[name].meta.update(metadata["meta"])
            result[name].description = metadata["description"]
            result[name].format = metadata["format"]
    validate(result)
    return result


def _json_scalar(value: Any, name: str) -> Any:
    if _is_absent_optional(value):
        return None
    if name in _JSON_INTEGER_COLUMNS:
        return int(value)
    if name in _JSON_FLOAT_COLUMNS:
        return _normalize_binary64(value, name)
    if name in _JSON_STRING_COLUMNS:
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_json_envelope(table: Any) -> dict[str, Any]:
    """Return a JSON-safe, schema-versioned envelope for a validated v1 table."""
    validate(table)
    columns = _column_names(table)
    values = {name: _column_values(table, name) for name in columns}
    estimate_kind = values.get(
        "estimate_kind", ["measurement"] * len(next(iter(values.values()), []))
    )
    rows = [
        [
            _json_scalar(
                _canonical_optional_value(
                    values[name][index], name, estimate_kind[index]
                )
                if name in _ROW_OPTIONAL_COLUMNS
                else values[name][index],
                name,
            )
            for name in columns
        ]
        for index in range(len(values[_REQUIRED_COLUMNS[0]]))
    ]
    return {"schema": SCHEMA_NAME, "columns": columns, "rows": rows}


def from_json_envelope(envelope: Mapping[str, Any]) -> pd.DataFrame:
    """Restore a v1 pandas DataFrame from a strict JSON-safe envelope."""
    if not isinstance(envelope, Mapping):
        raise TypeError("envelope must be a mapping")
    expected_fields = {"schema", "columns", "rows"}
    if set(envelope) != expected_fields:
        raise ValueError("envelope must contain only schema, columns, and rows")
    if envelope["schema"] != SCHEMA_NAME:
        raise ValueError(f"unsupported coupling segment schema {envelope['schema']!r}")
    columns = envelope["columns"]
    rows = envelope["rows"]
    if (
        not isinstance(columns, list)
        or not all(isinstance(name, str) for name in columns)
        or len(set(columns)) != len(columns)
    ):
        raise TypeError("envelope columns must be unique strings")
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) != len(columns) for row in rows
    ):
        raise TypeError("envelope rows must be lists matching envelope columns")
    # Object columns preserve JSON ``null`` as ``None`` instead of allowing
    # pandas to coerce mixed numeric optional metadata to floating ``NaN``.
    frame = pd.DataFrame(rows, columns=columns, dtype=object)
    frame = _normalize_optional_columns(
        frame, [name for name in _ROW_OPTIONAL_COLUMNS if name in frame]
    )
    validate(frame)
    return frame
