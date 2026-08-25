"""Versioned tabular schema for coupling-factor segments.

The public functions in this module intentionally operate on table-like
objects instead of introducing another table class. Pandas DataFrames are the
primary surface; Astropy Tables work through the same ``columns`` and
``__getitem__`` protocol when Astropy is available.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u

SCHEMA_NAME = "gwexpy.coupling.segment.v1"

__all__ = [
    "SCHEMA_NAME",
    "from_json_envelope",
    "from_result",
    "from_results",
    "to_json_envelope",
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
_KNOWN_COLUMNS = set(_REQUIRED_COLUMNS) | set(_OPTIONAL_COLUMNS)
_INT64_MAX = 2**63 - 1
_FREQUENCY_GRID_ULPS = 32
_FREQUENCY_GRID_RESOLUTION_FRACTION = 1e-9


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
    if hasattr(column, "tolist"):
        values = column.tolist()
        return list(values) if isinstance(values, Iterable) else [values]
    return list(column)


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


def _is_missing_optional(value: Any) -> bool:
    return _is_absent_optional(value) or (isinstance(value, str) and value == "")


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


def _validate_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
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


def _validate_frequency_column(table: Any, values: list[Any]) -> None:
    """Validate the declared frequency unit without changing the table."""
    column = table["frequency_hz"]
    missing_unit = object()
    column_unit = getattr(column, "unit", missing_unit)
    if column_unit is not missing_unit:
        if column_unit is None:
            raise ValueError("frequency_hz must use the canonical Hz unit")
        try:
            if u.Unit(column_unit) != u.Hz:
                raise ValueError("frequency_hz must use the canonical Hz unit")
        except (TypeError, ValueError, u.UnitsError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("frequency_hz"):
                raise
            raise ValueError("frequency_hz must use the canonical Hz unit") from exc
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
            missing_value = _is_missing_optional(value)
            if kind == "upper_limit":
                if missing_value or not isinstance(value, str) or not value.strip():
                    raise ValueError("limit_method is required for upper_limit")
            elif not _is_missing_optional(value):
                raise ValueError("limit_method is forbidden for measurement")
    elif "upper_limit" in estimate_kind:
        raise ValueError("limit_method is required for upper_limit")
    if "confidence_level" in names:
        for kind, value in zip(estimate_kind, columns["confidence_level"]):
            missing_value = _is_missing_optional(value)
            if kind == "upper_limit":
                if missing_value:
                    raise ValueError("confidence_level is required for upper_limit")
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise TypeError("confidence_level must be a real number")
                q = float(value)
                if not np.isfinite(q) or not 0 < q < 1:
                    raise ValueError("confidence_level must satisfy 0 < q < 1")
            elif not _is_missing_optional(value):
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


def _frequency_grid_tolerance(frequencies: np.ndarray) -> np.ndarray:
    """Return per-bin Hz tolerances tied to precision and bin resolution.

    The tolerance is the larger of 32 binary64 ULPs at a bin's magnitude and
    one billionth of its nearest positive neighbour spacing.  This admits unit
    conversion roundoff without merging physically distinct analysis bins.
    """
    scale = np.maximum(np.abs(frequencies), 1.0)
    ulp_tolerance = _FREQUENCY_GRID_ULPS * np.finfo(float).eps * scale
    if frequencies.size < 2:
        return ulp_tolerance

    differences = np.abs(np.diff(frequencies))
    resolutions = np.full(frequencies.shape, np.inf, dtype=float)
    positive = differences[differences > 0]
    if positive.size == 0:
        return ulp_tolerance
    resolutions[0] = differences[0] if differences[0] > 0 else positive.min()
    resolutions[-1] = differences[-1] if differences[-1] > 0 else positive.min()
    if frequencies.size > 2:
        left = np.where(differences[:-1] > 0, differences[:-1], positive.min())
        right = np.where(differences[1:] > 0, differences[1:], positive.min())
        resolutions[1:-1] = np.minimum(left, right)
    return np.maximum(ulp_tolerance, _FREQUENCY_GRID_RESOLUTION_FRACTION * resolutions)


def _frequency_grids_match(reference: np.ndarray, candidate: np.ndarray) -> bool:
    if reference.shape != candidate.shape:
        return False
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        return False
    return bool(
        np.all(np.abs(candidate - reference) <= _frequency_grid_tolerance(reference))
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_REQUIRED_COLUMNS, "estimate_kind"])


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
        q = float(confidence_level)
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
    Invalid bins are otherwise omitted. The returned frame contains no nulls;
    non-applicable mixed-kind metadata uses an empty string.
    """
    if isinstance(result, Mapping):
        raise TypeError("result mappings must be passed to from_results")
    if not hasattr(result, "cf"):
        raise TypeError("result must provide cf; use from_results for mappings")

    start, duration = _validate_result_options(
        start_gps_ns, duration_ns, limit_method, confidence_level
    )

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

    frame = pd.DataFrame(rows, columns=[*_REQUIRED_COLUMNS, "estimate_kind"])
    if frame.empty:
        return frame

    kinds = frame["estimate_kind"].tolist()
    if any(kind == "upper_limit" for kind in kinds):
        frame["limit_method"] = [
            limit_method if kind == "upper_limit" else "" for kind in kinds
        ]
        if confidence_level is not None:
            frame["confidence_level"] = [
                confidence_level if kind == "upper_limit" else "" for kind in kinds
            ]

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
    validated v1 frame with its declared columns preserved.
    """
    if not isinstance(results, Mapping):
        raise TypeError("results must be an estimate_coupling result mapping")
    _validate_result_options(start_gps_ns, duration_ns, limit_method, confidence_level)
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
        frame = _empty_frame()
    else:
        optional_columns = [
            name
            for name in _OPTIONAL_COLUMNS[1:]
            if any(name in item for item in frames)
        ]
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

    ``from_result`` deliberately omits optional columns when a target has no
    upper-limit rows.  Before heterogeneous factory frames are concatenated,
    add only optional columns that another target actually emitted and use
    ``None`` for non-applicable measurement cells.  Normalizing before concat
    prevents pandas from manufacturing floating ``NaN`` values.
    """
    if not optional_columns:
        return frame
    normalized = frame.copy(deep=True)
    measurement = normalized["estimate_kind"] == "measurement"
    for name in optional_columns:
        if name not in normalized:
            normalized[name] = pd.Series([None] * len(normalized), dtype=object)
        elif bool(measurement.any()):
            normalized[name] = normalized[name].astype(object)
            normalized.loc[measurement, name] = None
    return normalized


def _json_scalar(value: Any) -> Any:
    if _is_absent_optional(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_json_envelope(table: Any) -> dict[str, Any]:
    """Return a JSON-safe, schema-versioned envelope for a validated v1 table."""
    validate(table)
    columns = _column_names(table)
    values = {name: _column_values(table, name) for name in columns}
    rows = [
        [_json_scalar(values[name][index]) for name in columns]
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
    validate(frame)
    return frame
