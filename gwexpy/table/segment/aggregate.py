"""Eager, status-aware grouping and scalar aggregation for SegmentTable."""

from __future__ import annotations

import numbers
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from astropy import units as u

from gwexpy.table.segment.cell import CellStatus
from gwexpy.table.segment.stats import _ArrayStatisticReducer, _run_reducer
from gwexpy.table.segment.summary import SummaryTable

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable

_COUNT_COLUMNS = ("n_total", "n_valid", "n_missing", "n_failed", "n_masked")
_BUILTIN_AGGREGATES = frozenset({"count", "mean", "median", "min", "max"})


@dataclass(frozen=True)
class Quantile:
    """A quantile reducer specification for ``groupby().aggregate()``."""

    q: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.q) or not 0 <= self.q <= 1:
            raise ValueError("quantile probability must be in [0, 1]")


def quantile(q: float) -> Quantile:
    """Construct a quantile aggregate specification with probability *q*."""
    return Quantile(float(q))


class _ScalarAggregateReducer:
    """Reducer over validated numeric scalar rows in one group."""

    def __init__(self, method: str | Quantile):
        self.method = method

    def initialize(self) -> list[int | float]:
        """Return an independent empty state."""
        return []

    def update(self, state: list[int | float], batch: int | float) -> list[int | float]:
        """Add one scalar value."""
        state.append(batch)
        return state

    def merge(self, a: list[int | float], b: list[int | float]) -> list[int | float]:
        """Merge values in row order."""
        return a + b

    def finalize(self, state: list[int | float]) -> int | float:
        """Compute the requested aggregate."""
        if self.method == "count":
            return len(state)
        if not state:
            return np.nan
        if all(isinstance(item, numbers.Integral) for item in state):
            integers = [int(item) for item in state]
            if self.method == "min":
                return min(integers)
            if self.method == "max":
                return max(integers)
            if self.method == "mean":
                exact = Fraction(sum(integers), len(integers))
                return _integer_fraction_result(exact, "mean")
            if self.method == "median":
                return _integer_quantile(integers, 0.5)
            if isinstance(self.method, Quantile):
                return _integer_quantile(integers, self.method.q)
        if (
            any(isinstance(item, numbers.Integral) for item in state)
            and any(not isinstance(item, numbers.Integral) for item in state)
            and any(
                abs(int(item)) > 2**53
                for item in state
                if isinstance(item, numbers.Integral)
            )
        ):
            raise ValueError(
                "Mixed floating-point and large integer aggregate values can lose precision"
            )
        values = np.asarray(state)
        if values.dtype.kind not in "iuf":
            raise TypeError("Aggregate values must be real numeric scalars")
        if isinstance(self.method, Quantile):
            if self.method.q == 0.5:
                return _ArrayStatisticReducer("median").finalize([values])
            ordered = sorted(float(item) for item in state)
            rank = Fraction(str(self.method.q)) * (len(ordered) - 1)
            lower = rank.numerator // rank.denominator
            weight = float(rank - lower)
            result = (
                ordered[lower] * (1 - weight)
                + ordered[min(lower + 1, len(ordered) - 1)] * weight
            )
            if not np.isfinite(result):
                raise ValueError("Quantile aggregate produced a non-finite result")
            return result
        return _ArrayStatisticReducer(self.method).finalize([values])


def _integer_fraction_result(exact: Fraction, method: str) -> int | float:
    """Keep integral results exact and bound fractional conversion error."""
    if exact.denominator == 1:
        return exact.numerator
    approximate = float(exact)
    if not np.isfinite(approximate):
        raise ValueError(f"Integer {method} cannot be represented as float64")
    # Compare every conversion with the exact rational. Small decimal
    # quantiles (e.g. 0.1) have unavoidable binary64 rounding; allow at most
    # 1e-12 absolute error while rejecting lost large-integer fractions.
    error = abs(Fraction.from_float(approximate) - exact)
    if error > Fraction(1, 10**12):
        raise ValueError(
            f"Integer {method} cannot be represented within 1e-12 absolute error as float64"
        )
    return approximate


def _integer_quantile(values: list[int], q: float) -> int | float:
    """Interpolate integer quantiles without NumPy dtype coercion."""
    ordered = sorted(values)
    rank = Fraction(str(q)) * (len(ordered) - 1)
    lower = rank.numerator // rank.denominator
    fraction = rank - lower
    exact = (
        ordered[lower] * (1 - fraction)
        + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    )
    return _integer_fraction_result(exact, "quantile")


class SegmentTableGroupBy:
    """A detached eager grouping plan for metadata keys."""

    def __init__(self, table: SegmentTable, keys: Sequence[str]) -> None:
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("groupby requires distinct nonempty key names")
        for key in keys:
            if key not in table._schema:
                raise KeyError(f"Column {key!r} not found in SegmentTable")
            if key in table._payload:
                raise TypeError("groupby keys must be metadata columns")
        self._table = table.copy()
        self._keys = tuple(keys)

    def aggregate(
        self,
        *,
        skipna: bool = True,
        min_count: int = 1,
        **named: tuple[str, str | Quantile],
    ) -> SummaryTable:
        """Aggregate one scalar source column per group.

        Named specifications have the form ``output=(column, method)``.
        ``method`` is ``count``, ``mean``, ``median``, ``min``, ``max``, or
        :func:`quantile`. Counts classify the source column in each group;
        ``NOT_APPLICABLE`` contributes to ``n_missing``. Invalid source rows
        are excluded when ``skipna=True`` and rejected otherwise. ``min_count``
        applies to statistics, while ``count`` always reports ``n_valid``.
        """
        if not isinstance(skipna, bool):
            raise TypeError("skipna must be bool")
        if (
            isinstance(min_count, bool)
            or not isinstance(min_count, int)
            or min_count < 0
        ):
            raise ValueError("min_count must be a nonnegative integer")
        if not named:
            raise ValueError("aggregate requires at least one named specification")
        if set(named) & (set(self._keys) | set(_COUNT_COLUMNS)):
            raise ValueError(
                "Aggregate output names conflict with group or count columns"
            )

        sources: set[str] = set()
        for output, spec in named.items():
            if not isinstance(spec, tuple) or len(spec) != 2:
                raise TypeError(f"Aggregate {output!r} must be (column, method)")
            column, method = spec
            if not isinstance(column, str) or column not in self._table._schema:
                raise KeyError(f"Aggregate source column {column!r} does not exist")
            if column in self._table._payload:
                raise TypeError(
                    "Aggregate source must be a scalar metadata column; use extract_stat first"
                )
            if not (
                isinstance(method, Quantile)
                or isinstance(method, str)
                and method in _BUILTIN_AGGREGATES
            ):
                raise ValueError(f"Unsupported aggregate method {method!r}")
            sources.add(column)
        if len(sources) != 1:
            raise ValueError(
                "One aggregate call must use one source column for unambiguous status counts"
            )
        source = next(iter(sources))
        source_unit = self._table._units.get(source)
        valid_values = [
            self._table._meta.at[index, source]
            for index in range(len(self._table))
            if self._table._meta_row_status(source, index) == CellStatus.VALID
        ]
        quantities = [value for value in valid_values if isinstance(value, u.Quantity)]
        if quantities and source_unit is None:
            if len(quantities) != len(valid_values):
                raise ValueError(
                    "Aggregate source mixes quantities and unitless scalars"
                )
            source_unit = quantities[0].unit

        groups: OrderedDict[tuple[Any, ...], list[int]] = OrderedDict()
        for index in range(len(self._table)):
            if any(
                self._table._meta_row_status(name, index) != CellStatus.VALID
                for name in self._keys
            ):
                raise ValueError("groupby keys must have valid status")
            key = tuple(self._table._meta.at[index, name] for name in self._keys)
            for value in key:
                if not np.isscalar(value):
                    raise TypeError("groupby keys must be hashable scalar values")
                if pd.isna(value):
                    raise ValueError("groupby keys cannot be missing")
            try:
                groups.setdefault(key, []).append(index)
            except TypeError as exc:
                raise TypeError("groupby keys must be hashable scalar values") from exc

        rows: list[dict[str, Any]] = []
        for key, indices in groups.items():
            counts = {status: 0 for status in CellStatus}
            valid: list[int | float] = []
            for index in indices:
                status = self._table._meta_row_status(source, index)
                if status == CellStatus.VALID:
                    value = self._table._meta.at[index, source]
                    if isinstance(value, u.Quantity):
                        value = value.to_value(source_unit)
                    if not isinstance(value, numbers.Real) or not bool(
                        np.isfinite(np.asarray(value))
                    ):
                        raise ValueError(
                            f"Aggregate source {source!r} must contain finite real scalars"
                        )
                    valid.append(cast(int | float, value))
                counts[status] += 1
            n_invalid = len(indices) - len(valid)
            if n_invalid and not skipna:
                raise ValueError(
                    "aggregate encountered non-valid values with skipna=False"
                )
            row = dict(zip(self._keys, key, strict=True))
            row.update(
                n_total=len(indices),
                n_valid=len(valid),
                n_missing=counts[CellStatus.MISSING]
                + counts[CellStatus.NOT_APPLICABLE],
                n_failed=counts[CellStatus.FAILED],
                n_masked=counts[CellStatus.MASKED],
            )
            for output, (_, method) in named.items():
                if method == "count":
                    row[output] = len(valid)
                elif len(valid) < min_count:
                    row[output] = np.nan
                else:
                    row[output] = _run_reducer(_ScalarAggregateReducer(method), valid)
            rows.append(row)

        columns = [*self._keys, *_COUNT_COLUMNS, *named]
        frame = pd.DataFrame(rows, columns=columns)
        for output in named:
            values = [row[output] for row in rows]
            if any(
                isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                for value in values
            ):
                frame[output] = pd.Series(values, dtype=object)
        units = {
            key: self._table._units[key]
            for key in self._keys
            if key in self._table._units
        }
        if source_unit is not None:
            units.update(
                {
                    output: source_unit
                    for output, (_, method) in named.items()
                    if method != "count"
                }
            )
        return SummaryTable(frame, units=units)


class _GroupByMixin:
    """Eager grouping entry point for SegmentTable."""

    def groupby(self, keys: str | Sequence[str]) -> SegmentTableGroupBy:
        """Group rows by one or more scalar metadata columns."""
        from gwexpy.table.segment.core import SegmentTable

        names = [keys] if isinstance(keys, str) else list(keys)
        return SegmentTableGroupBy(cast(SegmentTable, self), names)
