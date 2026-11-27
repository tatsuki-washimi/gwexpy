"""Reducer contract and eager batch runner for SegmentTable statistics.

Reducers consume batches in source order. ``merge`` combines independently
computed states and must be associative for partitioned or parallel execution.
The caller decides which cells are valid before forming batches; this module
does not interpret :class:`~gwexpy.table.segment.cell.CellStatus`.
"""

from __future__ import annotations

import numbers
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
from astropy import units as u

from gwexpy.table.segment.cell import CellStatus
from gwexpy.table.segment.exceptions import CellLoadError

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable

BatchT = TypeVar("BatchT", contravariant=True)
StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT", covariant=True)


class Reducer(Protocol[BatchT, StateT, ResultT]):
    """Stateful reduction contract for eager, chunked, and later parallel work.

    ``initialize`` returns an independent empty state. ``update`` consumes one
    batch and returns its new state. ``merge`` combines two independent states
    in source order. ``finalize`` converts the completed state to the result.
    A reducer used with partitioning must make ``merge`` associative and use
    ``initialize`` as its identity. Inputs and state may carry physical units;
    the reducer is responsible for preserving them consistently.
    """

    def initialize(self) -> StateT:
        """Return an independent empty state."""
        ...

    def update(self, state: StateT, batch: BatchT) -> StateT:
        """Consume one batch and return the updated state."""
        ...

    def merge(self, a: StateT, b: StateT) -> StateT:
        """Combine independent states in source order."""
        ...

    def finalize(self, state: StateT) -> ResultT:
        """Return the final result from a completed state."""
        ...


def _run_reducer(
    reducer: Reducer[BatchT, StateT, ResultT],
    batches: Iterable[BatchT],
    *,
    batches_per_partition: int | None = None,
) -> ResultT:
    """Reduce batches without materializing the input iterable.

    When ``batches_per_partition`` is set, each partition starts with a fresh
    state and is merged into the accumulated state in input order. This path
    exercises the same merge contract needed by later parallel aggregations.
    An empty input finalizes one freshly initialized state.
    """
    if batches_per_partition is not None and (
        isinstance(batches_per_partition, bool)
        or not isinstance(batches_per_partition, int)
        or batches_per_partition <= 0
    ):
        raise ValueError("batches_per_partition must be a positive integer")

    combined = reducer.initialize()
    if batches_per_partition is None:
        for batch in batches:
            combined = reducer.update(combined, batch)
        return reducer.finalize(combined)

    partial = reducer.initialize()
    count = 0
    for batch in batches:
        partial = reducer.update(partial, batch)
        count += 1
        if count == batches_per_partition:
            combined = reducer.merge(combined, partial)
            partial = reducer.initialize()
            count = 0
    if count:
        combined = reducer.merge(combined, partial)
    return reducer.finalize(combined)


_BUILTIN_STATISTICS = frozenset({"median", "mean", "rms", "max", "min"})


class _ArrayStatisticReducer:
    """Reduce numeric payload samples with one of the #406 built-in aliases."""

    def __init__(self, statistic: str) -> None:
        if statistic not in _BUILTIN_STATISTICS:
            raise ValueError(f"Unknown statistic {statistic!r}.")
        self.statistic = statistic

    def initialize(self) -> list[np.ndarray[Any, Any]]:
        """Return an empty collection of sample batches."""
        return []

    def update(
        self, state: list[np.ndarray[Any, Any]], batch: np.ndarray[Any, Any]
    ) -> list[np.ndarray[Any, Any]]:
        """Append one numeric sample batch."""
        state.append(batch)
        return state

    def merge(
        self, a: list[np.ndarray[Any, Any]], b: list[np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        """Concatenate partial batches in source order."""
        return a + b

    def finalize(self, state: list[np.ndarray[Any, Any]]) -> Any:
        """Evaluate the requested statistic on the accumulated samples."""
        if not state:
            raise ValueError("Cannot extract a statistic from an empty payload.")
        values = np.concatenate(state)
        if values.size == 0:
            raise ValueError("Cannot extract a statistic from an empty payload.")
        if values.dtype.kind not in "biufc":
            raise TypeError("Built-in statistics require a numeric payload.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Built-in statistics require finite payload values.")
        if self.statistic == "median":
            if np.iscomplexobj(values):
                raise TypeError("median is undefined for complex payloads.")
            middle = values.size // 2
            ordered = np.partition(values, middle)
            if values.size % 2:
                # NumPy median converts even singleton int64/uint64 inputs
                # to float64, losing integers above 2**53.
                return ordered[middle].item()
            lower = ordered[:middle].max()
            upper = ordered[middle]
            if values.dtype.kind in "biu":
                exact = Fraction(int(lower) + int(upper), 2)
                if exact.denominator == 1:
                    return exact.numerator
                approximate = float(exact)
                if Fraction.from_float(approximate) != exact:
                    raise ValueError(
                        "Integer median is not exactly representable as float64; "
                        "use a custom statistic for exact half-integer output."
                    )
                return approximate
            with np.errstate(over="ignore"):
                midpoint = (lower + upper) / 2
            if np.isfinite(midpoint):
                return midpoint
            return lower / 2 + upper / 2
        if self.statistic == "mean":
            numeric = (
                values.astype(np.float64) if values.dtype.kind in "biu" else values
            )
            scale = np.max(np.abs(numeric))
            if scale == 0:
                return numeric.dtype.type(0)
            return scale * np.mean(numeric / scale)
        if self.statistic == "rms":
            # Scale before squaring so integer inputs and large finite floats
            # do not overflow an intermediate sum of squares.
            numeric = (
                values.astype(np.float64) if values.dtype.kind in "biu" else values
            )
            magnitudes = np.abs(numeric)
            scale = np.max(magnitudes)
            if scale == 0:
                return 0.0
            return scale * np.sqrt(np.mean((magnitudes / scale) ** 2))
        if np.iscomplexobj(values):
            raise TypeError(f"{self.statistic} is undefined for complex payloads.")
        if self.statistic == "max":
            return np.max(values)
        return np.min(values)


def _statistic_for_payload(payload: Any, statistic: str) -> Any:
    if (
        isinstance(payload, dict)
        or not hasattr(payload, "__array__")
        and not hasattr(payload, "value")
    ):
        raise TypeError("Built-in statistics require an array-like numeric payload.")
    unit = getattr(payload, "unit", None)
    raw_values = getattr(payload, "value", payload)
    if np.ma.isMaskedArray(raw_values) and bool(np.any(np.ma.getmaskarray(raw_values))):
        raise ValueError("Built-in statistics require unmasked payload samples.")
    values = np.asarray(raw_values).reshape(-1)
    result = _run_reducer(_ArrayStatisticReducer(statistic), [values])
    if not np.all(np.isfinite(result)):
        raise ValueError("Built-in statistic produced a non-finite result.")
    if unit is not None:
        return u.Quantity(result, unit=unit)
    return result


def _parallel_workers(parallel: bool | int) -> int:
    """Resolve the eager #406/#407 parallel policy to a worker count."""
    if parallel is False:
        return 1
    if parallel is True:
        return os.cpu_count() or 1
    if not isinstance(parallel, int) or parallel < 1:
        raise ValueError("parallel must be False, True, or a positive worker count")
    return parallel


class _StatMixin:
    """Per-segment eager scalar extraction operations."""

    _schema: dict[str, str]
    _payload: dict[str, Any]
    _meta: Any
    _units: dict[str, Any]
    _provenance: list[dict[str, Any]]

    def extract_stat(
        self,
        stat_fn: str | Callable[[Any], Any],
        column: str,
        out_column: str | None = None,
        *,
        parallel: bool | int = False,
    ) -> SegmentTable:
        """Extract a scalar statistic from each row without changing the source.

        Built-in aliases are ``"median"``, ``"mean"``, ``"rms"``, ``"max"``,
        and ``"min"``. They reduce finite numeric samples and preserve the
        payload's physical unit. A custom callable receives the whole payload
        object and must return one scalar (optionally a scalar Quantity).
        Missing, masked, and not-applicable cells yield ``NaN`` with a
        companion ``<out_column>_status`` column; failed cells raise.
        ``parallel=True`` uses available CPU threads, while a positive integer
        sets the worker count. Results retain input row order.
        """
        if column not in self._schema:
            raise KeyError(f"Column {column!r} not found in SegmentTable.")
        if not isinstance(stat_fn, str) and not callable(stat_fn):
            raise TypeError("stat_fn must be a built-in alias or callable")
        if isinstance(stat_fn, str) and stat_fn not in _BUILTIN_STATISTICS:
            raise ValueError(
                f"Unknown statistic {stat_fn!r}; expected one of "
                f"{sorted(_BUILTIN_STATISTICS)}."
            )
        workers = _parallel_workers(parallel)
        source_unit = (
            u.Unit(self._units[column])
            if isinstance(stat_fn, str) and column in self._units
            else None
        )
        output = (
            out_column or f"{column}_{stat_fn if isinstance(stat_fn, str) else 'stat'}"
        )
        status_column = f"{output}_status"
        if output in self._schema or status_column in self._schema:
            raise ValueError(
                f"Output column {output!r} or {status_column!r} already exists."
            )

        target = self.copy()  # type: ignore[attr-defined]

        def compute(index: int) -> tuple[Any, CellStatus]:
            if column in target._payload:
                cell = target._payload[column][index]
                if cell.status in (
                    CellStatus.MISSING,
                    CellStatus.MASKED,
                    CellStatus.NOT_APPLICABLE,
                ):
                    return np.nan, cell.status
            else:
                status = target._meta_row_status(column, index)
                if status == CellStatus.FAILED:
                    raise CellLoadError(
                        f"Column {column!r} row {index} is in FAILED state."
                    )
                if status != CellStatus.VALID:
                    return np.nan, status
            payload = target.row(index)[column]
            result = (
                _statistic_for_payload(payload, stat_fn)
                if isinstance(stat_fn, str)
                else stat_fn(payload)
            )
            if source_unit is not None:
                result = (
                    result.to(source_unit)
                    if isinstance(result, u.Quantity)
                    else u.Quantity(result, unit=source_unit)
                )
            if np.ndim(result) != 0:
                raise ValueError("extract_stat() requires one scalar result per row")
            scalar = result.value if isinstance(result, u.Quantity) else result
            if not isinstance(scalar, numbers.Number) or not bool(
                np.isfinite(np.asarray(scalar))
            ):
                raise ValueError(
                    "extract_stat() requires a finite numeric scalar result"
                )
            return result, CellStatus.VALID

        if workers == 1:
            extracted = [compute(i) for i in range(len(self._meta))]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                extracted = list(executor.map(compute, range(len(self._meta))))

        values = [value for value, _ in extracted]
        statuses = [status.value for _, status in extracted]
        units = [
            value.unit
            for value, status in extracted
            if status == CellStatus.VALID and isinstance(value, u.Quantity)
        ]
        nonquantities = any(
            status == CellStatus.VALID and not isinstance(value, u.Quantity)
            for value, status in extracted
        )
        if units and nonquantities:
            raise ValueError(
                "extract_stat() results mix physical quantities and unitless scalars"
            )
        common_unit = units[0] if units else source_unit
        if common_unit is not None:
            values = [
                value.to_value(common_unit) if isinstance(value, u.Quantity) else value
                for value in values
            ]
        target.add_column(output, values, unit=common_unit)
        # DataFrame assignment otherwise upcasts [int64_max, NaN] or
        # [int64_max, 1.5] to float64 before the value reaches HDF5.
        if any(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            target._meta[output] = pd.Series(
                values, index=target._meta.index, dtype=object
            )
        target.add_column(status_column, statuses)
        target._provenance.append(
            {
                "op": "extract_stat",
                "column": column,
                "out_column": output,
                "statistic": stat_fn
                if isinstance(stat_fn, str)
                else getattr(stat_fn, "__name__", "callable"),
                "parallel": parallel,
            }
        )
        return cast("SegmentTable", target)
