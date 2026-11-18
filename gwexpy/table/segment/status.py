"""Error-policy helpers and missing-data reporting / selection for SegmentTable."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd

from gwexpy.table.segment.cell import _UNSET, CellStatus, SegmentCell

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable

VALID_ERROR_POLICIES = ("raise", "record", "coerce", "skip")


def validate_errors_policy(errors: str) -> None:
    """Validate an ``errors=`` execution policy string."""
    if errors not in VALID_ERROR_POLICIES:
        raise ValueError(
            f"Invalid errors policy {errors!r}. "
            f"Must be one of: {', '.join(VALID_ERROR_POLICIES)}."
        )


def is_scalar_nan(val: Any) -> bool:
    """Return True if *val* is a scalar NA/NaN (without triggering array truthiness)."""
    if val is None or val is _UNSET:
        return True
    if isinstance(val, (float, np.floating)):
        return bool(np.isnan(val))
    try:
        res = pd.isna(val)
    except (TypeError, ValueError):
        return False
    return bool(res) if isinstance(res, (bool, np.bool_)) else False


def is_payload_nonfinite(val: Any) -> bool:
    """Return True if a materialized payload contains NaN or non-finite values."""
    if val is None or val is _UNSET:
        return True
    if isinstance(val, Mapping):
        return any(is_payload_nonfinite(v) for v in val.values())
    if hasattr(val, "value"):
        arr = np.asarray(val.value)
    elif isinstance(val, np.ndarray):
        arr = val
    elif isinstance(val, (bool, np.bool_)):
        return False
    elif isinstance(val, numbers.Complex):
        c = complex(val)
        return not (math.isfinite(c.real) and math.isfinite(c.imag))
    else:
        return is_scalar_nan(val)

    if arr.dtype.kind in ("f", "c", "i", "u"):
        return not bool(np.all(np.isfinite(arr)))
    if arr.dtype == object:
        return any(is_scalar_nan(x) for x in arr.flat)
    return False


class _MissingMixin:
    """Missing-data reporting and status-aware selection methods."""

    _meta: pd.DataFrame
    _payload: dict[str, list[SegmentCell]]
    _schema: dict[str, str]

    def _resolve_subset(
        self,
        subset: Optional[Union[str, Sequence[str]]],
        *,
        exclude_span_when_none: bool = False,
    ) -> list[str]:
        all_cols = list(self.columns)  # type: ignore[attr-defined]
        if subset is None:
            if exclude_span_when_none and len(all_cols) > 1:
                return [c for c in all_cols if c != "span"]
            return all_cols
        cols = [subset] if isinstance(subset, str) else list(subset)
        for col in cols:
            if col not in self._schema:
                raise KeyError(f"Column {col!r} not found in SegmentTable.")
        return cols

    def _meta_row_status(self, col: str, i: int) -> CellStatus:
        status_col = f"{col}_status"
        if status_col in self._meta.columns:
            raw_status = self._meta.at[i, status_col]
            if raw_status == CellStatus.FAILED.value:
                return CellStatus.FAILED
            if raw_status == CellStatus.MISSING.value:
                return CellStatus.MISSING
            if raw_status == CellStatus.MASKED.value:
                return CellStatus.MASKED
            if raw_status == CellStatus.NOT_APPLICABLE.value:
                return CellStatus.NOT_APPLICABLE
        val = self._meta.at[i, col]
        return CellStatus.MISSING if is_scalar_nan(val) else CellStatus.VALID

    def missing_report(self) -> pd.DataFrame:
        """Return a per-column summary of cell status counts without loading payloads.

        Returns
        -------
        pandas.DataFrame
            Columns: ``column``, ``total``, ``valid``, ``missing``, ``failed``,
            ``unloaded``, ``masked``, ``not_applicable``.

        """
        n = len(self._meta)
        rows: list[dict[str, Any]] = []
        for col in self.columns:  # type: ignore[attr-defined]
            counts = {status: 0 for status in CellStatus}
            if col in self._payload:
                for cell in self._payload[col]:
                    counts[cell.status] += 1
            else:
                for i in range(n):
                    counts[self._meta_row_status(col, i)] += 1
            rows.append(
                {
                    "column": col,
                    "total": n,
                    "valid": counts[CellStatus.VALID],
                    "missing": counts[CellStatus.MISSING],
                    "failed": counts[CellStatus.FAILED],
                    "unloaded": counts[CellStatus.UNLOADED],
                    "masked": counts[CellStatus.MASKED],
                    "not_applicable": counts[CellStatus.NOT_APPLICABLE],
                }
            )
        return pd.DataFrame(rows)

    def isna(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        include_failed: bool = True,
        deep: bool = False,
    ) -> pd.DataFrame:
        """Return a boolean DataFrame indicating missing (and optionally failed) cells.

        Parameters
        ----------
        columns:
            Column name or sequence of column names.  Defaults to all columns.
        include_failed:
            If ``True`` (default), ``FAILED`` cells are marked ``True``.
        deep:
            If ``True``, inspect payload values for ``NaN`` or non-finite
            numbers, loading ``UNLOADED`` cells if necessary.  Defaults to
            ``False`` so lazy payloads are never loaded unexpectedly.

        Returns
        -------
        pandas.DataFrame

        """
        cols = self._resolve_subset(columns)
        n = len(self._meta)
        data: dict[str, list[bool]] = {}
        for col in cols:
            if col in self._payload:
                col_flags: list[bool] = []
                for cell in self._payload[col]:
                    st = cell.status
                    if st == CellStatus.UNLOADED:
                        if deep and cell.loader is not None:
                            try:
                                val = cell.get()
                            except Exception:
                                col_flags.append(bool(include_failed))
                                continue
                            col_flags.append(is_payload_nonfinite(val))
                        elif deep and cell.loader is None:
                            col_flags.append(True)
                        else:
                            col_flags.append(False)
                    elif st == CellStatus.VALID:
                        if deep:
                            col_flags.append(is_payload_nonfinite(cell.get()))
                        else:
                            col_flags.append(False)
                    elif st == CellStatus.FAILED:
                        col_flags.append(bool(include_failed))
                    else:
                        # MISSING, MASKED, NOT_APPLICABLE
                        col_flags.append(True)
                data[col] = col_flags
            else:
                col_flags = []
                for i in range(n):
                    st = self._meta_row_status(col, i)
                    if st == CellStatus.FAILED:
                        col_flags.append(bool(include_failed))
                    else:
                        col_flags.append(st != CellStatus.VALID)
                data[col] = col_flags
        return pd.DataFrame(data, index=range(n))

    def select_valid(
        self, subset: Optional[Union[str, Sequence[str]]] = None
    ) -> SegmentTable:
        """Select rows where every column in *subset* is in ``VALID`` state.

        Does not trigger lazy loading (``UNLOADED`` cells are not considered valid).
        """
        cols = self._resolve_subset(subset)
        n = len(self._meta)
        keep: list[int] = []
        for i in range(n):
            ok = True
            for col in cols:
                if col in self._payload:
                    if self._payload[col][i].status != CellStatus.VALID:
                        ok = False
                        break
                else:
                    if self._meta_row_status(col, i) != CellStatus.VALID:
                        ok = False
                        break
            if ok:
                keep.append(i)
        return self._slice_rows(keep)  # type: ignore[attr-defined]

    def select_missing(
        self, subset: Optional[Union[str, Sequence[str]]] = None
    ) -> SegmentTable:
        """Select rows where any column in *subset* is in ``MISSING`` state."""
        cols = self._resolve_subset(subset)
        n = len(self._meta)
        keep: list[int] = []
        for i in range(n):
            hit = False
            for col in cols:
                if col in self._payload:
                    if self._payload[col][i].status == CellStatus.MISSING:
                        hit = True
                        break
                else:
                    if self._meta_row_status(col, i) == CellStatus.MISSING:
                        hit = True
                        break
            if hit:
                keep.append(i)
        return self._slice_rows(keep)  # type: ignore[attr-defined]

    def select_failed(
        self, subset: Optional[Union[str, Sequence[str]]] = None
    ) -> SegmentTable:
        """Select rows where any column in *subset* is in ``FAILED`` state."""
        cols = self._resolve_subset(subset)
        n = len(self._meta)
        keep: list[int] = []
        for i in range(n):
            hit = False
            for col in cols:
                if col in self._payload:
                    if self._payload[col][i].status == CellStatus.FAILED:
                        hit = True
                        break
                else:
                    if self._meta_row_status(col, i) == CellStatus.FAILED:
                        hit = True
                        break
            if hit:
                keep.append(i)
        return self._slice_rows(keep)  # type: ignore[attr-defined]

    def dropna(
        self,
        subset: Optional[Union[str, Sequence[str]]] = None,
        how: str = "any",
        load: bool = False,
    ) -> SegmentTable:
        """Return a new table with rows containing missing/failed cells removed.

        Parameters
        ----------
        subset:
            Column name or sequence of column names to inspect.  Defaults to
            all non-``span`` columns (or all columns if only ``span`` exists).
        how:
            ``"any"`` (default) drops a row if any inspected column is missing
            or failed; ``"all"`` drops only if all inspected columns are
            missing or failed.
        load:
            If ``False`` (default), ``UNLOADED`` cells are **preserved** and
            never treated as missing.  If ``True``, lazy loaders in *subset*
            are resolved on a working copy before filtering.

        Returns
        -------
        SegmentTable

        """
        if how not in ("any", "all"):
            raise ValueError(f"Invalid how {how!r}; expected 'any' or 'all'.")
        cols = self._resolve_subset(subset, exclude_span_when_none=(how == "all"))
        if not cols:
            return self.copy()  # type: ignore[attr-defined]

        working: SegmentTable = self.copy(deep=False) if load else self  # type: ignore[assignment, attr-defined]
        if load:
            for col in cols:
                if col not in working._payload:
                    continue
                for cell in working._payload[col]:
                    if cell.status == CellStatus.UNLOADED and cell.loader is not None:
                        try:
                            cell.get()
                        except Exception:
                            # cell.get() transitions cell.status -> FAILED
                            pass

        n = len(working._meta)
        keep: list[int] = []
        for i in range(n):
            drop_flags: list[bool] = []
            for col in cols:
                if col in working._payload:
                    st = working._payload[col][i].status
                    if st == CellStatus.UNLOADED:
                        # load=False preserves UNLOADED; load=True drops empty UNLOADED
                        drop_flags.append(load)
                    elif st == CellStatus.VALID:
                        drop_flags.append(False)
                    else:
                        # MISSING, FAILED, MASKED, NOT_APPLICABLE
                        drop_flags.append(True)
                else:
                    st = working._meta_row_status(col, i)
                    drop_flags.append(st != CellStatus.VALID)

            should_drop = any(drop_flags) if how == "any" else all(drop_flags)
            if not should_drop:
                keep.append(i)

        return working._slice_rows(keep)
