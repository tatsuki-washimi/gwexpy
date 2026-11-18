"""SegmentTable — segment-keyed analysis container.

Each row represents one analysis unit defined by a :class:`~gwpy.segments.Segment`
(a half-open time interval ``[start, end)``).  Columns may be lightweight *meta*
values or heavy *payload* objects such as :class:`~gwpy.timeseries.TimeSeries`.
Payload columns are stored as :class:`~gwexpy.table.segment.cell.SegmentCell`
instances that support lazy loading and optional caching.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
import pandas as pd

from gwexpy.table.segment.cell import _UNSET, CellStatus, SegmentCell
from gwexpy.table.segment.ops import _OpsMixin
from gwexpy.table.segment.status import (
    _MissingMixin,
    is_scalar_nan,
    validate_errors_policy,
)

if TYPE_CHECKING:
    from gwexpy.table.segment.expr import SegmentColumn

__all__ = ["RowProxy", "SegmentTable"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Column kinds for meta (lightweight) columns.
_META_KINDS = frozenset({"segment", "meta", "object"})

#: Column kinds for payload (heavy) columns.
_PAYLOAD_KINDS = frozenset(
    {
        "timeseries",
        "timeseriesdict",
        "frequencyseries",
        "frequencyseriesdict",
        "spectrogram",
        "matrix",
        "array",
        "scalar",
        "object",
    }
)

#: All valid kinds.
_ALL_KINDS = _META_KINDS | _PAYLOAD_KINDS


def _parse_span_string(value: str) -> Any:
    """Parse simple serialized Segment forms used in CSV files."""
    from gwpy.segments import Segment

    text = value.strip()
    if text.startswith("Segment(") and text.endswith(")"):
        inner = text[len("Segment(") : -1]
    elif len(text) >= 2 and text[0] in "([" and text[-1] in ")]":
        inner = text[1:-1]
    else:
        raise ValueError(
            "Could not parse SegmentTable 'span' value "
            f"{value!r}; expected '(start, end)', 'Segment(start, end)', "
            "or '[start ... end)'."
        )

    separator = "..." if "..." in inner else ","
    parts = [part.strip() for part in inner.split(separator)]
    if len(parts) != 2:
        raise ValueError(
            "Could not parse SegmentTable 'span' value "
            f"{value!r}; expected exactly two endpoints."
        )

    try:
        start, end = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "Could not parse SegmentTable 'span' value "
            f"{value!r}; endpoints must be numeric."
        ) from exc
    return Segment(start, end)


def _coerce_span_value(value: Any) -> Any:
    from gwpy.segments import Segment

    if isinstance(value, Segment):
        return value
    if isinstance(value, str):
        return _parse_span_string(value)
    return value


# ---------------------------------------------------------------------------
# RowProxy
# ---------------------------------------------------------------------------


class RowProxy:
    """Dict-like read-only proxy for a single row of a :class:`SegmentTable`.

    Meta columns return their value directly; payload columns are resolved
    through :meth:`SegmentCell.get`.

    Parameters
    ----------
    table:
        The parent :class:`SegmentTable`.
    index:
        0-based row index.

    """

    __slots__ = ("_table", "_index")

    def __init__(self, table: SegmentTable, index: int) -> None:
        self._table = table
        self._index = index

    # -- dict-like access ---------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        table = self._table
        if key not in table._schema:
            raise KeyError(f"Column {key!r} not found in SegmentTable")

        if key in table._payload:
            # Payload column
            return table._payload[key][self._index].get()

        # Meta column
        return table._meta.at[self._index, key]

    @property
    def index(self) -> int:
        """The 0-based row index within the parent :class:`SegmentTable`."""
        return self._index

    def keys(self) -> list[str]:
        """Return all column names."""
        return self._table.columns

    def __repr__(self) -> str:
        span = self._table._meta.at[self._index, "span"]
        return f"RowProxy(index={self._index}, span={span})"


# ---------------------------------------------------------------------------
# SegmentTable
# ---------------------------------------------------------------------------


class SegmentTable(_MissingMixin, _OpsMixin):
    """Segment-keyed analysis table.

    Each row holds exactly one ``span`` (:class:`gwpy.segments.Segment`)
    plus arbitrary meta columns and optional heavy payload columns.

    Parameters
    ----------
    meta:
        :class:`pandas.DataFrame` with at least a ``span`` column whose
        elements are :class:`gwpy.segments.Segment` objects.

    Raises
    ------
    ValueError
        If ``meta`` does not contain a ``span`` column.
    TypeError
        If any element of the ``span`` column is not a
        :class:`gwpy.segments.Segment`.

    Examples
    --------
    >>> from gwpy.segments import Segment
    >>> import pandas as pd
    >>> segs = [Segment(0, 4), Segment(4, 8)]
    >>> st = SegmentTable(pd.DataFrame({"span": segs}))
    >>> len(st)
    2

    """

    # ------------------------------------------------------------------
    # Internal storage
    # ------------------------------------------------------------------

    _meta: pd.DataFrame
    _payload: dict[str, list[SegmentCell]]
    _schema: dict[str, str]
    _units: dict[str, Any]
    _segment_index: Any
    _provenance: list[dict[str, Any]]
    _external_attrs: dict[str, Any]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, meta: pd.DataFrame) -> None:
        if "span" not in meta.columns:
            raise ValueError("'meta' DataFrame must contain a 'span' column.")

        # Validate span element types
        try:
            from gwpy.segments import Segment
        except ImportError:
            Segment = cast(Any, None)

        if Segment is not None:
            for i, s in enumerate(meta["span"]):
                if not isinstance(s, Segment):
                    raise TypeError(
                        f"All elements of 'span' must be gwpy.segments.Segment, "
                        f"got {type(s).__name__!r} at index {i}."
                    )

        self._meta = meta.reset_index(drop=True).copy()
        self._payload: dict[str, list[SegmentCell]] = {}
        self._units: dict[str, Any] = {}
        self._segment_index = None
        self._provenance = []
        self._external_attrs = {}

        # Build initial schema from meta columns
        schema: dict[str, str] = {}
        for col in self._meta.columns:
            schema[col] = "segment" if col == "span" else "meta"
        self._schema = schema

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_segments(
        cls,
        segments: Sequence[Any],
        **meta_columns: Sequence[Any],
    ) -> SegmentTable:
        """Create a :class:`SegmentTable` from a sequence of segments.

        Parameters
        ----------
        segments:
            Sequence of :class:`gwpy.segments.Segment` objects.
        **meta_columns:
            Extra meta columns supplied as keyword arguments.
            Each value must have the same length as *segments*.

        Returns
        -------
        SegmentTable

        Raises
        ------
        ValueError
            If any column length does not match ``len(segments)``.

        Examples
        --------
        >>> from gwpy.segments import Segment
        >>> segs = [Segment(0, 4), Segment(4, 8)]
        >>> st = SegmentTable.from_segments(segs, label=["a", "b"])

        """
        n = len(segments)
        for name, data in meta_columns.items():
            if len(data) != n:
                raise ValueError(
                    f"Length of '{name}' ({len(data)}) does not match "
                    f"number of segments ({n})."
                )
        df = pd.DataFrame(
            {"span": list(segments), **{k: list(v) for k, v in meta_columns.items()}}
        )
        return cls(df)

    @classmethod
    def from_table(
        cls,
        table: Any,
        span: str = "span",
    ) -> SegmentTable:
        """Create a :class:`SegmentTable` from an existing table.

        Parameters
        ----------
        table:
            A :class:`pandas.DataFrame` (or compatible object).
        span:
            Column name to use as the ``span`` column.  If different from
            ``"span"``, the column is renamed.

        Returns
        -------
        SegmentTable

        Raises
        ------
        ValueError
            If *span* column is not found in *table*.

        """
        if hasattr(table, "to_pandas"):
            df = table.to_pandas()
        else:
            df = pd.DataFrame(table)

        if span not in df.columns:
            raise ValueError(f"Column {span!r} not found in provided table.")
        if span != "span":
            df = df.rename(columns={span: "span"})
        return cls(df)

    @classmethod
    def read_csv(
        cls,
        filepath: str,
        span_cols: tuple[str, str] = ("start", "end"),
        **kwargs: Any,
    ) -> SegmentTable:
        """Read a :class:`SegmentTable` from a CSV file.

        Parameters
        ----------
        filepath:
            Path to the CSV file.
        span_cols:
            The two column names to use as the ``span`` start and end values.
            Ignored if a column named ``"span"`` already exists (containing
            Segment objects or compatible).
        **kwargs:
            Forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        SegmentTable

        """
        import pandas as pd
        from gwpy.segments import Segment

        df = pd.read_csv(filepath, **kwargs)
        if "span" not in df.columns:
            s1, s2 = span_cols
            df["span"] = [Segment(float(s), float(e)) for s, e in zip(df[s1], df[s2])]
        else:
            df["span"] = [_coerce_span_value(value) for value in df["span"]]
        return cls(df)

    # Alias for backward compatibility or ease of use
    read = read_csv

    # ------------------------------------------------------------------
    # Column management
    # ------------------------------------------------------------------

    def add_column(
        self,
        name: str,
        data: Sequence[Any],
        kind: str = "meta",
        unit: Any = None,
    ) -> None:
        """Add a lightweight meta column.

        Parameters
        ----------
        name:
            Column name.  Must not already exist.
        data:
            Sequence of values; length must equal ``len(self)``.
        kind:
            Column kind — ``"meta"`` or ``"object"``.
        unit:
            Optional unit metadata for the column.

        Raises
        ------
        ValueError
            If *name* already exists, length mismatch, or *kind* is not
            ``"meta"`` or ``"object"``.

        """
        if name in self._schema:
            raise ValueError(f"Column {name!r} already exists.")
        if kind not in ("meta", "object"):
            raise ValueError(
                f"add_column only accepts kind 'meta' or 'object', got {kind!r}. "
                "Use add_series_column for payload kinds."
            )
        if len(data) != len(self):
            raise ValueError(
                f"Length of data ({len(data)}) does not match "
                f"table row count ({len(self)})."
            )
        if unit is None and hasattr(data, "unit"):
            unit = data.unit
        self._meta[name] = list(data)
        self._schema[name] = kind
        if unit is not None:
            self._units[name] = unit

    def add_series_column(
        self,
        name: str,
        data: Optional[Sequence[Any]] = None,
        loader: Optional[
            Union[Sequence[Callable[[], Any]], Callable[[int], Callable[[], Any]]]
        ] = None,
        kind: str = "timeseries",
        unit: Any = None,
    ) -> None:
        """Add a payload column (lazy-loadable).

        Parameters
        ----------
        name:
            Column name.  Must not already exist.
        data:
            Sequence of concrete payload objects, one per row. Exclusive
            with *loader* (but one of the two must be provided).
        loader:
            Either a sequence of zero-argument callables (one per row), or
            a single callable that accepts the row index and returns a
            zero-argument callable.
        kind:
            Payload kind.  Must be one of:
            ``"timeseries"``, ``"timeseriesdict"``,
            ``"frequencyseries"``, ``"frequencyseriesdict"``,
            ``"spectrogram"``, ``"matrix"``, ``"array"``, ``"scalar"``,
            ``"object"``.
        unit:
            Optional unit metadata for the column.

        Raises
        ------
        ValueError
            If both *data* and *loader* are ``None``, or if length mismatches.
        ValueError
            If *kind* is not a valid payload kind.

        """
        if name in self._schema:
            raise ValueError(f"Column {name!r} already exists.")
        if kind not in _PAYLOAD_KINDS:
            raise ValueError(
                f"Invalid payload kind {kind!r}. "
                f"Must be one of: {sorted(_PAYLOAD_KINDS)}."
            )
        if data is None and loader is None:
            raise ValueError("At least one of 'data' or 'loader' must be provided.")

        n = len(self)

        cells: list[SegmentCell] = []

        if data is not None:
            if len(data) != n:
                raise ValueError(
                    f"Length of data ({len(data)}) does not match "
                    f"table row count ({n})."
                )
            if loader is not None:
                # Both provided: data takes priority; loader as fallback ignored
                # (simpler semantics for v0.1)
                for val in data:
                    cells.append(
                        val.clone()
                        if isinstance(val, SegmentCell)
                        else SegmentCell(value=val)
                    )
            else:
                for val in data:
                    cells.append(
                        val.clone()
                        if isinstance(val, SegmentCell)
                        else SegmentCell(value=val)
                    )
        else:
            # loader only
            assert loader is not None
            # Detect if loader is a sequence or a factory callable
            if callable(loader) and not _is_sequence(loader):
                # Standard pattern: loader(segment) -> payload
                for i in range(n):
                    seg = self._meta.at[i, "span"]

                    # Capture current segment in a closure
                    def _wrap_loader(s=seg, ldr=loader):
                        return ldr(s)

                    cells.append(SegmentCell(loader=_wrap_loader))
            else:
                # Sequence of callables
                loader_seq = list(cast(Sequence[Any], loader))
                if len(loader_seq) != n:
                    raise ValueError(
                        f"Length of loader sequence ({len(loader_seq)}) does not "
                        f"match table row count ({n})."
                    )
                for ldr in loader_seq:
                    cells.append(SegmentCell(loader=ldr))

        self._payload[name] = cells
        self._schema[name] = kind
        if unit is not None:
            self._units[name] = unit

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._meta)

    def __iter__(self):
        """Yield a RowProxy for each row."""
        for i in range(len(self)):
            yield self.row(i)

    @property
    def columns(self) -> list[str]:
        """Ordered list of all column names (meta then payload)."""
        meta_cols = list(self._meta.columns)
        payload_cols = [c for c in self._schema if c not in set(meta_cols)]
        return meta_cols + payload_cols

    @property
    def schema(self) -> dict[str, str]:
        """Mapping of column name → kind."""
        return dict(self._schema)

    def row(self, i: int) -> RowProxy:
        """Return a dict-like proxy for row *i*.

        Parameters
        ----------
        i:
            0-based row index.

        Raises
        ------
        IndexError
            If *i* is out of range.

        """
        n = len(self)
        if i < -n or i >= n:
            raise IndexError(
                f"Row index {i} out of range for SegmentTable with {n} rows."
            )
        if i < 0:
            i = n + i
        return RowProxy(self, i)

    def column(self, name: str) -> SegmentColumn:
        """Return a typed :class:`~gwexpy.table.segment.expr.SegmentColumn` view."""
        from gwexpy.table.segment.expr import SegmentColumn

        return SegmentColumn(self, name)

    def __getitem__(self, key: str) -> SegmentColumn:
        if not isinstance(key, str):
            raise TypeError(
                f"SegmentTable column access requires a str key, got {type(key).__name__!r}. "
                "Use table.filter(...) for row selection."
            )
        return self.column(key)

    # ------------------------------------------------------------------
    # Row-wise processing
    # ------------------------------------------------------------------

    def _slice_rows(
        self,
        indices: Sequence[int],
        *,
        share_cells: bool = False,
    ) -> SegmentTable:
        """Return a new table containing rows at *indices*."""
        idx_list = list(indices)
        new_meta = cast(pd.DataFrame, self._meta.iloc[idx_list].reset_index(drop=True))
        new_table = SegmentTable(new_meta)
        for col in new_meta.columns:
            if col != "span":
                new_table._schema[col] = self._schema.get(col, "meta")
        for col, cells in self._payload.items():
            if share_cells:
                new_table._payload[col] = [cells[i] for i in idx_list]
            else:
                new_table._payload[col] = [cells[i].clone() for i in idx_list]
            new_table._schema[col] = self._schema[col]
        new_table._units = dict(self._units)
        new_table._provenance = [dict(p) for p in self._provenance]
        new_table._external_attrs = dict(self._external_attrs)
        return new_table

    def _slice_inplace(self, indices: Sequence[int]) -> None:
        idx_list = list(indices)
        self._meta = cast(
            pd.DataFrame, self._meta.iloc[idx_list].reset_index(drop=True)
        )
        for col, cells in list(self._payload.items()):
            self._payload[col] = [cells[i] for i in idx_list]
        self._segment_index = None

    def _set_meta_column(
        self, name: str, values: Sequence[Any], kind: str = "meta"
    ) -> None:
        self._meta[name] = list(values)
        self._schema[name] = kind

    def _emit_error_columns(
        self,
        col_name: str,
        statuses: list[str],
        row_errors: list[Optional[BaseException]],
    ) -> None:
        self._set_meta_column(f"{col_name}_status", statuses, kind="meta")
        self._set_meta_column(
            f"{col_name}_error_type",
            [type(e).__name__ if e is not None else "" for e in row_errors],
            kind="meta",
        )
        self._set_meta_column(
            f"{col_name}_error_message",
            [str(e) if e is not None else "" for e in row_errors],
            kind="meta",
        )

    def _finalize_row_outputs(
        self,
        *,
        col_names: list[str],
        explicit_kinds: dict[str, str],
        row_dicts: list[Optional[dict[str, Any]]],
        row_errors: list[Optional[BaseException]],
        inplace: bool,
        errors: str,
    ) -> SegmentTable:
        n = len(self)
        if errors == "skip":
            kept_indices = [i for i in range(n) if row_errors[i] is None]
            if inplace:
                self._slice_inplace(kept_indices)
                target = self
            else:
                target = self._slice_rows(kept_indices)
            for col_name in col_names:
                vals = [
                    cast(dict[str, Any], row_dicts[i])[col_name] for i in kept_indices
                ]
                kind = explicit_kinds.get(col_name) or _infer_kind(vals)
                if kind == "meta":
                    target._set_meta_column(col_name, vals, kind=kind)
                else:
                    target._payload[col_name] = [
                        v.clone()
                        if isinstance(v, SegmentCell)
                        else SegmentCell(value=v)
                        for v in vals
                    ]
                    target._schema[col_name] = kind
            return target

        target = self if inplace else self.copy()
        for col_name in col_names:
            success_vals = [
                cast(dict[str, Any], row_dicts[i])[col_name]
                for i in range(n)
                if row_errors[i] is None
            ]
            kind = explicit_kinds.get(col_name) or _infer_kind(success_vals)
            if kind == "meta":
                meta_vals: list[Any] = []
                statuses: list[str] = []
                for i in range(n):
                    err = row_errors[i]
                    if err is not None:
                        meta_vals.append(np.nan)
                        statuses.append(CellStatus.FAILED.value)
                    else:
                        val = cast(dict[str, Any], row_dicts[i])[col_name]
                        meta_vals.append(val)
                        statuses.append(
                            CellStatus.MISSING.value
                            if is_scalar_nan(val)
                            else CellStatus.VALID.value
                        )
                target._set_meta_column(col_name, meta_vals, kind=kind)
                if errors == "record":
                    target._emit_error_columns(col_name, statuses, row_errors)
            else:
                cells: list[SegmentCell] = []
                statuses = []
                for i in range(n):
                    err = row_errors[i]
                    if err is not None:
                        cell = (
                            SegmentCell.failed(err)
                            if errors == "record"
                            else SegmentCell.missing()
                        )
                    else:
                        val = cast(dict[str, Any], row_dicts[i])[col_name]
                        cell = (
                            val.clone()
                            if isinstance(val, SegmentCell)
                            else SegmentCell(value=val)
                        )
                    cells.append(cell)
                    statuses.append(cell.status.value)
                target._payload[col_name] = cells
                target._schema[col_name] = kind
                if errors == "record":
                    target._emit_error_columns(col_name, statuses, row_errors)
        return target

    def apply(
        self,
        func: Callable[[RowProxy], dict[str, Any]],
        in_cols: Optional[list[str]] = None,
        out_cols: Optional[list[str]] = None,
        parallel: bool = False,
        inplace: bool = False,
        errors: str = "raise",
    ) -> SegmentTable:
        """Apply *func* to each row and collect the results as new columns.

        Parameters
        ----------
        func:
            Callable that receives a :class:`RowProxy` and returns a
            ``dict[str, object]``.
        in_cols:
            Hint for which columns are read (ignored in v0.1).
        out_cols:
            Expected keys of ``func``'s return dict.  If given, the actual
            keys must match exactly.
        parallel:
            If ``True``, falls back to sequential execution in v0.1.
        inplace:
            If ``True``, add columns to *self*; otherwise return a new
            :class:`SegmentTable`.
        errors:
            Failure policy across rows: ``"raise"`` (default), ``"record"``
            (mark failed cells ``FAILED`` and emit status/error columns),
            ``"coerce"`` (mark failed cells ``MISSING``), or ``"skip"`` (drop
            failing rows).

        Returns
        -------
        SegmentTable
            Either *self* (``inplace=True``) or a new table.

        Raises
        ------
        TypeError
            If *func* does not return a ``dict``.
        ValueError
            If ``out_cols`` is given and the actual keys do not match, or
            *errors* is invalid.

        """
        validate_errors_policy(errors)
        # parallel is intentionally ignored (v0.1: sequential fallback)
        n = len(self)
        if n == 0:
            return self if inplace else self.copy()

        row_dicts: list[Optional[dict[str, Any]]] = []
        row_errors: list[Optional[BaseException]] = []

        for i in range(n):
            row = self.row(i)
            if errors == "raise":
                ret = func(row)
                if not isinstance(ret, dict):
                    raise TypeError(
                        f"apply() expects func to return a dict, "
                        f"got {type(ret).__name__!r}."
                    )
                if out_cols is not None:
                    expected = set(out_cols)
                    actual = set(ret.keys())
                    if expected != actual:
                        raise ValueError(
                            f"out_cols {sorted(expected)} does not match "
                            f"func return keys {sorted(actual)} at row {i}."
                        )
                row_dicts.append(ret)
                row_errors.append(None)
            else:
                try:
                    ret = func(row)
                    if not isinstance(ret, dict):
                        raise TypeError(
                            f"apply() expects func to return a dict, "
                            f"got {type(ret).__name__!r}."
                        )
                    if out_cols is not None:
                        expected = set(out_cols)
                        actual = set(ret.keys())
                        if expected != actual:
                            raise ValueError(
                                f"out_cols {sorted(expected)} does not match "
                                f"func return keys {sorted(actual)} at row {i}."
                            )
                except Exception as exc:
                    row_dicts.append(None)
                    row_errors.append(exc)
                else:
                    row_dicts.append(ret)
                    row_errors.append(None)

        first_ok = next((d for d in row_dicts if d is not None), None)
        if first_ok is not None:
            all_keys = list(out_cols) if out_cols is not None else list(first_ok.keys())
        elif out_cols is not None:
            all_keys = list(out_cols)
        elif errors == "skip":
            if inplace:
                self._slice_inplace([])
                return self
            return self._slice_rows([])
        else:
            first_err = next(e for e in row_errors if e is not None)
            raise ValueError(
                "All rows failed in apply() and 'out_cols' was not provided to "
                "determine output column names."
            ) from first_err

        return self._finalize_row_outputs(
            col_names=all_keys,
            explicit_kinds={},
            row_dicts=row_dicts,
            row_errors=row_errors,
            inplace=inplace,
            errors=errors,
        )

    def map(
        self,
        column: str,
        func: Callable[[Any], Any],
        out_col: Optional[str] = None,
        inplace: bool = False,
        errors: str = "raise",
    ) -> SegmentTable:
        """Apply *func* to each value in *column*.

        Parameters
        ----------
        column:
            Source column name.
        func:
            Callable that receives the cell value and returns a new value.
        out_col:
            Output column name.  Defaults to ``column + "_mapped"``.
        inplace:
            Whether to modify *self* in place.
        errors:
            Failure policy across rows (``"raise"``, ``"record"``,
            ``"coerce"``, ``"skip"``).

        Returns
        -------
        SegmentTable

        """
        validate_errors_policy(errors)
        if column not in self._schema:
            raise KeyError(f"Column {column!r} not found.")
        if out_col is None:
            out_col = column + "_mapped"

        def _func(row: RowProxy) -> dict[str, Any]:
            return {out_col: func(row[column])}

        return self.apply(_func, out_cols=[out_col], inplace=inplace, errors=errors)

    # Sugar APIs ----------------------------------------------------------

    def crop(
        self,
        column: str,
        out_col: Optional[str] = None,
        span_col: str = "span",
        inplace: bool = False,
        errors: str = "raise",
    ) -> SegmentTable:
        """Crop each row's payload to its :class:`~gwpy.segments.Segment`.

        Parameters
        ----------
        column:
            Payload column containing ``TimeSeries`` or ``TimeSeriesDict``.
        out_col:
            Output column name; defaults to ``column + "_cropped"``.
        span_col:
            Column name for the span (default ``"span"``).
        inplace:
            Whether to modify *self* in place.
        errors:
            Failure policy across rows (``"raise"``, ``"record"``,
            ``"coerce"``, ``"skip"``).

        Raises
        ------
        TypeError
            If the payload is not ``TimeSeries`` or ``TimeSeriesDict``.

        """
        validate_errors_policy(errors)
        try:
            from gwpy.timeseries import TimeSeries, TimeSeriesDict
        except ImportError as exc:
            raise ImportError("gwpy is required for crop()") from exc

        kind = self._schema.get(column)
        if kind not in ("timeseries", "timeseriesdict"):
            raise TypeError(
                f"crop() requires a 'timeseries' or 'timeseriesdict' column, "
                f"got kind={kind!r} for column {column!r}."
            )

        if out_col is None:
            out_col = column + "_cropped"

        out_kind = kind  # preserve timeseries/timeseriesdict kind
        n = len(self)
        if n == 0:
            target = self if inplace else self.copy()
            target._payload[out_col] = []
            target._schema[out_col] = out_kind
            return target

        row_dicts: list[Optional[dict[str, Any]]] = []
        row_errors: list[Optional[BaseException]] = []

        for i in range(n):
            cell = self._payload[column][i]
            if cell.status in (
                CellStatus.MISSING,
                CellStatus.MASKED,
                CellStatus.NOT_APPLICABLE,
            ):
                row_dicts.append(
                    {out_col: SegmentCell(value=_UNSET, status=cell.status)}
                )
                row_errors.append(None)
                continue

            if errors == "raise":
                payload = self.row(i)[column]
                span = self.row(i)[span_col]
                if not isinstance(payload, (TimeSeries, TimeSeriesDict)):
                    raise TypeError(
                        f"crop() expected TimeSeries or TimeSeriesDict, "
                        f"got {type(payload).__name__!r}."
                    )
                row_dicts.append({out_col: payload.crop(span[0], span[1])})
                row_errors.append(None)
            else:
                try:
                    payload = self.row(i)[column]
                    span = self.row(i)[span_col]
                    if not isinstance(payload, (TimeSeries, TimeSeriesDict)):
                        raise TypeError(
                            f"crop() expected TimeSeries or TimeSeriesDict, "
                            f"got {type(payload).__name__!r}."
                        )
                    cropped = payload.crop(span[0], span[1])
                except Exception as exc:
                    row_dicts.append(None)
                    row_errors.append(exc)
                else:
                    row_dicts.append({out_col: cropped})
                    row_errors.append(None)

        return self._finalize_row_outputs(
            col_names=[out_col],
            explicit_kinds={out_col: out_kind},
            row_dicts=row_dicts,
            row_errors=row_errors,
            inplace=inplace,
            errors=errors,
        )

    def asd(
        self,
        column: str,
        out_col: Optional[str] = None,
        *,
        errors: str = "raise",
        inplace: bool = False,
        **kwargs: Any,
    ) -> SegmentTable:
        """Apply ASD to each row's ``TimeSeries`` or ``TimeSeriesDict``.

        Parameters
        ----------
        column:
            Payload column.
        out_col:
            Output column name; defaults to ``column + "_asd"``.
        errors:
            Failure policy across rows (``"raise"``, ``"record"``,
            ``"coerce"``, ``"skip"``).
        inplace:
            Whether to modify *self* in place.
        **kwargs:
            Forwarded to :meth:`~gwpy.timeseries.TimeSeries.asd`.

        Returns
        -------
        SegmentTable

        """
        validate_errors_policy(errors)
        if column not in self._schema:
            raise KeyError(f"Column {column!r} not found.")
        if out_col is None:
            out_col = column + "_asd"

        out_kind = "frequencyseries"
        kind = self._schema.get(column, "timeseries")
        if kind == "timeseriesdict":
            out_kind = "frequencyseriesdict"

        n = len(self)
        if n == 0:
            target = self if inplace else self.copy()
            target._payload[out_col] = []
            target._schema[out_col] = out_kind
            return target

        row_dicts: list[Optional[dict[str, Any]]] = []
        row_errors: list[Optional[BaseException]] = []

        for i in range(n):
            if column in self._payload:
                cell = self._payload[column][i]
                if cell.status in (
                    CellStatus.MISSING,
                    CellStatus.MASKED,
                    CellStatus.NOT_APPLICABLE,
                ):
                    row_dicts.append(
                        {out_col: SegmentCell(value=_UNSET, status=cell.status)}
                    )
                    row_errors.append(None)
                    continue

            if errors == "raise":
                payload = self.row(i)[column]
                row_dicts.append({out_col: payload.asd(**kwargs)})
                row_errors.append(None)
            else:
                try:
                    payload = self.row(i)[column]
                    res = payload.asd(**kwargs)
                except Exception as exc:
                    row_dicts.append(None)
                    row_errors.append(exc)
                else:
                    row_dicts.append({out_col: res})
                    row_errors.append(None)

        return self._finalize_row_outputs(
            col_names=[out_col],
            explicit_kinds={out_col: out_kind},
            row_dicts=row_dicts,
            row_errors=row_errors,
            inplace=inplace,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Selection and conversion
    # ------------------------------------------------------------------

    def select(
        self,
        mask: Optional[Sequence[bool]] = None,
        **conditions: Any,
    ) -> SegmentTable:
        """Select rows by boolean mask or column conditions.

        .. note::
            For backward compatibility, ``select()`` shares the underlying
            :class:`~gwexpy.table.segment.cell.SegmentCell` objects with the
            source table. Prefer :meth:`filter` for isolated cell status and
            richer column expressions.

        Parameters
        ----------
        mask:
            Boolean sequence of length ``len(self)``.
        **conditions:
            Simple equality conditions, e.g. ``label="glitch"``.

        Returns
        -------
        SegmentTable
            A new :class:`SegmentTable` with only the selected rows.

        Raises
        ------
        ValueError
            If *mask* length is not equal to ``len(self)``.
        KeyError
            If a condition column does not exist.

        """
        bool_mask: pd.Series[bool] = pd.Series([True] * len(self), dtype=bool)

        if mask is not None:
            if len(mask) != len(self):
                raise ValueError(
                    f"mask length ({len(mask)}) does not match "
                    f"table row count ({len(self)})."
                )
            bool_mask = bool_mask & pd.Series(list(mask), dtype=bool)

        for col, val in conditions.items():
            if col not in self._schema:
                raise KeyError(f"Column {col!r} not found.")
            if col in self._payload:
                raise KeyError(
                    f"select() does not support payload column {col!r} as a condition."
                )
            bool_mask = bool_mask & (self._meta[col] == val)

        indices = [i for i, keep in enumerate(bool_mask.tolist()) if keep]
        return self._slice_rows(indices, share_cells=True)

    def fetch(
        self,
        columns: Optional[list[str]] = None,
        errors: str = "raise",
    ) -> None:
        """Eagerly load all payload cells in *columns* (default: all).

        Parameters
        ----------
        columns:
            List of payload column names to load.  ``None`` loads all.
        errors:
            Failure policy across cells: ``"raise"`` (default), ``"record"``,
            ``"coerce"``, or ``"skip"``.

        """
        validate_errors_policy(errors)
        cols = list(columns) if columns is not None else list(self._payload.keys())
        n = len(self)
        failed_rows: set[int] = set()

        for col in cols:
            if col not in self._payload:
                continue
            cells = self._payload[col]
            col_errors: list[Optional[BaseException]] = [None] * n
            statuses: list[str] = []
            for i, cell in enumerate(cells):
                if cell.status in (
                    CellStatus.VALID,
                    CellStatus.MISSING,
                    CellStatus.MASKED,
                    CellStatus.NOT_APPLICABLE,
                ):
                    statuses.append(cell.status.value)
                    continue

                if errors == "raise":
                    cell.get()
                    statuses.append(cell.status.value)
                else:
                    try:
                        cell.get()
                    except Exception as exc:
                        failed_rows.add(i)
                        if errors == "record":
                            if cell.status != CellStatus.FAILED:
                                cell.status = CellStatus.FAILED
                                cell.error = exc
                            col_errors[i] = exc
                            statuses.append(CellStatus.FAILED.value)
                        elif errors == "coerce":
                            cells[i] = SegmentCell.missing()
                            statuses.append(CellStatus.MISSING.value)
                        else:
                            # skip
                            statuses.append(CellStatus.FAILED.value)
                    else:
                        statuses.append(cell.status.value)

            if errors == "record":
                self._emit_error_columns(col, statuses, col_errors)

        if errors == "skip" and failed_rows:
            kept = [i for i in range(n) if i not in failed_rows]
            self._slice_inplace(kept)

    def materialize(
        self,
        columns: Optional[list[str]] = None,
        inplace: bool = True,
        errors: str = "raise",
    ) -> Optional[SegmentTable]:
        """Materialize lazy payload cells (equivalent to :meth:`fetch` in v0.1).

        Parameters
        ----------
        columns:
            Payload columns to materialize.  ``None`` means all.
        inplace:
            If ``True`` (default), mutate *self*.  If ``False``, return a
            copy with the payload loaded.
        errors:
            Failure policy across cells: ``"raise"`` (default), ``"record"``,
            ``"coerce"``, or ``"skip"``.

        Returns
        -------
        SegmentTable or None
            *None* if ``inplace=True``; new table if ``inplace=False``.

        """
        validate_errors_policy(errors)
        target = self if inplace else self.copy(deep=True)
        target.fetch(columns=columns, errors=errors)
        if not inplace:
            return target
        return None

    def to_pandas(
        self,
        meta_only: bool = True,
        missing_as: str = "summary",
    ) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` representation.

        Parameters
        ----------
        meta_only:
            If ``True`` (default), return only the meta columns.  If
            ``False``, payload columns are appended as object columns
            containing the resolved cell values.
        missing_as:
            How non-valid payload cells (``UNLOADED``, ``MISSING``, ``FAILED``,
            ``MASKED``, ``NOT_APPLICABLE``) are represented when
            ``meta_only=False``: ``"summary"`` (default, human-readable status
            tag) or ``"nan"`` (``numpy.nan``).

        Returns
        -------
        pandas.DataFrame

        """
        if missing_as not in ("summary", "nan"):
            raise ValueError(
                f"Invalid missing_as {missing_as!r}; expected 'summary' or 'nan'."
            )
        df = self._meta.copy()
        if not meta_only:
            for col, cells in self._payload.items():
                kind = self._schema.get(col, "object")
                col_values: list[Any] = []
                for c in cells:
                    if c.status == CellStatus.VALID:
                        col_values.append(c.get())
                    elif missing_as == "summary":
                        col_values.append(c._summary(kind))
                    else:
                        col_values.append(np.nan)
                df[col] = col_values
        return df

    def copy(self, deep: bool = False) -> SegmentTable:
        """Return a copy of this table.

        Parameters
        ----------
        deep:
            If ``False`` (default), meta is copied and payload cells are
            cloned shallowly (sharing values/loaders, isolating cell state).
            If ``True``, payload values are also deep-copied where possible.

        Returns
        -------
        SegmentTable

        """
        new_meta = self._meta.copy(deep=True)
        new_table = SegmentTable(new_meta)
        new_table._schema = dict(self._schema)

        for col in new_meta.columns:
            if col != "span" and col in self._schema:
                new_table._schema[col] = self._schema[col]

        for col, cells in self._payload.items():
            new_table._payload[col] = [c.clone(deep=deep) for c in cells]
            new_table._schema[col] = self._schema[col]

        new_table._units = dict(self._units)
        new_table._provenance = [dict(p) for p in self._provenance]
        new_table._external_attrs = dict(self._external_attrs)
        return new_table

    def clear_cache(self) -> None:
        """Discard all cached payload values, forcing re-load on next access."""
        for cells in self._payload.values():
            for cell in cells:
                cell.clear()

    # ------------------------------------------------------------------
    # Drawing API — thin wrappers around segment_plot
    # ------------------------------------------------------------------

    def plot(
        self,
        column: Optional[str] = None,
        *,
        row: Optional[int] = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """General-purpose plot entry.  Requires both *column* and *row*."""
        from gwexpy.table.segment_plot import plot_segment_table

        return plot_segment_table(self, column=column, row=row, mode=mode, **kwargs)

    def scatter(
        self,
        x: str,
        y: str,
        color: Optional[str] = None,
        *,
        selection: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Scatter plot of two scalar columns."""
        from gwexpy.table.segment_plot import scatter_segment_table

        return scatter_segment_table(
            self, x, y, color=color, selection=selection, **kwargs
        )

    def hist(
        self,
        column: str,
        *,
        bins: int = 10,
        **kwargs: Any,
    ) -> Any:
        """Histogram of a scalar column."""
        from gwexpy.table.segment_plot import hist_segment_table

        return hist_segment_table(self, column, bins=bins, **kwargs)

    def segments(
        self,
        *,
        y: Optional[str] = None,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Draw each row's span as a horizontal bar.

        This is one of the two *representative* APIs of :class:`SegmentTable`
        (alongside :meth:`overlay_spectra`).
        """
        from gwexpy.table.segment_plot import segments_segment_table

        return segments_segment_table(self, y=y, color=color, **kwargs)

    def overlay(
        self,
        column: str,
        rows: list[int],
        *,
        separate: bool = False,
        sharex: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Overlay payload from multiple rows."""
        from gwexpy.table.segment_plot import overlay_segment_table

        return overlay_segment_table(
            self, column, rows, separate=separate, sharex=sharex, **kwargs
        )

    def overlay_spectra(
        self,
        column: str,
        *,
        channel: Optional[str] = None,
        rows: Optional[list[int]] = None,
        color_by: Optional[str] = None,
        sort_by: Optional[str] = None,
        cmap: str = "viridis",
        alpha: float = 0.7,
        linewidth: float = 0.8,
        colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        xscale: str = "log",
        yscale: str = "log",
        xlim: Optional[Any] = None,
        ylim: Optional[Any] = None,
        ax: Optional[Any] = None,
    ) -> Any:
        """Overlay frequency spectra with colour-graded lines.

        Representative API of :class:`SegmentTable` for frequency-domain
        visualisation (alongside :meth:`segments` for time-domain).
        """
        from gwexpy.table.segment_plot import overlay_spectra_segment_table

        return overlay_spectra_segment_table(
            self,
            column,
            channel=channel,
            rows=rows,
            color_by=color_by,
            sort_by=sort_by,
            cmap=cmap,
            alpha=alpha,
            linewidth=linewidth,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_rows = len(self)
        n_payload = len(self._payload)
        all_cols = self.columns
        n_cols = len(all_cols)

        # Summarise span range
        spans = list(self._meta["span"])
        if spans:
            span_summary = f"[{spans[0][0]} ... {spans[-1][1]}]"
        else:
            span_summary = "[]"

        col_display = (
            str(all_cols) if n_cols <= 8 else str(all_cols[:8])[:-1] + ", ...]"
        )
        return (
            f"SegmentTable(n_rows={n_rows}, n_cols={n_cols}, payload={n_payload}, "
            f"columns={col_display})\n"
            f"span={span_summary}"
        )

    def __str__(self) -> str:
        MAX_ROWS = 10
        MAX_COLS = 8
        df = self._meta.copy()

        shown_cols = list(df.columns)[:MAX_COLS]
        for col, cells in self._payload.items():
            if len(shown_cols) >= MAX_COLS:
                break
            kind = self._schema.get(col, "object")
            df[col] = [c._summary(kind) for c in cells]
            shown_cols.append(col)

        return df[shown_cols].head(MAX_ROWS).to_string()

    def _repr_html_(self) -> str:
        MAX_ROWS = 10
        MAX_COLS = 8
        df = self._meta.copy()

        shown_cols = list(df.columns)[:MAX_COLS]
        for col, cells in self._payload.items():
            if len(shown_cols) >= MAX_COLS:
                break
            kind = self._schema.get(col, "object")
            df[col] = [c._summary(kind) for c in cells]
            shown_cols.append(col)

        return df[shown_cols].head(MAX_ROWS).to_html()

    def display(
        self,
        max_rows: int = 20,
        max_cols: int = 8,
        meta_only: bool = False,
    ) -> pd.DataFrame:
        """Return a summary :class:`pandas.DataFrame` for interactive display.

        Parameters
        ----------
        max_rows:
            Maximum rows to include.
        max_cols:
            Maximum columns to include.
        meta_only:
            If ``True``, omit payload columns.

        Returns
        -------
        pandas.DataFrame

        """
        df = self._meta.copy()

        if not meta_only:
            for col, cells in self._payload.items():
                if len(df.columns) >= max_cols:
                    break
                kind = self._schema.get(col, "object")
                df[col] = [c._summary(kind) for c in cells]

        shown_cols = list(df.columns)[:max_cols]
        return cast(pd.DataFrame, df[shown_cols].head(max_rows))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_sequence(obj: Any) -> bool:
    """Return True if *obj* is a non-string sequence (list/tuple/array etc.)."""
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))


def _infer_kind(values: list[Any]) -> str:
    """Infer column kind from a list of values."""
    import numbers

    sample: Any = None
    for v in values:
        if isinstance(v, SegmentCell):
            if v.status == CellStatus.VALID:
                v = v.get()
            else:
                continue
        if v is None or is_scalar_nan(v):
            continue
        sample = v
        break

    if sample is None:
        return "object"

    if isinstance(sample, (str, bool, numbers.Number)):
        return "meta"

    try:
        from gwpy.frequencyseries import FrequencySeries, FrequencySeriesDict
        from gwpy.spectrogram import Spectrogram
        from gwpy.timeseries import TimeSeries, TimeSeriesDict

        if isinstance(sample, TimeSeriesDict):
            return "timeseriesdict"
        if isinstance(sample, TimeSeries):
            return "timeseries"
        if isinstance(sample, FrequencySeriesDict):
            return "frequencyseriesdict"
        if isinstance(sample, FrequencySeries):
            return "frequencyseries"
        if isinstance(sample, Spectrogram):
            return "spectrogram"
    except ImportError:
        pass

    try:
        from gwexpy.types import SeriesMatrix

        if isinstance(sample, SeriesMatrix):
            return "matrix"
    except ImportError:
        pass

    try:
        from gwpy.types import Array

        if isinstance(sample, Array):
            return "array"
    except ImportError:
        pass

    from astropy import units as u

    if isinstance(sample, (np.ndarray, u.Quantity)):
        return "array"

    return "object"
