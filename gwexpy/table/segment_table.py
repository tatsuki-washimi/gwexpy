"""SegmentTable — segment-keyed analysis container.

Each row represents one analysis unit defined by a :class:`~gwpy.segments.Segment`
(a half-open time interval ``[start, end)``).  Columns may be lightweight *meta*
values or heavy *payload* objects such as :class:`~gwpy.timeseries.TimeSeries`.
Payload columns are stored as :class:`~gwexpy.table.segment_cell.SegmentCell`
instances that support lazy loading and optional caching.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import pandas as pd

from gwexpy.table.segment_cell import SegmentCell

__all__ = ["RowProxy", "SegmentTable"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Column kinds for meta (lightweight) columns.
_META_KINDS = frozenset({"segment", "meta", "object"})

#: Column kinds for payload (heavy) columns.
_PAYLOAD_KINDS = frozenset(
    {"timeseries", "timeseriesdict", "frequencyseries", "frequencyseriesdict", "object"}
)

#: All valid kinds.
_ALL_KINDS = _META_KINDS | _PAYLOAD_KINDS


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
        kind = table._schema[key]
        if kind in _META_KINDS:
            return table._meta.at[self._index, key]
        # Payload column
        return table._payload[key][self._index].get()

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


class SegmentTable:
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
            Segment = None  # type: ignore[assignment]

        if Segment is not None:
            for i, s in enumerate(meta["span"]):
                if not isinstance(s, Segment):
                    raise TypeError(
                        f"All elements of 'span' must be gwpy.segments.Segment, "
                        f"got {type(s).__name__!r} at index {i}."
                    )

        self._meta = meta.reset_index(drop=True).copy()
        self._payload: dict[str, list[SegmentCell]] = {}

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
        df = pd.DataFrame({"span": list(segments), **{k: list(v) for k, v in meta_columns.items()}})
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
            raise ValueError(
                f"Column {span!r} not found in provided table."
            )
        if span != "span":
            df = df.rename(columns={span: "span"})
        return cls(df)

    # ------------------------------------------------------------------
    # Column management
    # ------------------------------------------------------------------

    def add_column(
        self,
        name: str,
        data: Sequence[Any],
        kind: str = "meta",
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
        self._meta[name] = list(data)
        self._schema[name] = kind

    def add_series_column(
        self,
        name: str,
        data: Optional[Sequence[Any]] = None,
        loader: Optional[Union[Sequence[Callable[[], Any]], Callable[[int], Callable[[], Any]]]] = None,
        kind: str = "timeseries",
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
            ``"object"``.

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
            raise ValueError(
                "At least one of 'data' or 'loader' must be provided."
            )

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
                    cells.append(SegmentCell(value=val))
            else:
                for val in data:
                    cells.append(SegmentCell(value=val))
        else:
            # loader only
            assert loader is not None
            # Detect if loader is a sequence or a factory callable
            if callable(loader) and not _is_sequence(loader):
                # Factory: loader(i) -> callable
                for i in range(n):
                    cells.append(SegmentCell(loader=loader(i)))
            else:
                # Sequence of callables
                loader_seq = list(loader)  # type: ignore[arg-type]
                if len(loader_seq) != n:
                    raise ValueError(
                        f"Length of loader sequence ({len(loader_seq)}) does not "
                        f"match table row count ({n})."
                    )
                for ldr in loader_seq:
                    cells.append(SegmentCell(loader=ldr))

        self._payload[name] = cells
        self._schema[name] = kind

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._meta)

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

    # ------------------------------------------------------------------
    # Row-wise processing
    # ------------------------------------------------------------------

    def apply(
        self,
        func: Callable[[RowProxy], dict[str, Any]],
        in_cols: Optional[list[str]] = None,
        out_cols: Optional[list[str]] = None,
        parallel: bool = False,
        inplace: bool = False,
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

        Returns
        -------
        SegmentTable
            Either *self* (``inplace=True``) or a new table.

        Raises
        ------
        TypeError
            If *func* does not return a ``dict``.
        ValueError
            If ``out_cols`` is given and the actual keys do not match.
        """
        # parallel is intentionally ignored (v0.1: sequential fallback)
        results: list[dict[str, Any]] = []
        for i in range(len(self)):
            row = self.row(i)
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
            results.append(ret)

        # Collect per-column lists
        if not results:
            return self.copy() if not inplace else self

        all_keys = list(results[0].keys())
        new_data: dict[str, list[Any]] = {k: [r[k] for r in results] for k in all_keys}

        target = self if inplace else self.copy()
        for col_name, values in new_data.items():
            # Infer kind: if values are payload-like wrap in cells, else meta
            kind = _infer_kind(values)
            if kind in _META_KINDS - {"segment"}:
                target.add_column(col_name, values, kind=kind)
            else:
                cells = [SegmentCell(value=v) for v in values]
                target._payload[col_name] = cells
                target._schema[col_name] = kind
        return target

    def map(
        self,
        column: str,
        func: Callable[[Any], Any],
        out_col: Optional[str] = None,
        inplace: bool = False,
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

        Returns
        -------
        SegmentTable
        """
        if column not in self._schema:
            raise KeyError(f"Column {column!r} not found.")
        if out_col is None:
            out_col = column + "_mapped"

        def _func(row: RowProxy) -> dict[str, Any]:
            return {out_col: func(row[column])}

        return self.apply(_func, inplace=inplace)

    # Sugar APIs ----------------------------------------------------------

    def crop(
        self,
        column: str,
        out_col: Optional[str] = None,
        span_col: str = "span",
        inplace: bool = False,
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

        Raises
        ------
        TypeError
            If the payload is not ``TimeSeries`` or ``TimeSeriesDict``.
        """
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

        def _crop(row: RowProxy) -> dict[str, Any]:
            payload = row[column]
            span = row[span_col]
            if not isinstance(payload, (TimeSeries, TimeSeriesDict)):
                raise TypeError(
                    f"crop() expected TimeSeries or TimeSeriesDict, "
                    f"got {type(payload).__name__!r}."
                )
            return {out_col: payload.crop(span[0], span[1])}

        out_kind = kind  # preserve timeseries/timeseriesdict kind
        target = self if inplace else self.copy()

        results = []
        for i in range(len(self)):
            row = self.row(i)
            results.append(_crop(row)[out_col])

        cells = [SegmentCell(value=v) for v in results]
        target._payload[out_col] = cells
        target._schema[out_col] = out_kind
        return target

    def asd(
        self,
        column: str,
        out_col: Optional[str] = None,
        **kwargs: Any,
    ) -> SegmentTable:
        """Apply ASD to each row's ``TimeSeries`` or ``TimeSeriesDict``.

        Parameters
        ----------
        column:
            Payload column.
        out_col:
            Output column name; defaults to ``column + "_asd"``.
        **kwargs:
            Forwarded to :meth:`~gwpy.timeseries.TimeSeries.asd`.

        Returns
        -------
        SegmentTable
        """
        if out_col is None:
            out_col = column + "_asd"

        def _asd(row: RowProxy) -> dict[str, Any]:
            payload = row[column]
            return {out_col: payload.asd(**kwargs)}

        target = self.copy()
        results = []
        for i in range(len(self)):
            row = self.row(i)
            results.append(_asd(row)[out_col])

        out_kind = "frequencyseries"
        kind = self._schema.get(column, "timeseries")
        if kind == "timeseriesdict":
            out_kind = "frequencyseriesdict"

        cells = [SegmentCell(value=v) for v in results]
        target._payload[out_col] = cells
        target._schema[out_col] = out_kind
        return target

    # ------------------------------------------------------------------
    # Selection and conversion
    # ------------------------------------------------------------------

    def select(
        self,
        mask: Optional[Sequence[bool]] = None,
        **conditions: Any,
    ) -> SegmentTable:
        """Select rows by boolean mask or column conditions.

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
        bool_mask = pd.Series([True] * len(self))

        if mask is not None:
            if len(mask) != len(self):
                raise ValueError(
                    f"mask length ({len(mask)}) does not match "
                    f"table row count ({len(self)})."
                )
            bool_mask = bool_mask & pd.Series(list(mask))

        for col, val in conditions.items():
            if col not in self._schema:
                raise KeyError(f"Column {col!r} not found.")
            if col in self._payload:
                raise KeyError(
                    f"select() does not support payload column {col!r} as a condition."
                )
            bool_mask = bool_mask & (self._meta[col] == val)

        indices = bool_mask[bool_mask].index.tolist()
        new_meta = self._meta.loc[indices].reset_index(drop=True)
        new_table = SegmentTable(new_meta)

        # Carry over non-span schema kinds
        for col in new_meta.columns:
            if col != "span":
                new_table._schema[col] = self._schema.get(col, "meta")

        # Slice payload columns
        for col, cells in self._payload.items():
            new_table._payload[col] = [cells[i] for i in indices]
            new_table._schema[col] = self._schema[col]

        return new_table

    def fetch(self, columns: Optional[list[str]] = None) -> None:
        """Eagerly load all payload cells in *columns* (default: all).

        Parameters
        ----------
        columns:
            List of payload column names to load.  ``None`` loads all.
        """
        cols = columns if columns is not None else list(self._payload.keys())
        for col in cols:
            if col not in self._payload:
                continue
            for cell in self._payload[col]:
                cell.get()

    def materialize(
        self,
        columns: Optional[list[str]] = None,
        inplace: bool = True,
    ) -> Optional[SegmentTable]:
        """Materialize lazy payload cells (equivalent to :meth:`fetch` in v0.1).

        Parameters
        ----------
        columns:
            Payload columns to materialize.  ``None`` means all.
        inplace:
            If ``True`` (default), mutate *self*.  If ``False``, return a
            copy with the payload loaded.

        Returns
        -------
        SegmentTable or None
            *None* if ``inplace=True``; new table if ``inplace=False``.
        """
        target = self if inplace else self.copy(deep=True)
        target.fetch(columns=columns)
        if not inplace:
            return target
        return None

    def to_pandas(self, meta_only: bool = True) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` representation.

        Parameters
        ----------
        meta_only:
            If ``True`` (default), return only the meta columns.  If
            ``False``, payload columns are appended as object columns
            containing the resolved cell values.

        Returns
        -------
        pandas.DataFrame
        """
        df = self._meta.copy()
        if not meta_only:
            for col, cells in self._payload.items():
                df[col] = [c.get() if c.is_loaded() else c._summary(self._schema.get(col, "object")) for c in cells]
        return df

    def copy(self, deep: bool = False) -> SegmentTable:
        """Return a copy of this table.

        Parameters
        ----------
        deep:
            If ``False`` (default), meta is copied and payload cells are
            referenced (shallow).  If ``True``, payload values are also
            deep-copied where possible.

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
            if deep:
                new_cells = [
                    SegmentCell(
                        value=copy.deepcopy(c.value) if c.value is not None else None,
                        loader=c.loader,
                        cacheable=c.cacheable,
                    )
                    for c in cells
                ]
                for nc, oc in zip(new_cells, cells):
                    nc._loaded = oc._loaded
            else:
                new_cells = [
                    SegmentCell(value=c.value, loader=c.loader, cacheable=c.cacheable)
                    for c in cells
                ]
                for nc, oc in zip(new_cells, cells):
                    nc._loaded = oc._loaded
            new_table._payload[col] = new_cells
            new_table._schema[col] = self._schema[col]

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

        return scatter_segment_table(self, x, y, color=color, selection=selection, **kwargs)

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

        return overlay_segment_table(self, column, rows, separate=separate, sharex=sharex, **kwargs)

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

        col_display = str(all_cols) if n_cols <= 8 else str(all_cols[:8])[:-1] + ", ...]"
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
        return df[shown_cols].head(max_rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_sequence(obj: Any) -> bool:
    """Return True if *obj* is a non-string sequence (list/tuple/array etc.)."""
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))


def _infer_kind(values: list[Any]) -> str:
    """Infer column kind from a list of values."""
    try:
        from gwpy.frequencyseries import FrequencySeries
        from gwpy.timeseries import TimeSeries, TimeSeriesDict
    except ImportError:
        return "object"

    sample = next((v for v in values if v is not None), None)
    if sample is None:
        return "object"
    if isinstance(sample, TimeSeriesDict):
        return "timeseriesdict"
    if isinstance(sample, TimeSeries):
        return "timeseries"
    if isinstance(sample, FrequencySeries):
        return "frequencyseries"
    import numbers
    if isinstance(sample, (str, bool, numbers.Number)):
        return "meta"
    return "object"
