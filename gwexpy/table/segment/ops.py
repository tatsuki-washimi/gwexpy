"""Row, column, and segment-aware lookup operations for SegmentTable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from gwexpy.table.filter import parse_column_filters
from gwexpy.table.segment.cell import CellStatus, SegmentCell
from gwexpy.table.segment.expr import ColumnRef, Expr, _safe_elementwise_compare
from gwexpy.table.segment.status import is_scalar_nan

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable


class _SegmentIndex:
    """Start-time-sorted binary-search index over ``span`` segments."""

    __slots__ = ("order", "starts", "ends", "max_duration")

    def __init__(self, spans: Sequence[Any]) -> None:
        n = len(spans)
        if n == 0:
            self.order = np.empty(0, dtype=np.intp)
            self.starts = np.empty(0, dtype=float)
            self.ends = np.empty(0, dtype=float)
            self.max_duration = 0.0
            return
        raw_starts = np.array([float(s[0]) for s in spans], dtype=float)
        raw_ends = np.array([float(s[1]) for s in spans], dtype=float)
        order = np.argsort(raw_starts, kind="mergesort")
        self.order = order
        self.starts = raw_starts[order]
        self.ends = raw_ends[order]
        self.max_duration = float(np.max(raw_ends - raw_starts))


def _coerce_segment_bounds(segment: Any) -> tuple[float, float]:
    """Extract numeric ``(start, end)`` bounds from a Segment or 2-tuple."""
    try:
        from gwpy.segments import Segment

        if isinstance(segment, Segment):
            return float(segment[0]), float(segment[1])
    except ImportError:
        pass
    if isinstance(segment, (tuple, list)) and len(segment) == 2:
        return float(segment[0]), float(segment[1])
    raise TypeError(
        f"Expected a gwpy.segments.Segment or (start, end) pair, "
        f"got {type(segment).__name__!r}."
    )


class _OpsMixin:
    """Row, column, and segment-lookup operations (all returning new tables)."""

    _meta: Any
    _payload: dict[str, list[SegmentCell]]
    _schema: dict[str, str]
    _units: dict[str, Any]
    _segment_index: Optional[_SegmentIndex]
    _provenance: list[dict[str, Any]]
    _external_attrs: dict[str, Any]

    def _get_segment_index(self) -> _SegmentIndex:
        if self._segment_index is None:
            self._segment_index = _SegmentIndex(list(self._meta["span"]))
        return self._segment_index

    # ------------------------------------------------------------------
    # Row operations
    # ------------------------------------------------------------------

    def filter(
        self,
        predicate: Union[Expr, str, Sequence[bool]],
        name: Optional[str] = None,
    ) -> SegmentTable:
        """Return a new :class:`SegmentTable` keeping rows where *predicate* is True.

        Parameters
        ----------
        predicate:
            Either a :func:`~gwexpy.table.col` expression, a filter string
            parsed via :func:`~gwexpy.table.filter.parse_column_filters`, or
            a boolean sequence of length ``len(self)``.
        name:
            Optional filter step label recorded in table provenance for
            cut-flow reporting.

        Returns
        -------
        SegmentTable

        """
        n = len(self._meta)
        mask: np.ndarray[Any, np.dtype[np.bool_]]
        if isinstance(predicate, Expr):
            for req in predicate.required_columns():
                if req not in self._schema:
                    raise KeyError(f"Column {req!r} not found in SegmentTable.")
            mask = np.asarray(predicate.evaluate(self), dtype=np.bool_)  # type: ignore[arg-type]
            if mask.shape != (n,):
                raise ValueError(
                    f"Filter expression evaluated to shape {mask.shape}, "
                    f"expected ({n},)."
                )
        elif isinstance(predicate, str):
            specs = parse_column_filters(predicate)
            mask = np.ones(n, dtype=np.bool_)
            for spec in specs:
                if spec.column not in self._schema:
                    raise KeyError(f"Column {spec.column!r} not found in SegmentTable.")
                if spec.column in self._payload:
                    raise KeyError(
                        f"String filter does not support payload column {spec.column!r}."
                    )
                col_arr = self._meta[spec.column].to_numpy()
                rhs = np.empty(n, dtype=object)
                rhs[:] = [spec.operand] * n
                if col_arr.dtype == object:
                    step = _safe_elementwise_compare(spec.operator, col_arr, rhs)
                else:
                    with np.errstate(invalid="ignore"):
                        step = np.asarray(
                            spec.operator(col_arr, spec.operand), dtype=np.bool_
                        )
                mask &= step
        else:
            mask_list = list(predicate)
            if len(mask_list) != n:
                raise ValueError(
                    f"mask length ({len(mask_list)}) does not match "
                    f"table row count ({n})."
                )
            mask = np.asarray(mask_list, dtype=np.bool_)

        indices = [i for i, keep in enumerate(mask.tolist()) if keep]
        result = self._slice_rows(indices)  # type: ignore[attr-defined]
        if name is not None:
            result._provenance.append(
                {
                    "op": "filter",
                    "name": name,
                    "n_in": n,
                    "n_out": len(result),
                }
            )
        return result

    def head(self, n: int = 5) -> SegmentTable:
        """Return the first *n* rows as a new :class:`SegmentTable`."""
        total = len(self._meta)
        end = max(0, total + n) if n < 0 else min(n, total)
        return self._slice_rows(list(range(end)))  # type: ignore[attr-defined]

    def tail(self, n: int = 5) -> SegmentTable:
        """Return the last *n* rows as a new :class:`SegmentTable`."""
        total = len(self._meta)
        if n < 0:
            start = min(total, -n)
        elif n == 0:
            start = total
        else:
            start = max(0, total - n)
        return self._slice_rows(list(range(start, total)))  # type: ignore[attr-defined]

    def take(self, indices: Sequence[int]) -> SegmentTable:
        """Return rows at integer positions *indices* as a new :class:`SegmentTable`."""
        total = len(self._meta)
        normalized: list[int] = []
        for idx in indices:
            i = int(idx)
            if i < -total or i >= total:
                raise IndexError(
                    f"Index {idx} out of range for SegmentTable with {total} rows."
                )
            if i < 0:
                i += total
            normalized.append(i)
        return self._slice_rows(normalized)  # type: ignore[attr-defined]

    def sample(
        self,
        n: Optional[int] = None,
        *,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> SegmentTable:
        """Return a random sample of rows as a new :class:`SegmentTable`."""
        total = len(self._meta)
        if n is not None and frac is not None:
            raise ValueError("Cannot specify both 'n' and 'frac'.")
        if frac is not None:
            if frac < 0:
                raise ValueError(f"frac must be non-negative, got {frac}.")
            count = int(round(frac * total))
        else:
            count = 1 if n is None else int(n)
        if count < 0:
            raise ValueError(f"Sample size n must be non-negative, got {count}.")
        if not replace and count > total:
            raise ValueError(
                f"Cannot sample {count} rows without replacement from {total} rows."
            )
        if count == 0:
            return self._slice_rows([])  # type: ignore[attr-defined]
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        chosen = rng.choice(total, size=count, replace=replace).tolist()
        return self._slice_rows([int(i) for i in chosen])  # type: ignore[attr-defined]

    def sort_values(
        self,
        by: Union[str, Expr, Sequence[Union[str, Expr]]],
        ascending: Union[bool, Sequence[bool]] = True,
        *,
        na_position: str = "last",
    ) -> SegmentTable:
        """Return a new :class:`SegmentTable` sorted by one or more columns/expressions."""
        if na_position not in ("first", "last"):
            raise ValueError(
                f"Invalid na_position {na_position!r}; expected 'first' or 'last'."
            )
        by_list: list[Union[str, Expr]] = (
            [by] if isinstance(by, (str, Expr)) else list(by)
        )
        if not by_list:
            raise ValueError("sort_values() requires at least one sort key.")
        if isinstance(ascending, bool):
            asc_list = [ascending] * len(by_list)
        else:
            asc_list = list(ascending)
            if len(asc_list) != len(by_list):
                raise ValueError(
                    f"Length of ascending ({len(asc_list)}) must match "
                    f"length of by ({len(by_list)})."
                )

        n = len(self._meta)
        key_arrays: list[list[Any]] = []
        for item in by_list:
            if isinstance(item, Expr):
                for req in item.required_columns():
                    if req not in self._schema:
                        raise KeyError(f"Column {req!r} not found in SegmentTable.")
                vals = item.evaluate(self).tolist()  # type: ignore[arg-type]
                key_arrays.append(vals)
            elif isinstance(item, str):
                if item not in self._schema:
                    raise KeyError(f"Column {item!r} not found in SegmentTable.")
                if item in self._payload:
                    raise TypeError(
                        f"Cannot sort directly by payload column {item!r}; "
                        "pass a col() expression such as col(name).max(load=True)."
                    )
                if item == "span":
                    key_arrays.append(
                        [(float(s[0]), float(s[1])) for s in self._meta["span"]]
                    )
                else:
                    key_arrays.append(self._meta[item].tolist())
            else:
                raise TypeError(f"Invalid sort key {item!r}; expected str or Expr.")

        order = list(range(n))
        for vals, asc in zip(reversed(key_arrays), reversed(asc_list)):
            non_na = [idx for idx in order if not is_scalar_nan(vals[idx])]
            na_idx = [idx for idx in order if is_scalar_nan(vals[idx])]
            non_na.sort(key=lambda idx: vals[idx], reverse=not asc)
            order = na_idx + non_na if na_position == "first" else non_na + na_idx

        return self._slice_rows(order)  # type: ignore[attr-defined]

    def drop_duplicates(
        self,
        on: Optional[Union[str, Sequence[str]]] = None,
        keep: Union[str, bool] = "first",
    ) -> SegmentTable:
        """Return a new :class:`SegmentTable` with duplicate rows removed."""
        if keep not in ("first", "last", False):
            raise ValueError(
                f"Invalid keep {keep!r}; expected 'first', 'last', or False."
            )
        if on is None:
            on_cols = list(self._meta.columns)
        elif isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)

        for col_name in on_cols:
            if col_name not in self._schema:
                raise KeyError(f"Column {col_name!r} not found in SegmentTable.")
            if col_name in self._payload:
                raise TypeError(f"Cannot deduplicate on payload column {col_name!r}.")

        dup_mask = self._meta.duplicated(subset=on_cols, keep=keep)
        kept = [i for i, is_dup in enumerate(dup_mask.tolist()) if not is_dup]
        return self._slice_rows(kept)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------

    def select_columns(self, *names: str) -> SegmentTable:
        """Return a new :class:`SegmentTable` retaining only the specified columns.

        The required ``"span"`` column is always preserved.
        """
        for name in names:
            if name not in self._schema:
                raise KeyError(f"Column {name!r} not found in SegmentTable.")

        ordered: list[str] = ["span"]
        seen = {"span"}
        for name in names:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        meta_cols = [c for c in ordered if c in self._meta.columns]
        new_meta = self._meta[meta_cols].copy(deep=True)
        new_table = type(self)(new_meta)  # type: ignore[call-arg]
        for col_name in meta_cols:
            if col_name != "span":
                new_table._schema[col_name] = self._schema[col_name]
                if col_name in self._units:
                    new_table._units[col_name] = self._units[col_name]

        for col_name in ordered:
            if col_name in self._payload:
                new_table._payload[col_name] = [
                    c.clone() for c in self._payload[col_name]
                ]
                new_table._schema[col_name] = self._schema[col_name]
                if col_name in self._units:
                    new_table._units[col_name] = self._units[col_name]

        new_table._provenance = [dict(p) for p in self._provenance]
        new_table._external_attrs = dict(self._external_attrs)
        return new_table  # type: ignore[return-value]

    def drop_columns(self, *names: str) -> SegmentTable:
        """Return a new :class:`SegmentTable` with *names* removed."""
        for name in names:
            if name == "span":
                raise ValueError("Cannot drop required 'span' column.")
            if name not in self._schema:
                raise KeyError(f"Column {name!r} not found in SegmentTable.")
        drop_set = set(names)
        keep = [c for c in self.columns if c not in drop_set]  # type: ignore[attr-defined]
        return self.select_columns(*keep)

    def rename_columns(
        self,
        mapping: Optional[Mapping[str, str]] = None,
        **kwargs: str,
    ) -> SegmentTable:
        """Return a new :class:`SegmentTable` with columns renamed."""
        renames: dict[str, str] = {}
        if mapping is not None:
            renames.update(mapping)
        renames.update(kwargs)

        for old_name in renames:
            if old_name not in self._schema:
                raise KeyError(f"Column {old_name!r} not found in SegmentTable.")
        if "span" in renames and renames["span"] != "span":
            raise ValueError("Cannot rename required 'span' column.")

        new_names = [renames.get(c, c) for c in self.columns]  # type: ignore[attr-defined]
        if len(set(new_names)) != len(new_names):
            raise ValueError(f"Duplicate column names after renaming: {new_names}.")

        meta_rename = {
            old: new for old, new in renames.items() if old in self._meta.columns
        }
        new_meta = self._meta.rename(columns=meta_rename).copy(deep=True)
        new_table = type(self)(new_meta)  # type: ignore[call-arg]

        for old_col in self._meta.columns:
            if old_col != "span":
                new_col = renames.get(old_col, old_col)
                new_table._schema[new_col] = self._schema[old_col]
                if old_col in self._units:
                    new_table._units[new_col] = self._units[old_col]

        for old_col, cells in self._payload.items():
            new_col = renames.get(old_col, old_col)
            new_table._payload[new_col] = [c.clone() for c in cells]
            new_table._schema[new_col] = self._schema[old_col]
            if old_col in self._units:
                new_table._units[new_col] = self._units[old_col]

        new_table._provenance = [dict(p) for p in self._provenance]
        new_table._external_attrs = dict(self._external_attrs)
        return new_table  # type: ignore[return-value]

    def assign(self, **exprs: Any) -> SegmentTable:
        """Return a new :class:`SegmentTable` with columns added or overwritten."""
        from gwexpy.table.segment.core import _infer_kind, _is_sequence

        target = self.copy(deep=False)  # type: ignore[attr-defined]
        n = len(target)

        for col_name, spec in exprs.items():
            if col_name == "span":
                raise ValueError(
                    "Cannot overwrite required 'span' column via assign()."
                )

            source_unit = (
                target._units.get(spec.name) if isinstance(spec, ColumnRef) else None
            )
            if isinstance(spec, ColumnRef) and spec.name in target._payload:
                target._payload[col_name] = [
                    c.clone() for c in target._payload[spec.name]
                ]
                target._schema[col_name] = target._schema[spec.name]
                if source_unit is None:
                    target._units.pop(col_name, None)
                else:
                    target._units[col_name] = source_unit
                if col_name in target._meta.columns:
                    target._meta = target._meta.drop(columns=[col_name])
                continue

            if isinstance(spec, Expr):
                for req in spec.required_columns():
                    if req not in target._schema:
                        raise KeyError(f"Column {req!r} not found in SegmentTable.")
                vals = spec.evaluate(target).tolist()
            elif callable(spec):
                raw = spec(target)
                vals = list(raw) if _is_sequence(raw) else [raw] * n
            elif _is_sequence(spec):
                vals = list(spec)
            else:
                vals = [spec] * n

            if len(vals) != n:
                raise ValueError(
                    f"Assigned column {col_name!r} has length {len(vals)}, "
                    f"expected {n}."
                )

            if source_unit is None:
                target._units.pop(col_name, None)
            else:
                target._units[col_name] = source_unit
            kind = _infer_kind(vals)
            if kind == "meta":
                if col_name in target._payload:
                    del target._payload[col_name]
                target._set_meta_column(col_name, vals, kind="meta")
            else:
                if col_name in target._meta.columns:
                    target._meta = target._meta.drop(columns=[col_name])
                target._payload[col_name] = [
                    v.clone() if isinstance(v, SegmentCell) else SegmentCell(value=v)
                    for v in vals
                ]
                target._schema[col_name] = kind

        return target

    def explode_channels(
        self,
        column: str,
        *,
        channel_col: str = "channel",
        load: bool = False,
    ) -> SegmentTable:
        """Expand a series-dictionary payload into one row per channel.

        Unloaded dictionaries require ``load=True``; their source cells remain
        unchanged. Missing, failed, masked, and not-applicable cells produce one row
        with a missing channel and the original cell status. Empty dictionaries
        produce one row with a missing channel and a missing payload.
        """
        kinds = {
            "timeseriesdict": "timeseries",
            "frequencyseriesdict": "frequencyseries",
        }
        kind = self._schema.get(column)
        if kind not in kinds or column not in self._payload:
            raise TypeError(
                "explode_channels requires a series-dictionary payload column"
            )
        if channel_col in self._schema:
            raise ValueError(f"Column {channel_col!r} already exists")
        indices: list[int] = []
        channels: list[Any] = []
        cells: list[SegmentCell] = []
        for index, original in enumerate(self._payload[column]):
            cell = original.clone()
            if cell.status in (
                CellStatus.MISSING,
                CellStatus.FAILED,
                CellStatus.MASKED,
                CellStatus.NOT_APPLICABLE,
            ):
                indices.append(index)
                channels.append(None)
                cells.append(cell)
                continue
            if cell.status == CellStatus.UNLOADED and not load:
                raise ValueError("Unloaded channel dictionaries require load=True")
            values = cell.get()
            if not isinstance(values, Mapping):
                raise TypeError("Channel payload must be a mapping")
            if not values:
                indices.append(index)
                channels.append(None)
                cells.append(SegmentCell.missing())
            for name, value in values.items():
                indices.append(index)
                channels.append(name)
                cells.append(SegmentCell.valid(value))
        result = self._slice_rows(indices)  # type: ignore[attr-defined]
        result._payload[column] = cells
        result._schema[column] = kinds[kind]
        result._set_meta_column(channel_col, channels)  # type: ignore[attr-defined]
        return result

    def expand_bands(
        self,
        column: str,
        bands: Mapping[str, tuple[Any, Any]],
        *,
        band_col: str = "band",
        load: bool = False,
    ) -> SegmentTable:
        """Expand frequency-series rows into named GWpy-cropped bands.

        Bounds and their endpoint semantics follow ``FrequencySeries.crop``.
        Unloaded payloads require ``load=True`` and source cells are unchanged.
        """
        from gwpy.frequencyseries import FrequencySeries

        if self._schema.get(column) != "frequencyseries" or column not in self._payload:
            raise TypeError("expand_bands requires a frequencyseries payload column")
        if band_col in self._schema:
            raise ValueError(f"Column {band_col!r} already exists")
        indices: list[int] = []
        labels: list[str] = []
        cells: list[SegmentCell] = []
        for index, original in enumerate(self._payload[column]):
            cell = original.clone()
            if cell.status == CellStatus.UNLOADED and not load and bands:
                raise ValueError("Unloaded frequency series require load=True")
            if cell.status in (
                CellStatus.MISSING,
                CellStatus.FAILED,
                CellStatus.MASKED,
                CellStatus.NOT_APPLICABLE,
            ):
                for label in bands:
                    indices.append(index)
                    labels.append(label)
                    cells.append(cell.clone())
                continue
            if not bands:
                continue
            spectrum = cell.get()
            if not isinstance(spectrum, FrequencySeries):
                raise TypeError("Band payload must be a FrequencySeries")
            for label, (low, high) in bands.items():
                indices.append(index)
                labels.append(label)
                cells.append(SegmentCell.valid(spectrum.crop(low, high)))
        result = self._slice_rows(indices)  # type: ignore[attr-defined]
        result._payload[column] = cells
        result._set_meta_column(band_col, labels)  # type: ignore[attr-defined]
        return result

    # ------------------------------------------------------------------
    # Segment-aware lookups (binary search on start-time-sorted index)
    # ------------------------------------------------------------------

    def overlapping(self, segment: Any) -> SegmentTable:
        """Select rows whose ``span`` overlaps half-open *segment* ``[start, end)``."""
        s, e = _coerce_segment_bounds(segment)
        if e <= s:
            return self._slice_rows([])  # type: ignore[attr-defined]
        idx = self._get_segment_index()
        if idx.order.size == 0:
            return self._slice_rows([])  # type: ignore[attr-defined]
        right = int(np.searchsorted(idx.starts, e, side="left"))
        left = max(
            0, int(np.searchsorted(idx.starts, s - idx.max_duration, side="left"))
        )
        hits = idx.order[left:right][idx.ends[left:right] > s]
        return self._slice_rows(sorted(int(i) for i in hits))  # type: ignore[attr-defined]

    def within(self, segment: Any) -> SegmentTable:
        """Select rows whose ``span`` is contained within ``[start, end)``."""
        s, e = _coerce_segment_bounds(segment)
        if e < s:
            return self._slice_rows([])  # type: ignore[attr-defined]
        idx = self._get_segment_index()
        if idx.order.size == 0:
            return self._slice_rows([])  # type: ignore[attr-defined]
        left = int(np.searchsorted(idx.starts, s, side="left"))
        right = int(np.searchsorted(idx.starts, e, side="right"))
        hits = idx.order[left:right][idx.ends[left:right] <= e]
        return self._slice_rows(sorted(int(i) for i in hits))  # type: ignore[attr-defined]

    def containing(self, t: float) -> SegmentTable:
        """Select rows whose half-open ``span`` ``[start, end)`` contains *t*."""
        t_f = float(t)
        idx = self._get_segment_index()
        if idx.order.size == 0:
            return self._slice_rows([])  # type: ignore[attr-defined]
        right = int(np.searchsorted(idx.starts, t_f, side="right"))
        left = max(
            0, int(np.searchsorted(idx.starts, t_f - idx.max_duration, side="left"))
        )
        hits = idx.order[left:right][idx.ends[left:right] > t_f]
        return self._slice_rows(sorted(int(i) for i in hits))  # type: ignore[attr-defined]

    def before(self, t: float, *, inclusive: bool = True) -> SegmentTable:
        """Select rows ending before *t* (``end <= t`` if *inclusive* else ``end < t``)."""
        t_f = float(t)
        idx = self._get_segment_index()
        if idx.order.size == 0:
            return self._slice_rows([])  # type: ignore[attr-defined]
        right = int(np.searchsorted(idx.starts, t_f, side="right"))
        mask = idx.ends[:right] <= t_f if inclusive else idx.ends[:right] < t_f
        hits = idx.order[:right][mask]
        return self._slice_rows(sorted(int(i) for i in hits))  # type: ignore[attr-defined]

    def after(self, t: float, *, inclusive: bool = True) -> SegmentTable:
        """Select rows starting after *t* (``start >= t`` if *inclusive* else ``start > t``)."""
        t_f = float(t)
        idx = self._get_segment_index()
        if idx.order.size == 0:
            return self._slice_rows([])  # type: ignore[attr-defined]
        left = int(
            np.searchsorted(idx.starts, t_f, side="left" if inclusive else "right")
        )
        hits = idx.order[left:]
        return self._slice_rows(sorted(int(i) for i in hits))  # type: ignore[attr-defined]

    def nearest(self, t: float, n: int = 1) -> SegmentTable:
        """Return the *n* rows whose ``span`` is closest to timestamp *t*.

        Segments containing *t* have distance ``0.0``. Ties are broken by
        ascending row index.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}.")
        total = len(self._meta)
        if n == 0 or total == 0:
            return self._slice_rows([])  # type: ignore[attr-defined]

        t_f = float(t)
        idx = self._get_segment_index()
        pos = int(np.searchsorted(idx.starts, t_f, side="left"))
        left_cont = int(
            np.searchsorted(idx.starts, t_f - idx.max_duration, side="left")
        )

        left = max(0, left_cont - n)
        right = min(total, pos + n)

        def _window_dists(lo: int, hi: int) -> np.ndarray:
            return np.maximum(
                0.0, np.maximum(idx.starts[lo:hi] - t_f, t_f - idx.ends[lo:hi])
            )

        # Expand window until no outside entry can match or beat the n-th distance
        while True:
            dists = _window_dists(left, right)
            if right - left >= n:
                nth_dist = float(np.partition(dists, n - 1)[n - 1])
            else:
                nth_dist = float("inf")
            expanded = False
            if left > 0 and (t_f - idx.starts[left - 1] - idx.max_duration) <= nth_dist:
                left = max(0, left - max(n, (right - left) // 2 + 1))
                expanded = True
            if right < total and (idx.starts[right] - t_f) <= nth_dist:
                right = min(total, right + max(n, (right - left) // 2 + 1))
                expanded = True
            if not expanded:
                break

        candidates = [
            (float(d), int(orig_idx))
            for d, orig_idx in zip(dists, idx.order[left:right])
        ]
        candidates.sort(key=lambda pair: (pair[0], pair[1]))
        chosen = [orig_idx for _, orig_idx in candidates[: min(n, len(candidates))]]
        return self._slice_rows(chosen)  # type: ignore[attr-defined]
