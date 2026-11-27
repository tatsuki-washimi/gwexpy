"""Table combination operations (``concat``, ``join``, ``add_segment_id``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
import pandas as pd

from gwexpy.table.segment.cell import SegmentCell
from gwexpy.table.segment.status import is_scalar_nan

if TYPE_CHECKING:
    from gwexpy.table.segment.core import SegmentTable

_VALID_VALIDATE_MODES = frozenset(
    {
        "1:1",
        "one_to_one",
        "1:m",
        "one_to_many",
        "m:1",
        "many_to_one",
        "m:m",
        "many_to_many",
    }
)


def _check_kind_compatibility(
    col: str, k1: str, k2: str, is_p1: bool, is_p2: bool
) -> str:
    """Verify two column kinds are compatible and return the unified kind."""
    if is_p1 != is_p2:
        raise ValueError(
            f"Incompatible column storage for {col!r}: meta vs payload ({k1!r} vs {k2!r})."
        )
    if k1 == k2:
        return k1
    if not is_p1:
        # Both are meta/segment/object in _meta
        return "meta"
    # Both are payload
    if k1 == "object":
        return k2
    if k2 == "object":
        return k1
    raise ValueError(
        f"Incompatible payload kinds for column {col!r}: {k1!r} vs {k2!r}."
    )


class _CombineMixin:
    """Concatenation, key-based join, and segment ID generation methods."""

    _meta: pd.DataFrame
    _payload: dict[str, list[SegmentCell]]
    _schema: dict[str, str]
    _units: dict[str, Any]
    _provenance: list[dict[str, Any]]
    _external_attrs: dict[str, Any]

    @classmethod
    def concat(
        cls,
        tables: Union[Sequence[SegmentTable], Mapping[str, SegmentTable]],
        schema: str = "strict",
        source_column: Optional[str] = None,
    ) -> SegmentTable:
        """Concatenate multiple :class:`SegmentTable` instances row-wise.

        Parameters
        ----------
        tables:
            Non-empty sequence or mapping of :class:`SegmentTable` objects.
            When a mapping is given and *source_column* is set, its keys
            populate *source_column*.
        schema:
            Column alignment policy: ``"strict"`` (default, identical schema
            required), ``"union"`` (missing meta becomes ``NaN``, missing
            payload becomes ``SegmentCell.missing()``), or ``"intersection"``
            (retain only shared columns).
        source_column:
            Optional meta column name recording which input table each row
            originated from.

        Returns
        -------
        SegmentTable

        Raises
        ------
        ValueError
            If *tables* is empty, *schema* is invalid, schemas conflict, or
            duplicate ``segment_id`` values occur when ``segment_id`` exists.

        """
        if schema not in ("strict", "union", "intersection"):
            raise ValueError(
                f"Invalid schema policy {schema!r}; "
                "expected 'strict', 'union', or 'intersection'."
            )

        if isinstance(tables, Mapping):
            items: list[tuple[Any, SegmentTable]] = list(tables.items())
        else:
            items = list(enumerate(tables))

        if not items:
            raise ValueError("concat() requires at least one SegmentTable.")

        for _, t in items:
            if not isinstance(t, cls):
                raise TypeError(
                    f"concat() expected SegmentTable instances, got {type(t).__name__!r}."
                )

        first = items[0][1]
        if source_column is not None:
            if source_column == "span":
                raise ValueError("source_column cannot be 'span'.")
            for _, t in items:
                if source_column in t._schema:
                    raise ValueError(
                        f"source_column {source_column!r} already exists in an input table."
                    )

        # Determine aligned column list and unified schema
        col_kinds: dict[str, str] = {}
        col_is_payload: dict[str, bool] = {}

        if schema == "strict":
            for _, t in items[1:]:
                if t._schema != first._schema or set(t._payload) != set(first._payload):
                    raise ValueError(
                        "Schemas do not match under schema='strict'. "
                        f"Expected {first._schema}, got {t._schema}."
                    )
            ordered_cols = list(first.columns)
            for c in ordered_cols:
                col_kinds[c] = first._schema[c]
                col_is_payload[c] = c in first._payload
        elif schema == "intersection":
            shared = set(first._schema.keys())
            for _, t in items[1:]:
                shared &= set(t._schema.keys())
            ordered_cols = [c for c in first.columns if c in shared]
            for c in ordered_cols:
                kind = first._schema[c]
                is_p = c in first._payload
                for _, t in items[1:]:
                    kind = _check_kind_compatibility(
                        c, kind, t._schema[c], is_p, c in t._payload
                    )
                col_kinds[c] = kind
                col_is_payload[c] = is_p
        else:
            # union
            seen_cols: set[str] = set()
            ordered_meta: list[str] = []
            ordered_payload: list[str] = []
            for _, t in items:
                for c in t.columns:
                    is_p = c in t._payload
                    k = t._schema[c]
                    if c not in seen_cols:
                        seen_cols.add(c)
                        col_kinds[c] = k
                        col_is_payload[c] = is_p
                        if is_p:
                            ordered_payload.append(c)
                        else:
                            ordered_meta.append(c)
                    else:
                        col_kinds[c] = _check_kind_compatibility(
                            c, col_kinds[c], k, col_is_payload[c], is_p
                        )
            ordered_cols = ordered_meta + ordered_payload

        meta_cols = [c for c in ordered_cols if not col_is_payload[c]]
        payload_cols = [c for c in ordered_cols if col_is_payload[c]]

        meta_data: dict[str, list[Any]] = {c: [] for c in meta_cols}
        if source_column is not None:
            meta_data[source_column] = []
        payload_data: dict[str, list[SegmentCell]] = {c: [] for c in payload_cols}
        units: dict[str, Any] = {}

        for src_label, t in items:
            n_rows = len(t)
            for c in meta_cols:
                if c in t._meta.columns:
                    meta_data[c].extend(t._meta[c].tolist())
                else:
                    meta_data[c].extend([np.nan] * n_rows)
                if c not in units and c in t._units:
                    units[c] = t._units[c]

            if source_column is not None:
                meta_data[source_column].extend([src_label] * n_rows)

            for c in payload_cols:
                if c in t._payload:
                    payload_data[c].extend([cell.clone() for cell in t._payload[c]])
                else:
                    payload_data[c].extend(
                        [SegmentCell.missing() for _ in range(n_rows)]
                    )
                if c not in units and c in t._units:
                    units[c] = t._units[c]

        # Segment ID collision check when segment_id column is present
        if "segment_id" in meta_data:
            seen_ids: set[Any] = set()
            for sid in meta_data["segment_id"]:
                if is_scalar_nan(sid):
                    continue
                if sid in seen_ids:
                    raise ValueError(
                        f"Duplicate 'segment_id' value {sid!r} encountered during concat()."
                    )
                seen_ids.add(sid)

        result = cast("SegmentTable", cls(pd.DataFrame(meta_data)))  # type: ignore[call-arg]
        for c in meta_cols:
            if c != "span":
                result._schema[c] = col_kinds[c]
        if source_column is not None:
            result._schema[source_column] = "meta"
        for c in payload_cols:
            result._payload[c] = payload_data[c]
            result._schema[c] = col_kinds[c]
        result._units = units
        return result

    def join(
        self,
        other: SegmentTable,
        on: Union[str, Sequence[str]],
        how: str = "left",
        validate: Optional[str] = None,
        tolerance: Optional[float] = None,
        suffixes: tuple[str, str] = ("_left", "_right"),
    ) -> SegmentTable:
        """Join *self* with *other* on explicit key column(s).

        Parameters
        ----------
        other:
            Right-hand :class:`SegmentTable`.
        on:
            Key column name or sequence of column names present in both tables.
            Implicit row-index joining is intentionally unsupported.
        how:
            Join type: ``"left"`` (default), ``"inner"``, or ``"outer"``.
        validate:
            Optional cardinality check: ``"1:1"`` / ``"one_to_one"``,
            ``"1:m"`` / ``"one_to_many"``, ``"m:1"`` / ``"many_to_one"``,
            or ``"m:m"`` / ``"many_to_many"``.
        tolerance:
            Required non-negative tolerance in seconds when ``"span"`` is
            included in *on*.
        suffixes:
            Suffixes applied to overlapping non-key columns.

        Returns
        -------
        SegmentTable

        """
        if not isinstance(other, type(self)):
            raise TypeError(
                f"join() expected SegmentTable, got {type(other).__name__!r}."
            )
        if how not in ("left", "inner", "outer"):
            raise ValueError(
                f"Invalid how {how!r}; expected 'left', 'inner', or 'outer'."
            )
        if validate is not None and validate not in _VALID_VALIDATE_MODES:
            raise ValueError(
                f"Invalid validate {validate!r}; expected one of {sorted(_VALID_VALIDATE_MODES)}."
            )
        if on is None:
            raise ValueError("join() requires an explicit 'on' column or columns.")

        on_cols = [on] if isinstance(on, str) else list(on)
        if not on_cols:
            raise ValueError("join() requires at least one key column in 'on'.")

        for k in on_cols:
            if k not in self._schema:
                raise KeyError(f"Join key {k!r} not found in left table.")
            if k not in other._schema:
                raise KeyError(f"Join key {k!r} not found in right table.")
            if k in self._payload or k in other._payload:
                raise TypeError(f"Cannot join on payload column {k!r}.")

        has_span = "span" in on_cols
        if has_span:
            if tolerance is None:
                raise ValueError(
                    "Joining on 'span' requires an explicit 'tolerance' (in seconds)."
                )
            if tolerance < 0:
                raise ValueError(f"tolerance must be non-negative, got {tolerance}.")
        elif tolerance is not None and tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}.")

        non_span_keys = [k for k in on_cols if k != "span"]
        n_left = len(self._meta)
        n_right = len(other._meta)

        # Build index over right rows by non_span_keys
        right_buckets: dict[tuple[Any, ...], list[int]] = {}
        right_has_na_key: list[bool] = [False] * n_right
        for j in range(n_right):
            key_vals = tuple(other._meta.at[j, k] for k in non_span_keys)
            if any(is_scalar_nan(v) for v in key_vals):
                right_has_na_key[j] = True
                continue
            right_buckets.setdefault(key_vals, []).append(j)

        # Optional binary-search acceleration when joining purely on span
        other_idx = (
            other._get_segment_index()  # type: ignore[attr-defined]
            if (has_span and not non_span_keys)
            else None
        )

        # Match pairs
        left_match_counts = [0] * n_left
        right_match_counts = [0] * n_right
        matched_pairs: list[tuple[Optional[int], Optional[int]]] = []
        tol = float(tolerance) if tolerance is not None else 0.0

        for i in range(n_left):
            key_vals = tuple(self._meta.at[i, k] for k in non_span_keys)
            if any(is_scalar_nan(v) for v in key_vals):
                if how in ("left", "outer"):
                    matched_pairs.append((i, None))
                continue

            if other_idx is not None:
                ls = self._meta.at[i, "span"]
                l_start, l_end = float(ls[0]), float(ls[1])
                lo = int(np.searchsorted(other_idx.starts, l_start - tol, side="left"))
                hi = int(np.searchsorted(other_idx.starts, l_start + tol, side="right"))
                candidates = [
                    int(orig_j)
                    for orig_j, r_end in zip(
                        other_idx.order[lo:hi], other_idx.ends[lo:hi]
                    )
                    if abs(l_end - float(r_end)) <= tol
                ]
                candidates.sort()
            else:
                candidates = right_buckets.get(key_vals, [])
                if has_span and candidates:
                    ls = self._meta.at[i, "span"]
                    l_start, l_end = float(ls[0]), float(ls[1])
                    candidates = [
                        j
                        for j in candidates
                        if abs(l_start - float(other._meta.at[j, "span"][0])) <= tol
                        and abs(l_end - float(other._meta.at[j, "span"][1])) <= tol
                    ]

            if candidates:
                left_match_counts[i] = len(candidates)
                for j in candidates:
                    right_match_counts[j] += 1
                    matched_pairs.append((i, j))
            elif how in ("left", "outer"):
                matched_pairs.append((i, None))

        if how == "outer":
            for j in range(n_right):
                if right_match_counts[j] == 0:
                    matched_pairs.append((None, j))

        # Cardinality validation
        if validate is not None:
            left_must_be_unique = validate in (
                "1:1",
                "one_to_one",
                "1:m",
                "one_to_many",
            )
            right_must_be_unique = validate in (
                "1:1",
                "one_to_one",
                "m:1",
                "many_to_one",
            )
            if left_must_be_unique:
                if any(c > 1 for c in right_match_counts):
                    raise ValueError(
                        f"Join validation '{validate}' failed: left keys are not unique "
                        "(multiple left rows matched a single right row)."
                    )
                if not has_span:
                    seen_l: set[tuple[Any, ...]] = set()
                    for i in range(n_left):
                        kv = tuple(self._meta.at[i, k] for k in non_span_keys)
                        if any(is_scalar_nan(v) for v in kv):
                            continue
                        if kv in seen_l:
                            raise ValueError(
                                f"Join validation '{validate}' failed: duplicate key {kv!r} in left table."
                            )
                        seen_l.add(kv)
            if right_must_be_unique:
                if any(c > 1 for c in left_match_counts):
                    raise ValueError(
                        f"Join validation '{validate}' failed: right keys are not unique "
                        "(multiple right rows matched a single left row)."
                    )
                if not has_span:
                    for kv, idxs in right_buckets.items():
                        if len(idxs) > 1:
                            raise ValueError(
                                f"Join validation '{validate}' failed: duplicate key {kv!r} in right table."
                            )

        # Build output schema & column mappings
        on_set = set(on_cols)
        left_other = [c for c in self.columns if c not in on_set and c != "span"]  # type: ignore[attr-defined]
        right_other = [c for c in other.columns if c not in on_set and c != "span"]
        overlap = set(left_other) & set(right_other)

        meta_out: dict[str, list[Any]] = {"span": []}
        for li, rj in matched_pairs:
            if li is not None:
                meta_out["span"].append(self._meta.at[li, "span"])
            else:
                assert rj is not None
                meta_out["span"].append(other._meta.at[rj, "span"])

        for k in non_span_keys:
            vals: list[Any] = []
            for li, rj in matched_pairs:
                if li is not None:
                    vals.append(self._meta.at[li, k])
                else:
                    assert rj is not None
                    vals.append(other._meta.at[rj, k])
            meta_out[k] = vals

        payload_out: dict[str, list[SegmentCell]] = {}
        out_schema: dict[str, str] = {"span": "segment"}
        for k in non_span_keys:
            out_schema[k] = self._schema[k]
        out_units: dict[str, Any] = {}
        for k in on_cols:
            if k in self._units:
                out_units[k] = self._units[k]

        def _append_side_cols(
            src: SegmentTable,
            cols: list[str],
            side_idx: int,
            suffix: str,
        ) -> None:
            for c in cols:
                out_name = f"{c}{suffix}" if c in overlap else c
                if out_name in out_schema:
                    raise ValueError(
                        f"Column name collision after applying suffix: {out_name!r}."
                    )
                out_schema[out_name] = src._schema[c]
                if c in src._units:
                    out_units[out_name] = src._units[c]
                if c in src._payload:
                    cells: list[SegmentCell] = []
                    for pair in matched_pairs:
                        row_idx = pair[side_idx]
                        if row_idx is None:
                            cells.append(SegmentCell.missing())
                        else:
                            cells.append(src._payload[c][row_idx].clone())
                    payload_out[out_name] = cells
                else:
                    mvals: list[Any] = []
                    for pair in matched_pairs:
                        row_idx = pair[side_idx]
                        if row_idx is None:
                            mvals.append(np.nan)
                        else:
                            mvals.append(src._meta.at[row_idx, c])
                    meta_out[out_name] = mvals

        _append_side_cols(self, left_other, 0, suffixes[0])  # type: ignore[arg-type]
        _append_side_cols(other, right_other, 1, suffixes[1])

        result = type(self)(pd.DataFrame(meta_out))  # type: ignore[call-arg]
        for col_name, kind in out_schema.items():
            if col_name != "span":
                result._schema[col_name] = kind
        for col_name, cells in payload_out.items():
            result._payload[col_name] = cells
        result._units = out_units
        return result  # type: ignore[return-value]

    def add_segment_id(
        self,
        fmt: Optional[str] = None,
        *,
        column: str = "segment_id",
        inplace: bool = False,
        overwrite: bool = False,
    ) -> SegmentTable:
        """Populate a deterministic string identifier column for each row.

        Parameters
        ----------
        fmt:
            Python format string evaluated per row with all meta column values
            plus ``start``, ``end``, ``duration``, and ``index``.  Defaults to
            ``"{detector}-{start:.0f}-{end:.0f}"`` when a ``detector`` column
            exists, or ``"{start:.0f}-{end:.0f}"`` otherwise.
        column:
            Target column name (default ``"segment_id"``).
        inplace:
            If ``True``, mutate *self*; otherwise return a new table.
        overwrite:
            If ``True``, allow replacing an existing *column*.

        Returns
        -------
        SegmentTable

        """
        if column == "span":
            raise ValueError("Cannot use 'span' as segment_id column name.")
        if column in self._schema and not overwrite:
            raise ValueError(
                f"Column {column!r} already exists; pass overwrite=True to replace."
            )
        if fmt is None:
            fmt = (
                "{detector}-{start:.0f}-{end:.0f}"
                if "detector" in self._meta.columns
                else "{start:.0f}-{end:.0f}"
            )

        n = len(self._meta)
        meta_cols = list(self._meta.columns)
        ids: list[str] = []
        for i in range(n):
            span = self._meta.at[i, "span"]
            ctx: dict[str, Any] = {c: self._meta.at[i, c] for c in meta_cols}
            ctx["start"] = float(span[0])
            ctx["end"] = float(span[1])
            ctx["duration"] = float(span[1] - span[0])
            ctx["index"] = i
            ids.append(fmt.format(**ctx))

        target: SegmentTable = self if inplace else self.copy(deep=False)  # type: ignore[assignment, attr-defined]
        target._set_meta_column(column, ids, kind="meta")  # type: ignore[attr-defined]
        return target
