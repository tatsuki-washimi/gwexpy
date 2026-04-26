from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from astropy import units as u

from .seriesmatrix_validation import _expand_key, _slice_metadata_dict

if TYPE_CHECKING:
    from gwpy.types.index import Index

    from gwexpy.types.metadata import MetaDataDict, MetaDataMatrix


def _resolve_label_key(key: Any, resolver: Any) -> Any:
    """Resolve string labels while leaving integer positions unchanged."""
    if isinstance(key, str):
        return resolver(key)
    if isinstance(key, (list, tuple)):
        return [resolver(item) if isinstance(item, str) else item for item in key]
    if isinstance(key, np.ndarray) and key.dtype.kind in "US":
        return [resolver(item) for item in key.tolist()]
    return key


def _is_advanced_axis_selector(key: Any) -> bool:
    """Return True for list-like row/column selectors."""
    if isinstance(key, np.ndarray):
        return key.ndim > 0
    return isinstance(key, (list, tuple))


def _as_position_list(key: Any) -> list[int]:
    """Normalize a list-like row/column selector into integer positions."""
    arr = np.asarray(key)
    if arr.dtype == np.bool_:
        return np.nonzero(arr)[0].tolist()
    return [int(item) for item in arr.tolist()]


class SeriesMatrixIndexingMixin:
    """Mixin for SeriesMatrix indexing and slicing operations."""

    if TYPE_CHECKING:
        _value: np.ndarray
        meta: MetaDataMatrix
        rows: MetaDataDict
        cols: MetaDataDict
        unit: u.Unit | None
        list_class: type[Any]
        series_class: type[Any] | None
        ndim: int
        shape: tuple[int, ...]

        @property
        def xindex(self) -> np.ndarray | u.Quantity | Index | None: ...  # noqa: D102

        @xindex.setter
        def xindex(self, value: np.ndarray | u.Quantity | Index | None) -> None: ...

        def row_index(self, key: Any) -> int: ...  # noqa: D102
        def col_index(self, key: Any) -> int: ...  # noqa: D102
        def view(self, dtype: Any = ..., type: type[Any] | None = ...) -> Any: ...  # noqa: D102

    def _get_series_kwargs(self, xindex, meta):
        """Return keyword arguments for Series constructor.

        Override in subclasses to use specific keys (e.g., 'times' or 'frequencies').
        """
        kwargs = {
            "xindex": xindex,
            "unit": meta.unit,
            "name": meta.name,
            "channel": meta.channel,
        }
        if xindex is None:
            kwargs["epoch"] = getattr(self, "epoch", None)
        return kwargs

    def __getitem__(self, key):
        """Apply label-based or position-based indexing with metadata preservation."""
        # 1. Expand key to full 3D coordinates
        expanded_key = _expand_key(key, 3)
        r, c, s = expanded_key

        # 2. Check if we are selecting a single cell (row key and col key are scalars)
        is_scalar_r = isinstance(r, (int, np.integer, str))
        is_scalar_c = isinstance(c, (int, np.integer, str))

        if is_scalar_r and is_scalar_c:
            # Single cell access -> return a Series
            ri = self.row_index(r) if isinstance(r, str) else r
            ci = self.col_index(c) if isinstance(c, str) else c

            # Index into values
            # Use cast to Any to avoid MyPy errors with super().__getitem__
            result = cast(Any, super()).__getitem__((ri, ci, s))

            # Retrieve metadata
            meta = self.meta[ri, ci]

            # Construct Series
            series_cls = getattr(self, "series_class", None)
            if series_cls is None:
                from gwpy.types.series import Series as _Series

                series_cls = _Series

            # Handle xindex slicing
            xidx = self.xindex
            if xidx is not None:
                new_xidx = xidx[s]
            else:
                new_xidx = None

            # Wrap in Series
            kwargs = self._get_series_kwargs(new_xidx, meta)
            return series_cls(result, **kwargs)

        # 3. Handle label-based slicing for rows/cols
        ri = _resolve_label_key(r, self.row_index)
        ci = _resolve_label_key(c, self.col_index)

        # 4. Perform actual ndarray slicing
        if _is_advanced_axis_selector(ri) and _is_advanced_axis_selector(ci):
            ri_pos = _as_position_list(ri)
            ci_pos = _as_position_list(ci)
            result = self.view(np.ndarray)[np.ix_(ri_pos, ci_pos)][:, :, s]
            meta_ri: Any = ri_pos
            meta_ci: Any = ci_pos
        else:
            new_key = (ri, ci, s)
            result = cast(Any, super()).__getitem__(new_key)
            meta_ri = ri
            meta_ci = ci

        if not isinstance(result, np.ndarray):
            return result

        if not isinstance(result, type(self)):
            result = result.view(type(self))
        result = cast(SeriesMatrixIndexingMixin, result)

        if result.ndim < 3:
            # Ensure result preserves 3D SeriesMatrix structure (Row, Col, X).
            # We restore dimensions dropped by scalar indices.
            new_shape = list(result.shape)
            if isinstance(s, (int, np.integer)):
                new_shape.append(1)
            if isinstance(ci, (int, np.integer)):
                new_shape.insert(1, 1)
            if isinstance(ri, (int, np.integer)):
                new_shape.insert(0, 1)

            if len(new_shape) != result.ndim:
                # result.reshape is overridden and broken for 2D. Use ndarray reshape.
                result = result.view(np.ndarray).reshape(new_shape).view(type(self))

        # 5. Slice internal metadata
        if _is_advanced_axis_selector(meta_ri) and _is_advanced_axis_selector(meta_ci):
            new_meta = self.meta[
                np.ix_(_as_position_list(meta_ri), _as_position_list(meta_ci))
            ]
        else:
            new_meta = self.meta[meta_ri, meta_ci]

        # Restore metadata dimensions
        if new_meta.ndim < 2:
            meta_shape = list(new_meta.shape)
            if isinstance(meta_ci, (int, np.integer)):
                meta_shape.insert(1, 1)
            if isinstance(meta_ri, (int, np.integer)):
                meta_shape.insert(0, 1)

            if len(meta_shape) != new_meta.ndim:
                new_meta = new_meta.reshape(meta_shape)
        new_rows = _slice_metadata_dict(self.rows, meta_ri, "row")
        new_cols = _slice_metadata_dict(self.cols, meta_ci, "col")

        # 6. Slice xindex
        if self.xindex is not None:
            if isinstance(s, (int, np.integer)):
                new_xindex = self.xindex[s : s + 1]
            else:
                new_xindex = self.xindex[s]
        else:
            new_xindex = None

        # 7. Update object attributes
        # __array_finalize__ will be called, but we need to re-assign sliced meta
        result.meta = new_meta
        result.rows = new_rows
        result.cols = new_cols
        result.xindex = new_xindex

        return result

    def __setitem__(self, key, value):
        """Assign values or Series objects into the matrix."""

        def _to_base(x):
            if hasattr(x, "value"):
                return x.value
            if hasattr(x, "to_value"):
                # Quantity
                return x.value
            return x

        expanded_key = _expand_key(key, 3)
        r, c, s = expanded_key

        ri = _resolve_label_key(r, self.row_index)
        ci = _resolve_label_key(c, self.col_index)

        if hasattr(value, "shape"):
            # If value is a Series or Matrix, ensure compatibility
            self.view(np.ndarray)[ri, ci, s] = _to_base(value)
        else:
            self.view(np.ndarray)[ri, ci, s] = value

    @property
    def loc(self):
        """Label-based indexer for direct value access."""

        class _LocAccessor:
            def __init__(self, parent):
                self._parent = parent

            def __getitem__(self, key):
                return self._parent[key]

            def __setitem__(self, key, value):
                self._parent[key] = value

        return _LocAccessor(self)

    def submatrix(self, row_keys, col_keys):
        """Extract a submatrix by selecting specific rows and columns."""
        # Validate keys
        if isinstance(row_keys, str):
            row_keys = [row_keys]
        if isinstance(col_keys, str):
            col_keys = [col_keys]

        ri = [self.row_index(k) for k in row_keys]
        ci = [self.col_index(k) for k in col_keys]

        # Apply row and column lists separately to get the Cartesian submatrix,
        # not NumPy's pairwise advanced-indexing result.
        return self[ri, :, :][:, ci, :]
