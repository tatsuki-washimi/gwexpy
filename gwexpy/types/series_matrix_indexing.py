from __future__ import annotations

import numpy as np

from .seriesmatrix_validation import _expand_key, _slice_metadata_dict


class SeriesMatrixIndexingMixin:
    """Mixin for SeriesMatrix indexing and slicing operations."""

    def _get_series_kwargs(self, xindex, meta):
        """
        Return keyword arguments for Series constructor.
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
        """
        Label-based or position-based indexing with metadata preservation.
        """
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
            result = super().__getitem__((ri, ci, s))

            # Retrieve metadata
            meta = self.meta[ri, ci]

            # Construct Series
            series_class = getattr(self, "series_class", None)
            if series_class is None:
                from gwpy.types.series import Series as series_class

            # Handle xindex slicing
            xidx = self.xindex
            if xidx is not None:
                new_xidx = xidx[s]
            else:
                new_xidx = None

            # Wrap in Series
            kwargs = self._get_series_kwargs(new_xidx, meta)
            return series_class(result, **kwargs)

        # 3. Handle label-based slicing for rows/cols
        ri = r
        if isinstance(r, (str, list)) or (
            isinstance(r, np.ndarray) and r.dtype.kind in "US"
        ):
            if isinstance(r, str):
                ri = self.row_index(r)
            else:
                ri = [self.row_index(k) for k in r]

        ci = c
        if isinstance(c, (str, list)) or (
            isinstance(c, np.ndarray) and c.dtype.kind in "US"
        ):
            if isinstance(c, str):
                ci = self.col_index(c)
            else:
                ci = [self.col_index(k) for k in c]

        # 4. Perform actual ndarray slicing
        new_key = (ri, ci, s)
        result = super().__getitem__(new_key)

        if not isinstance(result, np.ndarray):
            return result

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
        new_meta = self.meta[ri, ci]

        # Restore metadata dimensions
        if new_meta.ndim < 2:
            meta_shape = list(new_meta.shape)
            if isinstance(ci, (int, np.integer)):
                meta_shape.insert(1, 1)
            if isinstance(ri, (int, np.integer)):
                meta_shape.insert(0, 1)

            if len(meta_shape) != new_meta.ndim:
                new_meta = new_meta.reshape(meta_shape)
        new_rows = _slice_metadata_dict(self.rows, ri, "row")
        new_cols = _slice_metadata_dict(self.cols, ci, "col")

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
        """
        Assign values or Series to the matrix.
        """

        def _to_base(x):
            if hasattr(x, "value"):
                return x.value
            if hasattr(x, "to_value"):
                # Quantity
                return x.value
            return x

        expanded_key = _expand_key(key, 3)
        r, c, s = expanded_key

        ri = r
        if isinstance(r, (str, list)) or (
            isinstance(r, np.ndarray) and r.dtype.kind in "US"
        ):
            if isinstance(r, str):
                ri = self.row_index(r)
            else:
                ri = [self.row_index(k) for k in r]

        ci = c
        if isinstance(c, (str, list)) or (
            isinstance(c, np.ndarray) and c.dtype.kind in "US"
        ):
            if isinstance(c, str):
                ci = self.col_index(c)
            else:
                ci = [self.col_index(k) for k in c]

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

        # Use __getitem__ logic
        return self[ri, ci, :]
