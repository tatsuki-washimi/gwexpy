from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numpy as np
from astropy import units as u
from gwpy.types.index import Index
from gwpy.types.series import Series

if TYPE_CHECKING:
    from gwexpy.types.metadata import MetaDataMatrix


class SeriesMatrixCoreMixin:
    """Core properties and basic conversion methods for SeriesMatrix."""

    @property
    def _x_axis_index(self) -> int:
        """Index of the axis corresponding to the 'sample' (xindex). Default is last."""
        return -1

    @property
    def shape3D(self) -> tuple[int, int, int]:
        """Shape of the matrix as a 3-tuple (n_rows, n_cols, n_samples).
           For 4D matrices (spectrograms), the last dimension is likely frequency,
           so n_samples is determined by _x_axis_index.
        """
        # This is a legacy helper; for 4D matrices it might not be strictly 3D.
        # But for display/repr, we might want (Row, Col, Time).
        val = self._value
        if val.ndim == 3:
            return val.shape
        # If not 3D, return full shape? Or try to map to (Row, Col, Sample)?
        return val.shape

    @property
    def value(self) -> np.ndarray:
        """Underlying numpy array of data values."""
        return self._value

    @value.setter
    def value(self, new_value: np.ndarray) -> None:
        """Set the underlying data values."""
        self._value[:] = new_value

    ##### xindex Information #####
    @property
    def xindex(self) -> Any:
        """Sample axis index array."""
        return getattr(self, "_xindex", None)

    @xindex.setter
    def xindex(self, value: Any) -> None:
        if value is None:
            self._xindex = None
        else:
            if isinstance(value, Index):
                xi = value
            elif isinstance(value, u.Quantity):
                xi = value
            else:
                xi = np.asarray(value)
            try:
                n_samples = self._value.shape[self._x_axis_index]
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                n_samples = None
            suppress = getattr(self, "_suppress_xindex_check", False)
            if (not suppress) and n_samples is not None and hasattr(xi, "__len__") and len(xi) != n_samples:
                raise ValueError(f"xindex length mismatch: expected {n_samples}, got {len(xi)}")
            self._xindex = xi
        for attr in ("_dx", "_x0"):
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def x0(self) -> Any:
        """Starting value of the sample axis."""
        try:
            return self._x0
        except AttributeError:
            try:
                self._x0 = self.xindex[0]
            except (AttributeError, IndexError):
                self._x0 = u.Quantity(0, self.xunit)
            return self._x0

    @property
    def dx(self) -> Any:
        """Step size between samples on the x-axis."""
        try:
            return self._dx
        except AttributeError:
            if not hasattr(self, "xindex") or self.xindex is None:
                raise AttributeError("dx is undefined because xindex is not set")
            if hasattr(self.xindex, "regular") and not self.xindex.regular:
                raise AttributeError("This SeriesMatrix has an irregular x-axis index, so 'dx' is not well defined")
            dx = self.xindex[1] - self.xindex[0]
            if not isinstance(dx, u.Quantity):
                xunit = getattr(self.xindex, 'unit', u.dimensionless_unscaled)
                dx = u.Quantity(dx, xunit)
            self._dx = dx
            return self._dx

    @property
    def xspan(self) -> Any:
        """Full extent of the sample axis as a tuple (start, end)."""
        xindex = self.xindex
        try:
            if hasattr(xindex, "regular") and xindex.regular:
                return (xindex[0], xindex[-1] + self.dx)
            if len(xindex) > 1:
                step = xindex[-1] - xindex[-2]
                return (xindex[0], xindex[-1] + step)
            return (xindex[0], xindex[0])
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return (xindex[0], xindex[-1])

    @property
    def xunit(self) -> Any:
        # Priority: dx (regular), xindex (array), x0
        try:
            return self._dx.unit
        except AttributeError:
            if self.xindex is not None:
                return getattr(self.xindex, 'unit', u.dimensionless_unscaled)
            try:
                return self._x0.unit
            except AttributeError:
                return u.dimensionless_unscaled

    @property
    def N_samples(self) -> int:
        """Number of samples along the x-axis."""
        return len(self.xindex) if self.xindex is not None else 0

    @property
    def xarray(self) -> np.ndarray:
        """Return the sample axis values."""
        return self.xindex

    @property
    def duration(self) -> Any:
        """Duration covered by the samples."""
        if self.N_samples == 0:
            try:
                return 0 * self.xunit
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                return 0
        return self.xindex[-1] - self.xindex[0]

    ##### rows/cols Information #####
    def row_keys(self) -> tuple[Any, ...]:
        """Get the keys (labels) for all rows."""
        return tuple(self.rows.keys()) if hasattr(self, "rows") and self.rows else tuple()

    def col_keys(self) -> tuple[Any, ...]:
        """Get the keys (labels) for all columns."""
        return tuple(self.cols.keys()) if hasattr(self, "cols") and self.cols else tuple()

    def keys(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Get both row and column keys."""
        return (self.row_keys(), self.col_keys())

    def row_index(self, key: Any) -> int:
        """Get the integer index for a row key."""
        try:
            return list(self.row_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid row key: {key}")

    def col_index(self, key: Any) -> int:
        """Get the integer index for a column key."""
        try:
            return list(self.col_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid column key: {key}")

    def get_index(self, key_row: Any, key_col: Any) -> tuple[int, int]:
        """Get the (row, col) integer indices for given keys."""
        return self.row_index(key_row), self.col_index(key_col)

    ##### Elements Metadata #####
    @property
    def MetaDataMatrix(self) -> MetaDataMatrix:
        """Metadata matrix containing per-element metadata."""
        return self.meta

    @property
    def units(self) -> np.ndarray:
        """2D array of units for each matrix element."""
        return self.meta.units

    @property
    def names(self) -> np.ndarray:
        """2D array of names for each matrix element. Alias for channel_names if 1D."""
        return self.meta.names

    @property
    def channels(self) -> np.ndarray:
        """2D array of channel identifiers for each matrix element."""
        return self.meta.channels

    @property
    def channel_names(self) -> list[str]:
        """Flattened list of all element names."""
        return [n for row in self.names for n in row]

    @channel_names.setter
    def channel_names(self, values: list[str]) -> None:
        """Set flattened list of element names."""
        self.meta.names = np.asarray(values).reshape(self.meta.shape)

    ##### Conversions #####
    def to_series_2Dlist(self) -> list[list[Series]]:
        """Convert matrix to a 2D nested list of Series objects."""
        N, M = self._value.shape[:2]
        return [[self[i, j] for j in range(M)] for i in range(N)]

    def to_series_1Dlist(self) -> list[Series]:
        """Convert matrix to a flat 1D list of Series objects."""
        N, M = self._value.shape[:2]
        return [self[i, j] for i in range(N) for j in range(M)]

    def to_list(self) -> Any:
        """Convert matrix to an appropriate collection list (e.g. TimeSeriesList)."""
        items = self.to_series_1Dlist()
        list_class = getattr(self, "list_class", list)
        return list_class(items)

    def to_dict_flat(self) -> dict[str, Series]:
        """Convert matrix to a flat dictionary mapping name to Series."""
        names = self.channel_names
        items = self.to_series_1Dlist()
        return dict(zip(names, items))

    def to_dict(self) -> Any:
        """
        Convert matrix to an appropriate collection dict (e.g. TimeSeriesDict).
        Follows the matrix structure (row, col) unless it's a 1-column matrix.
        """
        r_keys = self.row_keys()
        c_keys = self.col_keys()

        dict_class = getattr(self, "dict_class", dict)

        results = {}
        for i, rk in enumerate(r_keys):
            for j, ck in enumerate(c_keys):
                if len(c_keys) == 1:
                    key = rk
                else:
                    key = (rk, ck)
                results[key] = self[i, j]

        return dict_class(results)

    def _get_meta_for_constructor(self, data: np.ndarray, xindex: Any) -> dict[str, Any]:
        """
        Prepare arguments for constructing a new instance of the same class.
        """
        return {
            "data": data,
            "xindex": xindex,
            "rows": getattr(self, "rows", None),
            "cols": getattr(self, "cols", None),
            "meta": getattr(self, "meta", None),
            "name": getattr(self, "name", None),
            "epoch": getattr(self, "epoch", None),
            "attrs": getattr(self, "attrs", None),
            "unit": getattr(self, "unit", None),
        }
