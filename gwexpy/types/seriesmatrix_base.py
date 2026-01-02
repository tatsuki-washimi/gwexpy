import warnings
import numpy as np
from collections import OrderedDict
from typing import Optional, Union, Any
from astropy import units as u
from gwpy.types.index import Index

from .metadata import MetaData, MetaDataDict, MetaDataMatrix
from .seriesmatrix_validation import (
    _normalize_input,
    _check_attribute_consistency,
    _fill_missing_attributes,
    _make_meta_matrix,
    _check_shape_consistency,
    build_index_if_needed,
    check_shape_xindex_compatibility,
    check_add_sub_compatibility
)

from .seriesmatrix_ops import SeriesMatrixOps
from ._stats import StatisticalMethodsMixin
from gwexpy.types.mixin import RegularityMixin

class PerformanceWarning(RuntimeWarning):
    """Warning raised when an operation falls back to a slower implementation."""
    pass

class SeriesMatrix(RegularityMixin, SeriesMatrixOps, StatisticalMethodsMixin, np.ndarray):
    def __new__(cls,
                data: Any = None,
                *,
                meta: Optional[Union["MetaDataMatrix", np.ndarray, list]] = None,
                unit: Optional[object] = None,
                units: Optional[np.ndarray] = None,
                names: Optional[np.ndarray] = None,
                channels: Optional[np.ndarray] = None,
                rows: Any = None,
                cols: Any = None,
                shape: Any = None,
                xindex: Any = None,
                dx: Any = None,
                x0: Any = None,
                xunit: Any = None,
                name: str = "",
                epoch: float = 0.0,
                attrs: Optional[dict[str, Any]] = None) -> "SeriesMatrix":
        """
        Create a SeriesMatrix with normalized inputs and metadata.
        """

        if unit is not None:
            if units is not None:
                raise ValueError("give only one of unit or units")
            units = unit

            units = unit

        if xindex is not None and xunit is not None:
            try:
                xindex = u.Quantity(xindex, xunit)
            except (TypeError, ValueError, AttributeError) as e:
                # If conversion fails or type is incompatible, we propagate the error
                # or assume caller knows what they are doing. But xunit implies intention.
                print(f"DEBUG: SeriesMatrix xindex conversion failed: {e}")
                pass  # Or raise u.UnitConversionError(f"xindex conversion failed: {e}")

        value_array, data_attrs, detected_xindex = _normalize_input(
            data=data,
            units=units,
            names=names,
            channels=channels,
            shape=shape,
            xindex=xindex,
            dx=dx,
            x0=x0,
            xunit=xunit
        )

        if meta is not None:
            if not isinstance(meta, MetaDataMatrix):
                try:
                    meta = MetaDataMatrix(meta)
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        "meta must be a MetaDataMatrix or a 2D array-like of MetaData/dict"
                    ) from e
            if units is None:
                data_attrs["unit"] = None
            if names is None:
                data_attrs["name"] = None
            if channels is None:
                data_attrs["channel"] = None
            _check_attribute_consistency(
                data_attrs=data_attrs,
                meta=meta
            )
            units_arr, names_arr, channels_arr = _fill_missing_attributes(
                data_attrs=data_attrs,
                meta=meta
            )
        else:
            units_arr = data_attrs.get("unit", None)
            names_arr = data_attrs.get("name", None)
            channels_arr = data_attrs.get("channel", None)

        meta_matrix = _make_meta_matrix(
            shape=value_array.shape[:2],
            units=units_arr,
            names=names_arr,
            channels=channels_arr
        )

        if xindex is None:
            if detected_xindex is not None:
                xindex = detected_xindex
            else:
                if value_array.shape[2] == 0 and dx is None and x0 is None:
                    xindex = np.asarray([])
                else:
                    xindex = build_index_if_needed(
                        xindex=None,
                        dx=dx,
                        x0=x0,
                        xunit=xunit,
                        length=value_array.shape[2]
                    )

        _check_shape_consistency(
            value_array=value_array,
            meta_matrix=meta_matrix,
            xindex=xindex
        )

        obj = np.asarray(value_array).view(cls)

        obj._value = obj.view(np.ndarray)

        obj.meta = meta_matrix
        N, M = value_array.shape[:2]
        if isinstance(rows, dict) and not isinstance(rows, OrderedDict):
            rows = OrderedDict(rows)
        if isinstance(cols, dict) and not isinstance(cols, OrderedDict):
            cols = OrderedDict(cols)
        obj.rows = MetaDataDict(rows, expected_size=N, key_prefix="row")
        obj.cols = MetaDataDict(cols, expected_size=M, key_prefix="col")
        obj.xindex = xindex
        obj.name = name
        obj.epoch = epoch
        obj.attrs = attrs or {}

        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self._value = self.view(np.ndarray)
        self._suppress_xindex_check = True
        self.xindex = getattr(obj, 'xindex', None)
        if hasattr(self, "_suppress_xindex_check"):
            delattr(self, "_suppress_xindex_check")
        self.meta   = getattr(obj, 'meta', None)
        self.rows   = getattr(obj, 'rows', None)
        self.cols   = getattr(obj, 'cols', None)
        self.name   = getattr(obj, 'name', "")
        self.epoch  = getattr(obj, 'epoch', 0.0)
        self.attrs  = getattr(obj, 'attrs', getattr(self, "attrs", {}))

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != '__call__':
            base_inputs = [inp.view(np.ndarray) if isinstance(inp, SeriesMatrix) else inp for inp in inputs]
            try:
                return np.ndarray.__array_ufunc__(self.view(np.ndarray), ufunc, method, *base_inputs, **kwargs)
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                return NotImplemented

        casted_inputs = []
        xindex = self.xindex
        shape  = self._value.shape
        rows   = self.rows
        cols   = self.cols
        epoch  = getattr(self, "epoch", 0.0)
        name   = getattr(self, "name", "")
        attrs  = getattr(self, "attrs", {})

        for inp in inputs:
            if isinstance(inp, SeriesMatrix):
                casted_inputs.append(inp)
            elif isinstance(inp, u.Quantity):
                val = np.asarray(inp.value)
                N, M, K = shape
                if val.ndim == 0:
                    arr = np.full(shape, val)
                elif val.ndim == 1:
                    if val.shape != (K,):
                        raise ValueError(f"1D Quantity must have length N_samples={K}, got {val.shape}")
                    arr = np.broadcast_to(val.reshape(1, 1, K), shape)
                elif val.ndim == 2:
                    if val.shape != (N, M):
                        raise ValueError(f"2D Quantity must have shape (Nrow,Ncol)={(N, M)}, got {val.shape}")
                    arr = np.broadcast_to(val.reshape(N, M, 1), shape)
                elif val.ndim == 3:
                    if val.shape != shape:
                        raise ValueError(f"3D Quantity must have shape {shape}, got {val.shape}")
                    arr = val
                else:
                    raise ValueError(f"Quantity with ndim={val.ndim} is not supported in __array_ufunc__")
                unit = inp.unit
                meta_array = np.empty(self._value.shape[:2], dtype=object)
                for i in range(self._value.shape[0]):
                    for j in range(self._value.shape[1]):
                        meta_array[i, j] = MetaData(unit=unit, name=f"s{i}{j}")
                meta_matrix = MetaDataMatrix(meta_array)
                casted_inputs.append(SeriesMatrix(arr, xindex=xindex, meta=meta_matrix, shape=self._value.shape))
            elif isinstance(inp, (float, int, complex)):
                arr = np.full(self._value.shape, inp)
                unit = u.dimensionless_unscaled
                meta_array = np.empty(self._value.shape[:2], dtype=object)
                for i in range(self._value.shape[0]):
                    for j in range(self._value.shape[1]):
                        meta_array[i, j] = MetaData(unit=unit, name=f"s{i}{j}")
                meta_matrix = MetaDataMatrix(meta_array)
                casted_inputs.append(SeriesMatrix(arr, xindex=xindex, meta=meta_matrix, shape=self._value.shape))
            elif isinstance(inp, np.ndarray):
                val = np.asarray(inp)
                N, M, K = shape
                if val.ndim == 0:
                    arr = np.full(shape, val)
                elif val.ndim == 1:
                    if val.shape != (K,):
                        raise ValueError(
                            f"1D ndarray must have length N_samples={K}, got {val.shape}"
                        )
                    arr = np.broadcast_to(val.reshape(1, 1, K), shape)
                elif val.ndim == 2:
                    if val.shape != (N, M):
                        raise ValueError(
                            f"2D ndarray must have shape (Nrow,Ncol)={(N, M)}, got {val.shape}"
                        )
                    arr = np.broadcast_to(val.reshape(N, M, 1), shape)
                elif val.ndim == 3:
                    if val.shape != shape:
                        raise ValueError(f"3D ndarray must have shape {shape}, got {val.shape}")
                    arr = val
                else:
                    raise ValueError(
                        f"ndarray with ndim={val.ndim} is not supported in __array_ufunc__"
                    )

                casted_inputs.append(SeriesMatrix(arr, xindex=xindex, shape=self._value.shape))
            else:
                return NotImplemented

        check_shape_xindex_compatibility(*casted_inputs)

        if ufunc in [np.add, np.subtract, np.less, np.less_equal, np.equal, np.not_equal, np.greater, np.greater_equal]:
            check_add_sub_compatibility(*casted_inputs)

        value_arrays = [inp.view(np.ndarray) for inp in casted_inputs]

        meta_matrices = [inp.meta for inp in casted_inputs]

        ufunc_kwargs = {k: v for k, v in kwargs.items() if k not in ('out', 'where')}

        N, M = self._value.shape[:2]
        logical_ufuncs = {
            np.logical_and, np.logical_or, np.logical_xor, np.logical_not,
            np.isfinite, np.isinf, np.isnan, np.isclose
        }
        meta_passthrough_ufuncs = logical_ufuncs | {
            np.sign, np.floor, np.ceil, np.trunc, np.rint,
            np.mod, np.remainder, np.clip
        }
        ufunc_name = getattr(ufunc, "__name__", None)
        meta_passthrough = ufunc in meta_passthrough_ufuncs or ufunc_name in {"clip"}
        try:
            probe_val_args = [v[0, 0] for v in value_arrays]
            probe_result = ufunc(*probe_val_args, **ufunc_kwargs)
            boolean_ufuncs = {np.less, np.less_equal, np.equal, np.not_equal,
                              np.greater, np.greater_equal,
                              np.logical_and, np.logical_or, np.logical_xor,
                              np.logical_not, np.isfinite, np.isinf, np.isnan,
                              np.isclose}
            if ufunc in boolean_ufuncs:
                result_dtype = np.bool_
            else:
                result_dtype = np.asarray(probe_result).dtype
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            result_dtype = self._value.dtype

        # Vectorized value calculation
        try:
            result_values = ufunc(*value_arrays, **ufunc_kwargs)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            # Fallback to loop if vectorized call fails (likely due to custom ufunc or mixed types)
            warnings.warn(f"ufunc {ufunc.__name__} failed vectorized execution; falling back to loop. Error: {e}", PerformanceWarning)
            result_values = np.empty(self._value.shape, dtype=result_dtype)
            for i in range(N):
                for j in range(M):
                    val_args = [v[i, j] for v in value_arrays]
                    result_values[i, j] = ufunc(*val_args, **ufunc_kwargs)

        result_meta   = np.empty(self._value.shape[:2], dtype=object)
        bool_result = np.issubdtype(result_dtype, np.bool_)

        # Optimize Metadata calculation
        # If all input matrices have uniform units, we can avoid the loop
        def _get_uniform_meta(meta_mat):
            first = meta_mat[0, 0]
            # Check if all elements are effectively the same in terms of ufunc result
            # For simplicity, we check if all have the same unit.
            if np.all(meta_mat.units == first.unit):
                return first
            return None

        uniform_metas = [_get_uniform_meta(m) for m in meta_matrices]
        all_uniform = all(m is not None for m in uniform_metas)

        if all_uniform and not (bool_result or meta_passthrough):
            try:
                # Compute resulting meta once
                res_meta_obj = ufunc(*uniform_metas, **ufunc_kwargs)
                result_meta = np.full((N, M), res_meta_obj, dtype=object)
            except Exception:
                all_uniform = False

        if not all_uniform:
            try:
                if bool_result or meta_passthrough:
                    # In these cases, we typically take the first operand's metadata
                    result_meta = meta_matrices[0].copy()
                else:
                    # Leverage MetaDataMatrix's vectorized ufunc support
                    result_meta = ufunc(*meta_matrices, **ufunc_kwargs)
            except Exception as e:
                 # Fallback to loop if vectorized meta-ufunc fails
                 warnings.warn(f"MetaData vectorized ufunc failed; falling back to loop. Error: {e}", PerformanceWarning)
                 result_meta = np.empty((N, M), dtype=object)
                 for i in range(N):
                     for j in range(M):
                         meta_args = [m[i, j] for m in meta_matrices]
                         if bool_result or meta_passthrough:
                             result_meta[i, j] = meta_args[0]
                         else:
                             result_meta[i, j] = ufunc(*meta_args, **ufunc_kwargs)

        result_meta_matrix = MetaDataMatrix(result_meta)
        result_units = result_meta_matrix.units
        return self.__class__(
            result_values,
            xindex=self.xindex,
            meta=result_meta_matrix,
            units=result_units,
            rows=rows,
            cols=cols,
            name=name,
            epoch=epoch,
            attrs=attrs
        )

    ##### xindex Information #####
    @property
    def xindex(self) -> Any:
        """Sample axis index array.

        This is the array of x-axis values (e.g., time or frequency) corresponding
        to each sample in the matrix. For time series, this represents timestamps;
        for frequency series, this represents frequency bins.

        Returns
        -------
        xindex : `~gwpy.types.Index`, `~astropy.units.Quantity`, or `numpy.ndarray`
            The sample axis values with appropriate units.
        """
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
                n_samples = self._value.shape[2]
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
        """Starting value of the sample axis.

        Returns
        -------
        x0 : `~astropy.units.Quantity`
            The first value of xindex, representing the start time/frequency.
        """
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
        """Step size between samples on the x-axis.

        For regularly-sampled data, this is the constant spacing between
        consecutive samples. For time series, this equals ``1/sample_rate``.

        Returns
        -------
        dx : `~astropy.units.Quantity`
            The sample spacing with appropriate units.

        Raises
        ------
        AttributeError
            If xindex is not set or if the series has irregular sampling.
        """
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
        """Full extent of the sample axis as a tuple (start, end).

        The end value is calculated as the last sample plus one step (dx),
        representing the exclusive upper bound of the data range.

        Returns
        -------
        xspan : tuple
            A 2-tuple of (start, end) values with appropriate units.
        """
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
        """Physical unit of the sample axis.

        Returns
        -------
        xunit : `~astropy.units.Unit`
            The unit of the x-axis (e.g., seconds for time series, Hz for frequency series).
        """
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
        """Number of samples along the x-axis.

        Returns
        -------
        int
            The length of the sample axis (third dimension of the matrix).
        """
        return len(self.xindex) if self.xindex is not None else 0

    @property
    def xarray(self) -> np.ndarray:
        """
        Return the sample axis values.
        """
        return self.xindex

    @property
    def duration(self) -> Any:
        """
        Duration covered by the samples.
        """
        if self.N_samples == 0:
            try:
                return 0 * self.xunit
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                return 0
        return self.xindex[-1] - self.xindex[0]

    def is_compatible(self, other: Any) -> bool:
        """
        Compatibility check.
        """
        if not isinstance(other, SeriesMatrix):
            arr = np.asarray(other)
            if arr.shape != self._value.shape:
                raise ValueError(f"shape does not match: {self._value.shape} vs {arr.shape}")
            return True

        if self._value.shape[:2] != other._value.shape[:2]:
            raise ValueError(f"matrix shape does not match: {self._value.shape[:2]} vs {other._value.shape[:2]}")

        xunit_self = getattr(self.xindex, "unit", None)
        xunit_other = getattr(other.xindex, "unit", None)
        if xunit_self is not None and xunit_other is not None:
            try:
                if not u.Unit(xunit_self).is_equivalent(u.Unit(xunit_other)):
                    raise ValueError(f"xindex unit does not match: {xunit_self} vs {xunit_other}")
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                raise ValueError(f"xindex unit does not match: {xunit_self} vs {xunit_other}")

        try:
            dx_self = self.dx
            dx_other = other.dx
            if dx_self != dx_other:
                raise ValueError(f"dx does not match: {dx_self} vs {dx_other}")
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            lhs = np.asarray(self.xindex)
            rhs = np.asarray(other.xindex)
            if not np.array_equal(lhs, rhs):
                raise ValueError("xindex does not match")

        for i in range(self._value.shape[0]):
            for j in range(self._value.shape[1]):
                u1 = self.meta[i, j].unit
                u2 = other.meta[i, j].unit
                if u1 != u2:
                    raise ValueError(f"unit does not match at ({i},{j}): {u1} vs {u2}")
        return True

    ##### rows/cols Information #####
    def row_keys(self) -> list[Any]:
        """Get the keys (labels) for all rows.

        Returns
        -------
        tuple
            A tuple of row keys in order.
        """
        return tuple(self.rows.keys())

    def col_keys(self) -> list[Any]:
        """Get the keys (labels) for all columns.

        Returns
        -------
        tuple
            A tuple of column keys in order.
        """
        return tuple(self.cols.keys())

    def keys(self) -> list[tuple[Any, Any]]:
        """Get both row and column keys.

        Returns
        -------
        tuple
            A 2-tuple of (row_keys, col_keys).
        """
        return (self.row_keys(), self.col_keys())

    def row_index(self, key: Any) -> int:
        """Get the integer index for a row key.

        Parameters
        ----------
        key : Any
            The row key to look up.

        Returns
        -------
        int
            The zero-based index of the row.

        Raises
        ------
        KeyError
            If the key is not found.
        """
        try:
            return list(self.row_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid row key: {key}")

    def col_index(self, key: Any) -> int:
        """Get the integer index for a column key.

        Parameters
        ----------
        key : Any
            The column key to look up.

        Returns
        -------
        int
            The zero-based index of the column.

        Raises
        ------
        KeyError
            If the key is not found.
        """
        try:
            return list(self.col_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid column key: {key}")

    def get_index(self, key_row: Any, key_col: Any) -> tuple[int, int]:
        """Get the (row, col) integer indices for given keys.

        Parameters
        ----------
        key_row : Any
            The row key.
        key_col : Any
            The column key.

        Returns
        -------
        tuple of int
            A 2-tuple of (row_index, col_index).
        """
        return self.row_index(key_row), self.col_index(key_col)

    ##### Elements Metadata #####
    @property
    def MetaDataMatrix(self) -> "MetaDataMatrix":
        """Metadata matrix containing per-element metadata.

        Returns
        -------
        MetaDataMatrix
            A 2D matrix of MetaData objects, one for each (row, col) element.
        """
        return self.meta

    @property
    def units(self) -> np.ndarray:
        """2D array of units for each matrix element.

        Returns
        -------
        numpy.ndarray
            A 2D object array of `astropy.units.Unit` instances.
        """
        return self.meta.units

    @property
    def names(self) -> np.ndarray:
        """2D array of names for each matrix element.

        Returns
        -------
        numpy.ndarray
            A 2D object array of string names.
        """
        return self.meta.names

    @property
    def channels(self) -> np.ndarray:
        """2D array of channel identifiers for each matrix element.

        Returns
        -------
        numpy.ndarray
            A 2D object array of channel names or Channel objects.
        """
        return self.meta.channels
