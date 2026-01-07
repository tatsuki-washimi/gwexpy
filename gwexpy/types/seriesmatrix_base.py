import warnings
import numpy as np
from collections import OrderedDict
from typing import Optional, Union, Any
from astropy import units as u

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

from .series_matrix_core import SeriesMatrixCoreMixin
from .series_matrix_indexing import SeriesMatrixIndexingMixin
from .series_matrix_io import SeriesMatrixIOMixin
from .series_matrix_math import SeriesMatrixMathMixin
from .series_matrix_analysis import SeriesMatrixAnalysisMixin
from .series_matrix_structure import SeriesMatrixStructureMixin
from .series_matrix_visualization import SeriesMatrixVisualizationMixin
from .series_matrix_validation_mixin import SeriesMatrixValidationMixin
from ._stats import StatisticalMethodsMixin
from gwexpy.types.mixin import RegularityMixin, InteropMixin

_ADD_SUB_COMPARISON_UFUNCS = {
    np.add, np.subtract, np.less, np.less_equal, np.equal,
    np.not_equal, np.greater, np.greater_equal
}

class PerformanceWarning(RuntimeWarning):
    """Warning raised when an operation falls back to a slower implementation."""
    pass

class SeriesMatrix(
    RegularityMixin,
    InteropMixin,
    SeriesMatrixCoreMixin,
    SeriesMatrixIndexingMixin,
    SeriesMatrixIOMixin,
    SeriesMatrixMathMixin,
    SeriesMatrixAnalysisMixin,
    SeriesMatrixStructureMixin,
    SeriesMatrixVisualizationMixin,
    SeriesMatrixValidationMixin,
    StatisticalMethodsMixin,
    np.ndarray,
):
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
                casted_inputs.append(self.__class__(arr, xindex=xindex, meta=meta_matrix, shape=self._value.shape))
            elif isinstance(inp, (float, int, complex)):
                arr = np.full(self._value.shape, inp)
                unit = u.dimensionless_unscaled
                meta_array = np.empty(self._value.shape[:2], dtype=object)
                for i in range(self._value.shape[0]):
                    for j in range(self._value.shape[1]):
                        meta_array[i, j] = MetaData(unit=unit, name=f"s{i}{j}")
                meta_matrix = MetaDataMatrix(meta_array)
                casted_inputs.append(self.__class__(arr, xindex=xindex, meta=meta_matrix, shape=self._value.shape))
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

                casted_inputs.append(self.__class__(arr, xindex=xindex, shape=self._value.shape))
            else:
                return NotImplemented

        check_shape_xindex_compatibility(*casted_inputs)

        if ufunc in _ADD_SUB_COMPARISON_UFUNCS:
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

