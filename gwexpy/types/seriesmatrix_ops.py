import json
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from astropy import units as u
from gwpy.types.series import Series
from gwpy.types.index import Index
from gwexpy.interop._optional import require_optional

from .metadata import MetaData, MetaDataDict, MetaDataMatrix
from .seriesmatrix_validation import _expand_key, _slice_metadata_dict

class SeriesMatrixOps:
    """Mixin class for SeriesMatrix operations."""
    
    ##### Elements (Series object) acess #####
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(row_key, (int, str)) and isinstance(col_key, (int, str)):
                i = self.row_index(row_key) if not isinstance(row_key, int) else row_key
                j = self.col_index(col_key) if not isinstance(col_key, int) else col_key
                meta = self.meta[i, j]
                values = np.ndarray.__getitem__(self._value, (i, j)) 
                return Series(values, xindex=self.xindex.copy(),
                              unit=meta.unit, name=meta.name, channel=meta.channel)

        result = np.ndarray.__getitem__(self, key)
        if hasattr(result, "__array_finalize__") and hasattr(self, "xindex"):
            # result is a SeriesMatrix view
            row_sel, col_sel, sample_sel = _expand_key(key, self.ndim)
            try:
                xi = self.xindex[sample_sel]
                if np.isscalar(xi):
                    xi = u.Quantity([xi]) if isinstance(xi, u.Quantity) else np.asarray([xi])
                result.xindex = xi
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                result.xindex = self.xindex

            try:
                meta_slice = self.meta[row_sel, col_sel]
                if meta_slice.ndim == 1:
                    if isinstance(row_sel, (int, np.integer)):
                        meta_slice = meta_slice.reshape(1, -1)
                    elif isinstance(col_sel, (int, np.integer)):
                        meta_slice = meta_slice.reshape(-1, 1)
                result.meta = MetaDataMatrix(meta_slice)
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                result.meta = self.meta

            try:
                result.rows = _slice_metadata_dict(self.rows, row_sel, "row")
                result.cols = _slice_metadata_dict(self.cols, col_sel, "col")
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                result.rows = self.rows
                result.cols = self.cols
            if result.ndim == 2:
                sample_is_scalar = isinstance(sample_sel, (int, np.integer))
                row_is_scalar = isinstance(row_sel, (int, np.integer))
                col_is_scalar = isinstance(col_sel, (int, np.integer))

                if sample_is_scalar:
                    result = result.reshape(result.shape[0], result.shape[1], 1).view(
                        type(self)
                    )
                    result.xindex = (
                        u.Quantity(result.xindex).reshape(1)
                        if isinstance(result.xindex, u.Quantity)
                        else np.asarray(result.xindex).reshape(1)
                    )
                else:
                    if col_is_scalar and not row_is_scalar:
                        # (nrow, nsample) -> (nrow, 1, nsample)
                        result = result.reshape(
                            result.shape[0], 1, result.shape[1]
                        ).view(type(self))
                    elif row_is_scalar and not col_is_scalar:
                        # (ncol, nsample) -> (1, ncol, nsample)
                        result = result.reshape(
                            1, result.shape[0], result.shape[1]
                        ).view(type(self))
                    else:
                        # Fallback: treat as single-sample matrix
                        result = result.reshape(result.shape[0], result.shape[1], 1).view(
                            type(self)
                        )
                        result.xindex = (
                            u.Quantity(result.xindex).reshape(1)
                            if isinstance(result.xindex, u.Quantity)
                            else np.asarray(result.xindex).reshape(1)
                        )

                result.meta = MetaDataMatrix(result.meta)
            if hasattr(result, "_value"):
                result._value = result.view(np.ndarray)
        return result
    
    def __setitem__(self, key, value):
        if not (isinstance(key, tuple) and len(key) == 2):
            return np.ndarray.__setitem__(self, key, value)
        
        row_key, col_key = key
        if not (isinstance(row_key, (int, str)) and isinstance(col_key, (int, str))):
            return np.ndarray.__setitem__(self, key, value)

        if not isinstance(value, Series):
            raise TypeError("Only Series objects can be assigned to SeriesMatrix elements.")

        row_is_index = isinstance(row_key, int)
        col_is_index = isinstance(col_key, int)
        row_missing = (not row_is_index) and (row_key not in self.row_keys())
        col_missing = (not col_is_index) and (col_key not in self.col_keys())

        if row_missing or col_missing:
            old_n, old_m, old_k = self._value.shape
            row_keys = list(self.row_keys())
            col_keys = list(self.col_keys())
            if row_missing:
                row_keys.append(row_key)
            if col_missing:
                col_keys.append(col_key)

            new_n = len(row_keys)
            new_m = len(col_keys)
            new_k = old_k
            new_xindex = self.xindex
            if old_k == 0:
                new_k = len(value)
                if hasattr(value, "xindex") and value.xindex is not None:
                    new_xindex = value.xindex
                elif new_xindex is None or len(new_xindex) != new_k:
                    new_xindex = np.arange(new_k)

            dtype = self._value.dtype if self._value.size else np.asarray(value.value).dtype
            new_value = np.zeros((new_n, new_m, new_k), dtype=dtype)
            if self._value.size:
                new_value[:old_n, :old_m, :old_k] = self._value

            new_meta = MetaDataMatrix(shape=(new_n, new_m))
            if old_n and old_m:
                new_meta[:old_n, :old_m] = self.meta

            new_rows = MetaDataDict(OrderedDict(self.rows))
            new_cols = MetaDataDict(OrderedDict(self.cols))
            if row_missing:
                new_rows[row_key] = MetaData()
            if col_missing:
                new_cols[col_key] = MetaData()

            try:
                self.resize(new_value.shape, refcheck=False)
                self[:] = new_value
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                self.__setstate__(new_value.__reduce__()[2])
            self._value = self.view(np.ndarray)
            self.xindex = new_xindex
            self.meta = new_meta
            self.rows = new_rows
            self.cols = new_cols

        i = self.row_index(row_key) if not row_is_index else row_key
        j = self.col_index(col_key) if not col_is_index else col_key
        if len(value) != self._value.shape[2]:
            raise ValueError("xindex length mismatch")
        if hasattr(value, "xindex") and self.xindex is not None:
            base_unit = getattr(self.xindex, "unit", None)

            def _to_base(x):
                if base_unit is None:
                    return np.asarray(x)
                return u.Quantity(x).to_value(base_unit)

            try:
                lhs = _to_base(value.xindex)
                rhs = _to_base(self.xindex)
                if not np.array_equal(lhs, rhs):
                    raise ValueError("Assigned Series has incompatible xindex")
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                raise ValueError("Assigned Series has incompatible xindex")
        self._value[i, j] = value.value
        self.meta[i, j] = MetaData(unit=value.unit, name=value.name, channel=value.channel)
       
    def plot(self, **kwargs):
        """
        Plot this SeriesMatrix using gwexpy.plot.Plot.
        """
        from gwexpy.plot import Plot
        return Plot(self, **kwargs)

    ##### as a Matrix #####
    def _all_element_units_equivalent(self):
        """
        Check whether all element units are mutually equivalent.
        """
        ref_unit = self.meta[0, 0].unit
        try:
            units = np.array(self.meta.units, dtype=object)
            if units.size == 0:
                return True, ref_unit
            def _eq(u_):
                try:
                    return u_.is_equivalent(ref_unit)
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    return False
            mask = np.vectorize(_eq)(units)
            if np.all(mask):
                return True, ref_unit
            return False, ref_unit
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            for meta in self.meta.flat:
                try:
                    if not meta.unit.is_equivalent(ref_unit):
                        return False, ref_unit
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    return False, ref_unit
            return True, ref_unit

    def _to_common_unit_values(self, ref_unit):
        """
        Convert all element values to a common reference unit.
        """
        N, M, K = self._value.shape
        units = np.array(self.meta.units, dtype=object)
        if units.size == 0:
            return np.array(self._value, copy=True)

        try:
            def _eq(u_):
                try:
                    return u_.is_equivalent(ref_unit)
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    return False
            mask = np.vectorize(_eq)(units)
            if np.all(mask):
                return u.Quantity(self._value, units[0, 0]).to_value(ref_unit)
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            pass

        result = np.empty((N, M, K), dtype=np.result_type(self._value, float))
        for i in range(N):
            for j in range(M):
                u_ij = self.meta[i, j].unit
                try:
                    vals = u.Quantity(self._value[i, j], u_ij).to_value(ref_unit)
                except Exception as e:
                    raise type(e)(f"Unit conversion failed at ({i},{j}): {e}")
                result[i, j] = vals
        return result

    @property
    def shape3D(self):
        return self._value.shape[0], self._value.shape[1], self.N_samples
        
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        np.copyto(self._value, new_value)

    @property
    def loc(self):
        class _LocAccessor:
            def __init__(self, parent):
                self._parent = parent

            def __getitem__(self, key):
                return self._parent._value[key]

            def __setitem__(self, key, value):
                self._parent._value[key] = value

        return _LocAccessor(self)

    def submatrix(self, row_keys, col_keys):
        row_indices = [self.row_index(k) for k in row_keys]
        col_indices = [self.col_index(k) for k in col_keys]
        new_data = self.value[np.ix_(row_indices, col_indices)]
        new_meta = MetaDataMatrix(self.meta[np.ix_(row_indices, col_indices)])
    
        new_rows = OrderedDict((k, self.rows[k]) for k in row_keys)
        new_cols = OrderedDict((k, self.cols[k]) for k in col_keys)
    
        return type(self)(new_data, xindex=self.xindex, name=self.name,
                            epoch=self.epoch, rows=new_rows, cols=new_cols,
                            meta=new_meta, attrs=self.attrs)

    def to_series_2Dlist(self):
        return [[self[row,col] for col in self.col_keys()] for row in self.row_keys()]

    def to_series_1Dlist(self):
        return [self[row,col] for col in self.col_keys() for row in self.row_keys()]

    ##### Mathematics #####
    def astype(self, dtype, copy=True):
        new_value = np.array(self.value, dtype=dtype, copy=copy)
        return self.__class__(new_value,
                              meta=self.meta.copy(),
                              xindex=self.xindex,
                              name=self.name,
                              epoch=self.epoch,
                              attrs=self.attrs,
                             )
    
    @property
    def real(self):
        new = (self + self.conj()) / 2
        return new.astype(float)
    
    @property
    def imag(self):
        new = (self - self.conj()) / (2j)
        return new.astype(float)

    def abs(self):
        new = abs(self)
        return new.astype(float)
        
    def angle(self, deg: bool = False):
        new = self.copy()
        new.meta = deepcopy(self.meta)
        new.value = np.angle(self.value, deg=deg)
        unit = u.deg if deg else u.rad
        for meta in new.meta.flat:
            meta["unit"] = unit
        return new.astype(float)

    @property
    def T(self):
        new_data = np.transpose(self._value, (1, 0, 2))
        new_meta = np.transpose(self.meta, (1, 0))
        return self.__class__(
            new_data,
            xindex=self.xindex,
            name=self.name,
            epoch=self.epoch,
            rows=self.cols,
            cols=self.rows,
            meta=MetaDataMatrix(new_meta),
            attrs=self.attrs,
        )

    def transpose(self):
        return self.T

    @property
    def dagger(self):
        return self.conj().T

    def trace(self):
        nrow, ncol, _ = self._value.shape
        if nrow != ncol:
            raise ValueError("trace requires a square matrix")
        ref_unit = self.meta[0, 0].unit
        diag_values = []
        for i in range(nrow):
            u_ii = self.meta[i, i].unit
            if not u_ii.is_equivalent(ref_unit):
                raise u.UnitConversionError(f"Diagonal units not equivalent: {u_ii} vs {ref_unit}")
            diag_values.append(u.Quantity(self._value[i, i], u_ii).to_value(ref_unit))
        summed = np.sum(diag_values, axis=0)
        try:
            xi = self.xindex.copy()
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            xi = deepcopy(self.xindex)
        name = f"trace({self.name})" if getattr(self, "name", "") else "trace"
        return Series(summed, xindex=xi, unit=ref_unit, name=name)

    def diagonal(self, output: str = "list"):
        nrow, ncol, nsamp = self._value.shape
        n = min(nrow, ncol)
        diag_series = []
        for i in range(n):
            meta = self.meta[i, i]
            try:
                xi = self.xindex.copy()
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                xi = deepcopy(self.xindex)
            diag_series.append(
                Series(
                    self._value[i, i],
                    xindex=xi,
                    unit=meta.unit,
                    name=meta.name,
                    channel=meta.channel,
                )
            )

        if output == "list":
            return diag_series

        if output == "vector":
            values = np.empty((n, 1, nsamp), dtype=self._value.dtype)
            meta_arr = np.empty((n, 1), dtype=object)
            for i in range(n):
                values[i, 0] = diag_series[i].value
                meta_arr[i, 0] = MetaData(**dict(self.meta[i, i]))
            rows_dict = _slice_metadata_dict(self.rows, list(range(n)), "row")
            cols_dict = MetaDataDict({"diag": MetaData()}, expected_size=1, key_prefix="col")
            return self.__class__(
                values,
                xindex=self.xindex,
                rows=rows_dict,
                cols=cols_dict,
                meta=MetaDataMatrix(meta_arr),
                name=getattr(self, "name", ""),
                epoch=getattr(self, "epoch", 0.0),
                attrs=getattr(self, "attrs", {}),
            )

        if output == "matrix":
            values = np.zeros_like(self._value)
            for i in range(n):
                values[i, i] = self._value[i, i]
            return self.__class__(
                values,
                xindex=self.xindex,
                rows=self.rows,
                cols=self.cols,
                meta=self.meta,
                name=getattr(self, "name", ""),
                epoch=getattr(self, "epoch", 0.0),
                attrs=getattr(self, "attrs", {}),
            )

        raise ValueError("output must be one of {'list', 'vector', 'matrix'}")

    def det(self):
        nrow, ncol, nsamp = self._value.shape
        if nrow != ncol:
            raise ValueError("det requires a square matrix")
        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError("All element units must be equivalent for det()")
        common = self._to_common_unit_values(ref_unit)
        mats = np.moveaxis(common, 2, 0)
        det_vals = np.linalg.det(mats)
        result_unit = ref_unit ** nrow
        try:
            xi = self.xindex.copy()
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            xi = deepcopy(self.xindex)
        name = f"det({self.name})" if getattr(self, "name", "") else "det"
        return Series(det_vals, xindex=xi, unit=result_unit, name=name)

    def inv(self, swap_rowcol: bool = True):
        nrow, ncol, nsamp = self._value.shape
        if nrow != ncol:
            raise ValueError("inv requires a square matrix")
        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError("All element units must be equivalent for inv()")
        common = self._to_common_unit_values(ref_unit)
        mats = np.moveaxis(common, 2, 0)
        inv_stack = np.linalg.inv(mats)
        inv_vals = np.moveaxis(inv_stack, 0, 2)

        inv_unit = ref_unit ** -1
        meta_arr = np.empty((nrow, ncol), dtype=object)
        for i in range(nrow):
            for j in range(ncol):
                meta_arr[i, j] = MetaData(unit=inv_unit, name="", channel=None)
        meta_matrix = MetaDataMatrix(meta_arr)

        def _copy_meta_dict(md: MetaDataDict, prefix: str):
            items = OrderedDict()
            for k, v in md.items():
                items[k] = MetaData(**dict(v))
            return MetaDataDict(items, expected_size=len(md), key_prefix=prefix)

        rows_out = _copy_meta_dict(self.cols, "row") if swap_rowcol else _copy_meta_dict(self.rows, "row")
        cols_out = _copy_meta_dict(self.rows, "col") if swap_rowcol else _copy_meta_dict(self.cols, "col")

        return self.__class__(
            inv_vals,
            xindex=self.xindex,
            rows=rows_out,
            cols=cols_out,
            meta=meta_matrix,
            name=f"inv({self.name})" if getattr(self, "name", "") else "inv",
            epoch=getattr(self, "epoch", 0.0),
            attrs=getattr(self, "attrs", {}),
        )

    def schur(self, keep_rows, keep_cols=None, eliminate_rows=None, eliminate_cols=None):
        nrow, ncol, nsamp = self._value.shape
        if keep_cols is None:
            keep_cols = keep_rows

        def _row_idx(k):
            return int(k) if isinstance(k, (int, np.integer)) else self.row_index(k)

        def _col_idx(k):
            return int(k) if isinstance(k, (int, np.integer)) else self.col_index(k)

        all_row_idx = list(range(nrow))
        all_col_idx = list(range(ncol))
        keep_rows_idx = [_row_idx(k) for k in keep_rows]
        keep_cols_idx = [_col_idx(k) for k in keep_cols]
        if eliminate_rows is None:
            eliminate_rows_idx = [i for i in all_row_idx if i not in keep_rows_idx]
        else:
            eliminate_rows_idx = [_row_idx(k) for k in eliminate_rows]
        if eliminate_cols is None:
            eliminate_cols_idx = [j for j in all_col_idx if j not in keep_cols_idx]
        else:
            eliminate_cols_idx = [_col_idx(k) for k in eliminate_cols]

        if len(eliminate_rows_idx) != len(eliminate_cols_idx):
            raise ValueError("Eliminated row/col sets must have the same size for Schur complement")
        if not keep_rows_idx or not keep_cols_idx:
            raise ValueError("Keep sets must be non-empty")

        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError("All element units must be equivalent for schur()")
        common = self._to_common_unit_values(ref_unit)

        r_keep = len(keep_rows_idx)
        c_keep = len(keep_cols_idx)

        if len(eliminate_rows_idx) == 0:
            result_vals = common[np.ix_(keep_rows_idx, keep_cols_idx)]
        else:
            stack = np.moveaxis(common, 2, 0)
            A = np.take(np.take(stack, keep_rows_idx, axis=1), keep_cols_idx, axis=2)
            B = np.take(np.take(stack, keep_rows_idx, axis=1), eliminate_cols_idx, axis=2)
            C = np.take(np.take(stack, eliminate_rows_idx, axis=1), keep_cols_idx, axis=2)
            D = np.take(np.take(stack, eliminate_rows_idx, axis=1), eliminate_cols_idx, axis=2)

            D_inv = np.linalg.inv(D)
            schur_block = A - np.matmul(np.matmul(B, D_inv), C)
            result_vals = np.moveaxis(schur_block, 0, 2)

        meta_arr = np.empty((r_keep, c_keep), dtype=object)
        for ii, ri in enumerate(keep_rows_idx):
            for jj, cj in enumerate(keep_cols_idx):
                base_meta = self.meta[ri, cj]
                meta_arr[ii, jj] = MetaData(unit=ref_unit, name=base_meta.name, channel=base_meta.channel)

        def _subset_meta_dict(md: MetaDataDict, indices, prefix):
            items = OrderedDict()
            keys = list(md.keys())
            for idx in indices:
                key = keys[idx]
                items[key] = MetaData(**dict(md[key]))
            return MetaDataDict(items, expected_size=len(indices), key_prefix=prefix)

        rows_out = _subset_meta_dict(self.rows, keep_rows_idx, "row")
        cols_out = _subset_meta_dict(self.cols, keep_cols_idx, "col")

        return self.__class__(
            result_vals,
            xindex=self.xindex,
            rows=rows_out,
            cols=cols_out,
            meta=MetaDataMatrix(meta_arr),
            name=f"schur({self.name})" if getattr(self, "name", "") else "schur",
            epoch=getattr(self, "epoch", 0.0),
            attrs=getattr(self, "attrs", {}),
        )

    ##### Edit forrwing the Sampling axis   ##### 
    def crop(self, start=None, end=None, copy=False):
        xindex = self.xindex
        xunit = getattr(xindex, "unit", None)
        xvalues = getattr(xindex, "value", np.asarray(xindex))

        def _as_base(val):
            if val is None:
                return None
            if xunit is not None:
                return u.Quantity(val, xunit).to_value(xunit)
            return u.Quantity(val).value if isinstance(val, u.Quantity) else val

        start_val = _as_base(start)
        end_val = _as_base(end)

        idx0 = np.searchsorted(xvalues, start_val, side='left') if start_val is not None else 0
        idx1 = np.searchsorted(xvalues, end_val, side='left') if end_val is not None else len(xvalues)
        new_data = self.value[:, :, idx0:idx1]
        if copy:
            new_data = np.array(new_data, copy=True)
        new_xindex = xindex[idx0:idx1]
        return self.__class__(
            new_data,
            xindex=new_xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs
        )

    def append(self, other, inplace=True, pad=None, gap=None, resize=True):
        if gap is None:
            gap = 'raise' if pad is None else 'pad'
        if pad is None and gap == 'pad':
            pad = 0.0

        target = self

        base_unit = getattr(target.xindex, "unit", getattr(other.xindex, "unit", None))

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return u.Quantity(val, base_unit).to_value(base_unit)

        def _is_regular(xidx):
            if hasattr(xidx, "regular"):
                return xidx.regular
            try:
                arr = np.asarray(xidx, dtype=float)
                if arr.size < 2:
                    return True
                diffs = np.diff(arr)
                return np.allclose(diffs, diffs[0])
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                return False

        def _concat_ignore(a, b) -> "SeriesMatrix":
            new_data = np.concatenate([a.value, b.value], axis=2)
            if base_unit is None:
                new_xindex = np.concatenate([np.asarray(a.xindex), np.asarray(b.xindex)])
            else:
                ax = u.Quantity(a.xindex).to_value(base_unit)
                bx = u.Quantity(b.xindex).to_value(base_unit)
                new_xindex = np.concatenate([ax, bx]) * base_unit
            return self.__class__(
                new_data,
                xindex=new_xindex,
                rows=a.rows,
                cols=a.cols,
                meta=a.meta,
                name=a.name,
                epoch=a.epoch,
                attrs=a.attrs,
            )

        cont = target.is_contiguous(other)
        if cont != 1:
            _gap_is_numeric = isinstance(gap, (int, float, np.number, u.Quantity))
            _use_pad = (gap == 'pad') or (_gap_is_numeric and pad is not None)
            if _use_pad:
                if not _is_regular(target.xindex):
                    raise ValueError("Padding gap requires regular xindex")
                s0, s1 = target.xspan
                o0, _ = other.xspan
                s0b = _to_base(s0)
                s1b = _to_base(s1)
                o0b = _to_base(o0)
                if s0b <= o0b < s1b:
                    raise ValueError("Cannot append overlapping SeriesMatrix")
                if o0b < s0b:
                    raise ValueError("Cannot append that starts before this one")

                gap_base = np.inf if gap == "pad" else _to_base(gap)
                out_full = target.append_exact(other, inplace=False, pad=pad, gap=gap_base)
            elif gap == 'ignore':
                out_full = _concat_ignore(target, other)
            else:
                s0, s1 = target.xspan
                o0, _ = other.xspan
                if _to_base(s0) < _to_base(o0) < _to_base(s1):
                    raise ValueError("Cannot append overlapping SeriesMatrix")
                raise ValueError("Cannot append discontiguous SeriesMatrix")
        else:
            out_full = target.append_exact(other, inplace=False, pad=None, gap=None)

        if not resize:
            keep = target._value.shape[2]
            if out_full._value.shape[2] > keep:
                out_full = out_full[:, :, -keep:]

        if not inplace:
            return out_full

        try:
            self.resize(out_full.shape, refcheck=False)
            self[:] = out_full[:]
            self.xindex = out_full.xindex
            self.meta = out_full.meta
            self.rows = out_full.rows
            self.cols = out_full.cols
            self.name = out_full.name
            self.epoch = out_full.epoch
            self.attrs = out_full.attrs
            self._value = self.view(np.ndarray)
            return self
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return out_full

    def append_exact(self, other, inplace=False, pad=None, gap=None, tol=1/2.**18):
        if self._value.shape[:2] != other._value.shape[:2]:
            raise ValueError(f"shape[:2] does not match: {self._value.shape[:2]} vs {other._value.shape[:2]}")
        if self._value.ndim != other._value.ndim:
            raise ValueError(f"ndim does not match: {self._value.ndim} vs {other._value.ndim}")

        base_unit = getattr(self.xindex, "unit", None)
        if base_unit is None:
            base_unit = getattr(other.xindex, "unit", None)

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return float(np.asarray(val))

        if hasattr(self.xindex, "regular") and self.xindex.regular:
            dx_q = self.dx
        else:
            if len(self.xindex) < 2:
                raise ValueError("Cannot infer dx from xindex with length < 2")
            dx_q = self.xindex[-1] - self.xindex[-2]
        dx_self = _to_base(dx_q)
        self_last = _to_base(self.xindex[-1])
        other_first = _to_base(other.xindex[0])
        diff = other_first - self_last

        if np.isclose(diff, dx_self, atol=tol) or abs(diff) < tol:
            new_data = np.concatenate([self.value, other.value], axis=2)
            if hasattr(self.xindex, 'regular') and self.xindex.regular:
                new_xindex = Index.define(self.xindex[0], dx_q, new_data.shape[2])
            else:
                new_xindex = np.concatenate([self.xindex, other.xindex])
        elif gap is not None and abs(diff - dx_self) <= gap:
            n_gap = int(np.round((diff - dx_self) / dx_self))
            pad_value = np.nan if pad is None or pad == "nan" else pad
            pad_block = np.full((self._value.shape[0], self._value.shape[1], n_gap), pad_value, dtype=self._value.dtype)
            if base_unit is not None:
                pad_times = self_last + dx_self * np.arange(1, n_gap+1)
                pad_xindex = pad_times * base_unit
            else:
                pad_xindex = self_last + dx_self * np.arange(1, n_gap+1)
            new_data = np.concatenate([self.value, pad_block, other.value], axis=2)
            if hasattr(self.xindex, 'regular') and self.xindex.regular:
                new_xindex = Index.define(self.xindex[0], dx_q, new_data.shape[2])
            else:
                new_xindex = np.concatenate([self.xindex, pad_xindex, other.xindex])
        else:
            raise ValueError(
                f"gap detected: {diff} [{base_unit if base_unit is not None else 'dimensionless'}], but gap={gap} specified."
            )
        result = self.__class__(
            new_data,
            xindex=new_xindex,
            rows=self.rows,
            cols=self.cols,
            meta=self.meta,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs
        )
        if inplace:
            self.value = result.value
            self.xindex = result.xindex
            return self
        return result

    def prepend(self, other, inplace=True, pad=None, gap=None, resize=True):
        out = other.append(self, inplace=False, pad=pad, gap=gap, resize=resize)
        if not inplace:
            return out
        try:
            self.resize(out.shape, refcheck=False)
            self[:] = out[:]
            self.xindex = out.xindex
            self.meta = out.meta
            self.rows = out.rows
            self.cols = out.cols
            self.name = out.name
            self.epoch = out.epoch
            self.attrs = out.attrs
            self._value = self.view(np.ndarray)
            return self
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return out

    def prepend_exact(self, other, inplace=False, pad=None, gap=None, tol=1/2.**18):
        return other.append_exact(self, inplace=inplace, pad=pad, gap=gap, tol=tol)

    def update(self, other, inplace=True, pad=None, gap=None):
        return self.append(other, inplace=inplace, pad=pad, gap=gap, resize=False)

    def diff(self, n=1, axis=2):
        if axis != 2:
            raise ValueError("SeriesMatrix.diff supports only axis=2 (sample axis)")
        if n < 0:
            raise ValueError("n must be non-negative")
        new_values = np.diff(self.value, n=n, axis=2)
        if hasattr(self.xindex, "__len__") and len(self.xindex) > n:
            try:
                dx = self.dx
                new_xindex = Index.define(self.xindex[n], dx, new_values.shape[2])
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                new_xindex = self.xindex[n:]
        else:
            new_xindex = self.xindex
        return self.__class__(
            new_values,
            xindex=new_xindex,
            meta=self.meta,
            rows=self.rows,
            cols=self.cols,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs,
        )

    def value_at(self, x):
        xidx = self.xindex
        base_unit = getattr(xidx, "unit", None)
        if base_unit is not None:
            x_val = u.Quantity(x, base_unit).to_value(base_unit)
            x_array = u.Quantity(xidx).to_value(base_unit)
        else:
            x_val = float(u.Quantity(x).value) if isinstance(x, u.Quantity) else x
            x_array = np.asarray(xidx)
        matches = np.nonzero(x_array == x_val)[0]
        if len(matches) == 0:
            raise ValueError(f"Value {x} not found in xindex")
        idx = matches[0]
        return self.value[:, :, idx]

    def pad(self, pad_width, **kwargs):
        kwargs.setdefault("mode", "constant")
        if isinstance(pad_width, int):
            before = after = pad_width
        else:
            try:
                before, after = pad_width
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                raise ValueError("pad_width must be int or tuple of (before, after)")
        pad_spec = ((0, 0), (0, 0), (before, after))
        new_values = np.pad(self.value, pad_spec, **kwargs)

        dx = self.dx
        start = self.xindex[0] - dx * before
        new_len = new_values.shape[2]
        new_xindex = Index.define(start, dx, new_len)

        result = self.__class__(
            new_values,
            xindex=new_xindex,
            meta=self.meta,
            rows=self.rows,
            cols=self.cols,
            name=self.name,
            epoch=self.epoch,
            attrs=self.attrs,
        )
        return result

    def shift(self, delta):
        xidx = self.xindex
        xunit = getattr(xidx, "unit", None)
        if xunit is not None:
            delta_val = u.Quantity(delta, xunit)
            new_xindex = xidx + delta_val
        else:
            delta_val = float(u.Quantity(delta).value) if isinstance(delta, u.Quantity) else delta
            new_xindex = np.asarray(xidx) + delta_val
        self.xindex = new_xindex
        try:
            self._x0 = new_xindex[0]
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            pass
        return self

    def copy(self, order='C'):
        new_values = np.array(self._value, copy=True, order=order)
        try:
            xindex_copy = self.xindex.copy()
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            xindex_copy = deepcopy(self.xindex)

        def _copy_meta_dict(md: MetaDataDict, prefix: str):
            items = OrderedDict()
            for k, v in md.items():
                items[k] = MetaData(**dict(v))
            return MetaDataDict(items, expected_size=len(md), key_prefix=prefix)

        rows_copy = _copy_meta_dict(self.rows, "row")
        cols_copy = _copy_meta_dict(self.cols, "col")

        meta_arr = np.empty(self.meta.shape, dtype=object)
        for idx, m in enumerate(self.meta.flat):
            meta_arr.flat[idx] = MetaData(**dict(m))
        meta_copy = MetaDataMatrix(meta_arr)

        attrs_copy = deepcopy(getattr(self, "attrs", {}))

        return self.__class__(
            new_values,
            xindex=xindex_copy,
            meta=meta_copy,
            rows=rows_copy,
            cols=cols_copy,
            name=self.name,
            epoch=self.epoch,
            attrs=attrs_copy,
        )

    def step(self, where="post", **kwargs):
        kwargs.setdefault("drawstyle", f"steps-{where}")
        return self.plot(method="plot", **kwargs)

    # -- I/O (HDF5) -------------------------------------------------
    def to_pandas(self, format="wide"):
        pd = require_optional("pandas")
        if format == "wide":
            N, M, K = self._value.shape
            val_T = np.moveaxis(self._value, -1, 0)
            val_flat = val_T.reshape(K, -1)
            r_keys = list(self.row_keys())
            c_keys = list(self.col_keys())
            col_names = [f"{r}_{c}" for r in r_keys for c in c_keys]
            xidx = self.xindex
            idx_name = "index"
            if isinstance(xidx, u.Quantity):
                idx_name = f"index [{xidx.unit}]"
                xidx = xidx.value
            df = pd.DataFrame(val_flat, index=xidx, columns=col_names)
            df.index.name = idx_name
            return df
        elif format == "long":
            N, M, K = self._value.shape
            r_keys = list(self.row_keys())
            c_keys = list(self.col_keys())
            xidx = self.xindex
            if isinstance(xidx, u.Quantity):
                xidx = xidx.value
            long_index = np.tile(xidx, N * M)
            val_list = []
            row_list = []
            col_list = []
            for r in r_keys:
                for c in c_keys:
                    i = self.row_index(r)
                    j = self.col_index(c)
                    val_list.append(self._value[i, j])
                    row_list.extend([r] * K)
                    col_list.extend([c] * K)
            long_values = np.concatenate(val_list)
            df = pd.DataFrame({
                "index": long_index,
                "row": row_list,
                "col": col_list,
                "value": long_values
            })
            return df
        else:
            raise ValueError(f"Unknown format: {format}")

    def write(self, target, format=None, **kwargs):
        from pathlib import Path
        if format is None:
            ext = Path(target).suffix.lower()
            if ext in [".h5", ".hdf5", ".hdf"]:
                format = "hdf5"
            elif ext == ".csv":
                format = "csv"
            elif ext in [".parquet", ".pq"]:
                format = "parquet"
            else:
                format = "hdf5"
        if format == "hdf5":
            return self.to_hdf5(target, **kwargs)
        elif format == "csv":
            df = self.to_pandas(format="wide")
            return df.to_csv(target, **kwargs)
        elif format == "parquet":
            df = self.to_pandas(format="wide")
            return df.to_parquet(target, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def to_hdf5(self, filepath, **kwargs):
        import h5py
        with h5py.File(filepath, "w", **kwargs) as f:
            f.attrs["name"] = str(getattr(self, "name", ""))
            try:
                f.attrs["epoch"] = float(getattr(self, "epoch", 0.0))
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                pass
            attrs_dict = getattr(self, "attrs", None)
            if attrs_dict is not None:
                try:
                    f.attrs["attrs_json"] = json.dumps(attrs_dict)
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    pass
            f.create_dataset("data", data=self.value)
            grp_x = f.create_group("xindex")
            if isinstance(self.xindex, u.Quantity):
                grp_x.create_dataset("value", data=np.asarray(self.xindex.value))
                grp_x.attrs["unit"] = str(self.xindex.unit)
            else:
                grp_x.create_dataset("value", data=np.asarray(self.xindex))
            meta_grp = f.create_group("meta")
            units = np.vectorize(lambda u_: "" if u_ is None else str(u_))(self.meta.units)
            names = np.vectorize(lambda n: "" if n is None else str(n))(self.meta.names)
            channels = np.vectorize(lambda c: "" if c is None else str(c))(self.meta.channels)
            meta_grp.create_dataset("units", data=units.astype("S"))
            meta_grp.create_dataset("names", data=names.astype("S"))
            meta_grp.create_dataset("channels", data=channels.astype("S"))
            row_grp = f.create_group("rows")
            row_grp.create_dataset("keys", data=np.array(list(self.rows.keys()), dtype="S"))
            row_grp.create_dataset("names", data=np.array([str(v.name) for v in self.rows.values()], dtype="S"))
            row_grp.create_dataset("units", data=np.array([str(v.unit) for v in self.rows.values()], dtype="S"))
            row_grp.create_dataset("channels", data=np.array([str(v.channel) for v in self.rows.values()], dtype="S"))
            col_grp = f.create_group("cols")
            col_grp.create_dataset("keys", data=np.array(list(self.cols.keys()), dtype="S"))
            col_grp.create_dataset("names", data=np.array([str(v.name) for v in self.cols.values()], dtype="S"))
            col_grp.create_dataset("units", data=np.array([str(v.unit) for v in self.cols.values()], dtype="S"))
            col_grp.create_dataset("channels", data=np.array([str(v.channel) for v in self.cols.values()], dtype="S"))

    ##### Visualizations #####
    def __repr__(self): 
        try:
            return f"<SeriesMatrix shape={self.shape3D} rows={self.row_keys()} cols={self.col_keys()}>"
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return "<SeriesMatrix (incomplete or empty)>"

    def __str__(self):
        info = (
            f"SeriesMatrix(shape={self._value.shape},  name='{self.name}')\n"
            f"  epoch   : {self.epoch}\n"
            f"  x0      : {self.x0}\n"
            f"  dx      : {self.dx}\n"
            f"  xunit   : {self.xunit}\n"
            f"  samples : {self.N_samples}\n"
        )
        info += "\n[ Row metadata ]\n" + str(self.rows)
        info += "\n\n[ Column metadata ]\n" + str(self.cols)
        if hasattr(self, 'meta'):
            info += "\n\n[ Elements metadata ]\n" + str(self.meta)
        return info
    
    def _repr_html_(self):
        html = f"<h3>SeriesMatrix: shape={self._value.shape}, name='{json.dumps(str(self.name))[1:-1]}'</h3>"
        html += f"<ul><li><b>epoch:</b> {self.epoch}</li><li><b>x0:</b> {self.x0}, <b>dx:</b> {self.dx}, <b>N_samples:</b> {self.N_samples}</li><li><b>xunit:</b> {self.xunit}</li></ul>"
        html += "<h4>Row Metadata</h4>" + self.rows._repr_html_()
        html += "<h4>Column Metadata</h4>" + self.cols._repr_html_()
        if hasattr(self, 'meta'):
            html += "<h4>Element Metadata</h4>" + self.meta._repr_html_()
        if self.attrs:
            html += f"<h4>Attributes</h4><pre>{json.dumps(self.attrs, indent=2)}</pre>"
        return html

    def is_contiguous(self, other, tol=1/2.**18):
        """
        Check contiguity using xspan endpoints (gwpy-like semantics).
        """
        if not isinstance(other, type(self).__mro__[1]):  # SeriesMatrix base
            raise TypeError("is_contiguous expects SeriesMatrix")
        self.is_compatible(other)

        base_unit = getattr(self.xindex, "unit", getattr(other.xindex, "unit", None))

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return float(np.asarray(val))

        span_self = self.xspan
        span_other = other.xspan
        diff1 = _to_base(span_self[1]) - _to_base(span_other[0])
        diff2 = _to_base(span_other[1]) - _to_base(span_self[0])
        if abs(diff1) < tol:
            return 1
        elif abs(diff2) < tol:
            return -1
        else:
            return 0

    def is_contiguous_exact(self, other, tol=1/2.**18):
        """
        Strict contiguity check (requires identical shape and matching endpoints).
        """
        if self._value.shape != other._value.shape:
            raise ValueError(f"shape does not match: {self._value.shape} vs {other._value.shape}")

        base_unit = getattr(self.xindex, "unit", None)
        if base_unit is None:
            base_unit = getattr(other.xindex, "unit", None)

        def _to_base(val):
            if base_unit is None:
                if isinstance(val, u.Quantity):
                    return float(val.value)
                return float(np.asarray(val))
            if isinstance(val, u.Quantity):
                return val.to_value(base_unit)
            return float(np.asarray(val))

        xspan_self = self.xspan
        xspan_other = other.xspan
        diff1 = _to_base(xspan_self[1]) - _to_base(xspan_other[0])
        diff2 = _to_base(xspan_other[1]) - _to_base(xspan_self[0])
        if abs(diff1) < tol:
            return 1
        elif abs(diff2) < tol:
            return -1
        else:
            return 0

    def is_compatible_exact(self, other):
        """
        Strict compatibility: requires identical shape, xindex, and metadata units.
        """
        base_unit = getattr(self.xindex, "unit", None)
        lhs = self.xindex
        rhs = other.xindex
        if base_unit is not None:
            try:
                lhs = lhs.to_value(base_unit)
                rhs = rhs.to_value(base_unit)
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                pass
        try:
            equal = np.array_equal(lhs, rhs)
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            equal = lhs == rhs
        if not equal:
            raise ValueError(f"xindex does not match: {self.xindex} vs {other.xindex}")

        if self._value.shape != other._value.shape:
            raise ValueError(f"shape does not match: {self._value.shape} vs {other._value.shape}")

        if list(self.rows.keys()) != list(other.rows.keys()):
            raise ValueError("row keys do not match")
        if list(self.cols.keys()) != list(other.cols.keys()):
            raise ValueError("col keys do not match")

        for (k1, meta1), (k2, meta2) in zip(self.rows.items(), other.rows.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"row {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
        for (k1, meta1), (k2, meta2) in zip(self.cols.items(), other.cols.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(f"col {k1} unit does not match: {meta1.unit} vs {meta2.unit}")
        return True

    @classmethod
    def read(cls, source, format=None, **kwargs):
        """
        Read a SeriesMatrix from file.
        
        Parameters
        ----------
        source : str or path-like
            Path to file to read.
        format : str, optional
            File format. If None, inferred from extension.
        **kwargs
            Additional arguments passed to the reader.
            
        Returns
        -------
        SeriesMatrix
            The loaded matrix.
        """
        import h5py
        from pathlib import Path
        
        if format is None:
            ext = Path(source).suffix.lower()
            if ext in [".h5", ".hdf5", ".hdf"]:
                format = "hdf5"
            else:
                format = "hdf5"
                
        if format != "hdf5":
            raise NotImplementedError(f"Format {format} is not supported for read")
            
        with h5py.File(source, "r") as f:
            data = f["data"][:]
            
            grp_x = f["xindex"]
            xindex_vals = grp_x["value"][:]
            xunit_str = grp_x.attrs.get("unit", None)
            if xunit_str:
                xindex = u.Quantity(xindex_vals, xunit_str)
            else:
                xindex = xindex_vals
                
            name = f.attrs.get("name", "")
            try:
                epoch = f.attrs.get("epoch", 0.0)
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                epoch = 0.0
                
            attrs_json = f.attrs.get("attrs_json", None)
            if attrs_json:
                try:
                    attrs = json.loads(attrs_json)
                except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                    attrs = {}
            else:
                attrs = {}
                
            meta_grp = f["meta"]
            units_raw = meta_grp["units"][:].astype(str)
            names_raw = meta_grp["names"][:].astype(str)
            channels_raw = meta_grp["channels"][:].astype(str)
            
            N, M = units_raw.shape
            meta_arr = np.empty((N, M), dtype=object)
            for i in range(N):
                for j in range(M):
                    unit_str = units_raw[i, j]
                    unit_val = u.Unit(unit_str) if unit_str else u.dimensionless_unscaled
                    meta_arr[i, j] = MetaData(
                        unit=unit_val,
                        name=names_raw[i, j] if names_raw[i, j] else None,
                        channel=channels_raw[i, j] if channels_raw[i, j] else None
                    )
            meta_matrix = MetaDataMatrix(meta_arr)
            
            row_grp = f["rows"]
            row_keys = [k.decode() if isinstance(k, bytes) else k for k in row_grp["keys"][:]]
            row_names = [n.decode() if isinstance(n, bytes) else n for n in row_grp["names"][:]]
            row_units = [u_.decode() if isinstance(u_, bytes) else u_ for u_ in row_grp["units"][:]]
            row_channels = [c.decode() if isinstance(c, bytes) else c for c in row_grp["channels"][:]]
            rows = OrderedDict()
            for k, n, u_, c in zip(row_keys, row_names, row_units, row_channels):
                rows[k] = MetaData(
                    unit=u.Unit(u_) if u_ else u.dimensionless_unscaled,
                    name=n if n else None,
                    channel=c if c else None
                )
            rows = MetaDataDict(rows, expected_size=len(row_keys), key_prefix="row")
            
            col_grp = f["cols"]
            col_keys = [k.decode() if isinstance(k, bytes) else k for k in col_grp["keys"][:]]
            col_names = [n.decode() if isinstance(n, bytes) else n for n in col_grp["names"][:]]
            col_units = [u_.decode() if isinstance(u_, bytes) else u_ for u_ in col_grp["units"][:]]
            col_channels = [c.decode() if isinstance(c, bytes) else c for c in col_grp["channels"][:]]
            cols = OrderedDict()
            for k, n, u_, c in zip(col_keys, col_names, col_units, col_channels):
                cols[k] = MetaData(
                    unit=u.Unit(u_) if u_ else u.dimensionless_unscaled,
                    name=n if n else None,
                    channel=c if c else None
                )
            cols = MetaDataDict(cols, expected_size=len(col_keys), key_prefix="col")
            
        return cls(
            data,
            xindex=xindex,
            meta=meta_matrix,
            rows=rows,
            cols=cols,
            name=name,
            epoch=epoch,
            attrs=attrs
        )
