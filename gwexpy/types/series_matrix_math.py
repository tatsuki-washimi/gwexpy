from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from astropy import units as u

from .metadata import MetaData, MetaDataMatrix


class SeriesMatrixMathMixin:
    """Mixin for SeriesMatrix math operations (linear algebra)."""

    if TYPE_CHECKING:
        from gwpy.types.index import Index

        from .metadata import MetaDataDict

        _value: np.ndarray
        meta: MetaDataMatrix
        rows: MetaDataDict
        cols: MetaDataDict
        name: str | None
        epoch: float | int | None
        attrs: dict[str, Any] | None
        dx: Any
        xspan: Any
        series_class: type[Any] | None

        @property
        def xindex(self) -> np.ndarray | u.Quantity | Index | None: ...

        @xindex.setter
        def xindex(self, value: np.ndarray | u.Quantity | Index | None) -> None: ...

        def row_index(self, key: Any) -> int: ...
        def col_index(self, key: Any) -> int: ...

    def _all_element_units_equivalent(self) -> tuple[bool, u.Unit | None]:
        """Check whether all element units are mutually equivalent."""
        units = self.meta.units
        if units.size == 0:
            return True, None
        first = units[0, 0]
        if first is None:
            first = u.dimensionless_unscaled

        def _eq(u_):
            if u_ is None:
                u_ = u.dimensionless_unscaled
            return u_.is_equivalent(first)

        v_eq = np.vectorize(_eq)
        if np.all(v_eq(units)):
            return True, first
        return False, None

    def _to_common_unit_values(self, ref_unit: u.Unit) -> np.ndarray:
        """Convert all element values to a common reference unit."""
        ref_unit = u.Unit(ref_unit)
        N, M, K = self._value.shape
        out = np.empty((N, M, K), dtype=self._value.dtype)

        units = self.meta.units

        def _eq(u_):
            if u_ is None:
                u_ = u.dimensionless_unscaled
            return u_ == ref_unit

        v_eq = np.vectorize(_eq)
        if np.all(v_eq(units)):
            return self._value.copy()

        for i in range(N):
            for j in range(M):
                u_ij = units[i, j]
                if u_ij is None:
                    u_ij = u.dimensionless_unscaled
                if u_ij == ref_unit:
                    out[i, j] = self._value[i, j]
                else:
                    out[i, j] = u.Quantity(self._value[i, j], u_ij).to_value(ref_unit)
        return out

    @staticmethod
    def _invert_with_rescale(mat: np.ndarray) -> np.ndarray:
        """Invert with preconditioning if the direct inverse is singular."""
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            sigma = np.nanmax(np.abs(mat))
            if not np.isfinite(sigma) or sigma == 0:
                raise
            mat_scaled = mat / sigma
            eye = np.eye(mat.shape[0], dtype=mat.dtype)
            inv_scaled = np.linalg.solve(mat_scaled, eye)
            return inv_scaled / sigma

    @classmethod
    def _invert_stack_with_rescale(cls, mats: np.ndarray) -> np.ndarray:
        inv_stack = np.empty_like(mats)
        for idx, mat in enumerate(mats):
            inv_stack[idx] = cls._invert_with_rescale(mat)
        return inv_stack

    def __matmul__(self, other):
        """
        Matrix multiplication (broadcasting over sample axis).
        """
        if not isinstance(other, type(self)):
            # If other is not a SeriesMatrix, try to use ndarray matmul
            return np.matmul(self, other)  # type: ignore[call-overload]

        if self._value.shape[2] != other._value.shape[2]:
            raise ValueError("Sample axis length mismatch in matrix multiplication")
        if self._value.shape[1] != other._value.shape[0]:
            raise ValueError(
                f"Matrix dimension mismatch: ({self._value.shape[0]}, {self._value.shape[1]}) @ ({other._value.shape[0]}, {other._value.shape[1]})"
            )

        # Move sample axis to front for np.matmul broadcasting
        a = np.moveaxis(self._value, 2, 0)
        b = np.moveaxis(other._value, 2, 0)
        res_stack = np.matmul(a, b)
        res_vals = np.moveaxis(res_stack, 0, 2)

        N = self._value.shape[0]
        M = other._value.shape[1]
        K = self._value.shape[1]

        # Compute metadata (units)
        # Result unit at (i, j) is sum_k (self[i, k].unit * other[k, j].unit)
        # We assume for each (i, j), the units for all k are equivalent.
        res_meta: np.ndarray = np.empty((N, M), dtype=object)
        for i in range(N):
            for j in range(M):
                # Calculate the unit of the first term k=0
                u0 = self.meta[i, 0].unit * other.meta[0, j].unit
                # Check consistency and assign
                for k in range(1, K):
                    uk = self.meta[i, k].unit * other.meta[k, j].unit
                    if not uk.is_equivalent(u0):
                        raise u.UnitConversionError(
                            f"Inconsistent units in matrix multiplication at result ({i},{j}): term 0 has {u0}, term {k} has {uk}"
                        )
                res_meta[i, j] = MetaData(unit=u0)

        matrix_cls = cast(type[Any], type(self))
        return matrix_cls(
            res_vals,
            xindex=self.xindex,
            rows=self.rows,
            cols=other.cols,
            meta=MetaDataMatrix(res_meta),
            name=f"({getattr(self, 'name', '')} @ {getattr(other, 'name', '')})",
            epoch=getattr(self, "epoch", 0.0),
            attrs=getattr(self, "attrs", {}),
        )

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """Compute the trace of the matrix (sum of diagonal elements)."""
        if offset != 0 or axis1 != 0 or axis2 != 1:
            raise NotImplementedError(
                "trace currently supports only offset=0, axis1=0, axis2=1"
            )
        if out is not None:
            raise NotImplementedError("trace does not support the 'out' argument")
        nrow, ncol, _ = self._value.shape
        if nrow != ncol:
            raise ValueError("trace requires a square matrix")
        ref_unit = self.meta[0, 0].unit
        diag_values = []
        for i in range(nrow):
            u_ii = self.meta[i, i].unit
            if not u_ii.is_equivalent(ref_unit):
                raise u.UnitConversionError(
                    f"Diagonal units not equivalent: {u_ii} vs {ref_unit}"
                )
            diag_values.append(u.Quantity(self._value[i, i], u_ii).to_value(ref_unit))
        summed = np.sum(diag_values, axis=0)
        if dtype is not None:
            summed = np.asarray(summed, dtype=dtype)

        # Result is a Series. Need to find base Series class.
        series_cls = getattr(self, "series_class", None)
        if series_cls is None:
            from gwpy.types.series import Series as _Series

            series_cls = _Series

        name = f"trace({self.name})" if getattr(self, "name", "") else "trace"
        return series_cls(summed, xindex=self.xindex, unit=ref_unit, name=name)

    def diagonal(self, offset=0, axis1=0, axis2=1, **kwargs):
        """Extract diagonal elements from the matrix."""
        output = kwargs.pop("output", "list")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        if offset != 0 or axis1 != 0 or axis2 != 1:
            raise NotImplementedError(
                "diagonal currently supports only offset=0, axis1=0, axis2=1"
            )
        from .metadata import MetaDataDict

        nrow, ncol, nsamp = self._value.shape
        n = min(nrow, ncol)
        diag_series = []

        series_cls = getattr(self, "series_class", None)
        if series_cls is None:
            from gwpy.types.series import Series as _Series

            series_cls = _Series

        for i in range(n):
            meta = self.meta[i, i]
            diag_series.append(
                series_cls(
                    self._value[i, i],
                    xindex=self.xindex,
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
            from .seriesmatrix_validation import _slice_metadata_dict

            rows_dict = _slice_metadata_dict(self.rows, list(range(n)), "row")
            cols_dict = MetaDataDict(
                {"diag": MetaData()}, expected_size=1, key_prefix="col"
            )
            matrix_cls = cast(type[Any], self.__class__)
            return matrix_cls(
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
            matrix_cls = cast(type[Any], self.__class__)
            return matrix_cls(
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
        """Compute the determinant of the matrix at each sample point."""
        nrow, ncol, nsamp = self._value.shape
        if nrow != ncol:
            raise ValueError("det requires a square matrix")

        # These helpers must be in the mixin or base class
        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError(
                "All element units must be equivalent for det()"
            )
        assert ref_unit is not None
        common = self._to_common_unit_values(ref_unit)
        mats = np.moveaxis(common, 2, 0)
        sign, logdet = np.linalg.slogdet(mats)
        det_vals = sign * np.exp(logdet)
        result_unit = ref_unit**nrow

        series_cls = getattr(self, "series_class", None)
        if series_cls is None:
            from gwpy.types.series import Series as _Series

            series_cls = _Series

        name = f"det({self.name})" if getattr(self, "name", "") else "det"
        return series_cls(det_vals, xindex=self.xindex, unit=result_unit, name=name)

    def inv(self, swap_rowcol: bool = True):
        """Compute the matrix inverse at each sample point."""
        from collections import OrderedDict

        from .metadata import MetaDataDict

        nrow, ncol, nsamp = self._value.shape
        if nrow != ncol:
            raise ValueError("inv requires a square matrix")
        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError(
                "All element units must be equivalent for inv()"
            )
        assert ref_unit is not None
        common = self._to_common_unit_values(ref_unit)
        mats = np.moveaxis(common, 2, 0)
        try:
            inv_stack = np.linalg.inv(mats)
        except np.linalg.LinAlgError:
            inv_stack = self._invert_stack_with_rescale(mats)
        inv_vals = np.moveaxis(inv_stack, 0, 2)

        inv_unit = ref_unit**-1
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

        rows_out = (
            _copy_meta_dict(self.cols, "row")
            if swap_rowcol
            else _copy_meta_dict(self.rows, "row")
        )
        cols_out = (
            _copy_meta_dict(self.rows, "col")
            if swap_rowcol
            else _copy_meta_dict(self.cols, "col")
        )

        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            inv_vals,
            xindex=self.xindex,
            rows=rows_out,
            cols=cols_out,
            meta=meta_matrix,
            name=f"inv({self.name})" if getattr(self, "name", "") else "inv",
            epoch=getattr(self, "epoch", 0.0),
            attrs=getattr(self, "attrs", {}),
        )

    def schur(
        self, keep_rows, keep_cols=None, eliminate_rows=None, eliminate_cols=None
    ):
        """Compute the Schur complement of a block matrix."""
        from collections import OrderedDict

        from .metadata import MetaDataDict

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
            raise ValueError(
                "Eliminated row/col sets must have the same size for Schur complement"
            )
        if not keep_rows_idx or not keep_cols_idx:
            raise ValueError("Keep sets must be non-empty")

        ok, ref_unit = self._all_element_units_equivalent()
        if not ok:
            raise u.UnitConversionError(
                "All element units must be equivalent for schur()"
            )
        common = self._to_common_unit_values(ref_unit)

        r_keep = len(keep_rows_idx)
        c_keep = len(keep_cols_idx)

        if len(eliminate_rows_idx) == 0:
            result_vals = common[np.ix_(keep_rows_idx, keep_cols_idx)]
        else:
            stack = np.moveaxis(common, 2, 0)
            A = np.take(np.take(stack, keep_rows_idx, axis=1), keep_cols_idx, axis=2)
            B = np.take(
                np.take(stack, keep_rows_idx, axis=1), eliminate_cols_idx, axis=2
            )
            C = np.take(
                np.take(stack, eliminate_rows_idx, axis=1), keep_cols_idx, axis=2
            )
            D = np.take(
                np.take(stack, eliminate_rows_idx, axis=1), eliminate_cols_idx, axis=2
            )

            try:
                D_inv = np.linalg.inv(D)
            except np.linalg.LinAlgError:
                D_inv = self._invert_stack_with_rescale(D)
            schur_block = A - np.matmul(np.matmul(B, D_inv), C)
            result_vals = np.moveaxis(schur_block, 0, 2)

        meta_arr = np.empty((r_keep, c_keep), dtype=object)
        for ii, ri in enumerate(keep_rows_idx):
            for jj, cj in enumerate(keep_cols_idx):
                base_meta = self.meta[ri, cj]
                meta_arr[ii, jj] = MetaData(
                    unit=ref_unit, name=base_meta.name, channel=base_meta.channel
                )

        def _subset_meta_dict(md: MetaDataDict, indices, prefix):
            items = OrderedDict()
            keys = list(md.keys())
            for idx in indices:
                key = keys[idx]
                items[key] = MetaData(**dict(md[key]))
            return MetaDataDict(items, expected_size=len(indices), key_prefix=prefix)

        rows_out = _subset_meta_dict(self.rows, keep_rows_idx, "row")
        cols_out = _subset_meta_dict(self.cols, keep_cols_idx, "col")

        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            result_vals,
            xindex=self.xindex,
            rows=rows_out,
            cols=cols_out,
            meta=MetaDataMatrix(meta_arr),
            name=f"schur({self.name})" if getattr(self, "name", "") else "schur",
            epoch=getattr(self, "epoch", 0.0),
            attrs=getattr(self, "attrs", {}),
        )

    def abs(self):
        """Return the absolute value of the matrix element-wise."""
        return np.abs(self)  # type: ignore[call-overload]

    def angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any):
        """
        Return the element-wise complex phase angle of the matrix.

        Notes
        -----
        - The phase is computed from the stored numeric values (not including units).
        - Output units are radians by default, or degrees if ``deg=True``.
        - If ``unwrap=True``, phase unwrapping is applied along the sample axis (axis=2).
        """
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

        # Keep typing loose here: NumPy stubs don't always preserve ndarray-ness through
        # angle/unwrap for our 3D (row, col, sample) arrays.
        vals: Any = np.angle(self._value)
        if unwrap:
            vals = np.unwrap(vals, axis=2)

        unit = u.deg if deg else u.rad
        if deg:
            vals = np.rad2deg(vals)

        meta_arr = np.empty(self.meta.shape, dtype=object)
        for i in range(self.meta.shape[0]):
            for j in range(self.meta.shape[1]):
                base_meta = self.meta[i, j]
                meta_arr[i, j] = MetaData(
                    unit=unit, name=base_meta.name, channel=base_meta.channel
                )

        matrix_cls = cast(type[Any], self.__class__)
        return matrix_cls(
            vals,
            xindex=self.xindex,
            rows=self.rows,
            cols=self.cols,
            meta=MetaDataMatrix(meta_arr),
            name=f"{self.name}.angle" if getattr(self, "name", "") else "",
            epoch=getattr(self, "epoch", 0.0),
            attrs=getattr(self, "attrs", {}),
        )
