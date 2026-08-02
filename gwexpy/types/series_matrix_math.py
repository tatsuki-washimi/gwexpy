from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from astropy import units as u

from .metadata import MetaData, MetaDataMatrix

_XINDEX_RTOL = 1e-9


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
        def xindex(self) -> np.ndarray | u.Quantity | Index | None: ...  # noqa: D102

        @xindex.setter
        def xindex(self, value: np.ndarray | u.Quantity | Index | None) -> None: ...

        def row_index(self, key: Any) -> int: ...  # noqa: D102
        def col_index(self, key: Any) -> int: ...  # noqa: D102
        def view(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: D102
        def copy(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: D102
        def astype(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: D102
        def _ufunc_dispatch(  # noqa: D102
            self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
        ) -> Any: ...

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

    @staticmethod
    def _xindex_equal(left: Any, right: Any) -> bool:
        """Return True when two sample axes represent the same coordinates.

        Unit-aware axes are compared after conversion to a shared unit. If one
        side is unitless, its raw numeric values are compared with the other
        side's values in that side's unit. Numeric axes use a relative
        tolerance of 1e-9 with no absolute tolerance, so zero-valued
        coordinates still require exact equality; two missing axes skip the
        check.
        """
        if left is None or right is None:
            return left is right

        base_unit = getattr(left, "unit", None)
        if base_unit is None:
            base_unit = getattr(right, "unit", None)

        def _axis_values(axis: Any) -> Any:
            if base_unit is not None and hasattr(axis, "to_value"):
                return axis.to_value(base_unit)
            return np.asarray(axis)

        try:
            left_values = np.asarray(_axis_values(left))
            right_values = np.asarray(_axis_values(right))
            if left_values.shape != right_values.shape:
                return False
            return bool(
                np.allclose(
                    left_values,
                    right_values,
                    rtol=_XINDEX_RTOL,
                    atol=0.0,
                )
            )
        except (u.UnitConversionError, TypeError, ValueError, AttributeError):
            return False

    @staticmethod
    def _xindex_summary(axis: Any) -> str:
        """Return a compact diagnostic summary for a sample axis."""
        if axis is None:
            return "None"
        unit = getattr(axis, "unit", None)
        unit_text = f", unit={unit}" if unit is not None else ""
        try:
            values = np.asarray(axis)
            preview = np.ravel(values)[:3]
            length = len(values)
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return repr(axis)
        return f"starts with {preview!r} (len={length}{unit_text})"

    # ------------------------------------------------------------------
    # Explicit operator suite
    #
    # ``SeriesMatrix.__array_ufunc__`` is ``None``, so NumPy refuses every
    # ufunc and the inherited ``ndarray`` operators raise ``TypeError``
    # immediately (they do *not* fall back to ``NotImplemented``).  Every
    # operator the class supports must therefore be spelled out here.  The
    # pay-off is that expressions with a foreign left operand -- ``Quantity``,
    # a bare ``Unit``, ``MaskedArray`` -- come back to us through Python's
    # reflected-operator protocol with the matrix intact (issue #575).
    # ------------------------------------------------------------------

    @staticmethod
    def _as_operand(other: Any) -> Any:
        """Normalise a bare ``Unit`` into a scale-1 ``Quantity``.

        A bare :class:`~astropy.units.UnitBase` carries no values, so it is
        only meaningful in multiplication and division; ``matrix + u.m`` stays
        unconverted here and is rejected downstream as an unsupported operand.
        """
        if isinstance(other, u.UnitBase):
            return u.Quantity(1, other)
        return other

    def _binary_op(self, ufunc: Any, other: Any, *, reflected: bool = False) -> Any:
        """Run a binary *ufunc* through the metadata-aware dispatcher."""
        operands = (other, self) if reflected else (self, other)
        return self._ufunc_dispatch(ufunc, "__call__", *operands)

    def _unary_op(self, ufunc: Any) -> Any:
        """Run a unary *ufunc* through the metadata-aware dispatcher."""
        return self._ufunc_dispatch(ufunc, "__call__", self)

    def _inplace_op(self, ufunc: Any, other: Any) -> Any:
        """Apply *ufunc* in place, committing only after every check passes.

        The result is computed out of place first, so a unit mismatch, a shape
        error or a zero divisor leaves ``self`` byte-for-byte unchanged.  Only
        once the values and the dtype are known to be acceptable is the
        existing buffer overwritten.
        """
        result = self._binary_op(ufunc, other)
        if result is NotImplemented:
            return NotImplemented
        if not hasattr(result, "meta"):
            raise TypeError(
                f"in-place {ufunc.__name__} produced a non-matrix result; "
                "use the out-of-place operator instead"
            )
        new_values = np.asarray(result.view(np.ndarray))
        if new_values.shape != self._value.shape:
            raise ValueError(
                f"in-place {ufunc.__name__} would change the shape from "
                f"{self._value.shape} to {new_values.shape}"
            )
        if not np.can_cast(new_values.dtype, self._value.dtype, casting="same_kind"):
            raise TypeError(
                f"Cannot apply in-place {ufunc.__name__}: result dtype "
                f"{new_values.dtype} cannot be cast to {self._value.dtype} "
                "under the 'same_kind' rule"
            )
        # Commit.  Nothing below here can fail.
        self._value[...] = new_values
        self.meta = result.meta
        if "unit" in self.__dict__:
            self.__dict__["unit"] = getattr(result, "unit", None)
        return self

    def __add__(self, other):
        """Add *other* elementwise, converting it into this matrix's units."""
        return self._binary_op(np.add, other)

    def __radd__(self, other):
        """Add this matrix to *other* (reflected ``+``)."""
        return self._binary_op(np.add, other, reflected=True)

    def __iadd__(self, other):
        """Add *other* in place."""
        return self._inplace_op(np.add, other)

    def __sub__(self, other):
        """Subtract *other* elementwise."""
        return self._binary_op(np.subtract, other)

    def __rsub__(self, other):
        """Subtract this matrix from *other* (reflected ``-``)."""
        return self._binary_op(np.subtract, other, reflected=True)

    def __isub__(self, other):
        """Subtract *other* in place."""
        return self._inplace_op(np.subtract, other)

    def __mul__(self, other):
        """Multiply elementwise, composing per-cell units."""
        return self._binary_op(np.multiply, self._as_operand(other))

    def __rmul__(self, other):
        """Multiply *other* by this matrix (reflected ``*``)."""
        return self._binary_op(np.multiply, self._as_operand(other), reflected=True)

    def __imul__(self, other):
        """Multiply by *other* in place."""
        return self._inplace_op(np.multiply, self._as_operand(other))

    def __truediv__(self, other):
        """Divide elementwise, composing per-cell units."""
        return self._binary_op(np.divide, self._as_operand(other))

    def __rtruediv__(self, other):
        """Divide *other* by this matrix (reflected ``/``)."""
        return self._binary_op(np.divide, self._as_operand(other), reflected=True)

    def __itruediv__(self, other):
        """Divide by *other* in place."""
        return self._inplace_op(np.divide, self._as_operand(other))

    def __floordiv__(self, other):
        """Floor-divide elementwise."""
        return self._binary_op(np.floor_divide, self._as_operand(other))

    def __rfloordiv__(self, other):
        """Floor-divide *other* by this matrix (reflected ``//``)."""
        return self._binary_op(np.floor_divide, self._as_operand(other), reflected=True)

    def __ifloordiv__(self, other):
        """Floor-divide by *other* in place."""
        return self._inplace_op(np.floor_divide, self._as_operand(other))

    def __mod__(self, other):
        """Take the elementwise remainder."""
        return self._binary_op(np.mod, other)

    def __rmod__(self, other):
        """Take the remainder of *other* by this matrix (reflected ``%``)."""
        return self._binary_op(np.mod, other, reflected=True)

    def __imod__(self, other):
        """Take the remainder by *other* in place."""
        return self._inplace_op(np.mod, other)

    def __divmod__(self, other):
        """Return ``(self // other, self % other)``."""
        quotient = self.__floordiv__(other)
        if quotient is NotImplemented:
            return NotImplemented
        return quotient, self.__mod__(other)

    def __rdivmod__(self, other):
        """Return ``(other // self, other % self)``."""
        quotient = self.__rfloordiv__(other)
        if quotient is NotImplemented:
            return NotImplemented
        return quotient, self.__rmod__(other)

    def __pow__(self, other, modulo=None):
        """Raise each element to the power *other*.

        The exponent must be a dimensionless scalar unless every cell of this
        matrix is already dimensionless, because a per-sample exponent would
        give each sample a different unit.
        """
        if modulo is not None:
            raise TypeError("SeriesMatrix does not support three-argument pow()")
        return self._binary_op(np.power, other)

    def __ipow__(self, other, modulo=None):
        """Raise each element to the power *other* in place."""
        if modulo is not None:
            raise TypeError("SeriesMatrix does not support three-argument pow()")
        return self._inplace_op(np.power, other)

    def __neg__(self):
        """Negate every element, preserving units."""
        return self._unary_op(np.negative)

    def __pos__(self):
        """Return a copy of this matrix, preserving units."""
        return self._unary_op(np.positive)

    def __abs__(self):
        """Return the elementwise absolute value, preserving units."""
        return self._unary_op(np.absolute)

    def __lt__(self, other):
        """Compare elementwise with ``<``."""
        return self._binary_op(np.less, other)

    def __le__(self, other):
        """Compare elementwise with ``<=``."""
        return self._binary_op(np.less_equal, other)

    def __eq__(self, other):
        """Compare elementwise with ``==``."""
        return self._binary_op(np.equal, other)

    def __ne__(self, other):
        """Compare elementwise with ``!=``."""
        return self._binary_op(np.not_equal, other)

    def __gt__(self, other):
        """Compare elementwise with ``>``."""
        return self._binary_op(np.greater, other)

    def __ge__(self, other):
        """Compare elementwise with ``>=``."""
        return self._binary_op(np.greater_equal, other)

    def __matmul__(self, other):
        """Perform matrix multiplication while broadcasting over the sample axis."""
        if not isinstance(other, type(self)):
            return NotImplemented

        if self._value.shape[2] != other._value.shape[2]:
            raise ValueError("Sample axis length mismatch in matrix multiplication")
        if not self._xindex_equal(self.xindex, other.xindex):
            raise ValueError(
                "xindex mismatch in matrix multiplication: "
                f"left xindex {self._xindex_summary(self.xindex)}, "
                f"right xindex {self._xindex_summary(other.xindex)}"
            )
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
            attrs=deepcopy(getattr(self, "attrs", {})),
        )

    def __rmatmul__(self, other):
        """Reject ``other @ matrix`` for non-matrix left operands.

        Reaching this method means *other* declined the operation, so it is not
        a ``SeriesMatrix``.  A plain array has no per-cell units and no sample
        axis, so there is no defensible way to contract it against this matrix.
        """
        raise TypeError(
            f"unsupported operand type(s) for @: {type(other).__name__!r} and "
            f"{type(self).__name__!r}; matrix multiplication requires two "
            "SeriesMatrix operands of the same class"
        )

    # ------------------------------------------------------------------
    # Reductions
    #
    # ``ndarray.sum`` and friends go through ``np.add.reduce``, which the
    # ufunc opt-out rejects.  These overrides keep the historical public API
    # working; they deliberately return bare NumPy values because a reduction
    # collapses the cells whose metadata would have to be preserved.
    # ------------------------------------------------------------------

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        """Sum the raw values.

        Returns a plain NumPy value: reductions mix cells whose units may
        differ, so no per-cell metadata is carried over. No warning is issued.
        """
        return np.asarray(self).sum(*args, **kwargs)

    def prod(self, *args: Any, **kwargs: Any) -> Any:
        """Multiply the raw values together, returning a plain NumPy value."""
        return np.asarray(self).prod(*args, **kwargs)

    def cumsum(self, *args: Any, **kwargs: Any) -> Any:
        """Return the cumulative sum of the raw values as a plain NumPy array."""
        return np.asarray(self).cumsum(*args, **kwargs)

    def any(self, *args: Any, **kwargs: Any) -> Any:
        """Return whether any raw value is truthy, as a plain NumPy value."""
        return np.asarray(self).any(*args, **kwargs)

    def all(self, *args: Any, **kwargs: Any) -> Any:
        """Return whether all raw values are truthy, as a plain NumPy value."""
        return np.asarray(self).all(*args, **kwargs)

    def cumprod(self, *args: Any, **kwargs: Any) -> Any:
        """Return the cumulative product of the raw values as a plain NumPy array."""
        return np.asarray(self).cumprod(*args, **kwargs)

    # ------------------------------------------------------------------
    # ndarray methods that are implemented with ufuncs
    #
    # Left un-overridden these raise, and ``np.clip``/``np.round`` then fall
    # back to NumPy's ``_wrapit`` path, which views the plain result as this
    # class and hands back an object whose ``meta``/``rows``/``cols`` are the
    # *same objects* as the source's.  Rebuilding explicitly keeps the historic
    # behaviour and keeps the metadata independent.
    # ------------------------------------------------------------------

    def _rebuild_with_values(self, values: Any, meta: Any = None) -> Any:
        """Return a same-class matrix holding *values* with independent metadata."""
        from .seriesmatrix_base import _copy_meta_cells, _copy_metadata_dict

        values = np.asarray(values)
        result = (
            self.copy()
            if values.dtype == np.asarray(self).dtype
            else self.astype(values.dtype)
        )
        result._value[...] = values
        result.meta = _copy_meta_cells(self.meta) if meta is None else meta
        if getattr(self, "rows", None):
            result.rows = _copy_metadata_dict(self.rows, "row")
        if getattr(self, "cols", None):
            result.cols = _copy_metadata_dict(self.cols, "col")
        result.attrs = deepcopy(getattr(self, "attrs", {}))
        return result

    def clip(self, min: Any = None, max: Any = None, out: Any = None, **kwargs: Any):
        """Clip the values elementwise, preserving per-cell units."""
        if out is not None:
            raise TypeError("SeriesMatrix.clip does not support the 'out' argument")
        return self._rebuild_with_values(np.clip(np.asarray(self), min, max, **kwargs))

    def round(self, decimals: int = 0, out: Any = None):
        """Round the values elementwise, preserving per-cell units."""
        if out is not None:
            raise TypeError("SeriesMatrix.round does not support the 'out' argument")
        return self._rebuild_with_values(np.round(np.asarray(self), decimals))

    def conjugate(self):
        """Return the elementwise complex conjugate, preserving per-cell units."""
        return self._unary_op(np.conjugate)

    def conj(self):
        """Alias for :meth:`conjugate`."""
        return self._unary_op(np.conjugate)

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
                attrs=deepcopy(getattr(self, "attrs", {})),
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
                attrs=deepcopy(getattr(self, "attrs", {})),
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
            attrs=deepcopy(getattr(self, "attrs", {})),
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
            attrs=deepcopy(getattr(self, "attrs", {})),
        )

    def abs(self):
        """Return the absolute value of the matrix element-wise."""
        return self._unary_op(np.absolute)

    def angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any):
        """Return the element-wise complex phase angle of the matrix.

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
            attrs=deepcopy(getattr(self, "attrs", {})),
        )


# ----------------------------------------------------------------------
# Operators that are deliberately unsupported.
#
# The ufunc opt-out means every inherited ``ndarray`` operator raises a
# generic "operand does not support ufuncs" TypeError.  Re-declaring the
# operators we do *not* implement lets us say why, instead of leaking that
# message.  Each entry maps the dunder to the explanation shown to the user.
# ----------------------------------------------------------------------
_UNSUPPORTED_OPERATORS: dict[str, str] = {
    "__and__": "bitwise AND (&) is undefined for values that carry units",
    "__rand__": "bitwise AND (&) is undefined for values that carry units",
    "__iand__": "bitwise AND (&) is undefined for values that carry units",
    "__or__": "bitwise OR (|) is undefined for values that carry units",
    "__ror__": "bitwise OR (|) is undefined for values that carry units",
    "__ior__": "bitwise OR (|) is undefined for values that carry units",
    "__xor__": "bitwise XOR (^) is undefined for values that carry units",
    "__rxor__": "bitwise XOR (^) is undefined for values that carry units",
    "__ixor__": "bitwise XOR (^) is undefined for values that carry units",
    "__invert__": "bitwise NOT (~) is undefined for values that carry units",
    "__lshift__": "the shift operators are undefined for values that carry units",
    "__rlshift__": "the shift operators are undefined for values that carry units",
    "__ilshift__": "the shift operators are undefined for values that carry units",
    "__rshift__": "the shift operators are undefined for values that carry units",
    "__rrshift__": "the shift operators are undefined for values that carry units",
    "__irshift__": "the shift operators are undefined for values that carry units",
    "__imatmul__": (
        "in-place matrix multiplication (@=) would change the column labels "
        "and the result shape, so it cannot reuse the existing buffer"
    ),
    "__rpow__": (
        "a matrix cannot be used as an exponent: each cell would raise the "
        "base to a different power along the sample axis"
    ),
}


def _make_unsupported_operator(op_name: str, reason: str) -> Any:
    """Build a dunder that refuses the operation with an explanatory error."""

    def _unsupported(self: Any, *args: Any) -> Any:
        raise TypeError(
            f"{type(self).__name__} does not support {op_name}: {reason}. "
            "Operate on 'matrix.value' if a raw NumPy result is what you want."
        )

    _unsupported.__name__ = op_name
    _unsupported.__qualname__ = f"SeriesMatrixMathMixin.{op_name}"
    _unsupported.__doc__ = f"Reject ``{op_name}``: {reason}."
    return _unsupported


for _op_name, _reason in _UNSUPPORTED_OPERATORS.items():
    setattr(
        SeriesMatrixMathMixin, _op_name, _make_unsupported_operator(_op_name, _reason)
    )
