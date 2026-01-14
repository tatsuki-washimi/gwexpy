from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u


class SeriesMatrixValidationMixin:
    """Mixin for SeriesMatrix compatibility and contiguity checks."""

    def is_contiguous(self, other: Any, tol: float = 1 / 2.0**18) -> int:
        """Check if this matrix is contiguous with another."""
        # Note: 'type(self).__mro__[1]' was used in SeriesMatrixOps to refer to SeriesMatrix base.
        # Here we can just check for existence of xspan or similar if we want to be generic,
        # but usually this is called on another SeriesMatrix.
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

    def is_contiguous_exact(self, other: Any, tol: float = 1 / 2.0**18) -> int:
        """Check contiguity with strict shape matching."""
        if self._value.shape != other._value.shape:
            raise ValueError(
                f"shape does not match: {self._value.shape} vs {other._value.shape}"
            )

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

    def is_compatible_exact(self, other: Any) -> bool:
        """Check strict compatibility with another matrix."""
        base_unit = getattr(self.xindex, "unit", None)
        lhs = self.xindex
        rhs = other.xindex
        if base_unit is not None:
            try:
                lhs = lhs.to_value(base_unit)
                rhs = rhs.to_value(base_unit)
            except Exception:
                pass
        try:
            equal = np.array_equal(lhs, rhs)
        except Exception:
            equal = lhs == rhs
        if not equal:
            raise ValueError(f"xindex does not match: {self.xindex} vs {other.xindex}")

        if self._value.shape != other._value.shape:
            raise ValueError(
                f"shape does not match: {self._value.shape} vs {other._value.shape}"
            )

        if list(self.rows.keys()) != list(other.rows.keys()):
            raise ValueError("row keys do not match")
        if list(self.cols.keys()) != list(other.cols.keys()):
            raise ValueError("col keys do not match")

        for (k1, meta1), (k2, meta2) in zip(self.rows.items(), other.rows.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(
                    f"row {k1} unit does not match: {meta1.unit} vs {meta2.unit}"
                )
        for (k1, meta1), (k2, meta2) in zip(self.cols.items(), other.cols.items()):
            if not meta1.unit.is_equivalent(meta2.unit):
                raise ValueError(
                    f"col {k1} unit does not match: {meta1.unit} vs {meta2.unit}"
                )
        return True

    def is_compatible(self, other: Any) -> bool:
        """Compatibility check."""
        from .seriesmatrix_base import SeriesMatrix

        if not isinstance(other, SeriesMatrix):
            arr = np.asarray(other)
            if arr.shape != self._value.shape:
                raise ValueError(
                    f"shape does not match: {self._value.shape} vs {arr.shape}"
                )
            return True

        if self._value.shape[:2] != other._value.shape[:2]:
            raise ValueError(
                f"matrix shape does not match: {self._value.shape[:2]} vs {other._value.shape[:2]}"
            )

        xunit_self = getattr(self.xindex, "unit", None)
        xunit_other = getattr(other.xindex, "unit", None)
        if xunit_self is not None and xunit_other is not None:
            try:
                if not u.Unit(xunit_self).is_equivalent(u.Unit(xunit_other)):
                    raise ValueError(
                        f"xindex unit does not match: {xunit_self} vs {xunit_other}"
                    )
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                raise ValueError(
                    f"xindex unit does not match: {xunit_self} vs {xunit_other}"
                )

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
