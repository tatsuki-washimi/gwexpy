from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from astropy.units import Quantity

from gwexpy.numerics import eps_for_dtype

__all__ = ["AxisDescriptor", "coerce_1d_quantity"]

_REGULAR_RTOL = 2.5e-14
_REGULAR_ATOL_ULPS = 1
# Least-squares normalized deviation (G3) constants from v0.3.0 Step 9 design:
# kappa=4 covers the 2-roundoff zero-crossing upper bound 4*ULP(max|x|)/|d|;
# floor >= 0.25 indicates the dtype cannot resolve the sample spacing.
_REGULAR_LS_KAPPA = 4.0
_REGULAR_MAX_FLOOR = 0.25


def _validate_nonneg_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a non-negative real number, got {type(value).__name__}"
        )
    val = float(value)
    if not np.isfinite(val) or val < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return val


def _ls_fit_metrics(
    values: np.ndarray,
) -> tuple[float | complex, float | complex, float, float]:
    """Compute centered least-squares slope, fitted intercept, E_LS, and floor.

    Uses offset subtraction (``y = x - x[0]``) and one step of iterative
    refinement so float64 summation noise stays bounded even at large ``N``.
    """
    n = values.size
    ic = np.arange(n, dtype=np.float64) - 0.5 * (n - 1)
    s_denom = (n * (n * n - 1)) / 12.0

    y: np.ndarray[Any, Any]
    if np.issubdtype(values.dtype, np.integer):
        y = np.asarray(values.astype(object) - values[0], dtype=np.float64)
        eps_dtype = 0.0
        x0_val: float | complex = float(values[0])
    elif np.issubdtype(values.dtype, np.complexfloating):
        y = np.asarray(values, dtype=np.complex128) - np.complex128(values[0])
        eps_dtype = eps_for_dtype(np.real(values[:1]).dtype)
        x0_val = complex(values[0])
    else:
        work_dtype = np.result_type(values.dtype, np.float64)
        y = np.asarray(values, dtype=work_dtype) - work_dtype.type(values[0])
        eps_dtype = eps_for_dtype(values.dtype)
        x0_val = float(values[0])

    y_mean = np.mean(y)
    slope = np.sum(ic * (y - y_mean)) / s_denom

    # 1-step iterative refinement suppresses O(N^2) float64 sum cancellation.
    residuals = (y - y_mean) - ic * slope
    y_corr = np.mean(residuals)
    slope = slope + np.sum(ic * (residuals - y_corr)) / s_denom
    y_mean = y_mean + y_corr
    residuals = (y - y_mean) - ic * slope

    x_fit_0 = x0_val + (y_mean - 0.5 * (n - 1) * slope)
    abs_slope = float(np.abs(slope))
    if abs_slope == 0.0 or not np.isfinite(abs_slope):
        return (
            complex(slope) if np.iscomplexobj(slope) else float(slope),
            complex(x_fit_0) if np.iscomplexobj(x_fit_0) else float(x_fit_0),
            float("inf"),
            float("inf"),
        )

    max_abs_x = float(np.max(np.abs(values)))
    e_ls = float(np.max(np.abs(residuals)) / abs_slope)
    floor = _REGULAR_LS_KAPPA * eps_dtype * max_abs_x / abs_slope
    return (
        complex(slope) if np.iscomplexobj(slope) else float(slope),
        complex(x_fit_0) if np.iscomplexobj(x_fit_0) else float(x_fit_0),
        e_ls,
        floor,
    )


def coerce_1d_quantity(index, unit=None) -> Quantity:
    """Ensure input is a 1D Quantity."""
    if not isinstance(index, Quantity):
        index = Quantity(index, unit=unit)
    elif unit is not None:
        index = index.to(unit)

    if index.ndim != 1:
        if index.ndim == 0:
            index = index.reshape(1)
        else:
            raise ValueError(f"Axis index must be 1D, got {index.ndim}D")
    return index


@dataclass
class AxisDescriptor:
    """Describe a named one-dimensional axis and its coordinate values."""

    name: str
    index: Quantity  # 1D

    def __post_init__(self):
        self.index = coerce_1d_quantity(self.index)

    @property
    def unit(self):
        """Return the axis unit."""
        return self.index.unit

    def __eq__(self, other):
        if not isinstance(other, AxisDescriptor):
            return NotImplemented
        try:
            return (
                self.name == other.name
                and self.index.shape == other.index.shape
                and np.all(self.index == other.index)
            )
        except (AttributeError, TypeError, ValueError):
            return False

    @property
    def size(self):
        """Return the number of axis samples."""
        return self.index.size

    @property
    def regular(self) -> bool:
        """Return whether the axis has regular linear spacing.

        Integer coordinates are compared exactly. Other numeric coordinates
        have adjacent intervals compared with a relative tolerance of
        ``2.5e-14`` and an absolute tolerance of one ULP at the largest
        represented interval magnitude. This keeps ``delta`` consistent with
        every represented interval while allowing bounded accumulated
        floating-point round-off.

        Logarithmic (equal-ratio) axes are not regular under this linear-spacing
        contract.
        """
        if self.size <= 1:
            return True
        values = np.asarray(self.index.to_value(self.unit))
        if not np.all(np.isfinite(values)):
            return False
        if np.issubdtype(values.dtype, np.integer):
            # Python integers preserve exact intervals and cannot overflow.
            diffs = np.diff(values.astype(object))
            return all(diff == diffs[0] for diff in diffs[1:])
        diffs = np.diff(values)
        interval_scale = np.max(np.abs(diffs))
        atol = _REGULAR_ATOL_ULPS * abs(np.spacing(interval_scale))
        return bool(
            np.allclose(
                diffs,
                diffs[0],
                rtol=_REGULAR_RTOL,
                atol=atol,
                equal_nan=False,
            )
        )

    @property
    def delta(self) -> Quantity | None:
        """Return the constant axis spacing when the axis is regular."""
        if self.size > 1 and self.regular:
            return self.index[1] - self.index[0]
        return None

    def mean_delta(self) -> Quantity | None:
        """Return the signed least-squares mean axis spacing.

        Computes the centered linear least-squares (G3) slope with offset
        removal and one-step iterative refinement. Returns ``None`` when the
        axis has fewer than 2 points or contains non-finite coordinates.
        """
        if self.size < 2:
            return None
        values = np.asarray(self.index.to_value(self.unit))
        if not np.all(np.isfinite(values)):
            return None
        slope, _, _, _ = _ls_fit_metrics(values)
        return Quantity(slope, unit=self.unit)

    def _regularity_report(
        self, *, max_deviation: float
    ) -> tuple[bool, float, float, float, float]:
        """Return ``(passed, e_ls, floor, mean_delta_val, fit_origin_val)``."""
        values = np.asarray(self.index.to_value(self.unit))
        if not np.all(np.isfinite(values)):
            return False, float("inf"), float("inf"), 0.0, 0.0
        if self.size <= 1:
            origin = (
                float(values[0])
                if self.size == 1 and not np.iscomplexobj(values)
                else 0.0
            )
            return True, 0.0, 0.0, 0.0, origin
        if np.issubdtype(values.dtype, np.complexfloating):
            passed = self.regular
            return passed, 0.0 if passed else float("inf"), 0.0, 0.0, 0.0

        slope, x_fit_0, e_ls, floor = _ls_fit_metrics(values)
        assert not isinstance(slope, complex) and not isinstance(x_fit_0, complex)
        if self.regular:
            return True, e_ls, floor, slope, x_fit_0
        if slope == 0.0 or not np.isfinite(e_ls) or not np.isfinite(floor):
            return False, e_ls, floor, slope, x_fit_0
        if floor >= _REGULAR_MAX_FLOOR:
            return False, e_ls, floor, slope, x_fit_0
        return bool(e_ls <= max_deviation + floor), e_ls, floor, slope, x_fit_0

    def is_regular(
        self,
        *,
        rtol: float | None = None,
        atol_ulps: float | None = None,
        max_deviation: float | None = None,
    ) -> bool:
        """Check whether the axis is regularly spaced with configurable tolerance.

        When all keyword arguments are ``None``, delegates directly to the
        strict :attr:`regular` property.

        Parameters
        ----------
        rtol : float, optional
            Relative tolerance on adjacent interval differences (for
            interval-based comparison).
        atol_ulps : float, optional
            Absolute tolerance in ULPs of the maximum interval magnitude.
        max_deviation : float, optional
            Maximum allowed normalized cumulative deviation
            ``E_LS = max|x_i - x_hat_i| / |mean_delta|`` from the least-squares
            linear grid, augmented by a dtype-dependent representation floor
            ``4 * eps_dtype * max|x| / |mean_delta|``. Axes where the floor
            reaches ``0.25`` are rejected unless strict :attr:`regular` passes.

        """
        if rtol is None and atol_ulps is None and max_deviation is None:
            return self.regular

        if max_deviation is not None and (rtol is not None or atol_ulps is not None):
            raise ValueError(
                "Cannot specify 'max_deviation' together with 'rtol' or 'atol_ulps'."
            )

        if max_deviation is not None:
            max_dev = _validate_nonneg_real(max_deviation, "max_deviation")
            passed, _, _, _, _ = self._regularity_report(max_deviation=max_dev)
            return passed

        effective_rtol = (
            _REGULAR_RTOL if rtol is None else _validate_nonneg_real(rtol, "rtol")
        )
        effective_atol_ulps = (
            float(_REGULAR_ATOL_ULPS)
            if atol_ulps is None
            else _validate_nonneg_real(atol_ulps, "atol_ulps")
        )

        values = np.asarray(self.index.to_value(self.unit))
        if not np.all(np.isfinite(values)):
            return False
        if self.size <= 1:
            return True

        if np.issubdtype(values.dtype, np.integer):
            diffs_obj = np.diff(values.astype(object))
            if all(diff == diffs_obj[0] for diff in diffs_obj[1:]):
                return True
            diffs_f = np.asarray(diffs_obj, dtype=np.float64)
            interval_scale = float(np.max(np.abs(diffs_f)))
            atol = (
                0.0
                if atol_ulps is None
                else effective_atol_ulps * abs(float(np.spacing(interval_scale)))
            )
            return bool(
                np.allclose(
                    diffs_f,
                    diffs_f[0],
                    rtol=effective_rtol,
                    atol=atol,
                    equal_nan=False,
                )
            )

        diffs = np.diff(values)
        interval_scale = np.max(np.abs(diffs))
        atol = effective_atol_ulps * abs(np.spacing(interval_scale))
        return bool(
            np.allclose(
                diffs,
                diffs[0],
                rtol=effective_rtol,
                atol=atol,
                equal_nan=False,
            )
        )

    def to_value(self, q):
        """Convert Quantity to axis unit value, or return float if dimensionless/compatible."""
        if isinstance(q, Quantity):
            return q.to_value(self.unit)
        return float(q)

    def iloc_nearest(self, value):
        """Return the integer index nearest to value."""
        val = self.to_value(value)
        # Assume monotonic for speed, or general search? User said "nearest/searchsorted"
        # but also "axis coordinates are not assumed to be regular".
        # If not regular, we can still use abs diffargmin.
        idx = np.abs(self.index.value - val).argmin()
        return idx

    def iloc_slice(self, s: slice):
        """Convert a coordinate slice (start, stop, step) to an integer slice.

        Coordinate start and stop bounds require an ascending axis because they
        are resolved with :func:`numpy.searchsorted`. Descending and unordered
        axes remain valid for :meth:`iloc_nearest`.
        """
        start_idx = None
        stop_idx = None
        step_idx = None

        # Handling start
        if s.start is not None or s.stop is not None:
            axis_vals = self.index.value
            if axis_vals.size >= 2 and np.any(np.diff(axis_vals) < 0):
                raise ValueError(
                    f"Coordinate range slicing on axis '{self.name}' requires an "
                    "ascending axis; use integer slicing (isel) on descending or "
                    "unordered axes."
                )

        if s.start is not None:
            val = self.to_value(s.start)
            # searchsorted works if sorted
            # If we assume sorted (ascending):
            start_idx = np.searchsorted(self.index.value, val, side="left")

        # Handling stop
        if s.stop is not None:
            val = self.to_value(s.stop)
            stop_idx = np.searchsorted(
                self.index.value, val, side="left"
            )  # 'right' might be more 'python slice' style?
            # Typically python slice includes start, excludes stop.
            # searchsorted('left') gives index i such that a[i-1] < v <= a[i].
            # If we want to exclude the value itself if it equals, left is good?
            # Actually, for `sel`, users expect range [start, stop).

        # Handling step (coordinate step -> integer step?)
        # This is tricky for non-regular. If regular, we can compute it.
        if s.step is not None:
            # If coordinate step is given, we need to map to integer step
            if self.regular and self.delta is not None and self.delta.value != 0:
                step_val = self.to_value(s.step)
                step_idx = int(round(step_val / self.delta.value))
            else:
                raise ValueError("Cannot use coordinate step slice on irregular axis")

        return slice(start_idx, stop_idx, step_idx)
