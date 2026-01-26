"""
Resampling and time-axis operations for TimeSeries.

This module provides resampling functionality as a mixin class:
- asfreq: Reindex to a new fixed-interval grid
- resample: Time-bin aggregation or signal processing resampling
- _resample_time_bin: Internal time-bin aggregation
- stlt: Short-Time Laplace Transform
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike

from ._typing import TimeSeriesAttrs

if TYPE_CHECKING:
    pass

NumberLike: TypeAlias = int | float | np.number
QuantityLike: TypeAlias = ArrayLike | u.Quantity
AggFunc: TypeAlias = Callable[[np.ndarray], float]


class TimeSeriesResamplingMixin(TimeSeriesAttrs):
    """
    Mixin class providing resampling methods for TimeSeries.

    This mixin is designed to be combined with TimeSeriesCore to create
    the full TimeSeries class.
    """

    # ===============================
    # asfreq - Reindex to New Grid
    # ===============================

    def asfreq(
        self,
        rule: str | NumberLike | u.Quantity,
        method: str | None = None,
        fill_value: Any = np.nan,
        *,
        origin: str | NumberLike | u.Quantity = "t0",
        offset: NumberLike | u.Quantity | None = None,
        align: Literal["ceil", "floor"] = "ceil",
        tolerance: float | u.Quantity | None = None,
        max_gap: float | u.Quantity | None = None,
        copy: bool = True,
    ) -> TimeSeriesResamplingMixin:
        """
        Reindex the TimeSeries to a new fixed-interval grid associated with the given rule.

        Parameters
        ----------
        rule : str, float, or Quantity
            Target time interval (e.g., '1s', 0.1, 0.5*u.s).
        method : str or None
            Fill method: None (exact), 'ffill', 'bfill', 'nearest'.
        fill_value : scalar
            Value for missing data points.
        origin : str
            Reference point for grid alignment: 't0' or 'gps0'.
        offset : Quantity
            Time offset from origin.
        align : str
            Grid alignment: 'ceil' or 'floor'.
        tolerance : float
            Tolerance for matching times.
        max_gap : float
            Maximum gap for interpolation.
        copy : bool
            Whether to copy data.

        Returns
        -------
        TimeSeries
            Reindexed series.
        """
        # 1. Parse rule to target dt (Quantity)
        if isinstance(rule, (int, float, np.number)):
            target_dt = u.Quantity(rule)
        elif isinstance(rule, str):
            try:
                target_dt = u.Quantity(rule)
            except (TypeError, ValueError):
                import re

                match = re.match(r"([0-9\.]+)([a-zA-Z]*)", rule)
                if match:
                    val, unit_str = match.groups()
                    if not unit_str:
                        target_dt = u.Quantity(float(val))
                    else:
                        target_dt = float(val) * u.Unit(unit_str)
                else:
                    raise ValueError(f"Could not parse rule: {rule}")
        elif isinstance(rule, u.Quantity):
            target_dt = rule
        else:
            raise TypeError("rule must be a string, number, or astropy Quantity.")

        # Validate rule unit compatibility
        is_time = target_dt.unit.physical_type == "time"
        is_dimless = (
            target_dt.unit is None
            or target_dt.unit == u.dimensionless_unscaled
            or target_dt.unit.physical_type == "dimensionless"
        )

        if not is_time and not is_dimless:
            raise ValueError("rule must be time-like or dimensionless")

        # 2. Determine Original Times and Span
        old_times_q = self.times
        old_times_val = old_times_q.value
        time_unit = old_times_q.unit
        unit_is_dimensionless = (
            time_unit is None
            or time_unit == u.dimensionless_unscaled
            or (
                hasattr(time_unit, "physical_type")
                and time_unit.physical_type == "dimensionless"
            )
        )
        if unit_is_dimensionless:
            time_unit = u.s

        is_dimensionless = unit_is_dimensionless

        target_dt_in_time_unit = None
        start_time_val = None
        stop_time_val = None

        if is_dimensionless and target_dt.unit.physical_type == "time":
            time_unit = u.s
            target_dt_in_time_unit = target_dt.to(u.s)

            start_time_val = old_times_q[0].value

            if hasattr(self, "dt") and self.dt is not None:
                dt_q = self.dt
                if dt_q.unit is None or dt_q.unit == u.dimensionless_unscaled:
                    dt_val = dt_q.value
                else:
                    dt_val = dt_q.to(u.s).value
                stop_time_val = start_time_val + (len(self) * dt_val)
            else:
                stop_time_val = old_times_q[-1].value
        else:
            safe_unit = time_unit if time_unit is not None else u.dimensionless_unscaled
            if target_dt.unit == u.dimensionless_unscaled:
                target_dt_in_time_unit = u.Quantity(target_dt.value, safe_unit)
                target_dt = target_dt_in_time_unit
            else:
                target_dt_in_time_unit = target_dt.to(safe_unit)

            if unit_is_dimensionless or old_times_q.unit == u.dimensionless_unscaled:
                start_time_val = old_times_q[0].value
            else:
                start_time_val = old_times_q[0].to(safe_unit).value

            if hasattr(self, "dt") and self.dt is not None:
                if self.dt.unit == u.dimensionless_unscaled:
                    dt_input = self.dt.value
                else:
                    dt_input = self.dt.to(safe_unit).value
                stop_time_val = start_time_val + (len(self) * dt_input)
            else:
                if (
                    unit_is_dimensionless
                    or old_times_q.unit == u.dimensionless_unscaled
                ):
                    stop_time_val = old_times_q[-1].value
                else:
                    stop_time_val = old_times_q[-1].to(safe_unit).value

        if target_dt_in_time_unit is None:
            safe_unit = time_unit if time_unit is not None else u.dimensionless_unscaled
            if isinstance(target_dt, u.Quantity):
                if target_dt.unit == u.dimensionless_unscaled or target_dt.unit is None:
                    target_dt_in_time_unit = u.Quantity(target_dt.value, safe_unit)
                else:
                    target_dt_in_time_unit = target_dt.to(safe_unit)
            else:
                target_dt_in_time_unit = u.Quantity(target_dt, safe_unit)

        dt_val = target_dt_in_time_unit.value

        def _to_time_value(val):
            q = u.Quantity(val)
            try:
                return q.to(time_unit).value
            except u.UnitConversionError:
                phys = getattr(q.unit, "physical_type", None)
                if q.unit == u.dimensionless_unscaled or phys == "dimensionless":
                    return q.value
                raise

        # 3. Determine Origin Base
        safe_unit = time_unit if time_unit is not None else u.dimensionless_unscaled

        if origin == "t0":
            origin_val = start_time_val
        elif origin == "gps0":
            origin_val = 0.0
        elif isinstance(origin, (u.Quantity, str)):
            try:
                origin_val = u.Quantity(origin).to(safe_unit).value
            except u.UnitConversionError:
                q_origin = u.Quantity(origin)
                if (
                    q_origin.unit == u.dimensionless_unscaled
                    and safe_unit.physical_type == "time"
                ):
                    raise TypeError(
                        "Cannot use dimensionless origin for time-based series."
                    )
                raise
        elif isinstance(origin, (int, float, np.number)):
            origin_val = float(origin)
        else:
            origin_val = 0.0

        if offset is None:
            offset_val = 0.0
        else:
            q_offset = u.Quantity(offset)
            try:
                offset_val = q_offset.to(safe_unit).value
            except u.UnitConversionError:
                phys = getattr(q_offset.unit, "physical_type", None)
                if q_offset.unit == u.dimensionless_unscaled or phys == "dimensionless":
                    offset_val = q_offset.value
                else:
                    raise

        base_val = origin_val + offset_val

        # 4. Generate New Grid
        if align == "ceil":
            k = np.ceil((start_time_val - base_val) / dt_val)
        elif align == "floor":
            k = np.floor((start_time_val - base_val) / dt_val)
        else:
            raise ValueError("align must be 'ceil' or 'floor'")

        grid_start = base_val + k * dt_val

        duration = stop_time_val - grid_start
        if duration <= 0:
            n_points = 0
        else:
            n_points = int(np.ceil(duration / dt_val))

        new_times_val = grid_start + np.arange(n_points) * dt_val
        new_times_val = new_times_val[new_times_val < stop_time_val]

        # 5. Reindex / Interpolate
        if len(new_times_val) == 0:
            safe_unit = time_unit if time_unit is not None else u.dimensionless_unscaled
            safe_t0 = u.Quantity(grid_start, safe_unit)
            return self.__class__(
                [],
                t0=safe_t0,
                dt=target_dt,
                channel=self.channel,
                name=self.name,
                unit=self.unit,
            )

        out_dtype: np.dtype[Any] = np.dtype(self.dtype)
        if fill_value is None:
            if self.dtype.kind in ("f", "c"):
                fill_value = np.nan
            else:
                fill_value = np.nan
                out_dtype = np.dtype(np.float64)

        is_fill_nan = False
        try:
            is_fill_nan = np.isnan(fill_value)
        except (TypeError, ValueError):
            pass
        if is_fill_nan and self.dtype.kind not in ("f", "c"):
            out_dtype = np.dtype(np.float64)

        out_dtype = np.dtype(out_dtype)
        new_data = np.full(len(new_times_val), fill_value, dtype=out_dtype)
        if out_dtype.kind == "c":
            new_data = new_data.astype(np.complex128)

        if method == "interpolate":
            raise ValueError(
                "asfreq does not interpolate; use resample() for interpolation or filtering."
            )
        else:
            idx_right = np.searchsorted(old_times_val, new_times_val, side="left")
            np.clip(idx_right - 1, 0, len(old_times_val) - 1)
            np.clip(idx_right, 0, len(old_times_val) - 1)

            if method == "ffill" or method == "pad":
                idx_side_right = np.searchsorted(
                    old_times_val, new_times_val, side="right"
                )
                fill_idx = idx_side_right - 1

                valid_f = fill_idx >= 0
                if tolerance is not None:
                    tol_val = _to_time_value(tolerance)
                    dt_diff = new_times_val - old_times_val[np.clip(fill_idx, 0, None)]
                    valid_f &= dt_diff <= tol_val
                if max_gap is not None:
                    limit = _to_time_value(max_gap)
                    dt_diff = new_times_val - old_times_val[np.clip(fill_idx, 0, None)]
                    valid_f &= dt_diff <= limit

                valid_out_indices = np.where(valid_f)[0]
                src_indices = fill_idx[valid_f]
                new_data[valid_out_indices] = self.value[src_indices]

            elif method == "bfill" or method == "backfill":
                idx_side_left = np.searchsorted(
                    old_times_val, new_times_val, side="left"
                )
                fill_idx = idx_side_left

                valid_b = fill_idx < len(old_times_val)

                if tolerance is not None:
                    tol_val = _to_time_value(tolerance)
                    dt_diff = (
                        old_times_val[np.clip(fill_idx, 0, len(old_times_val) - 1)]
                        - new_times_val
                    )
                    valid_b &= dt_diff <= tol_val
                if max_gap is not None:
                    limit = _to_time_value(max_gap)
                    dt_diff = (
                        old_times_val[np.clip(fill_idx, 0, len(old_times_val) - 1)]
                        - new_times_val
                    )
                    valid_b &= dt_diff <= limit

                valid_out_indices = np.where(valid_b)[0]
                src_indices = fill_idx[valid_b]
                new_data[valid_out_indices] = self.value[src_indices]

            elif method == "nearest":
                idx_side_left = np.searchsorted(
                    old_times_val, new_times_val, side="left"
                )

                idx_L = np.clip(idx_side_left - 1, 0, len(old_times_val) - 1)
                idx_R = np.clip(idx_side_left, 0, len(old_times_val) - 1)

                dist_L = np.abs(new_times_val - old_times_val[idx_L])
                dist_R = np.abs(new_times_val - old_times_val[idx_R])

                use_L = dist_L < dist_R

                chosen_idx = np.where(use_L, idx_L, idx_R)
                chosen_dist = np.where(use_L, dist_L, dist_R)

                valid_n = np.ones(len(new_times_val), dtype=bool)
                if tolerance is not None:
                    tol_val = _to_time_value(tolerance)
                    valid_n &= chosen_dist <= tol_val

                if max_gap is not None:
                    limit = _to_time_value(max_gap)
                    valid_n &= chosen_dist <= limit

                valid_out = np.where(valid_n)[0]
                src_idx = chosen_idx[valid_n]
                new_data[valid_out] = self.value[src_idx]

            elif method is None:
                tol_val = 1e-9 if tolerance is None else _to_time_value(tolerance)

                idx_side_left = np.searchsorted(
                    old_times_val, new_times_val, side="left"
                )

                idx_L = np.clip(idx_side_left - 1, 0, len(old_times_val) - 1)
                idx_R = np.clip(idx_side_left, 0, len(old_times_val) - 1)
                dist_L = np.abs(new_times_val - old_times_val[idx_L])
                dist_R = np.abs(new_times_val - old_times_val[idx_R])

                min_dist = np.minimum(dist_L, dist_R)
                chosen_from_min = np.where(dist_L < dist_R, idx_L, idx_R)

                valid_exact = min_dist <= tol_val

                valid_out = np.where(valid_exact)[0]
                src_idx = chosen_from_min[valid_exact]
                new_data[valid_out] = self.value[src_idx]
            else:
                raise ValueError(f"Unknown asfreq method: {method}")

        # Construct new TimeSeries
        # Construct new TimeSeries
        # Convert dt back to original time unit to avoid plotting unit mismatch
        final_dt = target_dt
        if time_unit is not None and not is_dimless:
            try:
                final_dt = target_dt.to(time_unit)
            except (ValueError, u.UnitConversionError):
                pass

        return self.__class__(
            new_data,
            t0=u.Quantity(new_times_val[0], time_unit),
            dt=final_dt,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
        )

    # ===============================
    # resample - High-level Resample
    # ===============================

    def resample(
        self,
        rate: str | NumberLike | u.Quantity,
        *args: Any,
        ignore_nan: bool | None = None,
        **kwargs: Any,
    ) -> TimeSeriesResamplingMixin:
        """
        Resample the TimeSeries.

        If 'rate' is a time-string (e.g. '1s') or time Quantity, performs time-bin aggregation.
        Otherwise, performs signal processing resampling (gwpy standard).
        """
        is_time_bin = False
        if isinstance(rate, str):
            is_time_bin = True
        elif isinstance(rate, u.Quantity):
            if rate.unit.physical_type == "time":
                is_time_bin = True

        if is_time_bin:
            return self._resample_time_bin(rate, ignore_nan=ignore_nan, **kwargs)
        else:
            self._check_regular("Signal processing resample")
            from gwpy.timeseries import TimeSeries as BaseTimeSeries

            return BaseTimeSeries.resample(self, rate, *args, **kwargs)

    # ===============================
    # _resample_time_bin - Internal
    # ===============================

    def _resample_time_bin(
        self,
        rule: str | NumberLike | u.Quantity,
        *,
        agg: str | AggFunc = "mean",
        closed: Literal["left", "right"] = "left",
        label: Literal["left", "right", "center"] = "left",
        origin: Literal["t0", "gps0"] = "t0",
        offset: NumberLike | u.Quantity | None = None,
        align: Literal["floor", "ceil"] = "floor",
        min_count: int = 1,
        nan_policy: Literal["omit", "propagate"] = "omit",
        inplace: bool = False,
        ignore_nan: bool | None = None,
    ) -> TimeSeriesResamplingMixin:
        """Internal: Bin-based resampling."""
        if ignore_nan is not None:
            nan_policy = "omit" if ignore_nan else "propagate"
        # Default offset
        if offset is None:
            offset = 0 * u.s

        # 1. Parse rule to dt
        if isinstance(rule, str):
            import re

            match = re.match(r"([0-9\.]+)([a-zA-Z]+)", rule)
            if match:
                val, unit_str = match.groups()
                bin_dt = float(val) * u.Unit(unit_str)
            else:
                bin_dt = u.Quantity(rule)
        else:
            bin_dt = rule

        # 2. Setup Bins
        old_times_q = self.times
        time_unit = old_times_q.unit
        unit_is_dimensionless = (
            time_unit is None
            or time_unit == u.dimensionless_unscaled
            or time_unit.physical_type == "dimensionless"
        )
        if unit_is_dimensionless:
            time_unit = u.s

        is_dimensionless = unit_is_dimensionless

        bin_dt_val = None
        start_time_val = None
        stop_time_val = None

        if is_dimensionless and bin_dt.unit.physical_type == "time":
            time_unit = u.s
            bin_dt_val = bin_dt.to(u.s).value
            start_time_val = old_times_q[0].value
            stop_time_val = (
                self.span[1].value if hasattr(self.span[1], "value") else self.span[1]
            )
        else:
            bin_dt_val = u.Quantity(bin_dt, time_unit).value
            start_time_val = u.Quantity(old_times_q[0], time_unit).value
            stop_time_val = u.Quantity(self.span[1], time_unit).value

        # Origin logic
        if origin == "t0":
            origin_val = start_time_val
        elif origin == "gps0":
            origin_val = 0.0
        else:
            origin_val = 0.0

        try:
            offset_val = u.Quantity(offset).to(time_unit).value
        except u.UnitConversionError:
            offset_val = float(u.Quantity(offset).value)
        base_val = origin_val + offset_val

        # Grid alignment
        if align == "floor":
            k = np.floor((start_time_val - base_val) / bin_dt_val)
        elif align == "ceil":
            k = np.ceil((start_time_val - base_val) / bin_dt_val)
        else:
            k = np.floor((start_time_val - base_val) / bin_dt_val)

        grid_start = base_val + k * bin_dt_val

        # Create bin edges
        duration = stop_time_val - grid_start
        n_bins = int(np.ceil(duration / bin_dt_val))

        if n_bins <= 0:
            return self.__class__([], dt=bin_dt, unit=self.unit, name=self.name)

        edges = grid_start + np.arange(n_bins + 1) * bin_dt_val

        # 3. Aggregate
        if is_dimensionless and bin_dt.unit.physical_type == "time":
            old_times_val = old_times_q.value
        else:
            old_times_val = old_times_q.to(time_unit).value

        bin_indices = np.floor((old_times_val - grid_start) / bin_dt_val).astype(int)

        # Clip or filter valid bins
        valid_mask = (bin_indices >= 0) & (bin_indices < n_bins)

        valid_indices = bin_indices[valid_mask]
        valid_values = self.value[valid_mask]

        # Handle nan_policy
        if hasattr(self.value, "dtype") and (
            self.value.dtype.kind == "f" or self.value.dtype.kind == "c"
        ):
            has_nan = np.isnan(valid_values)
            if np.any(has_nan):
                if nan_policy == "omit":
                    non_nan_mask = ~has_nan
                    valid_indices = valid_indices[non_nan_mask]
                    valid_values = valid_values[non_nan_mask]

        # Bincount for aggregation
        if agg == "mean":
            if np.iscomplexobj(valid_values):
                sums = np.bincount(
                    valid_indices, weights=valid_values.real, minlength=n_bins
                ) + 1j * np.bincount(
                    valid_indices, weights=valid_values.imag, minlength=n_bins
                )
            else:
                sums = np.bincount(
                    valid_indices, weights=valid_values, minlength=n_bins
                )

            counts = np.bincount(valid_indices, minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = sums / counts
            if min_count > 0:
                means[counts < min_count] = np.nan
            out_data = means

        elif agg == "sum":
            if np.iscomplexobj(valid_values):
                sums = np.bincount(
                    valid_indices, weights=valid_values.real, minlength=n_bins
                ) + 1j * np.bincount(
                    valid_indices, weights=valid_values.imag, minlength=n_bins
                )
            else:
                sums = np.bincount(
                    valid_indices, weights=valid_values, minlength=n_bins
                )

            counts = np.bincount(valid_indices, minlength=n_bins)
            if min_count > 0:
                sums[counts < min_count] = np.nan
            out_data = sums

        elif agg in ["median", "min", "max", "std"]:
            try:
                from scipy.stats import binned_statistic

                if np.iscomplexobj(valid_values):
                    raise NotImplementedError(
                        f"Aggregation '{agg}' not supported for complex data yet"
                    )

                stat_val, _, _ = binned_statistic(
                    valid_indices,
                    valid_values,
                    statistic=agg,
                    bins=np.arange(n_bins + 1),
                )
                out_data = stat_val

                if min_count > 1:
                    counts = np.bincount(valid_indices, minlength=n_bins)
                    out_data[counts < min_count] = np.nan

            except ImportError:
                out_data = np.full(n_bins, np.nan)
                import warnings

                warnings.warn(
                    f"Using slow fallback for {agg} resampling (install scipy for speed)."
                )
                func: Callable[[np.ndarray], float]
                if agg == "median":
                    func = np.median
                elif agg == "min":
                    func = np.min
                elif agg == "max":
                    func = np.max
                elif agg == "std":
                    func = np.std
                else:

                    def func(x: np.ndarray) -> float:
                        return float(np.nan)

                for i in range(n_bins):
                    in_bin = valid_values[valid_indices == i]
                    if len(in_bin) >= min_count:
                        out_data[i] = func(in_bin)

        elif callable(agg):
            out_data = np.full(n_bins, np.nan)
            for i in range(n_bins):
                in_bin = valid_values[valid_indices == i]
                if len(in_bin) >= min_count:
                    out_data[i] = agg(in_bin)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        # 4. Result Times
        if label == "left":
            final_t0 = u.Quantity(edges[0], time_unit)
        elif label == "right":
            final_t0 = u.Quantity(edges[1], time_unit)
        else:
            final_t0 = u.Quantity(edges[0] + bin_dt_val / 2, time_unit)

        out_unit = self.unit
        if agg == "count":
            out_unit = u.dimensionless_unscaled

        # Convert dt back to original time unit to avoid plotting unit mismatch
        final_dt = bin_dt
        if time_unit is not None and not is_dimensionless:
            try:
                final_dt = bin_dt.to(time_unit)
            except (ValueError, u.UnitConversionError):
                pass

        return self.__class__(
            out_data,
            t0=final_t0,
            dt=final_dt,
            unit=out_unit,
            name=self.name,
            channel=self.channel,
        )
