from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    SupportsIndex,
    TypeVar,
    Union,
    cast,
    overload,
)

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
from astropy import units as u

# Import low-level algorithms from signal.preprocessing
from gwexpy.signal.preprocessing import (
    StandardizationModel,
    WhiteningModel,
)

from .utils import _coerce_t0_gps

if TYPE_CHECKING:
    from .matrix import TimeSeriesMatrix
    from .timeseries import TimeSeries

TimeSeriesT = TypeVar("TimeSeriesT", bound="TimeSeries")
TimeSeriesSequence = Sequence["TimeSeries"]

NumericArray: TypeAlias = npt.NDArray[np.number[Any]]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
SliceKey: TypeAlias = Union[SupportsIndex, slice, npt.NDArray[np.integer[Any]]]
BaseImputeMethod = Literal[
    "linear",
    "nearest",
    "slinear",
    "quadratic",
    "cubic",
    "ffill",
    "bfill",
    "mean",
    "median",
]
ImputeMethod = Union[BaseImputeMethod, Literal["interpolate"]]


def _limit_mask(
    nans: npt.NDArray[np.bool_],
    limit: int | None,
    *,
    direction: Literal["forward", "backward"] = "forward",
) -> npt.NDArray[np.bool_]:
    if limit is None:
        return np.zeros_like(nans, dtype=bool)
    limit = int(limit)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return nans.copy()
    mask = np.zeros_like(nans, dtype=bool)
    run_start = None
    n = len(nans)
    for i in range(n):
        if nans[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len > limit:
                    if direction == "backward":
                        mask[run_start : i - limit] = True
                    else:
                        mask[run_start + limit : i] = True
                run_start = None
    if run_start is not None:
        run_len = n - run_start
        if run_len > limit:
            if direction == "backward":
                mask[run_start : n - limit] = True
            else:
                mask[run_start + limit : n] = True
    return mask


def _ffill_numpy(val: NumericArray, limit: int | None = None) -> NumericArray:
    out = val.copy()
    have_last = False
    last_val = None
    run = 0
    for i in range(len(out)):
        if np.isnan(out[i]):
            if not have_last:
                continue
            if limit is None or run < limit:
                assert last_val is not None
                out[i] = last_val
            run += 1
        else:
            last_val = out[i]
            have_last = True
            run = 0
    return out


def _bfill_numpy(val: NumericArray, limit: int | None = None) -> NumericArray:
    out = val.copy()
    have_next = False
    next_val = None
    run = 0
    for i in range(len(out) - 1, -1, -1):
        if np.isnan(out[i]):
            if not have_next:
                continue
            if limit is None or run < limit:
                assert next_val is not None
                out[i] = next_val
            run += 1
        else:
            next_val = out[i]
            have_next = True
            run = 0
    return out


def align_timeseries_collection(
    series_list: TimeSeriesSequence,
    *,
    how: Literal["intersection", "union"] = "intersection",
    fill_value: Any = np.nan,
    method: str | None = None,
    tolerance: float | None = None,
) -> tuple[npt.NDArray[Any], u.Quantity, dict[str, Any]]:
    """
    Align a collection of TimeSeries to a common time axis.

    All time spans are treated as semi-open intervals [start, end) (end exclusive).
    The common time grid uses ``n_samples = ceil((end - start) / dt)``, so samples
    are generated at ``start + k * dt`` for ``k = 0..n_samples-1`` and never
    include the end point, consistent with [start, end).

    Parameters
    ----------
    series_list : list of TimeSeries
        Input series to align.
    how : str, optional
        "intersection" (default) or "union".
    fill_value : float, optional
        Value to fill missing data with when how="union". Default is np.nan.
    method : str, optional
        Interpolation method for resampling/alignment (e.g. 'linear', 'nearest', 'pad').
        Passed to TimeSeries.asfreq.
    tolerance : float, optional
        Tolerance for time comparison in seconds. Passed to TimeSeries.asfreq.

    Returns
    -------
    values : np.ndarray
        Shape (n_samples, n_channels).
    times : np.ndarray
        Common time axis (Time array).
    meta : dict
        Metadata including 'dt', 'epoch', 'channel_names', etc.

    Notes
    -----
    - ``how="intersection"`` computes the semi-open intersection of spans and
      raises ``ValueError`` if the intersection is empty.
    - ``how="union"`` spans the semi-open union of all series and uses
      ``fill_value`` outside each series' coverage.
    """
    if not series_list:
        raise ValueError("No timeseries provided to align.")

    # 1. Determine common sample rate (minimum rate / maximum dt)
    dts = []
    has_time_dt = False
    for ts in series_list:
        # Check regularity safely
        from .timeseries import TimeSeries

        if isinstance(ts, TimeSeries):
            is_regular = ts.is_regular
        else:
            # Fallback for BaseTimeSeries or other objects
            is_regular = getattr(ts, "regular", True)

        if not is_regular:
            # For irregular series, we can't use .dt safely.
            # We estimate an average dt for alignment purposes.
            times_val = np.asarray(ts.times)
            if len(times_val) > 1:
                avg_dt = (times_val[-1] - times_val[0]) / (len(times_val) - 1)
                dt_q = u.Quantity(avg_dt, ts.times.unit or u.s)
            else:
                # fallback if 1 point
                dt_q = u.Quantity(1.0, ts.times.unit or u.s)
        else:
            if ts.dt is None:
                raise ValueError(
                    "align_timeseries_collection requires dt for all series"
                )
            # Ensure dt is a Quantity
            dt_q = (
                ts.dt
                if isinstance(ts.dt, u.Quantity)
                else u.Quantity(ts.dt, u.dimensionless_unscaled)
            )

        dt_vals = np.asanyarray(dt_q.value)
        if np.any(dt_vals <= 0):
            raise ValueError("align_timeseries_collection requires dt > 0")
        dts.append(dt_q)

        # Check physical type safely
        dt_unit = getattr(dt_q, "unit", None)
        if dt_unit is not None and getattr(dt_unit, "physical_type", None) == "time":
            has_time_dt = True

    # 2. Determine time bounds and common unit
    # Check if we should operate in physical time (seconds)
    is_time_based = has_time_dt
    if not is_time_based:
        # If any series has time unit in its axis, force time-based
        for ts in series_list:
            t_unit = getattr(ts.times, "unit", None)
            if t_unit is not None and getattr(t_unit, "physical_type", None) == "time":
                is_time_based = True
                break
    if not is_time_based:
        # Treat dimensionless axes as GPS seconds by default
        all_dimless = True
        for dt_q in dts:
            unit = getattr(dt_q, "unit", None)
            phys = getattr(unit, "physical_type", None) if unit is not None else None
            if (
                unit is not None
                and unit != u.dimensionless_unscaled
                and phys != "dimensionless"
            ):
                all_dimless = False
                break
        if all_dimless:
            for ts in series_list:
                t_unit = getattr(ts.times, "unit", None)
                phys = (
                    getattr(t_unit, "physical_type", None)
                    if t_unit is not None
                    else None
                )
                if (
                    t_unit is not None
                    and t_unit != u.dimensionless_unscaled
                    and phys != "dimensionless"
                ):
                    all_dimless = False
                    break
        if all_dimless:
            is_time_based = True

    if is_time_based:
        # Determine base time unit. Prefer seconds but keep original if all are same.
        time_units = set()
        for dt_q in dts:
            if getattr(dt_q.unit, "physical_type", None) == "time":
                time_units.add(dt_q.unit)
        for ts in series_list:
            t_unit = getattr(ts.times, "unit", None)
            if t_unit is not None and getattr(t_unit, "physical_type", None) == "time":
                time_units.add(t_unit)

        common_time_unit = u.s
        if time_units and (len(time_units) > 1 or list(time_units)[0] != u.s):
            warnings.warn(
                f"Converting time units {time_units} to GPS seconds for alignment."
            )

        dt_candidates = []
        for dt_q in dts:
            # Safer access to physical_type
            phys = getattr(dt_q.unit, "physical_type", None)
            if dt_q.unit == u.dimensionless_unscaled or phys == "dimensionless":
                # Interpret dimensionless as common_time_unit
                dt_candidates.append(u.Quantity(dt_q.value, common_time_unit))
            elif phys == "time":
                dt_candidates.append(dt_q.to(common_time_unit))
            else:
                raise ValueError(
                    f"align_timeseries_collection requires time-like dt when time-based alignment is used: {dt_q}"
                )
        target_dt = max(dt_candidates)
    else:
        # Fallback to first unit found in dts or times
        common_time_unit = u.dimensionless_unscaled
        for dt_q in dts:
            if dt_q.unit != u.dimensionless_unscaled:
                common_time_unit = dt_q.unit
                break
        if common_time_unit == u.dimensionless_unscaled:
            for ts in series_list:
                t_unit = getattr(ts.times, "unit", None)
                if t_unit is not None and t_unit != u.dimensionless_unscaled:
                    common_time_unit = t_unit
                    break

        # Convert all to common_time_unit, treating dimensionless specifically
        dt_candidates = []
        for dt_q in dts:
            if dt_q.unit == u.dimensionless_unscaled:
                dt_candidates.append(u.Quantity(dt_q.value, common_time_unit))
            else:
                try:
                    dt_candidates.append(dt_q.to(common_time_unit))
                except u.UnitConversionError as exc:
                    raise ValueError(
                        f"Incompatible dt unit in collection: {dt_q.unit} vs {common_time_unit}"
                    ) from exc
        target_dt = max(dt_candidates)

    # Helper to get start/end in common unit
    def get_span_val(ts: "TimeSeries") -> tuple[float, float]:  # noqa: UP037
        ts_u = ts.times.unit if ts.times.unit is not None else u.dimensionless_unscaled

        # t0 conversion
        if is_time_based:
            t0_q = _coerce_t0_gps(ts.t0)
            if t0_q is None:
                raise ValueError("Time zero for TimeSeries could not be resolved.")
            if hasattr(t0_q, "to"):
                t0 = t0_q.to(common_time_unit).value
            else:
                t0 = float(t0_q)
        elif ts_u != common_time_unit:
            try:
                t0 = ts.t0.to(common_time_unit).value
            except u.UnitConversionError:
                raise ValueError(
                    f"Incompatible time units in collection: {ts_u} vs {common_time_unit}"
                )
        else:
            t0 = ts.t0.value

        # End conversion
        end_q = ts.span[1]

        # If end_q is dimensionless quantity and we are in time mode, treat as seconds value
        if is_time_based and (
            not hasattr(end_q, "unit")
            or end_q.unit == u.dimensionless_unscaled
            or end_q.unit is None
        ):
            dt_q = getattr(ts, "dt", None)
            if dt_q is not None:
                dt_unit = getattr(dt_q, "unit", None)
                phys = (
                    getattr(dt_unit, "physical_type", None)
                    if dt_unit is not None
                    else None
                )
                if (
                    dt_unit is None
                    or dt_unit == u.dimensionless_unscaled
                    or phys == "dimensionless"
                ):
                    dt_base = dt_q.value if hasattr(dt_q, "value") else dt_q
                    dt_val = u.Quantity(dt_base, u.s).to(common_time_unit).value
                else:
                    dt_val = u.Quantity(dt_q).to(common_time_unit).value
                end = t0 + (len(ts) * dt_val)
            else:
                end = end_q.value if hasattr(end_q, "value") else end_q
        elif hasattr(end_q, "to"):
            try:
                end = end_q.to(common_time_unit).value
            except u.UnitConversionError:
                # Backup: if unit mismatch but one is None? already checked.
                raise ValueError(
                    f"Incompatible span unit: {end_q.unit} vs {common_time_unit}"
                )
        else:
            # float or similar
            end = end_q

        return t0, end

    starts = []
    ends = []
    for ts in series_list:
        s, e = get_span_val(ts)
        starts.append(s)
        ends.append(e)

    def float_min(x: Sequence[float]) -> float:
        return min(x)

    def float_max(x: Sequence[float]) -> float:
        return max(x)

    if how == "intersection":
        common_t0 = float_max(starts)
        common_end = float_min(ends)
        # Semi-open intersection is empty when end <= start.
        if common_end <= common_t0:
            raise ValueError(
                f"No overlap found. common_t0={common_t0}, common_end={common_end}"
            )
    elif how == "union":
        common_t0 = float_min(starts)
        common_end = float_max(ends)
    else:
        raise ValueError(f"Unknown alignment how='{how}'.")

    # 3. Create common time axis
    # Use common_time_unit for output
    out_unit = common_time_unit

    duration = common_end - common_t0
    target_dt_s = target_dt.to(common_time_unit).value

    if duration <= 0:
        n_samples = 0
    else:
        # Use ceil to ensure we cover the full range, matching asfreq behavior
        n_samples = int(np.ceil(duration / target_dt_s))

    # Create common times in Seconds
    common_times_s = common_t0 + np.arange(n_samples) * target_dt_s
    # Convert to output unit
    common_times = (common_times_s * common_time_unit).to(out_unit)

    if n_samples <= 0 and how == "intersection":
        # Fallback for empty
        pass

    # 4. Fill matrix
    n_channels = len(series_list)
    # Determine output dtype (promote if needed)
    # For now assume float or complex
    is_complex = any(ts.dtype.kind == "c" for ts in series_list)
    dtype = np.complex128 if is_complex else np.float64

    if fill_value is None:
        fill_value = np.nan

    values: npt.NDArray[Any] = np.full((n_samples, n_channels), fill_value, dtype=dtype)

    for i, ts in enumerate(series_list):
        # Align using asfreq
        # We specify the target grid by using origin=common_t0.
        # This ensures grid points match common_times exactly phase-wise.

        # Origin must be compatible with ts.times unit or convertible
        # If is_time_based=True, asfreq will coerce dimensionless ts to seconds.
        # So we should pass origin in seconds (common_time_unit).

        if is_time_based:
            # Regardless of ts.unit, we pass origin as Quantity(common_time_unit)
            origin_val = u.Quantity(common_t0, common_time_unit)
        elif ts.times.unit is None:
            origin_val = u.Quantity(common_t0, u.dimensionless_unscaled)
        else:
            origin_val = u.Quantity(common_t0, common_time_unit).to(ts.times.unit)

        # We process the whole series onto the grid defined by common_t0
        # asfreq returns the coverage of the original series but on the new grid.

        ts_aligned = ts.asfreq(
            target_dt,
            method=method,
            fill_value=fill_value,
            origin=origin_val,
            tolerance=tolerance,
            align="floor",
        )

        # Calculate offset of ts_aligned.t0 relative to common_t0
        # Both are on the grid defined by common_t0 and target_dt.
        if hasattr(ts_aligned.t0, "to"):
            t0_aligned_s = ts_aligned.t0.to(common_time_unit).value
        else:
            t0_aligned_s = float(ts_aligned.t0)

        # Index offset
        # Since we aligned to the grid, the difference should be integer multiple of dt
        # We use floor(x + 0.5) to snap to nearest integer safely.
        offset = int(np.floor((t0_aligned_s - common_t0) / target_dt_s + 0.5))

        # Copy valid overlap into values buffer
        # Buffer range: [0, n_samples)
        # TS range: [offset, offset + len)

        # Overlap in buffer coordinates
        buf_start = max(0, offset)
        buf_end = min(n_samples, offset + len(ts_aligned))

        # Overlap in TS coordinates
        ts_start = max(0, -offset)
        ts_end = ts_start + (buf_end - buf_start)

        if buf_end > buf_start:
            values[buf_start:buf_end, i] = ts_aligned.value[ts_start:ts_end]

    meta = {
        "t0": u.Quantity(common_t0, common_time_unit).to(out_unit),
        "dt": target_dt,
        "channel_names": [ts.name for ts in series_list],
    }

    return values, common_times, meta


def _impute_1d(
    val_1d: NumericArray,
    x: npt.NDArray[Any],
    method: BaseImputeMethod,
    has_gap_constraint: bool,
    gap_threshold: float | None,
    limit: int | None = None,
) -> NumericArray:
    """Internal 1D imputation core."""

    nans_1d = np.isnan(val_1d)
    if not np.any(nans_1d):
        return val_1d

    valid_1d = ~nans_1d
    if not np.any(valid_1d):
        return val_1d  # All NaNs

    x_valid = x[valid_1d]
    y_valid = val_1d[valid_1d]
    apply_limit_mask = True

    fill_value: Any
    if has_gap_constraint:
        fill_value = (np.nan, np.nan)
    else:
        fill_value = "extrapolate"

    if method in ["linear", "nearest", "slinear", "quadratic", "cubic"]:
        from scipy.interpolate import interp1d

        if np.iscomplexobj(val_1d):
            f_real = interp1d(
                x_valid,
                y_valid.real,
                kind=method,
                bounds_error=False,
                fill_value=fill_value,
            )
            f_imag = interp1d(
                x_valid,
                y_valid.imag,
                kind=method,
                bounds_error=False,
                fill_value=fill_value,
            )
            val_1d[nans_1d] = f_real(x[nans_1d]) + 1j * f_imag(x[nans_1d])
        else:
            f = interp1d(
                x_valid, y_valid, kind=method, bounds_error=False, fill_value=fill_value
            )
            val_1d[nans_1d] = f(x[nans_1d])
    elif method == "ffill":
        try:
            import pandas as pd
        except ImportError:
            val_1d[:] = _ffill_numpy(val_1d, limit=limit)
        else:
            val_1d[:] = pd.Series(val_1d).ffill(limit=limit).values
        apply_limit_mask = False
    elif method == "bfill":
        try:
            import pandas as pd
        except ImportError:
            val_1d[:] = _bfill_numpy(val_1d, limit=limit)
        else:
            val_1d[:] = pd.Series(val_1d).bfill(limit=limit).values
        apply_limit_mask = False
    elif method == "mean":
        mean_val = float(np.nanmean(val_1d))  # type: ignore[arg-type]
        val_1d[nans_1d] = mean_val
    elif method == "median":
        median_val = float(np.nanmedian(val_1d))  # type: ignore[arg-type]
        val_1d[nans_1d] = median_val

    if has_gap_constraint and gap_threshold is not None:
        valid_indices = np.where(valid_1d)[0]
        if len(valid_indices) > 1:
            diffs = np.diff(x_valid)
            big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]
            for idx in big_gaps:
                t_start = x_valid[idx]
                t_end = x_valid[idx + 1]
                mask = (x > t_start) & (x < t_end)
                val_1d[mask] = np.nan
        if len(valid_indices) > 0:
            val_1d[: valid_indices[0]] = np.nan
            val_1d[valid_indices[-1] + 1 :] = np.nan

    if limit is not None and apply_limit_mask:
        limit_mask = _limit_mask(nans_1d, limit, direction="forward")
        if np.any(limit_mask):
            val_1d[limit_mask] = np.nan

    return val_1d


@overload
def impute_timeseries(
    ts: TimeSeriesT,
    *,
    method: str = ...,
    limit: int | None = ...,
    axis: int | str = ...,
    max_gap: float | u.Quantity | None = ...,
    **kwargs: Any,
) -> TimeSeriesT: ...


@overload
def impute_timeseries(
    ts: npt.ArrayLike,
    *,
    method: str = ...,
    limit: int | None = ...,
    axis: int | str = ...,
    max_gap: float | u.Quantity | None = ...,
    **kwargs: Any,
) -> np.ndarray: ...


def impute_timeseries(
    ts: TimeSeriesT | npt.ArrayLike,
    *,
    method: str = "linear",
    limit: int | None = None,
    axis: int | str = -1,
    max_gap: float | u.Quantity | None = None,
    **kwargs: Any,
) -> TimeSeriesT | np.ndarray:
    """Impute missing values in a TimeSeries or array. Supports multi-dimensional data.

    Notes
    -----
    - ``method="interpolate"`` is normalized to ``"linear"``.
    - ``axis="time"`` is treated as ``axis=-1``.
    - ``limit`` caps the number of consecutive NaNs that are filled; any
      positions beyond ``limit`` in a run are reverted back to NaN.
    - ``max_gap`` sets a threshold on the spacing between valid samples: after
      interpolation, any interior region between valid points with a gap larger
      than ``max_gap`` is reverted to NaN, and leading/trailing extrapolated
      regions are also reverted to NaN.
    - When a ``TimeSeries`` is provided, metadata (times/t0/dt/name/unit/channel)
      are preserved. If imputation promotes integer data to float, a new
      ``TimeSeries`` is constructed with the same metadata.
    """
    from gwexpy.timeseries.timeseries import TimeSeries

    ts_obj: TimeSeries | None
    if isinstance(ts, TimeSeries):
        ts_obj = ts
        val = ts.value.copy()
    else:
        ts_obj = None
        # Use value attribute if available (proxy objects), otherwise assume array-like
        val = np.asarray(getattr(ts, "value", ts)).copy()

    if val.ndim == 0:
        val = val.reshape(-1)

    if method == "interpolate":
        method = "linear"
    method = cast(BaseImputeMethod, method)

    if axis == "time":
        axis_idx = -1
    else:
        axis_idx = int(axis)
    axis_idx %= val.ndim

    def _axis_indexer(mask: SliceKey) -> tuple[SliceKey, ...]:
        return tuple(mask if i == axis_idx else slice(None) for i in range(val.ndim))

    times_val: npt.NDArray[Any] | None = None
    time_unit = None
    if ts_obj is not None:
        try:
            times_val = ts_obj.times.value
            time_unit = ts_obj.times.unit
        except AttributeError:
            pass

    if times_val is None:
        times_val = np.arange(val.shape[axis_idx])

    gap_threshold: float | None = None
    if max_gap is not None:
        if isinstance(max_gap, u.Quantity):
            if time_unit is not None:
                gap_threshold = max_gap.to(time_unit).value
            else:
                gap_threshold = (
                    max_gap.to(u.s).value
                    if max_gap.unit.physical_type == "time"
                    else float(max_gap.value)
                )
        else:
            gap_threshold = float(max_gap)
    has_gap_constraint = gap_threshold is not None

    nans = np.isnan(val)
    if not np.any(nans):
        if ts_obj is not None:
            return cast(TimeSeriesT, ts_obj.copy())
        return val

    other_axes = [i for i in range(val.ndim) if i != axis_idx]
    nans_common: bool
    if other_axes:
        nans_reduced = np.any(nans, axis=tuple(other_axes))
        reshape_dims = [1] * val.ndim
        reshape_dims[axis_idx] = -1
        nans_broadcast = nans_reduced.reshape(reshape_dims)
        nans_common = bool(np.all(nans == nans_broadcast))
    else:
        nans_common = True
        nans_reduced = nans

    if nans_common and method not in ["ffill", "bfill", "mean", "median"]:
        valid_mask = ~nans_reduced
        if not np.any(valid_mask):
            if ts_obj is not None:
                return cast(TimeSeriesT, ts_obj.copy())
            return val

        x_valid = times_val[valid_mask]
        y_valid = val[_axis_indexer(valid_mask)]

        fill_value: Any
        if has_gap_constraint:
            fill_value = (np.nan, np.nan)
        else:
            fill_value = "extrapolate"

        if method in ["linear", "nearest", "slinear", "quadratic", "cubic"]:
            from scipy.interpolate import interp1d

            if np.iscomplexobj(val):
                f_real = interp1d(
                    x_valid,
                    y_valid.real,
                    kind=method,
                    axis=axis_idx,
                    bounds_error=False,
                    fill_value=fill_value,
                )
                f_imag = interp1d(
                    x_valid,
                    y_valid.imag,
                    kind=method,
                    axis=axis_idx,
                    bounds_error=False,
                    fill_value=fill_value,
                )
                nan_mask_idx = np.where(nans_reduced)[0]
                val[_axis_indexer(nan_mask_idx)] = f_real(
                    times_val[nan_mask_idx]
                ) + 1j * f_imag(times_val[nan_mask_idx])
            else:
                f = interp1d(
                    x_valid,
                    y_valid,
                    kind=method,
                    axis=axis_idx,
                    bounds_error=False,
                    fill_value=fill_value,
                )
                nan_mask_idx = np.where(nans_reduced)[0]
                val[_axis_indexer(nan_mask_idx)] = f(times_val[nan_mask_idx])

        if has_gap_constraint:
            assert gap_threshold is not None
            diffs = np.diff(x_valid)
            big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]
            for idx in big_gaps:
                t_start = x_valid[idx]
                t_end = x_valid[idx + 1]
                mask_idx = np.where((times_val > t_start) & (times_val < t_end))[0]
                val[_axis_indexer(mask_idx)] = np.nan
            if len(x_valid) > 0:
                lead_mask = np.where(times_val < x_valid[0])[0]
                tail_mask = np.where(times_val > x_valid[-1])[0]
                if lead_mask.size:
                    val[_axis_indexer(lead_mask)] = np.nan
                if tail_mask.size:
                    val[_axis_indexer(tail_mask)] = np.nan

        if limit is not None:
            limit_mask = _limit_mask(nans_reduced, limit, direction="forward")
            if np.any(limit_mask):
                val[_axis_indexer(limit_mask)] = np.nan
    else:
        it = np.ndindex(tuple(s for i, s in enumerate(val.shape) if i != axis_idx))
        for idxs in it:
            slc: list[SliceKey] = [slice(None)] * val.ndim
            j = 0
            for i in range(val.ndim):
                if i != axis_idx:
                    slc[i] = idxs[j]
                    j += 1
            val[tuple(slc)] = _impute_1d(
                val[tuple(slc)],
                times_val,
                method,
                has_gap_constraint,
                gap_threshold,
                limit=limit,
            )

    if ts_obj is not None:
        target_dtype = val.dtype
        needs_cast = np.issubdtype(
            ts_obj.value.dtype, np.integer
        ) and target_dtype.kind in (
            "f",
            "c",
        )
        if needs_cast:
            val = val.astype(np.result_type(val, np.float64))
            if getattr(ts_obj, "dt", None) is None or (
                hasattr(ts_obj, "is_regular") and not ts_obj.is_regular
            ):
                new_ts = ts_obj.__class__(
                    val,
                    times=ts_obj.times,
                    name=ts_obj.name,
                    unit=ts_obj.unit,
                    channel=getattr(ts_obj, "channel", None),
                )
            else:
                new_ts = ts_obj.__class__(
                    val,
                    t0=ts_obj.t0,
                    dt=ts_obj.dt,
                    name=ts_obj.name,
                    unit=ts_obj.unit,
                    channel=getattr(ts_obj, "channel", None),
                )
        else:
            new_ts = ts_obj.copy()
            new_ts.value[:] = val
        return cast(TimeSeriesT, new_ts)

    return val


def standardize_timeseries(
    ts: TimeSeries,
    *,
    method: str = "zscore",
    ddof: int = 0,
    robust: bool | None = None,
) -> tuple[TimeSeries, StandardizationModel]:
    """
    Standardize a TimeSeries.

    This function standardizes the input time series data by removing the
    location (mean or median) and scaling by a dispersion measure (standard
    deviation or MAD), resulting in a dimensionless output.

    Parameters
    ----------
    ts : TimeSeries
        Input time series.
    method : str, optional
        Standardization method. Default is ``'zscore'``.

        - ``'zscore'``: Subtract the mean (`nanmean`) and divide by the
          standard deviation (`nanstd` with ``ddof``).
        - ``'robust'``: Subtract the median (`nanmedian`) and divide by
          the scale defined as ``1.4826 * MAD``, where MAD is the median
          absolute deviation from the median.

    ddof : int, optional
        Delta degrees of freedom for standard deviation calculation when
        ``method='zscore'``. Default is 0.
    robust : bool or None, optional
        If True, forces ``method='robust'``. Deprecated; prefer using
        ``method='robust'`` directly.

    Returns
    -------
    standardized_ts : TimeSeries
        Standardized time series. The unit is always ``dimensionless_unscaled``.
        All metadata (``t0``, ``dt``, ``times``, ``name``, ``channel``) are
        preserved from the original series.
    model : StandardizationModel
        Model object containing the location (``mean``) and scale parameters.
        Can be used to inverse-transform back to original scale.

    Notes
    -----
    - NaN values are ignored during computation and remain NaN in the output.
    - If the scale (std or MAD) is zero (e.g., constant series), scale is set
      to 1.0 to avoid division by zero, resulting in an all-zero output.
      A warning is emitted in this case.
    - For ``method='zscore'``: ``standardized = (x - nanmean(x)) / nanstd(x, ddof)``
    - For ``method='robust'``: ``standardized = (x - nanmedian(x)) / (1.4826 * MAD)``
    - Output unit is always ``u.dimensionless_unscaled`` regardless of input unit.
    - Integer input dtypes are promoted to float64.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries
    >>> from gwexpy.timeseries.preprocess import standardize_timeseries
    >>> ts = TimeSeries([1, 2, 3, 4, 5], dt=1)
    >>> ts_std, model = standardize_timeseries(ts, method='zscore')
    >>> print(ts_std.unit)  # dimensionless
    """
    if robust is True:
        method = "robust"

    val = ts.value
    if robust or method == "robust":
        med = np.nanmedian(val)
        mad = np.nanmedian(np.abs(val - med))
        scale = 1.4826 * mad
        if scale == 0:
            warnings.warn(
                "MAD is zero, setting scale to 1.0 to avoid division by zero."
            )
            scale = 1.0
    elif method == "zscore":
        med = np.nanmean(val)
        scale = np.nanstd(val, ddof=ddof)
        if scale == 0:
            warnings.warn(
                "Standard deviation is zero, setting scale to 1.0 to avoid division by zero."
            )
            scale = 1.0
    else:
        raise ValueError(
            f"Unknown standardization method '{method}'. "
            f"Supported methods are 'zscore', 'robust'."
        )

    # Handle dtype. If input is integer, standardization results require float.
    if np.issubdtype(ts.value.dtype, np.integer):
        val_float = ts.value.astype("float64")
        new_ts = ts.__class__(
            val_float,
            t0=ts.t0,
            dt=ts.dt,
            name=ts.name,
            unit=u.dimensionless_unscaled,
            channel=getattr(ts, "channel", None),
        )
    else:
        new_ts = ts.copy()

    # Always set unit to dimensionless
    if hasattr(new_ts, "override_unit"):
        new_ts.override_unit(u.dimensionless_unscaled)
    else:
        try:
            new_ts.unit = u.dimensionless_unscaled
        except AttributeError:
            pass

    new_ts.value[:] = (val - med) / scale

    model = StandardizationModel(
        mean=np.array([med]), scale=np.array([scale]), axis="time"
    )
    return new_ts, model


def standardize_matrix(
    matrix: TimeSeriesMatrix,
    *,
    axis: Literal["time", "channel"] = "time",
    method: str = "zscore",
    ddof: int = 0,
    robust: bool | None = None,
) -> TimeSeriesMatrix:
    """
    Standardize a TimeSeriesMatrix.

    This function standardizes the input matrix data by removing the location
    (mean or median) and scaling by a dispersion measure (standard deviation
    or MAD), resulting in a dimensionless output.

    Parameters
    ----------
    matrix : TimeSeriesMatrix
        Input time series matrix with shape ``(n_rows, n_cols, n_time)`` or
        ``(n_channels, n_time)``.
    axis : str
        Axis along which to standardize. Must be one of:

        - ``'time'``: Standardize along the time axis (last axis). Each
          channel/element is standardized independently over time.
        - ``'channel'``: Standardize along the channel axes ``(0, 1)`` or
          ``(0,)`` for 2D. Each time sample is standardized independently
          across all channels.

        Any other value raises ``ValueError``.

    method : str, optional
        Standardization method. Default is ``'zscore'``.

        - ``'zscore'``: Subtract the mean (`nanmean`) and divide by the
          standard deviation (`nanstd` with ``ddof``).
        - ``'robust'``: Subtract the median (`nanmedian`) and divide by
          the scale defined as ``1.4826 * MAD``, where MAD is the median
          absolute deviation from the median.

    ddof : int, optional
        Delta degrees of freedom for standard deviation calculation when
        ``method='zscore'``. Default is 0.
    robust : bool or None, optional
        If True, forces ``method='robust'``. Deprecated; prefer using
        ``method='robust'`` directly.

    Returns
    -------
    standardized_matrix : TimeSeriesMatrix
        Standardized matrix. The unit is always ``dimensionless_unscaled``.
        Metadata (``t0``, ``dt``, ``channel_names``) are preserved.

    Notes
    -----
    - NaN values are ignored during computation and remain NaN in the output.
    - If the scale (std or MAD) is zero for any slice, scale is set to 1.0
      to avoid division by zero.
    - For ``method='zscore'``: ``standardized = (x - nanmean) / nanstd``
    - For ``method='robust'``: ``standardized = (x - nanmedian) / (1.4826 * MAD)``
    - Output unit is always ``dimensionless_unscaled`` regardless of input unit.

    Raises
    ------
    ValueError
        If ``axis`` is not ``'time'`` or ``'channel'``.
        If ``method`` is not ``'zscore'`` or ``'robust'``.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeriesMatrix
    >>> from gwexpy.timeseries.preprocess import standardize_matrix
    >>> mat = TimeSeriesMatrix(np.random.randn(2, 3, 100), dt=0.01)
    >>> mat_std = standardize_matrix(mat, axis='time')
    """
    # Validate axis strictly
    if axis not in ("time", "channel"):
        raise ValueError(
            f"axis must be 'time' or 'channel', got '{axis}'. "
            f"Use axis='time' to standardize each channel over time, or "
            f"axis='channel' to standardize each time sample across channels."
        )

    if robust is True:
        method = "robust"

    if method not in ("zscore", "robust"):
        raise ValueError(
            f"Unknown standardization method '{method}'. "
            f"Supported methods are 'zscore', 'robust'."
        )

    val = matrix.value.copy()

    # TimeSeriesMatrix layout: (rows, cols, time) or (channels, time).
    # Time is always the last axis (-1).
    # "time" -> operate along time axis (normalize each channel independently)
    # "channel" -> operate along all non-time axes (normalize each time sample)
    np_axis: int | tuple[int, ...]
    if axis == "time":
        np_axis = -1
    else:  # axis == "channel"
        # For 3D: (0, 1), for 2D: (0,)
        np_axis = tuple(range(val.ndim - 1)) if val.ndim > 1 else 0

    if method == "robust":
        med = np.nanmedian(val, axis=np_axis, keepdims=True)
        mad = np.nanmedian(np.abs(val - med), axis=np_axis, keepdims=True)
        scale = 1.4826 * mad
    else:  # zscore
        med = np.nanmean(val, axis=np_axis, keepdims=True)
        scale = np.nanstd(val, axis=np_axis, ddof=ddof, keepdims=True)

    scale = np.where(scale == 0, 1.0, scale)

    # Create new matrix with same metadata
    if np.issubdtype(matrix.value.dtype, np.integer):
        val_float = matrix.value.astype("float64")
        new_mat = matrix.__class__(val_float, t0=matrix.t0, dt=matrix.dt)
        if hasattr(matrix, "channel_names"):
            try:
                channel_names = getattr(matrix, "channel_names")
            except AttributeError:
                channel_names = None
            if channel_names is not None:
                new_mat.channel_names = channel_names
    else:
        new_mat = matrix.copy()

    # Update values
    new_mat.value[:] = (val - med) / scale

    # Set unit to dimensionless
    if hasattr(new_mat, "unit"):
        try:
            new_mat.unit = u.dimensionless_unscaled
        except AttributeError:
            pass

    return new_mat


def whiten_matrix(
    matrix: TimeSeriesMatrix,
    *,
    method: Literal["pca", "zca"] = "pca",
    eps: float | str | None = "auto",
    n_components: int | None = None,
) -> tuple[TimeSeriesMatrix, WhiteningModel]:
    """
    Whiten a TimeSeriesMatrix using PCA or ZCA whitening.

    Whitening transforms the data so that the covariance matrix becomes the
    identity matrix. This is useful for decorrelating features before machine
    learning or for visualization.

    Parameters
    ----------
    matrix : TimeSeriesMatrix
        Input time series matrix with shape ``(n_rows, n_cols, n_time)`` or
        ``(n_channels, n_time)``. The last axis is assumed to be time.
    method : str, optional
        Whitening method. Default is ``'pca'``.

        - ``'pca'``: PCA whitening, where ``W = D^{-1/2} @ U^T``.
          The output is in the principal component space and always has
          a flattened shape ``(n_features, 1, n_time)`` or
          ``(n_components, 1, n_time)`` if dimensionality is reduced.
        - ``'zca'``: ZCA (Zero-phase Component Analysis) whitening, where
          ``W = U @ D^{-1/2} @ U^T``. When ``n_components=None``, the output
          retains the original shape ``(n_rows, n_cols, n_time)``. When
          ``n_components`` is specified, the output is flattened to
          ``(n_components, 1, n_time)``.

    eps : float or str or None, optional
        Small constant added to eigenvalues for numerical stability
        (prevents division by zero). If ``'auto'`` or ``None`` (default),
        it is determined from data variance using ``safe_epsilon``.
    n_components : int or None, optional
        Number of components to keep. If None (default), all components are
        retained.

    Returns
    -------
    whitened_matrix : TimeSeriesMatrix
        Whitened matrix. The unit is always ``dimensionless_unscaled``.

        - For PCA: Always ``(n_features, 1, n_time)`` or ``(n_components, 1, n_time)``.
        - For ZCA with ``n_components=None``: Same shape as input.
        - For ZCA with ``n_components`` specified: ``(n_components, 1, n_time)``.

    model : WhiteningModel
        Model object containing the mean and whitening matrix ``W``.
        Can be used to inverse-transform back to original space.

    Notes
    -----
    - The input matrix is flattened to ``(n_time, n_features)`` where
      ``n_features = n_rows * n_cols`` for computation.
    - Covariance is computed as ``cov(X_centered)``, then SVD is applied:
      ``U, S, Vt = svd(cov)``.
    - Regularization via ``eps`` ensures stability when eigenvalues are small:
      ``D^{-1/2} = diag(1 / sqrt(S + eps))``.
    - Output unit is always ``dimensionless_unscaled``.
    - After whitening, ``cov(X_whitened) ≈ I`` (identity matrix).

    Raises
    ------
    ValueError
        If ``method`` is not ``'pca'`` or ``'zca'``.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeriesMatrix
    >>> from gwexpy.timeseries.preprocess import whiten_matrix
    >>> mat = TimeSeriesMatrix(np.random.randn(2, 3, 100), dt=0.01)
    >>> mat_w, model = whiten_matrix(mat, method='pca')
    >>> # PCA output is flattened
    >>> print(mat_w.shape)  # (6, 1, 100)
    """
    from gwexpy.numerics import SAFE_FLOOR_STRAIN

    if method not in ("pca", "zca"):
        raise ValueError(f"method must be 'pca' or 'zca', got '{method}'")

    # Store original shape for ZCA reshape
    original_shape = matrix.shape  # e.g., (n_rows, n_cols, n_time)
    n_time = original_shape[-1]
    n_features = int(np.prod(original_shape[:-1]))

    # Reshape to (features, time) -> (time, features)
    X_features = matrix.value.reshape(-1, n_time)
    X_T = X_features.T  # (time, features)

    mean = np.mean(X_T, axis=0)
    X_centered = X_T - mean

    # Resolve eps
    if eps is None or (isinstance(eps, str) and eps == "auto"):
        # For whitening, we want eps to be relative to the eigenvalues of the covariance matrix.
        # Eigenvalues have units of variance. std(X)**2 is a good proxy for the scale of eigenvalues.
        var = np.nanvar(X_centered)
        eps_val = max(SAFE_FLOOR_STRAIN, var * 1e-6)
    else:
        eps_val = float(cast(float, eps))

    # Handle 1D case (single feature)
    if n_features == 1:
        cov = np.array([[np.var(X_centered)]])
    else:
        cov = np.cov(X_centered, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])

    U, S, Vt = np.linalg.svd(cov)

    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + eps_val))

    if method == "pca":
        W = S_inv_sqrt @ U.T
    else:  # zca
        W = U @ S_inv_sqrt @ U.T

    # Apply dimensionality reduction
    if n_components is not None:
        if method == "zca":
            warnings.warn(
                "n_components with ZCA whitening produces flattened output; "
                "original spatial structure is lost."
            )
        W = W[:n_components, :]
    X_whitened = X_centered @ W.T  # (time, features)

    cls = matrix.__class__

    # Determine output shape based on method and n_components
    if method == "pca":
        # PCA always produces flattened output: (features, 1, time)
        new_data = X_whitened.T[:, None, :]  # (output_features, 1, time)
        new_mat = cls(new_data, t0=matrix.t0, dt=matrix.dt)
    else:  # zca
        if n_components is None:
            # ZCA with full components: reshape to original shape
            new_val = X_whitened.T.reshape(original_shape)
            new_mat = matrix.copy()
            new_mat.value[:] = new_val
        else:
            # ZCA with dimensionality reduction: flattened output
            new_data = X_whitened.T[:, None, :]  # (n_components, 1, time)
            new_mat = cls(new_data, t0=matrix.t0, dt=matrix.dt)

    # Set unit to dimensionless
    if hasattr(new_mat, "unit"):
        try:
            new_mat.unit = u.dimensionless_unscaled
        except AttributeError:
            pass

    model = WhiteningModel(mean, W)
    return new_mat, model
