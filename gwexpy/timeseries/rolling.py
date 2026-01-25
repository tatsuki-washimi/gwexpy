from __future__ import annotations

from collections.abc import Callable

import numpy as np
from astropy import units as u

try:
    import bottleneck as bn  # type: ignore

    _BN_AVAILABLE = True
except ImportError:  # pragma: no cover - bottleneck is optional
    _BN_AVAILABLE = False

from .collections import TimeSeriesDict, TimeSeriesList
from .matrix import TimeSeriesMatrix
from .timeseries import TimeSeries


def _resolve_window(window, dt):
    if isinstance(window, u.Quantity):
        if dt is None:
            raise ValueError("Cannot use time-like window without dt information.")
        dt_q = dt if isinstance(dt, u.Quantity) else u.Quantity(dt, "s")
        samples = window.to(dt_q.unit).value / dt_q.value
        win = int(np.round(samples))
    else:
        win = int(window)
    if win <= 0:
        raise ValueError("window must be a positive integer or time-like quantity.")
    return win


def _rolling_1d(
    arr, window: int, op: str, center: bool, min_count: int, nan_policy: str, ddof: int
):
    n = arr.shape[0]
    result = np.full(n, np.nan, dtype=np.result_type(arr, float))
    for i in range(n):
        if center:
            left = (window - 1) // 2
            right = window - left
            start = i - left
            end = i + right
        else:
            start = i - window + 1
            end = i + 1
        start = max(start, 0)
        end = min(end, n)
        window_vals = arr[start:end]
        if nan_policy == "propagate" and np.isnan(window_vals).any():
            continue
        valid = (
            window_vals
            if nan_policy == "propagate"
            else window_vals[~np.isnan(window_vals)]
        )
        if valid.size < min_count:
            continue
        if op == "mean":
            result[i] = np.mean(valid)
        elif op == "std":
            result[i] = np.std(valid, ddof=ddof)
        elif op == "median":
            result[i] = np.median(valid)
        elif op == "min":
            result[i] = np.min(valid)
        elif op == "max":
            result[i] = np.max(valid)
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"Unsupported op '{op}'")
    return result


def _rolling_numpy(
    data, window: int, op: str, center: bool, min_count: int, nan_policy: str, ddof: int
):
    flat = np.asarray(data)
    reshaped = flat.reshape(-1, flat.shape[-1])
    out = np.empty_like(reshaped, dtype=np.result_type(flat, float))
    for idx in range(reshaped.shape[0]):
        out[idx] = _rolling_1d(
            reshaped[idx], window, op, center, min_count, nan_policy, ddof
        )
    return out.reshape(flat.shape)


def _rolling_backend(
    data,
    window: int,
    op: str,
    center: bool,
    min_count: int,
    nan_policy: str,
    ddof: int,
    backend: str,
):
    if backend not in ("auto", "bottleneck", "numpy"):
        raise ValueError("backend must be 'auto', 'bottleneck', or 'numpy'")

    use_bn = (
        backend in ("auto", "bottleneck") and _BN_AVAILABLE and nan_policy == "omit"
    )
    if use_bn:
        bn_name = f"move_{op}"
        if not hasattr(bn, bn_name):
            use_bn = False
        else:
            func = getattr(bn, bn_name)
            kwargs = {"window": window, "min_count": min_count, "center": center}
            if op == "std":
                kwargs["ddof"] = ddof
            try:
                return func(data, **kwargs)
            except TypeError:
                # Older bottleneck versions may not support all kwargs; fall back to numpy.
                use_bn = False

    return _rolling_numpy(data, window, op, center, min_count, nan_policy, ddof)


def _apply_matrix_op(
    matrix: TimeSeriesMatrix,
    window,
    op: str,
    center: bool,
    min_count: int,
    nan_policy: str,
    ddof: int,
    backend: str,
):
    win = _resolve_window(window, matrix.dt)
    vals = _rolling_backend(
        matrix.value, win, op, center, min_count, nan_policy, ddof, backend
    )
    new_mat = matrix.copy()
    new_mat.value[:] = vals
    return new_mat


def _apply_timeseries_op(
    ts: TimeSeries,
    window,
    op: str,
    center: bool,
    min_count: int,
    nan_policy: str,
    ddof: int,
    backend: str,
):
    win = _resolve_window(window, ts.dt)
    vals = _rolling_backend(
        ts.value, win, op, center, min_count, nan_policy, ddof, backend
    )
    return ts.__class__(
        vals,
        t0=ts.t0,
        dt=ts.dt,
        unit=ts.unit,
        name=ts.name,
        channel=getattr(ts, "channel", None),
    )


def _map_collection(obj, func: Callable[[TimeSeries], TimeSeries]):
    if isinstance(obj, TimeSeriesDict):
        return obj.__class__({k: func(v) for k, v in obj.items()})
    if isinstance(obj, TimeSeriesList):
        return obj.__class__([func(v) for v in obj])
    raise TypeError(f"Unsupported collection type: {type(obj)}")


def rolling_mean(
    x,
    window,
    *,
    center: bool = False,
    min_count: int = 1,
    nan_policy: str = "omit",
    backend: str = "auto",
    ignore_nan: bool | None = None,
):
    """
    Rolling mean over the time axis.
    """
    if ignore_nan is not None:
        nan_policy = "omit" if ignore_nan else "propagate"

    if isinstance(x, TimeSeries):
        return _apply_timeseries_op(
            x, window, "mean", center, min_count, nan_policy, 0, backend
        )
    if isinstance(x, TimeSeriesMatrix):
        return _apply_matrix_op(
            x, window, "mean", center, min_count, nan_policy, 0, backend
        )
    if isinstance(x, (TimeSeriesDict, TimeSeriesList)):
        return _map_collection(
            x,
            lambda ts: rolling_mean(
                ts,
                window,
                center=center,
                min_count=min_count,
                nan_policy=nan_policy,
                backend=backend,
                ignore_nan=ignore_nan,
            ),
        )
    raise TypeError(f"Unsupported type for rolling_mean: {type(x)}")


def rolling_std(
    x,
    window,
    *,
    center: bool = False,
    min_count: int = 1,
    nan_policy: str = "omit",
    backend: str = "auto",
    ddof: int = 0,
    ignore_nan: bool | None = None,
):
    """
    Rolling standard deviation over the time axis.
    """
    if ignore_nan is not None:
        nan_policy = "omit" if ignore_nan else "propagate"

    if isinstance(x, TimeSeries):
        return _apply_timeseries_op(
            x, window, "std", center, min_count, nan_policy, ddof, backend
        )
    if isinstance(x, TimeSeriesMatrix):
        return _apply_matrix_op(
            x, window, "std", center, min_count, nan_policy, ddof, backend
        )
    if isinstance(x, (TimeSeriesDict, TimeSeriesList)):
        return _map_collection(
            x,
            lambda ts: rolling_std(
                ts,
                window,
                center=center,
                min_count=min_count,
                nan_policy=nan_policy,
                backend=backend,
                ddof=ddof,
                ignore_nan=ignore_nan,
            ),
        )
    raise TypeError(f"Unsupported type for rolling_std: {type(x)}")


def rolling_median(
    x,
    window,
    *,
    center: bool = False,
    min_count: int = 1,
    nan_policy: str = "omit",
    backend: str = "auto",
    ignore_nan: bool | None = None,
):
    """
    Rolling median over the time axis.
    """
    if ignore_nan is not None:
        nan_policy = "omit" if ignore_nan else "propagate"

    if isinstance(x, TimeSeries):
        return _apply_timeseries_op(
            x, window, "median", center, min_count, nan_policy, 0, backend
        )
    if isinstance(x, TimeSeriesMatrix):
        return _apply_matrix_op(
            x, window, "median", center, min_count, nan_policy, 0, backend
        )
    if isinstance(x, (TimeSeriesDict, TimeSeriesList)):
        return _map_collection(
            x,
            lambda ts: rolling_median(
                ts,
                window,
                center=center,
                min_count=min_count,
                nan_policy=nan_policy,
                backend=backend,
                ignore_nan=ignore_nan,
            ),
        )
    raise TypeError(f"Unsupported type for rolling_median: {type(x)}")


def rolling_min(
    x,
    window,
    *,
    center: bool = False,
    min_count: int = 1,
    nan_policy: str = "omit",
    backend: str = "auto",
    ignore_nan: bool | None = None,
):
    """
    Rolling minimum over the time axis.
    """
    if ignore_nan is not None:
        nan_policy = "omit" if ignore_nan else "propagate"

    if isinstance(x, TimeSeries):
        return _apply_timeseries_op(
            x, window, "min", center, min_count, nan_policy, 0, backend
        )
    if isinstance(x, TimeSeriesMatrix):
        return _apply_matrix_op(
            x, window, "min", center, min_count, nan_policy, 0, backend
        )
    if isinstance(x, (TimeSeriesDict, TimeSeriesList)):
        return _map_collection(
            x,
            lambda ts: rolling_min(
                ts,
                window,
                center=center,
                min_count=min_count,
                nan_policy=nan_policy,
                backend=backend,
                ignore_nan=ignore_nan,
            ),
        )
    raise TypeError(f"Unsupported type for rolling_min: {type(x)}")


def rolling_max(
    x,
    window,
    *,
    center: bool = False,
    min_count: int = 1,
    nan_policy: str = "omit",
    backend: str = "auto",
    ignore_nan: bool | None = None,
):
    """
    Rolling maximum over the time axis.
    """
    if ignore_nan is not None:
        nan_policy = "omit" if ignore_nan else "propagate"

    if isinstance(x, TimeSeries):
        return _apply_timeseries_op(
            x, window, "max", center, min_count, nan_policy, 0, backend
        )
    if isinstance(x, TimeSeriesMatrix):
        return _apply_matrix_op(
            x, window, "max", center, min_count, nan_policy, 0, backend
        )
    if isinstance(x, (TimeSeriesDict, TimeSeriesList)):
        return _map_collection(
            x,
            lambda ts: rolling_max(
                ts,
                window,
                center=center,
                min_count=min_count,
                nan_policy=nan_policy,
                backend=backend,
                ignore_nan=ignore_nan,
            ),
        )
    raise TypeError(f"Unsupported type for rolling_max: {type(x)}")
