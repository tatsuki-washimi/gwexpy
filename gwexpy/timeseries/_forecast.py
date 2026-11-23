"""Optional Chronos forecast support for GWexpy time series."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from astropy import units as u

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
_CHRONOS2_MODEL_ID = "amazon/chronos-2"
_CHRONOS2_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
# Chronos-2 trained quantile range for amazon/chronos-2 at the pinned revision.
# Used as a fallback if a pipeline instance does not expose `.quantiles`.
_CHRONOS2_DEFAULT_QUANTILE_RANGE = (0.01, 0.99)


def _normalize_quantiles(
    quantiles: tuple[float, ...] | list[float] | None,
) -> tuple[float, ...]:
    """Validate quantile levels and normalize them to unique float keys."""
    if quantiles is None:
        values: tuple[object, ...] = DEFAULT_QUANTILES
    else:
        if isinstance(quantiles, (str, bytes)):
            raise ValueError("quantiles must be a sequence of real numeric levels")
        try:
            values = tuple(quantiles)
        except TypeError as exc:
            raise ValueError(
                "quantiles must be a sequence of real numeric levels"
            ) from exc
    if not values:
        raise ValueError("quantiles must contain levels strictly between 0 and 1")

    levels: list[float] = []
    for level in values:
        if isinstance(level, (bool, np.bool_)) or not isinstance(level, Real):
            raise ValueError("quantiles must contain only real numeric levels")
        converted = float(level)
        if not np.isfinite(converted) or not 0 < converted < 1:
            raise ValueError(
                "quantiles must contain finite levels strictly between 0 and 1"
            )
        levels.append(converted)
    if len(set(levels)) != len(levels):
        raise ValueError("quantiles must not contain duplicate levels")
    if 0.5 not in levels:
        raise ValueError("quantiles must include 0.5 so point uses the median")
    return tuple(levels)


def _normalize_horizon_seconds(horizon: Real | u.Quantity) -> float:
    """Validate time-span ``horizon`` and convert it to positive finite seconds."""
    if isinstance(horizon, (bool, np.bool_)):
        raise TypeError(
            "horizon must be a positive real number (seconds) or time Quantity"
        )
    if isinstance(horizon, u.Quantity):
        if horizon.ndim != 0:
            raise ValueError("horizon must be a scalar time span")
        raw = horizon.value
        if isinstance(raw, (bool, np.bool_)) or np.iscomplexobj(raw):
            raise TypeError(
                "horizon must be a positive real number (seconds) or time Quantity"
            )
        try:
            horizon_s = float(horizon.to_value(u.s))
        except u.UnitConversionError as exc:
            raise ValueError(
                f"horizon Quantity must have time units convertible to seconds, got {horizon.unit!r}"
            ) from exc
    elif isinstance(horizon, (Real, np.integer, np.floating)):
        horizon_s = float(horizon)
    else:
        raise TypeError(
            "horizon must be a positive real number (seconds) or time Quantity"
        )
    if not np.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("horizon must be a positive finite time span")
    return horizon_s


def _horizon_to_steps(horizon_s: float, dt: u.Quantity) -> int:
    """Compute ``N_pred = floor(T / dt)`` with floating-point undercounting guard.

    Provisional reversible behavior (``asm-06`` / ``unk-004``): when
    ``0 < horizon < dt`` so ``N_pred == 0``, raises ``ValueError``.
    """
    dt_s = float(dt.to_value(u.s))
    ratio = horizon_s / dt_s
    nearest = round(ratio)
    if nearest >= 1 and math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12):
        n_pred = int(nearest)
    else:
        n_pred = int(math.floor(ratio))
    if n_pred <= 0:
        raise ValueError(
            f"horizon ({horizon_s:g} s) is shorter than one sample interval dt "
            f"({dt_s:g} s), yielding N_pred = 0"
        )
    return n_pred


def _validate_chronos_quantiles(levels: tuple[float, ...], pipeline: object) -> None:
    """Reject quantile levels outside Chronos-2's supported quantile range.

    Chronos-2 clamps out-of-range quantile requests to its boundary trained
    quantiles rather than raising. Validating against the pipeline's trained
    quantile bounds prevents silently returning e.g. Q10 under a Q05 key.
    """
    supported = getattr(pipeline, "quantiles", None)
    if supported is None:
        model = getattr(pipeline, "model", None)
        config = getattr(model, "chronos_config", None)
        supported = getattr(config, "quantiles", None)

    if supported is not None:
        try:
            supported_vals = [float(q) for q in supported]
        except TypeError:
            supported_vals = []
    else:
        supported_vals = []

    if supported_vals:
        min_q = min(supported_vals)
        max_q = max(supported_vals)
    else:
        min_q, max_q = _CHRONOS2_DEFAULT_QUANTILE_RANGE

    out_of_range = [level for level in levels if level < min_q or level > max_q]
    if out_of_range:
        raise ValueError(
            f"Requested quantile level(s) {out_of_range} are outside Chronos-2's "
            f"supported quantile range [{min_q:g}, {max_q:g}]; silent clamping "
            f"to boundary quantiles is not allowed."
        )


def _default_interval_levels(levels: tuple[float, ...]) -> tuple[float, float] | None:
    """Select ``(low, high)`` interval levels from normalized quantile levels."""
    if 0.1 in levels and 0.9 in levels:
        return (0.1, 0.9)
    if len(levels) >= 2:
        return (min(levels), max(levels))
    return None


def _real_finite_values(series: TimeSeries) -> np.ndarray:
    """Return a non-empty, finite, real-valued one-dimensional input array."""
    values = np.asarray(series.value)
    if np.iscomplexobj(values):
        raise ValueError("Chronos forecast requires real-valued input series")
    if values.ndim != 1 or not len(values):
        raise ValueError("Chronos forecast requires a non-empty 1-D TimeSeries")
    if not np.all(np.isfinite(values)):
        raise ValueError("Chronos forecast requires finite input values")
    return values


def _forecast_axis_kwargs(series: TimeSeries) -> dict[str, object]:
    """Validate a regular time axis and build its exact forecast start args."""
    try:
        explicit_xindex = getattr(series, "_xindex", None)
        if explicit_xindex is not None:
            from gwexpy.types.axis import AxisDescriptor

            explicit_axis_is_regular = AxisDescriptor(
                name="time", index=explicit_xindex
            ).regular
        else:
            explicit_axis_is_regular = True
        if not series.is_regular or not explicit_axis_is_regular:
            raise ValueError("Chronos forecast requires a regular time axis")
        dt = series.dt
        dt_seconds = float(dt.to_value(u.s))
    except (AttributeError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "regular time axis" in str(exc):
            raise
        raise ValueError("Chronos forecast requires a regular time axis") from exc
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("Chronos forecast requires a positive finite sample interval")

    offset = len(series) * dt
    kwargs: dict[str, object] = {"dt": dt}
    exact_epoch = getattr(series, "_gwex_t0_gps_ns", None)
    if exact_epoch is not None:
        from ._epoch import _integral_dt_gps_ns

        try:
            offset_ns = _integral_dt_gps_ns(offset)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Chronos forecast start offset must be exactly representable in integer GPS nanoseconds"
            ) from exc
        kwargs["t0_ns"] = int(exact_epoch) + offset_ns
    else:
        kwargs["x0"] = series.x0 + offset
    return kwargs


def _normalize_backend_output(
    quantile_predictions: object,
    mean_predictions: object,
    *,
    batch_size: int,
    horizon: int,
    n_quantiles: int,
) -> np.ndarray:
    """Normalize Chronos-2's per-task tensors to (batch, horizon, quantiles)."""
    if (
        not isinstance(quantile_predictions, (list, tuple))
        or len(quantile_predictions) != batch_size
    ):
        raise ValueError("Chronos backend returned an unexpected quantile batch")
    if (
        not isinstance(mean_predictions, (list, tuple))
        or len(mean_predictions) != batch_size
    ):
        raise ValueError("Chronos backend returned an unexpected mean batch")
    normalized = []
    for quantile_item, mean_item in zip(quantile_predictions, mean_predictions):
        quantile_array = np.asarray(quantile_item)
        mean_array = np.asarray(mean_item)
        if quantile_array.shape != (1, horizon, n_quantiles) or mean_array.shape != (
            1,
            horizon,
        ):
            raise ValueError("Chronos backend returned an unexpected prediction shape")
        normalized.append(quantile_array[0])
    return np.stack(normalized, axis=0)


def _build_forecast_series(
    series: TimeSeries,
    quantile_matrix: np.ndarray,
    *,
    axis_kwargs: dict[str, object],
    levels: tuple[float, ...],
) -> TimeSeries:
    """Build a direct ``TimeSeries`` forecast with uncertainty metadata attached.

    Chronos-2 returns the 0.5 quantile (median) as its point estimate and does
    not provide an independent predicted mean; therefore ``center_kind`` is
    ``"median"`` and ``estimate_mean`` is explicitly ``None``.
    """
    from gwexpy.time._reference import inherit_time_reference

    from .timeseries import TimeSeries

    output_kwargs = {
        **axis_kwargs,
        "unit": series.unit,
        "name": series.name,
        "channel": series.channel,
    }

    def make_plain(data: np.ndarray) -> TimeSeries:
        return inherit_time_reference(TimeSeries(data, **output_kwargs), series)

    quantile_outputs = {
        float(level): make_plain(quantile_matrix[:, i])
        for i, level in enumerate(levels)
    }
    median_idx = levels.index(0.5)
    result = make_plain(quantile_matrix[:, median_idx])
    result._set_uncertainty(
        quantiles=quantile_outputs,
        estimate_mean=None,
        center_kind="median",
        uncertainty_kind="predictive",
        interval_levels=_default_interval_levels(levels),
    )
    return result


def forecast_series(
    series: TimeSeries,
    *,
    horizon: Real | u.Quantity,
    quantiles: tuple[float, ...] | list[float] | None,
) -> TimeSeries:
    """Forecast one series using the optional Chronos-2 backend."""
    horizon_s = _normalize_horizon_seconds(horizon)
    levels = _normalize_quantiles(quantiles)
    values = _real_finite_values(series)
    axis_kwargs = _forecast_axis_kwargs(series)
    n_pred = _horizon_to_steps(horizon_s, axis_kwargs["dt"])  # type: ignore[arg-type]

    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos forecasting requires the 'chronos-forecasting' package; "
            "install it with `pip install 'gwexpy[chronos]'`."
        ) from exc

    pipeline = Chronos2Pipeline.from_pretrained(
        _CHRONOS2_MODEL_ID, revision=_CHRONOS2_REVISION
    )
    _validate_chronos_quantiles(levels, pipeline)
    quantile_predictions, mean_predictions = pipeline.predict_quantiles(
        [np.asarray(values)],
        prediction_length=n_pred,
        quantile_levels=[float(level) for level in levels],
        cross_learning=False,
    )
    quantile_values = _normalize_backend_output(
        quantile_predictions,
        mean_predictions,
        batch_size=1,
        horizon=n_pred,
        n_quantiles=len(levels),
    )[0]
    return _build_forecast_series(
        series, quantile_values, axis_kwargs=axis_kwargs, levels=levels
    )


def forecast_collection(
    collection: TimeSeriesDict,
    *,
    horizon: Real | u.Quantity,
    quantiles: tuple[float, ...] | list[float] | None,
    targets: Sequence[str] | None = None,
) -> TimeSeriesDict:
    """Forecast collection members as independent univariate batch tasks."""
    from .collections import TimeSeriesDict

    if targets is not None:
        raise NotImplementedError(
            "Chronos-2 backend currently supports independent SISO batch "
            "forecasting (targets=None) only; joint MISO/MIMO forecasting via "
            "'targets' is not yet supported."
        )

    items = list(collection.items())
    if not items:
        raise ValueError("Chronos forecast requires a non-empty TimeSeriesDict")
    horizon_s = _normalize_horizon_seconds(horizon)
    levels = _normalize_quantiles(quantiles)

    values_by_key: dict[str, np.ndarray] = {}
    axis_kwargs_by_key: dict[str, dict[str, object]] = {}
    n_pred_by_key: dict[str, int] = {}
    for key, series in items:
        values_by_key[key] = _real_finite_values(series)
        axis_kwargs = _forecast_axis_kwargs(series)
        axis_kwargs_by_key[key] = axis_kwargs
        n_pred_by_key[key] = _horizon_to_steps(
            horizon_s,
            axis_kwargs["dt"],  # type: ignore[arg-type]
        )

    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos forecasting requires the 'chronos-forecasting' package; "
            "install it with `pip install 'gwexpy[chronos]'`."
        ) from exc

    pipeline = Chronos2Pipeline.from_pretrained(
        _CHRONOS2_MODEL_ID, revision=_CHRONOS2_REVISION
    )
    _validate_chronos_quantiles(levels, pipeline)

    # Group keys by prediction_length so equal-dt collections remain a single
    # backend batch call while heterogeneous-dt collections compute exact spans.
    groups: dict[int, list[str]] = {}
    for key, _ in items:
        groups.setdefault(n_pred_by_key[key], []).append(key)

    series_by_key = dict(items)
    forecasts_by_key: dict[str, TimeSeries] = {}
    for n_pred, group_keys in groups.items():
        batch_inputs = [values_by_key[k] for k in group_keys]
        quantile_predictions, mean_predictions = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=n_pred,
            quantile_levels=[float(level) for level in levels],
            cross_learning=False,
        )
        quantile_values = _normalize_backend_output(
            quantile_predictions,
            mean_predictions,
            batch_size=len(group_keys),
            horizon=n_pred,
            n_quantiles=len(levels),
        )
        for i, key in enumerate(group_keys):
            forecasts_by_key[key] = _build_forecast_series(
                series_by_key[key],
                quantile_values[i],
                axis_kwargs=axis_kwargs_by_key[key],
                levels=levels,
            )

    return TimeSeriesDict((key, forecasts_by_key[key]) for key, _ in items)


# Resolve public annotations at runtime after the package has defined these
# classes. This does not load the optional Chronos backend.
from .collections import TimeSeriesDict  # noqa: E402
from .timeseries import TimeSeries  # noqa: E402
