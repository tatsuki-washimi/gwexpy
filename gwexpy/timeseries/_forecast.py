"""Optional Chronos forecast support for GWexpy time series."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from astropy import units as u

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


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


@dataclass
class ForecastResult:
    """Point forecast and requested quantile forecasts.

    ``point`` is always the 0.5 quantile. For a single series, values are
    ``TimeSeries`` objects; for a collection they are ``TimeSeriesDict``.
    """

    point: TimeSeries | TimeSeriesDict
    quantiles: dict[float, TimeSeries | TimeSeriesDict]


def forecast_series(
    series: TimeSeries,
    *,
    horizon: int,
    quantiles: tuple[float, ...] | list[float] | None,
) -> ForecastResult:
    """Forecast one series using the optional Chronos-2 backend."""
    from .timeseries import TimeSeries

    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be a positive integer")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    levels = _normalize_quantiles(quantiles)
    values = _real_finite_values(series)
    axis_kwargs = _forecast_axis_kwargs(series)
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos forecasting requires the 'chronos-forecasting' package; "
            "GWexpy's planned [chronos] extra is not registered yet."
        ) from exc

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", revision="29ec3766d36d6f73f0696f85560a422f50e8498c"
    )
    quantile_predictions, mean_predictions = pipeline.predict_quantiles(
        [np.asarray(values)],
        prediction_length=int(horizon),
        quantile_levels=[float(level) for level in levels],
        cross_learning=False,
    )
    quantile_values = _normalize_backend_output(
        quantile_predictions,
        mean_predictions,
        batch_size=1,
        horizon=int(horizon),
        n_quantiles=len(levels),
    )[0]
    output_kwargs = {
        **axis_kwargs,
        "unit": series.unit,
        "name": series.name,
        "channel": series.channel,
    }

    def make_output(data: np.ndarray) -> TimeSeries:
        return TimeSeries(data, **output_kwargs)

    outputs = {
        float(level): make_output(quantile_values[:, i])
        for i, level in enumerate(levels)
    }
    return ForecastResult(point=outputs[0.5], quantiles=outputs)


def forecast_collection(
    collection: TimeSeriesDict,
    *,
    horizon: int,
    quantiles: tuple[float, ...] | list[float] | None,
) -> ForecastResult:
    """Forecast collection members as independent univariate batch tasks."""
    from .collections import TimeSeriesDict

    items = list(collection.items())
    if not items:
        raise ValueError("Chronos forecast requires a non-empty TimeSeriesDict")
    # A single pipeline call keeps this a backend batch while each item remains
    # a distinct one-dimensional task with cross-learning disabled.
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be a positive integer")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    levels = _normalize_quantiles(quantiles)
    values = []
    axis_kwargs_by_key = {}
    for key, series in items:
        data = _real_finite_values(series)
        axis_kwargs_by_key[key] = _forecast_axis_kwargs(series)
        values.append(data)
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos forecasting requires the 'chronos-forecasting' package; "
            "GWexpy's planned [chronos] extra is not registered yet."
        ) from exc
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", revision="29ec3766d36d6f73f0696f85560a422f50e8498c"
    )
    quantile_predictions, mean_predictions = pipeline.predict_quantiles(
        values,
        prediction_length=int(horizon),
        quantile_levels=[float(level) for level in levels],
        cross_learning=False,
    )
    quantile_values = _normalize_backend_output(
        quantile_predictions,
        mean_predictions,
        batch_size=len(items),
        horizon=int(horizon),
        n_quantiles=len(levels),
    )
    point_values = quantile_values[:, :, levels.index(0.5)]

    from .timeseries import TimeSeries

    def output_series(key: str, series: TimeSeries, data: np.ndarray) -> TimeSeries:
        kwargs = {
            **axis_kwargs_by_key[key],
            "unit": series.unit,
            "name": series.name,
            "channel": series.channel,
        }
        return TimeSeries(data, **kwargs)

    point = TimeSeriesDict(
        (key, output_series(key, series, point_values[i]))
        for i, (key, series) in enumerate(items)
    )
    quantile_results = {
        float(level): TimeSeriesDict(
            (key, output_series(key, series, quantile_values[i, :, j]))
            for i, (key, series) in enumerate(items)
        )
        for j, level in enumerate(levels)
    }
    return ForecastResult(point=point, quantiles=quantile_results)


# Resolve public annotations at runtime after the package has defined these
# classes. This does not load the optional Chronos backend.
from .collections import TimeSeriesDict  # noqa: E402
from .timeseries import TimeSeries  # noqa: E402
