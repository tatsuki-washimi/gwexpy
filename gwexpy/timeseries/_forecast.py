"""Optional pretrained forecast support (Chronos-2, TimesFM 2.5) for GWexpy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from astropy import units as u

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
SUPPORTED_BACKENDS = ("chronos", "timesfm", "chronos-bolt", "timestral")

_BOLT_ID = "amazon/chronos-bolt-small"
_BOLT_REVISION = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"
_TIMESTRAL_ID = "mistralai/Timestral-209M"
_TIMESTRAL_REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"

_CHRONOS2_MODEL_ID = "amazon/chronos-2"
_CHRONOS2_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
# Chronos-2 trained quantile range for amazon/chronos-2 at the pinned revision.
# Used as a fallback if a pipeline instance does not expose `.quantiles`.
_CHRONOS2_DEFAULT_QUANTILE_RANGE = (0.01, 0.99)

_TIMESFM25_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
_TIMESFM25_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
_TIMESFM25_QUANTILES: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)
_TIMESFM25_INPUT_PATCH_LEN = 32
_TIMESFM25_OUTPUT_PATCH_LEN = 128
_TIMESFM25_MAX_CONTINUOUS_HORIZON = 1024
_TIMESFM25_CONTEXT_LIMIT = 16384
_TIMESFM25_MAX_BATCH_SIZE = 32


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


def _same_forecast_axis(first: TimeSeries, other: TimeSeries) -> bool:
    """Compare regular axes without discarding an exact epoch."""
    if len(first) != len(other) or first.dt != other.dt:
        return False
    first_exact = getattr(first, "_gwex_t0_gps_ns", None)
    other_exact = getattr(other, "_gwex_t0_gps_ns", None)
    if first_exact is None and other_exact is None:
        return bool(first.x0 == other.x0)
    from ._epoch import _integer_gps_ns

    try:
        first_ns = first_exact if first_exact is not None else _integer_gps_ns(first.x0)
        other_ns = other_exact if other_exact is not None else _integer_gps_ns(other.x0)
    except (TypeError, ValueError):
        return False
    return bool(first_ns == other_ns)


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


def _validate_timesfm_quantiles(levels: tuple[float, ...]) -> None:
    """Reject quantile levels not in TimesFM 2.5's 9 trained quantiles."""
    unsupported = [level for level in levels if level not in _TIMESFM25_QUANTILES]
    if unsupported:
        raise ValueError(
            f"Requested quantile level(s) {unsupported} are not supported by "
            f"TimesFM 2.5; supported quantiles are {_TIMESFM25_QUANTILES} and "
            "silent substitution is not allowed."
        )


def _validate_timesfm_horizon(n_pred: int) -> None:
    """Enforce TimesFM 2.5 continuous quantile head horizon ceiling."""
    if n_pred > _TIMESFM25_MAX_CONTINUOUS_HORIZON:
        raise ValueError(
            f"TimesFM 2.5 backend with continuous quantile head supports at most "
            f"{_TIMESFM25_MAX_CONTINUOUS_HORIZON} predicted samples, got {n_pred}."
        )


def _prescale_series_float64(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Center and scale a 1-D series in float64 to avoid float32 underflow/overflow.

    TimesFM 2.5 converts inputs to ``float32`` before its internal RevIN layer,
    and treats standard deviations below ``1e-6`` (``_TOLERANCE``) as unscaled.
    Pre-scaling in ``float64`` preserves physical strain-scale signals
    (e.g. ``~1e-21``).

    Returns ``(scaled, mu, scale)`` satisfying ``values = mu + scale * scaled``
    with ``scale > 0``. For constant series (zero variance):

    - If ``mu != 0`` (non-zero constant), falls back to ``scale = abs(mu)`` so
      inverse scaling stays proportional to the physical magnitude of the
      observed signal rather than inflating to ``O(1)``.
    - If ``mu == 0`` (all-zero series), falls back explicitly to ``scale = 1.0``
      because no positive physical scale can be inferred from the input.
    """
    arr = np.asarray(values, dtype=np.float64)
    magnitude = float(np.max(np.abs(arr)))
    if magnitude == 0.0:
        return np.zeros_like(arr), 0.0, 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        ordinary_mu = float(np.mean(arr))
        ordinary_centered = arr - ordinary_mu
    if np.isfinite(ordinary_mu) and np.all(np.isfinite(ordinary_centered)):
        max_abs = float(np.max(np.abs(ordinary_centered)))
        if max_abs == 0.0:
            return np.zeros_like(arr), ordinary_mu, abs(ordinary_mu) or 1.0
        std = float(np.std(ordinary_centered / max_abs)) * max_abs
        scale = std if np.isfinite(std) and std > 0.0 else max_abs
        return ordinary_centered / scale, ordinary_mu, scale
    normalized = arr / magnitude
    normalized_mean = float(np.mean(normalized))
    mu = normalized_mean * magnitude
    centered = normalized - normalized_mean
    max_abs = float(np.max(np.abs(centered)))
    if max_abs == 0.0:
        return np.zeros_like(arr), mu, abs(mu)
    normalized_std = float(np.std(centered / max_abs)) * max_abs
    scale = normalized_std * magnitude
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("input scale cannot be represented in float64")
    return centered / normalized_std, mu, scale


def _restore_scaled_forecast(values: np.ndarray, mu: float, scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result = mu + scale * values
        if np.any(~np.isfinite(result)):
            magnitude = max(abs(mu), scale)
            fallback = magnitude * (mu / magnitude + (scale / magnitude) * values)
            result = np.where(np.isfinite(result), result, fallback)
    return result


def _compute_timesfm_compile_dims(
    input_lengths: Sequence[int], n_pred: int
) -> tuple[int, int]:
    """Compute patch-aligned ``(max_context, max_horizon)`` within context limit."""
    max_horizon = (
        math.ceil(n_pred / _TIMESFM25_OUTPUT_PATCH_LEN) * _TIMESFM25_OUTPUT_PATCH_LEN
    )
    max_allowed_context = (
        (_TIMESFM25_CONTEXT_LIMIT - max_horizon) // _TIMESFM25_INPUT_PATCH_LEN
    ) * _TIMESFM25_INPUT_PATCH_LEN
    needed_context = max(max(input_lengths), _TIMESFM25_INPUT_PATCH_LEN)
    rounded_context = (
        math.ceil(needed_context / _TIMESFM25_INPUT_PATCH_LEN)
        * _TIMESFM25_INPUT_PATCH_LEN
    )
    return min(rounded_context, max_allowed_context), max_horizon


def _normalize_timesfm_output(
    point_forecast: object,
    quantile_forecast: object,
    *,
    batch_size: int,
    horizon: int,
    levels: tuple[float, ...],
) -> np.ndarray:
    """Extract requested quantile columns from TimesFM 2.5's 10-column output.

    When ``ForecastConfig.output_quantiles`` is left as ``None``, TimesFM 2.5
    returns ``quantile_forecast`` with shape ``(batch_size, horizon, 10)`` where
    index 0 is the point head and indices 1..9 correspond to ``0.1..0.9``.
    """
    point_arr = np.asarray(point_forecast, dtype=np.float64)
    quantile_arr = np.asarray(quantile_forecast, dtype=np.float64)
    expected_cols = len(_TIMESFM25_QUANTILES) + 1
    if point_arr.shape != (batch_size, horizon):
        raise ValueError("TimesFM backend returned an unexpected point forecast shape")
    if quantile_arr.shape != (batch_size, horizon, expected_cols):
        raise ValueError(
            "TimesFM backend returned an unexpected quantile forecast shape"
        )
    col_indices = [_TIMESFM25_QUANTILES.index(level) + 1 for level in levels]
    return quantile_arr[:, :, col_indices]


def _load_chronos_pipeline() -> object:
    """Lazily import and load the pinned Chronos-2 pipeline."""
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos forecasting requires the 'chronos-forecasting' package; "
            "install it with `pip install 'gwexpy[chronos]'`."
        ) from exc

    return Chronos2Pipeline.from_pretrained(
        _CHRONOS2_MODEL_ID, revision=_CHRONOS2_REVISION
    )


def _load_bolt_pipeline() -> object:
    try:
        from chronos import ChronosBoltPipeline
    except ImportError as exc:
        raise ImportError("Chronos-Bolt requires gwexpy[chronos]") from exc
    return ChronosBoltPipeline.from_pretrained(_BOLT_ID, revision=_BOLT_REVISION)


def _load_timestral_model() -> object:
    try:
        from timestral import TimestralForecaster
    except ImportError as exc:
        raise ImportError(
            "Timestral requires Python >= 3.10 and the upstream package; install "
            "it with `pip install git+https://github.com/mistralai/timestral.git` "
            "(no gwexpy extra provides it). The model weights have separate "
            "license terms."
        ) from exc
    return TimestralForecaster.from_pretrained(
        _TIMESTRAL_ID, revision=_TIMESTRAL_REVISION
    )


def _run_additional_backend(
    model: object,
    backend: str,
    values: np.ndarray,
    n_pred: int,
    levels: tuple[float, ...],
) -> np.ndarray:
    context_limit = getattr(
        model,
        "model_context_length" if backend == "chronos-bolt" else "max_context",
        None,
    )
    visible = values[-context_limit:] if context_limit else values
    scaled, mu, scale = _prescale_series_float64(visible)
    if backend == "chronos-bolt":
        import torch

        supported = getattr(model, "quantiles", (0.1, 0.9))
        if min(levels) < min(supported) or max(levels) > max(supported):
            raise ValueError(
                "Requested quantiles are outside Chronos-Bolt's supported range"
            )
        quantiles, point = model.predict_quantiles(  # type: ignore[attr-defined]
            [torch.as_tensor(scaled, dtype=torch.float32)],
            prediction_length=n_pred,
            quantile_levels=list(levels),
        )
        quantiles = np.asarray(quantiles)
        if quantiles.shape != (1, n_pred, len(levels)) or np.asarray(point).shape != (
            1,
            n_pred,
        ):
            raise ValueError(
                "Chronos-Bolt backend returned an unexpected prediction shape"
            )
        result = quantiles[0]
    else:
        output = model.forecast(  # type: ignore[attr-defined]
            scaled.reshape(1, 1, -1), prediction_length=n_pred, clip_nonnegative=False
        )
        supported = np.asarray(output.quantile_levels, dtype=np.float64)
        indices = []
        for level in levels:
            matches = np.flatnonzero(np.isclose(supported, level, rtol=0.0, atol=1e-9))
            if matches.size != 1:
                raise ValueError(
                    f"Quantile level {level:g} is not supported by Timestral; "
                    f"supported levels are {tuple(float(q) for q in supported)}"
                )
            indices.append(int(matches[0]))
        quantiles = np.asarray(output.quantiles)
        if quantiles.shape != (1, 1, n_pred, len(supported)):
            raise ValueError(
                "Timestral backend returned an unexpected prediction shape"
            )
        result = quantiles[0, 0][:, indices]
    restored = _restore_scaled_forecast(result, mu, scale)
    _require_finite_forecast(restored, backend)
    # Upstream clipping is disabled because the model sees centered data. Clip in
    # physical coordinates only when the whole input is non-negative: a series
    # that was ever negative is not treated as a non-negative quantity, even if
    # the visible context window happens to be non-negative.
    if backend == "timestral" and np.all(values >= 0):
        restored = np.maximum(restored, 0.0)
    return restored


def _require_finite_forecast(values: np.ndarray, backend: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{backend} backend returned a non-finite forecast")


def _load_timesfm_model() -> tuple[object, object]:
    """Lazily import and load the pinned TimesFM 2.5 PyTorch model."""
    try:
        import timesfm
    except ImportError as exc:
        raise ImportError(
            "TimesFM forecasting requires the 'timesfm[torch]' package; "
            "install it with `pip install 'gwexpy[timesfm]'`."
        ) from exc

    try:
        model_cls = timesfm.TimesFM_2p5_200M_torch
        config_cls = timesfm.ForecastConfig
    except AttributeError as exc:
        raise ImportError(
            "TimesFM 2.5 PyTorch backend requires 'timesfm[torch]' with PyTorch "
            "installed; install it with `pip install 'gwexpy[timesfm]'`."
        ) from exc

    model = model_cls.from_pretrained(
        _TIMESFM25_MODEL_ID,
        revision=_TIMESFM25_REVISION,
        torch_compile=False,
    )
    return model, config_cls


def _run_timesfm_batch(
    model: object,
    config_cls: object,
    batch_values: list[np.ndarray],
    *,
    n_pred: int,
    levels: tuple[float, ...],
) -> np.ndarray:
    """Run one equal-horizon TimesFM 2.5 batch with float64 pre/post scaling."""
    max_context, max_horizon = _compute_timesfm_compile_dims(
        [len(arr) for arr in batch_values], n_pred
    )
    scaled_inputs: list[np.ndarray] = []
    stats: list[tuple[float, float]] = []
    for arr in batch_values:
        # Truncate to the exact tail window TimesFM will actually observe
        # before computing scaling statistics, so discarded history cannot
        # distort the visible window's scale.
        visible = arr[-max_context:]
        scaled, mu, scale = _prescale_series_float64(visible)
        scaled_inputs.append(scaled)
        stats.append((mu, scale))

    forecast_config = config_cls(  # type: ignore[operator]
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=True,
        per_core_batch_size=min(len(scaled_inputs), _TIMESFM25_MAX_BATCH_SIZE),
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
    model.compile(forecast_config)  # type: ignore[attr-defined]
    point_forecast, quantile_forecast = model.forecast(  # type: ignore[attr-defined]
        horizon=n_pred,
        inputs=scaled_inputs,
    )
    scaled_quantiles = _normalize_timesfm_output(
        point_forecast,
        quantile_forecast,
        batch_size=len(scaled_inputs),
        horizon=n_pred,
        levels=levels,
    )
    restored = np.empty_like(scaled_quantiles, dtype=np.float64)
    for i, (mu, scale) in enumerate(stats):
        restored[i] = _restore_scaled_forecast(scaled_quantiles[i], mu, scale)
    return restored


def _build_forecast_series(
    series: TimeSeries,
    quantile_matrix: np.ndarray,
    *,
    axis_kwargs: dict[str, object],
    levels: tuple[float, ...],
) -> TimeSeries:
    """Build a direct ``TimeSeries`` forecast with uncertainty metadata attached.

    Both Chronos-2 and the TimesFM 2.5 adapter use the 0.5 quantile (median) as
    the representative series value (``center_kind="median"``) and set
    ``estimate_mean=None``.
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
    backend: str = "chronos",
    horizon: Real | u.Quantity,
    quantiles: tuple[float, ...] | list[float] | None,
) -> TimeSeries:
    """Forecast one series using an optional pretrained backend."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported forecast backend: {backend!r}")
    horizon_s = _normalize_horizon_seconds(horizon)
    levels = _normalize_quantiles(quantiles)
    if backend == "timesfm":
        _validate_timesfm_quantiles(levels)
    values = _real_finite_values(series)
    axis_kwargs = _forecast_axis_kwargs(series)
    n_pred = _horizon_to_steps(horizon_s, axis_kwargs["dt"])  # type: ignore[arg-type]

    if backend in ("chronos-bolt", "timestral"):
        model = (
            _load_bolt_pipeline()
            if backend == "chronos-bolt"
            else _load_timestral_model()
        )
        quantile_values = _run_additional_backend(
            model, backend, values, n_pred, levels
        )
        return _build_forecast_series(
            series, quantile_values, axis_kwargs=axis_kwargs, levels=levels
        )

    if backend == "timesfm":
        _validate_timesfm_horizon(n_pred)
        model, config_cls = _load_timesfm_model()
        quantile_values = _run_timesfm_batch(
            model,
            config_cls,
            [values],
            n_pred=n_pred,
            levels=levels,
        )[0]
        return _build_forecast_series(
            series, quantile_values, axis_kwargs=axis_kwargs, levels=levels
        )

    pipeline = _load_chronos_pipeline()
    _validate_chronos_quantiles(levels, pipeline)
    quantile_predictions, mean_predictions = pipeline.predict_quantiles(  # type: ignore[attr-defined]
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
    backend: str = "chronos",
    horizon: Real | u.Quantity,
    quantiles: tuple[float, ...] | list[float] | None,
    targets: Sequence[str] | None = None,
) -> TimeSeriesDict:
    """Forecast collection members independently or jointly.

    With ``targets=None`` every member is an independent univariate task. With
    ``targets`` (Chronos-2 only) the selected members are forecast jointly and
    the remaining members are passed as past-only covariates.
    """
    from .collections import TimeSeriesDict

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported forecast backend: {backend!r}")

    if targets is not None and backend != "chronos":
        raise NotImplementedError(
            f"{backend} backend currently supports independent SISO batch "
            "forecasting (targets=None) only; joint MISO/MIMO forecasting via "
            "'targets' is not yet supported."
        )

    items = list(collection.items())
    if not items:
        raise ValueError("Forecast requires a non-empty TimeSeriesDict")
    horizon_s = _normalize_horizon_seconds(horizon)
    levels = _normalize_quantiles(quantiles)
    if backend == "timesfm":
        _validate_timesfm_quantiles(levels)

    values_by_key: dict[str, np.ndarray] = {}
    axis_kwargs_by_key: dict[str, dict[str, object]] = {}
    n_pred_by_key: dict[str, int] = {}
    for key, series in items:
        values_by_key[key] = _real_finite_values(series)
        axis_kwargs = _forecast_axis_kwargs(series)
        axis_kwargs_by_key[key] = axis_kwargs
        n_pred = _horizon_to_steps(
            horizon_s,
            axis_kwargs["dt"],  # type: ignore[arg-type]
        )
        if backend == "timesfm":
            _validate_timesfm_horizon(n_pred)
        n_pred_by_key[key] = n_pred

    if targets is not None:
        from gwexpy.time._reference import require_uniform_time_reference

        if isinstance(targets, (str, bytes)):
            raise ValueError("targets must be a non-empty sequence of keys")
        selected = list(targets)
        try:
            unique = len(set(selected)) == len(selected)
        except TypeError:
            raise ValueError("targets must contain hashable collection keys") from None
        if not selected or not unique:
            raise ValueError("targets must contain unique keys and must not be empty")
        missing = [key for key in selected if key not in collection]
        if missing:
            raise ValueError(
                f"targets contains keys absent from the collection: {missing}"
            )
        require_uniform_time_reference(series for _, series in items)
        first_key, first = items[0]
        for key, series in items[1:]:
            if not _same_forecast_axis(first, series):
                raise ValueError(
                    "joint forecasting requires identical time axes; "
                    f"{key!r} differs from {first_key!r}"
                )
        pipeline = _load_chronos_pipeline()
        _validate_chronos_quantiles(levels, pipeline)
        context_limit = getattr(pipeline, "model_context_length", None)
        scaled_by_key = {}
        scales_by_key = {}
        for key, _ in items:
            values = values_by_key[key]
            visible = values[-context_limit:] if context_limit else values
            scaled, mu, scale = _prescale_series_float64(visible)
            scaled_by_key[key] = scaled
            scales_by_key[key] = (mu, scale)
        context = {
            "target": np.stack([scaled_by_key[key] for key in selected]),
            "past_covariates": {
                key: scaled_by_key[key] for key, _ in items if key not in selected
            },
        }
        n_pred = n_pred_by_key[selected[0]]
        predictions, means = pipeline.predict_quantiles(  # type: ignore[attr-defined]
            [context],
            prediction_length=n_pred,
            quantile_levels=list(levels),
            cross_learning=False,
        )
        if len(predictions) != 1 or len(means) != 1:
            raise ValueError("Chronos backend returned an unexpected prediction batch")
        predictions = np.asarray(predictions[0])
        means = np.asarray(means[0])
        if predictions.shape != (len(selected), n_pred, len(levels)) or means.shape != (
            len(selected),
            n_pred,
        ):
            raise ValueError("Chronos backend returned an unexpected prediction shape")
        predictions = predictions.astype(np.float64)
        for index, key in enumerate(selected):
            mu, scale = scales_by_key[key]
            predictions[index] = _restore_scaled_forecast(predictions[index], mu, scale)
        _require_finite_forecast(predictions, backend)
        return TimeSeriesDict(
            (
                key,
                _build_forecast_series(
                    collection[key],
                    predictions[i],
                    axis_kwargs=axis_kwargs_by_key[key],
                    levels=levels,
                ),
            )
            for i, key in enumerate(selected)
        )

    # Group keys by prediction_length so equal-dt collections remain a single
    # backend batch call while heterogeneous-dt collections compute exact spans.
    groups: dict[int, list[str]] = {}
    for key, _ in items:
        groups.setdefault(n_pred_by_key[key], []).append(key)

    series_by_key = dict(items)
    forecasts_by_key: dict[str, TimeSeries] = {}

    if backend in ("chronos-bolt", "timestral"):
        model = (
            _load_bolt_pipeline()
            if backend == "chronos-bolt"
            else _load_timestral_model()
        )
        return TimeSeriesDict(
            (
                key,
                _build_forecast_series(
                    series,
                    _run_additional_backend(
                        model, backend, values_by_key[key], n_pred_by_key[key], levels
                    ),
                    axis_kwargs=axis_kwargs_by_key[key],
                    levels=levels,
                ),
            )
            for key, series in items
        )

    if backend == "timesfm":
        model, config_cls = _load_timesfm_model()
        for n_pred, group_keys in groups.items():
            batch_inputs = [values_by_key[k] for k in group_keys]
            quantile_values = _run_timesfm_batch(
                model,
                config_cls,
                batch_inputs,
                n_pred=n_pred,
                levels=levels,
            )
            for i, key in enumerate(group_keys):
                forecasts_by_key[key] = _build_forecast_series(
                    series_by_key[key],
                    quantile_values[i],
                    axis_kwargs=axis_kwargs_by_key[key],
                    levels=levels,
                )
        return TimeSeriesDict((key, forecasts_by_key[key]) for key, _ in items)

    pipeline = _load_chronos_pipeline()
    _validate_chronos_quantiles(levels, pipeline)
    for n_pred, group_keys in groups.items():
        batch_inputs = [values_by_key[k] for k in group_keys]
        quantile_predictions, mean_predictions = pipeline.predict_quantiles(  # type: ignore[attr-defined]
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


def _resolve_percentiles(
    quantiles: tuple[float, ...] | list[float] | None,
    percentiles: tuple[float, ...] | list[float] | None,
) -> tuple[float, ...] | list[float] | None:
    """Convert percentile requests to the canonical quantile representation."""
    if quantiles is not None and percentiles is not None:
        raise ValueError("Specify either quantiles or percentiles, not both")
    if percentiles is not None:
        if any(
            isinstance(level, bool) or not isinstance(level, Real)
            for level in percentiles
        ):
            raise ValueError("Percentile levels must be real numbers")
        return [level / 100 for level in percentiles]
    return quantiles
