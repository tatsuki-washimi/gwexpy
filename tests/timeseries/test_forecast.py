from __future__ import annotations

import inspect
import subprocess
import sys
import types

import numpy as np
import pytest
from astropy import units as u

import gwexpy.timeseries as gwex_ts
from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def test_missing_timestral_reports_install_command(monkeypatch):
    import gwexpy.timeseries._forecast as forecast

    monkeypatch.setitem(sys.modules, "timestral", None)
    with pytest.raises(ImportError, match="github.com/mistralai/timestral"):
        forecast._load_timestral_model()


def test_joint_axis_compares_original_epoch():
    from gwexpy.timeseries._forecast import _same_forecast_axis

    legacy = TimeSeries([1.0, 2.0], t0=1.0, dt=1.0)
    exact = TimeSeries([1.0, 2.0], t0_ns=1_000_000_000, dt=1.0)
    assert _same_forecast_axis(legacy, exact)
    first = TimeSeries([1.0, 2.0], x0=0.0, dt=1.0)
    shifted = TimeSeries([1.0, 2.0], x0=1e-17, dt=1.0)
    assert not _same_forecast_axis(first, shifted)


def test_prescale_large_finite_constant():
    from gwexpy.timeseries._forecast import _prescale_series_float64

    scaled, center, scale = _prescale_series_float64(np.array([1e308, 1e308]))
    np.testing.assert_array_equal(scaled, [0.0, 0.0])
    assert center == 1e308
    assert scale == 1e308


def test_prescale_extreme_range():
    from gwexpy.timeseries._forecast import (
        _prescale_series_float64,
        _restore_scaled_forecast,
    )

    scaled, center, scale = _prescale_series_float64(
        np.array([-1.7e308, 1.7e308, 1.7e308])
    )
    assert np.all(np.isfinite(scaled))
    assert np.isfinite(center) and np.isfinite(scale)
    np.testing.assert_allclose(scaled, [-np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_allclose(
        _restore_scaled_forecast(scaled, center, scale) / 1.7e308, [-1.0, 1.0, 1.0]
    )


@pytest.mark.parametrize("backend", ["chronos-bolt", "timestral"])
def test_additional_backend_forecast_contract(monkeypatch, backend):
    import gwexpy.timeseries._forecast as forecast

    class Bolt:
        quantiles = [0.1, 0.5, 0.9]

        def predict_quantiles(self, inputs, *, prediction_length, quantile_levels):
            assert len(inputs) == 1
            return np.ones((1, prediction_length, len(quantile_levels))), np.ones(
                (1, prediction_length)
            )

    class Timestral:
        def forecast(self, inputs, *, prediction_length, clip_nonnegative):
            assert inputs.shape == (1, 1, 3)
            assert clip_nonnegative is False
            return types.SimpleNamespace(
                quantiles=np.ones((1, 1, prediction_length, 3)),
                quantile_levels=(0.1, 0.5, 0.9),
            )

    monkeypatch.setattr(forecast, "_load_bolt_pipeline", lambda: Bolt())
    monkeypatch.setattr(forecast, "_load_timestral_model", lambda: Timestral())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            as_tensor=lambda values, dtype: np.asarray(values, dtype=dtype),
            float32=np.float32,
        ),
    )
    source = TimeSeries([1.0, 2.0, 3.0], dt=1.0, unit="V")
    result = source.forecast(backend=backend, horizon=2.0)
    assert len(result) == 2
    assert result.unit == u.V
    assert result.center_kind == "median"
    assert list(result.quantiles) == [0.1, 0.5, 0.9]


@pytest.mark.parametrize("backend", ["chronos-bolt", "timestral"])
def test_additional_backend_scales_only_visible_context(monkeypatch, backend):
    import gwexpy.timeseries._forecast as forecast

    class Model:
        model_context_length = 3
        max_context = 3
        quantiles = [0.1, 0.5, 0.9]

        def predict_quantiles(self, inputs, *, prediction_length, quantile_levels):
            np.testing.assert_allclose(inputs[0], [-1.2247449, 0, 1.2247449])
            return np.zeros((1, prediction_length, 3)), np.zeros((1, prediction_length))

        def forecast(self, inputs, *, prediction_length, clip_nonnegative):
            np.testing.assert_allclose(inputs[0, 0], [-1.2247449, 0, 1.2247449])
            return types.SimpleNamespace(
                quantiles=np.zeros((1, 1, prediction_length, 3)),
                quantile_levels=(0.1, 0.5, 0.9),
            )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            as_tensor=lambda values, dtype: np.asarray(values, dtype=dtype),
            float32=np.float32,
        ),
    )
    result = forecast._run_additional_backend(
        Model(), backend, np.array([1e30, 1.0, 2.0, 3.0]), 1, (0.1, 0.5, 0.9)
    )
    np.testing.assert_allclose(result, 2.0)


@pytest.mark.parametrize("backend", ["chronos-bolt", "timestral"])
@pytest.mark.parametrize("scale", [1e-46, 1e40])
def test_additional_backend_inverse_scaling_float32(monkeypatch, backend, scale):
    import gwexpy.timeseries._forecast as forecast

    class Model:
        quantiles = [0.1, 0.5, 0.9]

        def predict_quantiles(self, inputs, **kwargs):
            return np.ones((1, 1, 3), dtype=np.float32), np.ones(
                (1, 1), dtype=np.float32
            )

        def forecast(self, inputs, **kwargs):
            return types.SimpleNamespace(
                quantiles=np.ones((1, 1, 1, 3), dtype=np.float32),
                quantile_levels=(0.1, 0.5, 0.9),
            )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            as_tensor=lambda values, dtype: np.asarray(values, dtype=dtype),
            float32=np.float32,
        ),
    )
    result = forecast._run_additional_backend(
        Model(), backend, np.array([-scale, scale]), 1, (0.1, 0.5, 0.9)
    )
    assert result.dtype == np.float64
    np.testing.assert_allclose(result / scale, 1.0)


@pytest.mark.parametrize(
    ("values", "expected"), [([0.1, 2.0], 0.0), ([-2.0, -2.0], -6.0)]
)
def test_timestral_clips_in_original_coordinates(values, expected):
    import gwexpy.timeseries._forecast as forecast

    class Model:
        def forecast(self, inputs, *, prediction_length, clip_nonnegative):
            assert not clip_nonnegative
            return types.SimpleNamespace(
                quantiles=np.full((1, 1, 1, 1), -2.0), quantile_levels=(0.5,)
            )

    result = forecast._run_additional_backend(
        Model(), "timestral", np.array(values), 1, (0.5,)
    )
    np.testing.assert_allclose(result, expected)


def test_timestral_clipping_uses_full_input_when_context_is_truncated():
    import gwexpy.timeseries._forecast as forecast

    class Model:
        max_context = 2

        def forecast(self, inputs, *, prediction_length, clip_nonnegative):
            assert clip_nonnegative is False
            return types.SimpleNamespace(
                quantiles=np.full((1, 1, 1, 1), -2.0), quantile_levels=(0.5,)
            )

    result = forecast._run_additional_backend(
        Model(), "timestral", np.array([-1.0, 0.1, 2.0]), 1, (0.5,)
    )
    assert result[0, 0] < 0


@pytest.mark.parametrize("magnitude", [1e-46, 1e40])
def test_joint_forecast_scales_targets_and_covariates(monkeypatch, magnitude):
    import gwexpy.timeseries._forecast as forecast

    class Model:
        model_context_length = 2

        def predict_quantiles(self, inputs, **kwargs):
            np.testing.assert_array_equal(inputs[0]["target"], [[-1.0, 1.0]])
            np.testing.assert_array_equal(
                inputs[0]["past_covariates"]["b"], [-1.0, 1.0]
            )
            return [np.ones((1, 1, 3), dtype=np.float32)], [
                np.ones((1, 1), dtype=np.float32)
            ]

    monkeypatch.setattr(forecast, "_load_chronos_pipeline", lambda: Model())
    collection = TimeSeriesDict(
        {
            "a": TimeSeries([1e50, -magnitude, magnitude], dt=1),
            "b": TimeSeries([1e50, -2 * magnitude, 2 * magnitude], dt=1),
        }
    )
    result = collection.forecast(horizon=1.0, targets=["a"])
    np.testing.assert_allclose(result["a"].value / magnitude, [1.0])


class MockChronosPipeline:
    calls: list[dict[str, object]] = []
    loads: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> MockChronosPipeline:
        assert model_id == "amazon/chronos-2"
        assert kwargs["revision"] == "29ec3766d36d6f73f0696f85560a422f50e8498c"
        cls.loads.append({"model_id": model_id, **kwargs})
        return cls()

    def predict_quantiles(
        self,
        inputs: list[np.ndarray],
        *,
        prediction_length: int,
        quantile_levels: list[float],
        cross_learning: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self.calls.append(
            {
                "inputs": [np.array(item, copy=True) for item in inputs],
                "prediction_length": prediction_length,
                "quantile_levels": quantile_levels,
                "cross_learning": cross_learning,
            }
        )
        point = np.array(
            [[item[-1] + i + 1 for i in range(prediction_length)] for item in inputs]
        )
        quantiles = np.stack(
            [point + (level - 0.5) for level in quantile_levels], axis=-1
        )
        quantile_tensors = [quantiles[i : i + 1] for i in range(len(inputs))]
        # Chronos-2 returns median-derived values in mean_predictions; gwexpy
        # must NOT expose these as .estimate_mean.
        mean_tensors = [point[i : i + 1] + 100 for i in range(len(inputs))]
        return quantile_tensors, mean_tensors


def install_mock_chronos(monkeypatch: pytest.MonkeyPatch) -> None:
    MockChronosPipeline.calls.clear()
    MockChronosPipeline.loads.clear()
    module = types.ModuleType("chronos")
    module.Chronos2Pipeline = MockChronosPipeline
    monkeypatch.setitem(sys.modules, "chronos", module)


def test_forecast_result_is_retired() -> None:
    assert not hasattr(gwex_ts, "ForecastResult")
    assert "ForecastResult" not in gwex_ts.__all__


def test_single_series_forecast_returns_timeseries_with_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(
        [1.0, 2.0, 3.0],
        t0_ns=epoch_ns,
        dt=7 * u.ns,
        unit="V",
        name="sensor",
        channel="H1:TEST",
    )

    result = series.forecast(horizon=14 * u.ns)

    assert isinstance(result, TimeSeries)
    assert result.name == "sensor"
    assert result.channel == series.channel
    assert result.unit == u.V
    assert result.dt == 7 * u.ns
    assert result.t0_gps_ns == epoch_ns + 3 * 7
    np.testing.assert_array_equal(result.value, [4.0, 5.0])
    assert result.center_kind == "median"
    assert result.uncertainty_kind == "predictive"
    assert result.interval_levels == (0.1, 0.9)
    assert result.estimate_mean is None
    assert result.quantiles is not None
    assert tuple(result.quantiles) == (0.1, 0.5, 0.9)
    np.testing.assert_array_equal(result.value, result.quantiles[0.5].value)
    assert result.lower is result.quantiles[0.1]
    assert result.upper is result.quantiles[0.9]
    assert MockChronosPipeline.calls[0]["cross_learning"] is False
    assert MockChronosPipeline.calls[0]["prediction_length"] == 2


def test_horizon_time_span_floor_and_fp_undercount_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    # 0.3 / 0.1 == 2.9999999999999996 in IEEE 754 float64; must yield 3, not 2.
    series = TimeSeries([1.0, 2.0, 3.0], dt=0.1 * u.s, t0=10.0)

    exact_fp = series.forecast(horizon=0.3)
    assert len(exact_fp) == 3

    # Non-integer horizon / dt uses floor(T / dt) without raising an error.
    non_integer = series.forecast(horizon=0.37 * u.s)
    assert len(non_integer) == 3

    # Astropy time Quantity in minutes.
    slow = TimeSeries([1.0, 2.0], dt=10 * u.s, t0=0.0)
    qty_res = slow.forecast(horizon=0.5 * u.min)
    assert len(qty_res) == 3


def test_horizon_shorter_than_dt_raises_value_error_provisionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s, t0=0.0)

    with pytest.raises(ValueError, match="N_pred = 0"):
        series.forecast(horizon=0.5)

    with pytest.raises(ValueError, match="N_pred = 0"):
        series.forecast(horizon=500 * u.ms)

    assert MockChronosPipeline.loads == []


@pytest.mark.parametrize(
    ("bad_horizon", "exc_type"),
    [
        (True, TypeError),
        ("2s", TypeError),
        (0, ValueError),
        (-1.0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (1.0 * u.m, ValueError),
    ],
)
def test_invalid_horizon_fails_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
    bad_horizon: object,
    exc_type: type[Exception],
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0, 2.0], dt=1.0 * u.s)

    with pytest.raises(exc_type):
        series.forecast(horizon=bad_horizon)  # type: ignore[arg-type]

    assert MockChronosPipeline.loads == []


def test_dict_forecast_batches_independent_series_and_handles_heterogeneous_dt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    first = TimeSeries([1.0, 2.0], t0=100, dt=0.25, name="one", unit="m")
    second = TimeSeries([10.0, 20.0], t0=200, dt=0.5, name="two", unit="s")
    collection = TimeSeriesDict({"a": first, "b": second})

    result = collection.forecast(horizon=1.0)

    assert isinstance(result, TimeSeriesDict)
    assert tuple(result) == ("a", "b")
    assert isinstance(result["a"], TimeSeries)
    assert result["a"].name == "one"
    assert result["b"].unit == u.s
    assert len(result["a"]) == 4  # 1.0 / 0.25
    assert len(result["b"]) == 2  # 1.0 / 0.5
    assert result["a"].t0 == first.t0 + 2 * first.dt
    assert result["b"].t0 == second.t0 + 2 * second.dt
    assert result["a"].estimate_mean is None
    assert result["b"].estimate_mean is None
    assert all(call["cross_learning"] is False for call in MockChronosPipeline.calls)


def test_dict_forecast_single_batch_for_equal_dt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    collection = TimeSeriesDict(
        {
            "a": TimeSeries([1.0, 2.0], t0=10, dt=0.5),
            "b": TimeSeries([10.0, 20.0], t0=20, dt=0.5),
        }
    )

    result = collection.forecast(horizon=1.5)

    assert len(MockChronosPipeline.calls) == 1
    assert MockChronosPipeline.calls[0]["prediction_length"] == 3
    assert len(MockChronosPipeline.calls[0]["inputs"]) == 2
    assert len(result["a"]) == 3
    assert len(result["b"]) == 3


def test_dict_forecast_joint_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gwexpy.timeseries._forecast as forecast

    class JointPipeline:
        def predict_quantiles(
            self, inputs, *, prediction_length, quantile_levels, cross_learning
        ):
            np.testing.assert_array_equal(inputs[0]["target"], [[-1, 1]])
            np.testing.assert_array_equal(inputs[0]["past_covariates"]["b"], [-1, 1])
            return [np.arange(6).reshape(1, 2, 3)], [np.zeros((1, 2))]

    monkeypatch.setattr(forecast, "_load_chronos_pipeline", lambda: JointPipeline())
    sig = inspect.signature(TimeSeriesDict.forecast)
    assert "mode" not in sig.parameters
    assert "targets" in sig.parameters

    collection = TimeSeriesDict(
        {"a": TimeSeries([1.0, 2.0], dt=1.0), "b": TimeSeries([3.0, 4.0], dt=1.0)}
    )
    result = collection.forecast(horizon=2.0, targets=["a"])
    assert list(result) == ["a"]
    np.testing.assert_array_equal(result["a"].value, [2, 3.5])
    with pytest.raises(ValueError, match="unique"):
        collection.forecast(horizon=2.0, targets=["a", "a"])
    with pytest.raises(ValueError, match="absent"):
        collection.forecast(horizon=2.0, targets=["missing"])
    collection["b"] = TimeSeries([3.0, 4.0], dt=2.0)
    with pytest.raises(ValueError, match="identical time axes"):
        collection.forecast(horizon=2.0, targets=["a"])
    from gwexpy.time._reference import set_time_reference

    collection["b"] = TimeSeries([3.0, 4.0], dt=1.0)
    set_time_reference(collection["b"], "relative")
    with pytest.raises(ValueError, match="mixed time_reference"):
        collection.forecast(horizon=2.0, targets=["a"])


def test_joint_mimo_preserves_target_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import gwexpy.timeseries._forecast as forecast

    class Pipeline:
        def predict_quantiles(self, inputs, **kwargs):
            np.testing.assert_array_equal(inputs[0]["target"], [[-1, 1], [-1, 1]])
            assert inputs[0]["past_covariates"] == {}
            return [np.arange(12).reshape(2, 2, 3)], [np.zeros((2, 2))]

    monkeypatch.setattr(forecast, "_load_chronos_pipeline", lambda: Pipeline())
    collection = TimeSeriesDict(
        {
            "a": TimeSeries([1.0, 2.0], dt=1.0, unit="V"),
            "b": TimeSeries([3.0, 4.0], dt=1.0, unit="m"),
        }
    )
    result = collection.forecast(horizon=2.0, targets=["b", "a"])
    assert list(result) == ["b", "a"]
    np.testing.assert_array_equal(result["b"].value, [4, 5.5])
    np.testing.assert_array_equal(result["a"].value, [5, 6.5])
    assert result["a"].unit == u.V
    assert result["b"].unit == u.m
    assert result["a"].center_kind == "median"
    assert result["a"].uncertainty_kind == "predictive"
    assert result["a"].interval_levels == (0.1, 0.9)
    assert result["a"].estimate_mean is None
    np.testing.assert_array_equal(result["a"].quantiles[0.1].value, [4.5, 6])
    np.testing.assert_array_equal(result["a"].quantiles[0.9].value, [5.5, 7])


def test_dict_series_changes_do_not_cross_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    original = TimeSeriesDict(
        {"a": TimeSeries([1.0, 2.0], dt=1.0), "b": TimeSeries([10.0, 20.0], dt=1.0)}
    )
    changed = TimeSeriesDict(
        {"a": TimeSeries([1.0, 200.0], dt=1.0), "b": TimeSeries([10.0, 20.0], dt=1.0)}
    )

    before = original.forecast(horizon=1.0)
    after = changed.forecast(horizon=1.0)

    assert MockChronosPipeline.calls[0]["inputs"][1].tolist() == [10.0, 20.0]
    assert MockChronosPipeline.calls[1]["inputs"][1].tolist() == [10.0, 20.0]
    np.testing.assert_array_equal(before["b"].value, after["b"].value)


def test_missing_chronos_names_future_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "chronos", raising=False)
    monkeypatch.setattr("builtins.__import__", _import_without_chronos)

    with pytest.raises(ImportError, match=r"\[chronos\]"):
        TimeSeries([1.0, 2.0], dt=1.0).forecast(horizon=1.0)


def _import_without_chronos(name: str, *args: object, **kwargs: object) -> object:
    if name == "chronos":
        raise ImportError("mock missing chronos")
    return _original_import(name, *args, **kwargs)


_original_import = __import__


def test_importing_timeseries_surface_does_not_import_chronos() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from typing import get_type_hints; "
            "from gwexpy.timeseries import TimeSeries, TimeSeriesDict; "
            "get_type_hints(TimeSeries.forecast); "
            "get_type_hints(TimeSeriesDict.forecast); "
            "assert 'chronos' not in sys.modules; "
            "assert 'timesfm' not in sys.modules",
        ],
        check=True,
    )


def test_irregular_single_axis_fails_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0, 2.0, 3.0], times=np.array([0.0, 1.0, 3.0]) * u.s)

    with pytest.raises(ValueError, match="regular time axis"):
        series.forecast(horizon=1.0)

    assert MockChronosPipeline.loads == []


def test_irregular_nanosecond_axis_fails_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0, 2.0, 3.0], times=np.array([0, 1, 3]) * u.ns)

    with pytest.raises(ValueError, match="regular time axis"):
        series.forecast(horizon=1 * u.ns)

    assert MockChronosPipeline.loads == []


def test_irregular_dict_axis_fails_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    collection = TimeSeriesDict(
        {
            "regular": TimeSeries([1.0, 2.0], dt=1 * u.s),
            "irregular": TimeSeries(
                [1.0, 2.0, 3.0], times=np.array([0.0, 1.0, 3.0]) * u.s
            ),
        }
    )

    with pytest.raises(ValueError, match="regular time axis"):
        collection.forecast(horizon=1.0)

    assert MockChronosPipeline.loads == []


def test_subnanosecond_interval_with_integral_total_offset_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries([1.0, 2.0], t0_ns=epoch_ns, dt=0.5 * u.ns)

    result = series.forecast(horizon=0.5 * u.ns)

    assert result.t0_gps_ns == epoch_ns + 1
    assert len(MockChronosPipeline.loads) == 1


def test_subnanosecond_forecast_start_fails_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0], t0_ns=100, dt=0.5 * u.ns)

    with pytest.raises(ValueError, match="integer GPS nanoseconds"):
        series.forecast(horizon=0.5 * u.ns)

    assert MockChronosPipeline.loads == []


@pytest.mark.parametrize("as_collection", [False, True])
def test_complex_values_fail_before_backend_load(
    monkeypatch: pytest.MonkeyPatch, as_collection: bool
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries(np.array([1.0 + 1.0j, 2.0 + 0.0j]), dt=1 * u.s)
    target = TimeSeriesDict({"complex": series}) if as_collection else series

    with pytest.raises(ValueError, match="real-valued"):
        target.forecast(horizon=1.0)

    assert MockChronosPipeline.loads == []


@pytest.mark.parametrize("as_collection", [False, True])
@pytest.mark.parametrize(
    "levels",
    [
        (0.1, 0.5, "0.1"),
        (0.1, 0.5, np.float64(0.1)),
        (True, 0.5, 0.9),
        (0.1, 0.5, np.nan),
        (0.1, 0.5, 1.2),
    ],
)
def test_invalid_quantiles_fail_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
    as_collection: bool,
    levels: tuple[object, ...],
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0, 2.0], dt=1 * u.s)
    target = TimeSeriesDict({"series": series}) if as_collection else series

    with pytest.raises(ValueError, match="quantiles"):
        target.forecast(horizon=1.0, quantiles=levels)  # type: ignore[arg-type]

    assert MockChronosPipeline.loads == []


def test_relative_axis_preserves_subnanosecond_forecast_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0], x0=0 * u.s, dt=0.5 * u.ns)

    result = series.forecast(horizon=0.5 * u.ns)

    assert result.x0.to_value(u.ns) == pytest.approx(0.5)


def test_out_of_range_chronos_quantiles_raise_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s)

    # Default Chronos-2 fallback range is [0.01, 0.99]
    with pytest.raises(ValueError, match="supported quantile range"):
        series.forecast(horizon=2.0, quantiles=(0.001, 0.5, 0.9))

    # Pipeline-reported quantiles take precedence (e.g. [0.1, 0.5, 0.9] rejects 0.05)
    monkeypatch.setattr(
        MockChronosPipeline, "quantiles", [0.1, 0.5, 0.9], raising=False
    )
    with pytest.raises(ValueError, match="supported quantile range"):
        series.forecast(horizon=2.0, quantiles=(0.05, 0.5, 0.9))

    collection = TimeSeriesDict({"a": series})
    with pytest.raises(ValueError, match="supported quantile range"):
        collection.forecast(horizon=2.0, quantiles=(0.1, 0.5, 0.95))


# ---------------------------------------------------------------------------
# TimesFM 2.5 backend tests
# ---------------------------------------------------------------------------


class MockTimesFMForecastConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockTimesFMModel:
    loads: list[dict[str, object]] = []
    compiles: list[MockTimesFMForecastConfig] = []
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> MockTimesFMModel:
        assert model_id == "google/timesfm-2.5-200m-pytorch"
        assert kwargs["revision"] == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
        assert kwargs["torch_compile"] is False
        cls.loads.append({"model_id": model_id, **kwargs})
        return cls()

    def compile(self, forecast_config: MockTimesFMForecastConfig) -> None:
        self.compiles.append(forecast_config)
        self._config = forecast_config

    def forecast(
        self, *, horizon: int, inputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(
            {
                "horizon": horizon,
                "inputs": [np.array(item, copy=True) for item in inputs],
            }
        )
        # Simulate TimesFM's internal float32 conversion.
        batch_points = []
        batch_quantiles = []
        for item in inputs:
            arr32 = np.asarray(item, dtype=np.float32)
            base = float(arr32[-1]) + np.linspace(
                0.1, 0.1 * horizon, horizon, dtype=np.float32
            )
            # Column 0 is the independent point head (intentionally distinct
            # to verify it is NOT leaked into .value or .estimate_mean).
            # Columns 1..9 correspond to quantiles 0.1..0.9.
            cols = [base + np.float32(999.0)]
            for q_idx in range(1, 10):
                q_level = q_idx / 10.0
                cols.append(base + np.float32(q_level - 0.5))
            q_mat = np.stack(cols, axis=-1)
            batch_points.append(q_mat[:, 5])
            batch_quantiles.append(q_mat)
        return np.stack(batch_points, axis=0), np.stack(batch_quantiles, axis=0)


def install_mock_timesfm(monkeypatch: pytest.MonkeyPatch) -> None:
    MockTimesFMModel.loads.clear()
    MockTimesFMModel.compiles.clear()
    MockTimesFMModel.calls.clear()
    module = types.ModuleType("timesfm")
    module.TimesFM_2p5_200M_torch = MockTimesFMModel
    module.ForecastConfig = MockTimesFMForecastConfig
    monkeypatch.setitem(sys.modules, "timesfm", module)


def test_timesfm_single_series_forecast_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_timesfm(monkeypatch)
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries(
        [10.0, 20.0, 30.0, 40.0],
        t0_ns=epoch_ns,
        dt=5 * u.ns,
        unit="V",
        name="darm",
        channel="K1:CAL-CS_PROC_DARM",
    )

    result = series.forecast(backend="timesfm", horizon=15 * u.ns)

    assert isinstance(result, TimeSeries)
    assert len(result) == 3
    assert result.name == "darm"
    assert result.channel == series.channel
    assert result.unit == u.V
    assert result.dt == 5 * u.ns
    assert result.t0_gps_ns == epoch_ns + 4 * 5
    assert result.center_kind == "median"
    assert result.uncertainty_kind == "predictive"
    assert result.interval_levels == (0.1, 0.9)
    assert result.estimate_mean is None
    assert result.quantiles is not None
    assert tuple(result.quantiles) == (0.1, 0.5, 0.9)
    np.testing.assert_array_equal(result.value, result.quantiles[0.5].value)
    assert result.lower is result.quantiles[0.1]
    assert result.upper is result.quantiles[0.9]
    assert np.all(result.lower.value <= result.value)
    assert np.all(result.value <= result.upper.value)
    assert np.all(result.value < 500.0)  # index 0 (+999) must not leak

    # Verify ForecastConfig defaults
    assert len(MockTimesFMModel.compiles) == 1
    cfg = MockTimesFMModel.compiles[0].kwargs
    assert cfg["normalize_inputs"] is True
    assert cfg["fix_quantile_crossing"] is True
    assert cfg["force_flip_invariance"] is True
    assert cfg["infer_is_positive"] is False
    assert cfg["use_continuous_quantile_head"] is True
    assert cfg["max_context"] == 32
    assert cfg["max_horizon"] == 128
    assert cfg["per_core_batch_size"] == 1

    # to_timeseriesdict() works seamlessly
    parts = result.to_timeseriesdict()
    assert set(parts.keys()) >= {"value", "lower", "upper", "q0.1", "q0.5", "q0.9"}
    assert "estimate_mean" not in parts


@pytest.mark.parametrize("amplitude", [1.0, 1e-21, 1e18])
def test_timesfm_float64_prescaling_restores_scale_and_order(
    monkeypatch: pytest.MonkeyPatch,
    amplitude: float,
) -> None:
    install_mock_timesfm(monkeypatch)
    raw = amplitude * np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    series = TimeSeries(raw, dt=0.5 * u.s, t0=100.0, unit="m")

    result = series.forecast(backend="timesfm", horizon=2.0)

    # Pre-scaled input passed to TimesFM must be O(1) regardless of physical scale.
    passed = MockTimesFMModel.calls[0]["inputs"][0]
    assert np.std(passed) == pytest.approx(1.0, rel=1e-12)
    assert np.mean(passed) == pytest.approx(0.0, abs=1e-12)

    # Restored outputs preserve original amplitude scale, unit, and quantile ordering.
    assert result.unit == u.m
    assert len(result) == 4
    assert np.all(np.isfinite(result.value))
    assert np.all(result.lower.value < result.value)
    assert np.all(result.value < result.upper.value)
    # Expected median in original scale: mu + scale * (z[-1] + 0.1 * k)
    mu = float(np.mean(raw))
    scale = float(np.std(raw))
    z_last = (raw[-1] - mu) / scale
    expected_median = mu + scale * (
        z_last + np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    )
    np.testing.assert_allclose(result.value, expected_median, rtol=1e-5)
    np.testing.assert_allclose(
        result.lower.value, expected_median - 0.4 * scale, rtol=1e-5
    )
    np.testing.assert_allclose(
        result.upper.value, expected_median + 0.4 * scale, rtol=1e-5
    )


def test_timesfm_constant_series_prescaling_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_timesfm(monkeypatch)
    # Non-zero constant at strain scale (~1e-21): restored interval must stay
    # proportional to abs(mu) instead of inflating to O(1).
    tiny_const = 1e-21
    tiny_series = TimeSeries([tiny_const] * 8, dt=1.0 * u.s, unit="m")
    tiny_result = tiny_series.forecast(backend="timesfm", horizon=2.0)

    passed_tiny = MockTimesFMModel.calls[0]["inputs"][0]
    np.testing.assert_array_equal(passed_tiny, np.zeros(8))
    assert tiny_result.unit == u.m
    assert np.all(np.isfinite(tiny_result.value))
    assert np.all(tiny_result.lower.value <= tiny_result.value)
    assert np.all(tiny_result.value <= tiny_result.upper.value)
    assert np.all(np.abs(tiny_result.value - tiny_const) < 1e-20)
    assert np.all((tiny_result.upper.value - tiny_result.lower.value) < 1e-20)

    # All-zero constant series falls back to scale=1.0 without division by zero.
    zero_series = TimeSeries([0.0] * 8, dt=1.0 * u.s, unit="V")
    zero_result = zero_series.forecast(backend="timesfm", horizon=2.0)
    assert np.all(np.isfinite(zero_result.value))
    assert np.all(zero_result.lower.value <= zero_result.value)
    assert np.all(zero_result.value <= zero_result.upper.value)


def test_timesfm_prescales_after_context_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_timesfm(monkeypatch)
    # For len=16000 and N_pred=1024, max_context=15360; the leading 640 samples
    # are outside TimesFM's context window and must NOT affect scaling.
    tail = np.sin(np.linspace(0.0, 40.0, 15360, dtype=np.float64))
    clean_prefix = np.ones(640, dtype=np.float64)
    huge_prefix = np.full(640, 1e12, dtype=np.float64)

    ts_clean = TimeSeries(np.concatenate([clean_prefix, tail]), dt=1.0 * u.s, unit="m")
    ts_huge = TimeSeries(np.concatenate([huge_prefix, tail]), dt=1.0 * u.s, unit="m")

    res_clean = ts_clean.forecast(backend="timesfm", horizon=1024.0)
    res_huge = ts_huge.forecast(backend="timesfm", horizon=1024.0)

    passed_clean = MockTimesFMModel.calls[0]["inputs"][0]
    passed_huge = MockTimesFMModel.calls[1]["inputs"][0]
    assert len(passed_clean) == 15360
    assert len(passed_huge) == 15360
    assert np.std(passed_huge) == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_array_equal(passed_clean, passed_huge)
    np.testing.assert_array_equal(res_clean.value, res_huge.value)
    np.testing.assert_array_equal(res_clean.lower.value, res_huge.lower.value)
    np.testing.assert_array_equal(res_clean.upper.value, res_huge.upper.value)


def test_timesfm_custom_quantiles_and_unsupported_quantiles_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_timesfm(monkeypatch)
    series = TimeSeries([1.0, 2.0, 3.0, 4.0], dt=1.0 * u.s)

    result = series.forecast(backend="timesfm", horizon=2.0, quantiles=(0.2, 0.5, 0.8))
    assert result.quantiles is not None
    assert tuple(result.quantiles) == (0.2, 0.5, 0.8)
    assert result.interval_levels == (0.2, 0.8)
    assert result.lower is result.quantiles[0.2]
    assert result.upper is result.quantiles[0.8]

    MockTimesFMModel.loads.clear()
    for bad_q in ((0.05, 0.5, 0.95), (0.25, 0.5, 0.75)):
        with pytest.raises(ValueError, match="not supported by TimesFM 2.5"):
            series.forecast(backend="timesfm", horizon=2.0, quantiles=bad_q)
    assert MockTimesFMModel.loads == []


def test_timesfm_horizon_1024_boundary_and_context_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_timesfm(monkeypatch)
    long_series = TimeSeries(np.linspace(0.0, 1.0, 16000), dt=1.0 * u.s)

    # 1024 samples succeeds and caps max_context so max_context + max_horizon <= 16384.
    ok = long_series.forecast(backend="timesfm", horizon=1024.0)
    assert len(ok) == 1024
    cfg = MockTimesFMModel.compiles[-1].kwargs
    assert cfg["max_horizon"] == 1024
    assert cfg["max_context"] == 15360
    assert cfg["max_context"] + cfg["max_horizon"] == 16384

    # 1025 samples raises ValueError before model load.
    MockTimesFMModel.loads.clear()
    with pytest.raises(ValueError, match="at most 1024"):
        long_series.forecast(backend="timesfm", horizon=1025.0)
    assert MockTimesFMModel.loads == []

    # 0 < horizon < dt keeps provisional ValueError.
    with pytest.raises(ValueError, match="N_pred = 0"):
        long_series.forecast(backend="timesfm", horizon=0.5)
    assert MockTimesFMModel.loads == []


def test_timesfm_dict_forecast_siso_batches_and_targets_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_timesfm(monkeypatch)
    collection = TimeSeriesDict(
        {
            "fast1": TimeSeries([1.0, 2.0, 3.0], t0=10, dt=0.5, unit="m"),
            "fast2": TimeSeries([10.0, 20.0, 30.0], t0=20, dt=0.5, unit="V"),
            "slow": TimeSeries([1e-21, 2e-21, 3e-21], t0=30, dt=1.0),
        }
    )

    result = collection.forecast(backend="timesfm", horizon=2.0)

    assert isinstance(result, TimeSeriesDict)
    assert tuple(result) == ("fast1", "fast2", "slow")
    assert len(result["fast1"]) == 4
    assert len(result["fast2"]) == 4
    assert len(result["slow"]) == 2
    assert result["fast1"].unit == u.m
    assert result["fast2"].unit == u.V
    assert result["slow"].estimate_mean is None
    assert len(MockTimesFMModel.loads) == 1
    assert len(MockTimesFMModel.calls) == 2  # grouped by n_pred: 4 and 2

    MockTimesFMModel.loads.clear()
    with pytest.raises(NotImplementedError, match="targets"):
        collection.forecast(backend="timesfm", horizon=2.0, targets=["fast1"])
    assert MockTimesFMModel.loads == []


def test_missing_timesfm_and_missing_torch_raise_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s)

    def _import_without_timesfm(name: str, *args: object, **kwargs: object) -> object:
        if name == "timesfm":
            raise ImportError("mock missing timesfm")
        return _original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "timesfm", raising=False)
    monkeypatch.setattr("builtins.__import__", _import_without_timesfm)
    with pytest.raises(ImportError, match=r"\[timesfm\]"):
        series.forecast(backend="timesfm", horizon=1.0)

    # Simulate timesfm installed without PyTorch extra (raises AttributeError on access)
    monkeypatch.undo()
    partial_timesfm = types.ModuleType("timesfm")
    monkeypatch.setitem(sys.modules, "timesfm", partial_timesfm)
    with pytest.raises(ImportError, match=r"\[timesfm\]"):
        series.forecast(backend="timesfm", horizon=1.0)


def test_unsupported_forecast_backend_raises_value_error() -> None:
    series = TimeSeries([1.0, 2.0], dt=1.0 * u.s)
    with pytest.raises(ValueError, match="Unsupported forecast backend"):
        series.forecast(backend="unknown", horizon=1.0)
    collection = TimeSeriesDict({"a": series})
    with pytest.raises(ValueError, match="Unsupported forecast backend"):
        collection.forecast(backend="unknown", horizon=1.0)


def test_joint_forecast_percentiles_roundtrip_hdf5(monkeypatch, tmp_path):
    import gwexpy.timeseries._forecast as forecast

    class Pipeline:
        def predict_quantiles(self, inputs, *, quantile_levels, **kwargs):
            assert quantile_levels == [0.1, 0.5, 0.9]
            assert set(inputs[0]["past_covariates"]) == {"aux"}
            return [np.arange(6, dtype=float).reshape(1, 2, 3)], [np.zeros((1, 2))]

    monkeypatch.setattr(forecast, "_load_chronos_pipeline", lambda: Pipeline())
    collection = TimeSeriesDict(
        {
            "DARM": TimeSeries([1.0, 2.0], t0=100, dt=1.0, unit="V", name="DARM"),
            "aux": TimeSeries([3.0, 4.0], t0=100, dt=1.0, unit="m", name="aux"),
        }
    )
    result = collection.forecast(
        horizon=2.0, targets=["DARM"], percentiles=[10, 50, 90]
    )
    assert list(result) == ["DARM"]
    assert result["DARM"].t0 == 102 * u.s
    np.testing.assert_allclose(
        result["DARM"].percentiles[90].value, result["DARM"].quantiles[0.9].value
    )

    path = tmp_path / "joint-forecast.h5"
    result.write(path, format="hdf5")
    restored = TimeSeriesDict.read(path, format="hdf5", names=["DARM"])
    np.testing.assert_allclose(restored["DARM"].value, result["DARM"].value)
    np.testing.assert_allclose(
        restored["DARM"].percentiles[10].value, result["DARM"].percentiles[10].value
    )
    assert restored["DARM"].center_kind == "median"
    assert restored["DARM"].uncertainty_kind == "predictive"


def _timestral_model(levels, values=None):
    class Timestral:
        def forecast(self, inputs, *, prediction_length, clip_nonnegative):
            data = np.zeros((1, 1, prediction_length, len(levels)))
            if values is not None:
                data[...] = values
            return types.SimpleNamespace(quantiles=data, quantile_levels=levels)

    return Timestral()


def test_timestral_matches_levels_within_tolerance(monkeypatch):
    import gwexpy.timeseries._forecast as forecast

    levels = (0.1, 0.15000000000000002, 0.5, 0.85, 0.9)
    monkeypatch.setattr(
        forecast, "_load_timestral_model", lambda: _timestral_model(levels)
    )
    result = TimeSeries([1.0, 2.0, 4.0], dt=1.0).forecast(
        backend="timestral", horizon=1.0, quantiles=(0.15, 0.5, 0.85)
    )
    assert list(result.quantiles) == [0.15, 0.5, 0.85]
    with pytest.raises(ValueError, match="0.3 is not supported by Timestral"):
        TimeSeries([1.0, 2.0, 4.0], dt=1.0).forecast(
            backend="timestral", horizon=1.0, quantiles=(0.3, 0.5)
        )


def test_additional_backend_rejects_non_finite_output(monkeypatch):
    import gwexpy.timeseries._forecast as forecast

    monkeypatch.setattr(
        forecast,
        "_load_timestral_model",
        lambda: _timestral_model((0.1, 0.5, 0.9), values=np.nan),
    )
    with pytest.raises(ValueError, match="non-finite forecast"):
        TimeSeries([1.0, 2.0, 4.0], dt=1.0).forecast(backend="timestral", horizon=1.0)


def test_joint_forecast_names_mismatched_member_and_bad_targets():
    collection = TimeSeriesDict(
        {
            "a": TimeSeries([1.0, 2.0], dt=1.0),
            "b": TimeSeries([1.0, 2.0, 3.0], dt=1.0),
        }
    )
    with pytest.raises(ValueError, match="'b' differs from 'a'"):
        collection.forecast(horizon=1.0, targets=["a"])
    with pytest.raises(ValueError, match="absent from the collection"):
        collection.forecast(horizon=1.0, targets=["missing"])
    with pytest.raises(ValueError, match="hashable"):
        collection.forecast(horizon=1.0, targets=[["a"]])


def test_chronos_bolt_accepts_percentiles(monkeypatch):
    import gwexpy.timeseries._forecast as forecast

    class Bolt:
        quantiles = [0.1, 0.5, 0.9]

        def predict_quantiles(self, inputs, *, prediction_length, quantile_levels):
            assert quantile_levels == [0.1, 0.5, 0.9]
            return np.zeros((1, prediction_length, 3)), np.zeros((1, prediction_length))

    monkeypatch.setattr(forecast, "_load_bolt_pipeline", lambda: Bolt())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            as_tensor=lambda values, dtype: np.asarray(values, dtype=dtype),
            float32=np.float32,
        ),
    )
    result = TimeSeries([1.0, 2.0, 4.0], dt=1.0).forecast(
        backend="chronos-bolt", horizon=1.0, percentiles=[10, 50, 90]
    )
    assert list(result.quantiles) == [0.1, 0.5, 0.9]
    assert result.percentiles[90] is result.quantiles[0.9]
    with pytest.raises(ValueError, match="outside Chronos-Bolt"):
        TimeSeries([1.0, 2.0, 4.0], dt=1.0).forecast(
            backend="chronos-bolt", horizon=1.0, percentiles=[5, 50, 95]
        )
