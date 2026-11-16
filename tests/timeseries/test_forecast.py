from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


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
        mean_tensors = [point[i : i + 1] + 100 for i in range(len(inputs))]
        return quantile_tensors, mean_tensors


def install_mock_chronos(monkeypatch: pytest.MonkeyPatch) -> None:
    MockChronosPipeline.calls.clear()
    MockChronosPipeline.loads.clear()
    module = types.ModuleType("chronos")
    module.Chronos2Pipeline = MockChronosPipeline
    monkeypatch.setitem(sys.modules, "chronos", module)


def test_single_series_forecast_metadata_and_quantiles(
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

    result = series.forecast(horizon=2)

    assert result.point.name == "sensor"
    assert result.point.channel == series.channel
    assert result.point.unit == u.V
    assert result.point.dt == 7 * u.ns
    assert result.point.t0_gps_ns == epoch_ns + 3 * 7
    np.testing.assert_array_equal(result.point.value, [4.0, 5.0])
    assert tuple(result.quantiles) == (0.1, 0.5, 0.9)
    np.testing.assert_array_equal(result.point.value, result.quantiles[0.5].value)
    assert MockChronosPipeline.calls[0]["cross_learning"] is False


def test_dict_forecast_batches_independent_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    first = TimeSeries([1.0, 2.0], t0=100, dt=0.25, name="one", unit="m")
    second = TimeSeries([10.0, 20.0], t0=200, dt=0.5, name="two", unit="s")
    collection = TimeSeriesDict({"a": first, "b": second})

    result = collection.forecast(horizon=1)

    assert tuple(result.point) == ("a", "b")
    assert result.point["a"].name == "one"
    assert result.point["b"].unit == u.s
    assert result.point["a"].t0 == first.t0 + 2 * first.dt
    assert result.point["b"].t0 == second.t0 + 2 * second.dt
    call = MockChronosPipeline.calls[0]
    assert call["cross_learning"] is False
    assert len(call["inputs"]) == 2
    assert all(np.ndim(item) == 1 for item in call["inputs"])


def test_dict_series_changes_do_not_cross_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    original = TimeSeriesDict(
        {"a": TimeSeries([1.0, 2.0]), "b": TimeSeries([10.0, 20.0])}
    )
    changed = TimeSeriesDict(
        {"a": TimeSeries([1.0, 200.0]), "b": TimeSeries([10.0, 20.0])}
    )

    before = original.forecast(horizon=1)
    after = changed.forecast(horizon=1)

    assert MockChronosPipeline.calls[0]["inputs"][1].tolist() == [10.0, 20.0]
    assert MockChronosPipeline.calls[1]["inputs"][1].tolist() == [10.0, 20.0]
    np.testing.assert_array_equal(before.point["b"].value, after.point["b"].value)


def test_missing_chronos_names_future_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "chronos", raising=False)
    monkeypatch.setattr("builtins.__import__", _import_without_chronos)

    with pytest.raises(ImportError, match=r"\[chronos\]"):
        TimeSeries([1.0, 2.0]).forecast(horizon=1)


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
            "from gwexpy.timeseries import ForecastResult, TimeSeries, TimeSeriesDict; "
            "get_type_hints(TimeSeries.forecast); "
            "get_type_hints(TimeSeriesDict.forecast); "
            "get_type_hints(ForecastResult); "
            "assert 'chronos' not in sys.modules",
        ],
        check=True,
    )


def test_irregular_single_axis_fails_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0, 2.0, 3.0], times=np.array([0.0, 1.0, 3.0]) * u.s)

    with pytest.raises(ValueError, match="regular time axis"):
        series.forecast(horizon=1)

    assert MockChronosPipeline.loads == []


def test_irregular_nanosecond_axis_fails_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0, 2.0, 3.0], times=np.array([0, 1, 3]) * u.ns)

    with pytest.raises(ValueError, match="regular time axis"):
        series.forecast(horizon=1)

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
        collection.forecast(horizon=1)

    assert MockChronosPipeline.loads == []


def test_subnanosecond_interval_with_integral_total_offset_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    epoch_ns = 1_234_567_890_123_456_789
    series = TimeSeries([1.0, 2.0], t0_ns=epoch_ns, dt=0.5 * u.ns)

    result = series.forecast(horizon=1)

    assert result.point.t0_gps_ns == epoch_ns + 1
    assert len(MockChronosPipeline.loads) == 1


def test_subnanosecond_forecast_start_fails_before_backend_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0], t0_ns=100, dt=0.5 * u.ns)

    with pytest.raises(ValueError, match="integer GPS nanoseconds"):
        series.forecast(horizon=1)

    assert MockChronosPipeline.loads == []


@pytest.mark.parametrize("as_collection", [False, True])
def test_complex_values_fail_before_backend_load(
    monkeypatch: pytest.MonkeyPatch, as_collection: bool
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries(np.array([1.0 + 1.0j, 2.0 + 0.0j]), dt=1 * u.s)
    target = TimeSeriesDict({"complex": series}) if as_collection else series

    with pytest.raises(ValueError, match="real-valued"):
        target.forecast(horizon=1)

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
        target.forecast(horizon=1, quantiles=levels)  # type: ignore[arg-type]

    assert MockChronosPipeline.loads == []


def test_relative_axis_preserves_subnanosecond_forecast_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_mock_chronos(monkeypatch)
    series = TimeSeries([1.0], x0=0 * u.s, dt=0.5 * u.ns)

    result = series.forecast(horizon=1)

    assert result.point.x0.to_value(u.ns) == pytest.approx(0.5)
