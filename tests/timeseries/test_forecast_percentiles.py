"""Tests for percentile aliases on forecasts and uncertainty results."""

import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries._forecast import _resolve_percentiles


def test_percentiles_view_uses_quantile_series():
    series = TimeSeries([1, 2], dt=1)
    quantile = TimeSeries([2, 3], dt=1)
    assert series.percentiles is None
    series.quantiles = {0.9: quantile}
    assert series.percentiles[90] is series.quantiles[0.9]


@pytest.mark.parametrize("collection", [False, True])
def test_forecast_rejects_both_aliases(collection):
    series = TimeSeries([1, 2], dt=1)
    source = TimeSeriesDict({"x": series}) if collection else series
    with pytest.raises(ValueError, match="either quantiles or percentiles"):
        source.forecast(horizon=1, quantiles=[0.5], percentiles=[50])


def test_percentile_conversion():
    assert _resolve_percentiles(None, [10, 50, 90]) == [0.1, 0.5, 0.9]
    with pytest.raises(ValueError, match="real numbers"):
        _resolve_percentiles(None, [True])


@pytest.mark.parametrize("collection", [False, True])
def test_forecast_passes_canonical_quantiles(monkeypatch, collection):
    from gwexpy.timeseries import _forecast

    captured = {}

    def fake_forecast(source, **kwargs):
        captured.update(kwargs)
        return source

    name = "forecast_collection" if collection else "forecast_series"
    monkeypatch.setattr(_forecast, name, fake_forecast)
    series = TimeSeries([1, 2], dt=1)
    source = TimeSeriesDict({"x": series}) if collection else series
    source.forecast(horizon=1, percentiles=[10, 50, 90])
    assert captured["quantiles"] == [0.1, 0.5, 0.9]
