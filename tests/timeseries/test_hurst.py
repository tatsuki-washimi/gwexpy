from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries import hurst as hurst_module


def test_hurst_nan_policy_raise():
    ts = TimeSeries([1.0, np.nan, 2.0], t0=0, dt=1, unit=u.dimensionless_unscaled)
    with pytest.raises(ValueError):
        hurst_module.hurst(ts, method="rs", nan_policy="raise")


def test_hurst_auto_backend_selection(monkeypatch):
    ts = TimeSeries(np.arange(8.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_rs(x, kind, simplified):
        return 0.42, "rs", "hurst", {"c": 1.1}

    monkeypatch.setattr(hurst_module, "_get_hurst_rs", fake_rs)

    out = hurst_module.hurst(ts, method="auto", return_details=True)
    assert out.H == pytest.approx(0.42)
    assert out.method == "rs"
    assert out.backend == "hurst"
    assert out.details["c"] == pytest.approx(1.1)


def test_hurst_impute_calls_preprocess(monkeypatch):
    ts = TimeSeries([1.0, np.nan, 3.0], t0=0, dt=1, unit=u.dimensionless_unscaled)
    called = {"impute": False}

    def fake_impute(series, **kwargs):
        called["impute"] = True
        return TimeSeries([1.0, 2.0, 3.0], t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_rs(x, kind, simplified):
        return 0.55, "rs", "hurst", {}

    monkeypatch.setattr("gwexpy.timeseries.preprocess.impute_timeseries", fake_impute)
    monkeypatch.setattr(hurst_module, "_get_hurst_rs", fake_rs)

    out = hurst_module.hurst(ts, method="rs", nan_policy="impute")
    assert called["impute"] is True
    assert out == pytest.approx(0.55)


def test_local_hurst_window_center(monkeypatch):
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.25

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    out = hurst_module.local_hurst(ts, window=4, step=2, center=True)
    assert np.allclose(out.value, 0.25)
    assert np.allclose(out.times.value, [2.0, 4.0, 6.0, 8.0])
