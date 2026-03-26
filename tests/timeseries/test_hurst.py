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


# --- HurstResult ---

def test_hurst_result_summary_dict():
    r = hurst_module.HurstResult(H=0.7, method="rs", backend="hurst", details={"c": 1.5})
    d = r.summary_dict()
    assert d["H"] == pytest.approx(0.7)
    assert d["method"] == "rs"
    assert d["backend"] == "hurst"
    assert d["c"] == pytest.approx(1.5)


def test_hurst_result_summary_dict_empty_details():
    r = hurst_module.HurstResult(H=0.5, method="exp", backend="exp-hurst", details={})
    d = r.summary_dict()
    assert set(d.keys()) == {"H", "method", "backend"}


# --- hurst() ---

def test_hurst_unknown_method():
    ts = TimeSeries(np.arange(8.0), t0=0, dt=1, unit=u.dimensionless_unscaled)
    with pytest.raises(ValueError, match="Unknown method"):
        hurst_module.hurst(ts, method="bogus")


def test_hurst_return_float(monkeypatch):
    ts = TimeSeries(np.arange(8.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_rs(x, kind, simplified):
        return 0.6, "rs", "hurst", {}

    monkeypatch.setattr(hurst_module, "_get_hurst_rs", fake_rs)
    out = hurst_module.hurst(ts, method="rs", return_details=False)
    assert isinstance(out, float)
    assert out == pytest.approx(0.6)


def test_hurst_method_standard(monkeypatch):
    ts = TimeSeries(np.arange(8.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_exp(x, method):
        return 0.45, "standard", "hurst-exponent", {}

    monkeypatch.setattr(hurst_module, "_get_hurst_exponent", fake_exp)
    out = hurst_module.hurst(ts, method="standard")
    assert out == pytest.approx(0.45)


def test_hurst_method_exp(monkeypatch):
    ts = TimeSeries(np.arange(8.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_exp(x):
        return 0.3, "exp", "exp-hurst", {}

    monkeypatch.setattr(hurst_module, "_get_exp_hurst", fake_exp)
    out = hurst_module.hurst(ts, method="exp")
    assert out == pytest.approx(0.3)


def test_hurst_auto_fallback_chain(monkeypatch):
    """auto mode falls through rs -> standard -> exp when each raises ImportError."""
    ts = TimeSeries(np.arange(8.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fail_rs(x, kind, simplified):
        raise ImportError("no hurst")

    def fail_std(x, method):
        raise ImportError("no hurst-exponent")

    def fake_exp(x):
        return 0.35, "exp", "exp-hurst", {}

    monkeypatch.setattr(hurst_module, "_get_hurst_rs", fail_rs)
    monkeypatch.setattr(hurst_module, "_get_hurst_exponent", fail_std)
    monkeypatch.setattr(hurst_module, "_get_exp_hurst", fake_exp)

    out = hurst_module.hurst(ts, method="auto")
    assert out == pytest.approx(0.35)


def test_hurst_auto_all_backends_missing(monkeypatch):
    """auto mode raises ImportError when all backends are missing."""
    ts = TimeSeries(np.arange(8.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def raise_import(*a, **kw):
        raise ImportError("not installed")

    monkeypatch.setattr(hurst_module, "_get_hurst_rs", raise_import)
    monkeypatch.setattr(hurst_module, "_get_hurst_exponent", raise_import)
    monkeypatch.setattr(hurst_module, "_get_exp_hurst", raise_import)

    with pytest.raises(ImportError):
        hurst_module.hurst(ts, method="auto")


# --- local_hurst() ---

def test_local_hurst_center_false(monkeypatch):
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    out = hurst_module.local_hurst(ts, window=4, step=2, center=False)
    # With center=False, t = t0 + start * dt = 0, 2, 4, 6
    assert np.allclose(out.times.value, [0.0, 2.0, 4.0, 6.0])


def test_local_hurst_default_step(monkeypatch):
    ts = TimeSeries(np.arange(12.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    # step defaults to window // 2 = 2
    out = hurst_module.local_hurst(ts, window=4)
    assert len(out) > 0
    assert np.all(np.isfinite(out.value))


def test_local_hurst_nan_window_raises(monkeypatch):
    """Windows containing NaN with raise policy raises ValueError."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    ts = TimeSeries(data, t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    with pytest.raises(ValueError, match="NaN found in window"):
        hurst_module.local_hurst(ts, window=4, step=1, nan_policy="raise")


def test_local_hurst_nan_window_skipped(monkeypatch):
    """Windows containing NaN with non-raise policy yield NaN in output."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    ts = TimeSeries(data, t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    # nan_policy != 'raise' causes NaN windows to be skipped (NaN output)
    out = hurst_module.local_hurst(ts, window=4, step=1, nan_policy="skip")
    assert np.any(np.isnan(out.value))


def test_local_hurst_window_float(monkeypatch):
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    # window as float should be treated as samples
    out = hurst_module.local_hurst(ts, window=4.0, step=2)
    assert len(out) > 0
