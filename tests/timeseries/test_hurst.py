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


# ---------------------------------------------------------------------------
# Backend ImportError paths
# ---------------------------------------------------------------------------


def test_get_hurst_rs_import_error():
    # Lines 44-50 — ImportError from hurst package
    import sys
    sys.modules["hurst"] = None  # type: ignore
    try:
        with pytest.raises(ImportError, match="hurst package"):
            hurst_module._get_hurst_rs(np.arange(10.0), "random_walk", True)
    finally:
        del sys.modules["hurst"]


def test_get_hurst_exponent_import_error():
    # Lines 54-57 — ImportError from hurst_exponent package
    import sys
    sys.modules["hurst_exponent"] = None  # type: ignore
    try:
        with pytest.raises(ImportError, match="hurst-exponent"):
            hurst_module._get_hurst_exponent(np.arange(10.0), "standard")
    finally:
        del sys.modules["hurst_exponent"]


def test_get_exp_hurst_import_error():
    # Lines 87-91 — ImportError from exp_hurst package
    import sys
    sys.modules["exp_hurst"] = None  # type: ignore
    try:
        with pytest.raises(ImportError, match="exp-hurst"):
            hurst_module._get_exp_hurst(np.arange(10.0))
    finally:
        del sys.modules["exp_hurst"]


def test_get_hurst_exponent_standard_tuple(monkeypatch):
    # Lines 63-73 — standard method, tuple return
    fake_mod = type("fake", (), {})()
    fake_mod.standard_hurst = lambda x: (0.6, 1.1, 2.2)
    monkeypatch.setitem(__import__("sys").modules, "hurst_exponent", fake_mod)
    H, meth, backend, det = hurst_module._get_hurst_exponent(np.arange(10.0), "standard")
    assert H == pytest.approx(0.6)
    assert "fit" in det


def test_get_hurst_exponent_standard_scalar(monkeypatch):
    # Lines 67, 70-72 — standard method, scalar return
    fake_mod = type("fake", (), {})()
    fake_mod.standard_hurst = lambda x: 0.55
    monkeypatch.setitem(__import__("sys").modules, "hurst_exponent", fake_mod)
    H, meth, backend, det = hurst_module._get_hurst_exponent(np.arange(10.0), "standard")
    assert H == pytest.approx(0.55)
    assert det == {}


def test_get_hurst_exponent_generalized_tuple(monkeypatch):
    # Lines 75-83 — generalized method, tuple return
    fake_mod = type("fake", (), {})()
    fake_mod.generalized_hurst = lambda x: (0.7, 0.5)
    monkeypatch.setitem(__import__("sys").modules, "hurst_exponent", fake_mod)
    H, meth, backend, det = hurst_module._get_hurst_exponent(np.arange(10.0), "generalized")
    assert H == pytest.approx(0.7)


def test_get_hurst_exponent_generalized_scalar(monkeypatch):
    # Lines 77, 80-82 — generalized method, scalar return
    fake_mod = type("fake", (), {})()
    fake_mod.generalized_hurst = lambda x: 0.65
    monkeypatch.setitem(__import__("sys").modules, "hurst_exponent", fake_mod)
    H, meth, backend, det = hurst_module._get_hurst_exponent(np.arange(10.0), "generalized")
    assert H == pytest.approx(0.65)
    assert det == {}


def test_get_exp_hurst_ok(monkeypatch):
    # Lines 87-94 — exp_hurst available
    fake_mod = type("fake", (), {})()
    fake_mod.hurst = lambda x: 0.45
    monkeypatch.setitem(__import__("sys").modules, "exp_hurst", fake_mod)
    H, meth, backend, det = hurst_module._get_exp_hurst(np.arange(10.0))
    assert H == pytest.approx(0.45)
    assert meth == "exp"


# ---------------------------------------------------------------------------
# local_hurst — additional branch coverage
# ---------------------------------------------------------------------------


def test_local_hurst_window_as_quantity(monkeypatch):
    # Lines 246-252 — window as Quantity
    ts = TimeSeries(np.arange(12.0), t0=0, dt=1 * u.s, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    out = hurst_module.local_hurst(ts, window=4 * u.s, step=2)
    assert len(out) > 0


def test_local_hurst_step_as_quantity(monkeypatch):
    # Lines 267-273 — step as Quantity
    ts = TimeSeries(np.arange(12.0), t0=0, dt=1 * u.s, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    out = hurst_module.local_hurst(ts, window=4 * u.s, step=2 * u.s)
    assert len(out) > 0


def test_local_hurst_step_float(monkeypatch):
    # Lines 276-280 — step as float
    ts = TimeSeries(np.arange(12.0), t0=0, dt=1 * u.s, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    out = hurst_module.local_hurst(ts, window=4, step=2.0)
    assert len(out) > 0


def test_local_hurst_step_else_branch(monkeypatch):
    # Line 282 — step as other type (fallback int())
    ts = TimeSeries(np.arange(12.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    # Step as numpy int scalar (not int or np.integer — use a custom obj)
    class FakeStep:
        def __index__(self):
            return 2
        def __int__(self):
            return 2

    out = hurst_module.local_hurst(ts, window=4, step=FakeStep())
    assert len(out) > 0


def test_local_hurst_step_small_clamp(monkeypatch):
    # Line 285 — step_samples < 1 → clamp to 1
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    # step=0 → should be clamped to 1
    out = hurst_module.local_hurst(ts, window=4, step=0)
    assert len(out) > 0


def test_local_hurst_impute_policy(monkeypatch):
    # Lines 302-307 — nan_policy='impute' calls impute_timeseries
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ts = TimeSeries(data, t0=0, dt=1, unit=u.dimensionless_unscaled)

    imputed = TimeSeries(np.arange(1.0, 11.0), t0=0, dt=1, unit=u.dimensionless_unscaled)
    called = {"impute": False}

    def fake_impute(series, **kwargs):
        called["impute"] = True
        return imputed

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr("gwexpy.timeseries.preprocess.impute_timeseries", fake_impute)
    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    out = hurst_module.local_hurst(ts, window=4, step=2, nan_policy="impute")
    assert called["impute"] is True
    assert len(out) > 0


def test_local_hurst_dt_not_quantity(monkeypatch):
    # Line 242 — dt is not Quantity (fallback to float)
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    # Override dt to be a plain float
    class FakeTS:
        value = np.arange(10.0)
        dt = 1.0  # plain float, not Quantity

        class _T0:
            value = 0.0
        t0 = _T0()

        class _Times:
            unit = u.s
        times = _Times()

    out = hurst_module.local_hurst(FakeTS(), window=4, step=2, center=True)
    assert len(out) > 0


def test_local_hurst_window_float_non_time_dt(monkeypatch):
    # Lines 259-260 — float window, dt has non-time physical_type
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    # Use plain float dt (not Quantity with time physical_type)
    class FakeTS:
        value = np.arange(10.0)
        dt = 1.0

        class _T0:
            value = 0.0
        t0 = _T0()

        class _Times:
            unit = u.dimensionless_unscaled
        times = _Times()

    out = hurst_module.local_hurst(FakeTS(), window=4.0, step=2)
    assert len(out) > 0


def test_local_hurst_hurst_raises_valueerror(monkeypatch):
    # Lines 349-350 — hurst raises ValueError → NaN in output
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def failing_hurst(_ts, **kwargs):
        raise ValueError("test error")

    monkeypatch.setattr(hurst_module, "hurst", failing_hurst)

    out = hurst_module.local_hurst(ts, window=4, step=2)
    assert np.all(np.isnan(out.value))


def test_local_hurst_window_else_fallback(monkeypatch):
    # Line 261 — window is not int/float/Quantity → fallback int(window)
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    class FakeWindow:
        def __int__(self):
            return 4

    out = hurst_module.local_hurst(ts, window=FakeWindow(), step=2)
    assert len(out) > 0


def test_local_hurst_step_float_with_time_dt(monkeypatch):
    # Line 278 — step is float AND dt is time Quantity
    ts = TimeSeries(np.arange(12.0), t0=0, dt=1 * u.s, unit=u.dimensionless_unscaled)

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    out = hurst_module.local_hurst(ts, window=4, step=2.0)
    assert len(out) > 0


def test_local_hurst_step_float_non_time_dt(monkeypatch):
    # Line 280 — step is float AND dt is NOT time Quantity
    class FakeTS:
        value = np.arange(12.0)
        dt = 1.0  # plain float, not Quantity

        class _T0:
            value = 0.0
        t0 = _T0()

        class _Times:
            unit = u.dimensionless_unscaled
        times = _Times()

    def fake_hurst(_ts, **kwargs):
        return 0.5

    monkeypatch.setattr(hurst_module, "hurst", fake_hurst)

    out = hurst_module.local_hurst(FakeTS(), window=4, step=2.0)
    assert len(out) > 0


def test_get_hurst_rs_with_mock_compute(monkeypatch):
    # Lines 49-50 — compute_Hc called successfully
    fake_hurst_mod = type("fake_hurst", (), {})()
    fake_hurst_mod.compute_Hc = lambda x, kind, simplified: (0.55, 1.2, [1, 2, 3])
    monkeypatch.setitem(__import__("sys").modules, "hurst", fake_hurst_mod)
    H, meth, backend, det = hurst_module._get_hurst_rs(np.arange(10.0), "random_walk", True)
    assert H == pytest.approx(0.55)
    assert meth == "rs"
    assert backend == "hurst"
    assert det["c"] == pytest.approx(1.2)
