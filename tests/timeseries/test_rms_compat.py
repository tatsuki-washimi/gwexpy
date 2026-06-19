"""gwpy-compatibility tests for ``TimeSeries.rms`` (issue #451).

gwpy's ``TimeSeries.rms(stride)`` takes ``stride`` (seconds) as the first
positional argument and returns a new ``TimeSeries`` holding one RMS value per
``stride``-second window (``dt = stride``).  These tests pin that behaviour for
the gwexpy override, plus the documented gwexpy enhancements (time ``Quantity``
stride and unit preservation).
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries


def _series(data, sample_rate=100, **kwargs):
    return TimeSeries(np.asarray(data, dtype=float), sample_rate=sample_rate, **kwargs)


# ---------------------------------------------------------------------------
# core gwpy semantics
# ---------------------------------------------------------------------------

def test_rms_stride_returns_trend_timeseries():
    ts = _series(np.arange(1000.0), sample_rate=100)  # 10 s
    out = ts.rms(2)
    assert isinstance(out, TimeSeries)
    assert out.dt.to("s").value == pytest.approx(2.0)
    assert out.sample_rate.to("Hz").value == pytest.approx(0.5)
    assert out.size == 5  # 10 s / 2 s
    assert out.t0 == ts.t0


def test_rms_positional_int_does_not_raise():
    # gwpy: data.rms(10) -> trend.  Regression for the AxisError in #451.
    ts = _series(np.arange(1000.0), sample_rate=100)
    out = ts.rms(10)
    assert out.size == 1
    assert out.dt.to("s").value == pytest.approx(10.0)


def test_rms_values_match_manual_windows():
    rng = np.random.default_rng(1)
    arr = rng.normal(size=1000)
    ts = _series(arr, sample_rate=100)
    out = ts.rms(2)  # 200 samples per window
    expected = np.array(
        [np.sqrt(np.mean(np.abs(arr[i * 200:(i + 1) * 200]) ** 2)) for i in range(5)]
    )
    np.testing.assert_allclose(out.value, expected)


def test_rms_default_stride_is_one_second():
    ts = _series(np.arange(500.0), sample_rate=100)  # 5 s
    out = ts.rms()
    assert out.size == 5
    assert out.dt.to("s").value == pytest.approx(1.0)


def test_rms_matches_gwpy_reference():
    gwpy_ts = pytest.importorskip("gwpy.timeseries").TimeSeries
    rng = np.random.default_rng(7)
    arr = rng.normal(size=10_000)
    ours = _series(arr, sample_rate=256).rms(4)
    ref = gwpy_ts(arr, sample_rate=256).rms(4)
    np.testing.assert_allclose(ours.value, ref.value)
    assert ours.size == ref.size
    assert ours.dt.to("s").value == pytest.approx(ref.dt.to("s").value)


# ---------------------------------------------------------------------------
# gwexpy enhancements (documented divergences from gwpy)
# ---------------------------------------------------------------------------

def test_rms_accepts_time_quantity_stride():
    ts = _series(np.arange(1000.0), sample_rate=100)
    np.testing.assert_allclose(ts.rms(2).value, ts.rms(2 * u.s).value)
    # other time units convert too
    np.testing.assert_allclose(ts.rms(2).value, ts.rms(2000 * u.ms).value)


def test_rms_preserves_unit():
    ts = _series(np.arange(1000.0), sample_rate=100, unit=u.m)
    out = ts.rms(2)
    assert out.unit == u.m


def test_rms_preserves_derived_unit():
    ts = _series(np.arange(1000.0), sample_rate=100, unit=u.m / u.s)
    assert ts.rms(2).unit == u.m / u.s


def test_rms_accepts_dimensionless_quantity_stride():
    # a dimensionless Quantity is read as a number of seconds (gwpy parity)
    ts = _series(np.arange(1000.0), sample_rate=100)
    np.testing.assert_allclose(
        ts.rms(2).value, ts.rms(2 * u.dimensionless_unscaled).value
    )


def test_rms_non_time_quantity_stride_raises():
    ts = _series(np.arange(100.0), sample_rate=10)
    with pytest.raises(ValueError):
        ts.rms(2 * u.Hz)


# ---------------------------------------------------------------------------
# edge cases / metadata
# ---------------------------------------------------------------------------

def test_rms_trailing_window_dropped():
    ts = _series(np.arange(100.0), sample_rate=10)  # 10 s
    out = ts.rms(0.7)  # 7 samples / window -> 14 full windows, 2 samples dropped
    assert out.size == 14


def test_rms_stride_longer_than_data_is_empty():
    ts = _series(np.arange(100.0), sample_rate=10)  # 10 s
    out = ts.rms(1000)
    assert out.size == 0


def test_rms_sub_sample_stride_raises():
    ts = _series(np.arange(100.0), sample_rate=10)
    with pytest.raises(ValueError, match="shorter than one sample"):
        ts.rms(0.05)  # 0.05 s < 0.1 s sample period


def test_rms_zero_or_negative_stride_raises():
    ts = _series(np.arange(100.0), sample_rate=10)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="shorter than one sample"):
            ts.rms(bad)


def test_rms_irregular_series_raises():
    # an irregularly-sampled (times-indexed) series has no sample_rate
    times = np.array([0.0, 0.1, 0.3, 0.7, 1.5])
    ts = TimeSeries(np.arange(5.0), times=times)
    with pytest.raises(ValueError, match="regularly-sampled"):
        ts.rms(1)


def test_rms_propagates_nan_per_window():
    # gwpy uses plain mean (no nan-ignore): a NaN taints only its own window.
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ts = _series(data, sample_rate=5)  # 2 s -> two 1 s windows
    out = ts.rms(1)
    assert np.isnan(out.value[0])
    assert np.isfinite(out.value[1])


def test_rms_name_and_channel_metadata():
    ts = _series(np.arange(100.0), sample_rate=10, name="X1:SIG", channel="X1:SIG")
    out = ts.rms(2)
    assert out.name is not None and "X1:SIG" in out.name
    assert str(out.channel) == "X1:SIG"


def test_rms_unnamed_series_has_none_name():
    ts = _series(np.arange(100.0), sample_rate=10)
    assert ts.rms(2).name is None
