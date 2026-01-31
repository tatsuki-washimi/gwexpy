import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries._core import TimeSeriesCore


def test_timeseries_tail_none_returns_self():
    ts = TimeSeries(np.arange(5.0), t0=0, dt=1, name="demo")
    out = ts.tail(None)
    assert out is ts


def test_timeseries_tail_nonpositive_returns_empty():
    ts = TimeSeries(np.arange(5.0), t0=0, dt=1, name="demo")
    out = ts.tail(0)
    assert len(out) == 0
    assert isinstance(out, TimeSeries)


def test_timeseries_tail_positive_values():
    ts = TimeSeries(np.arange(6.0), t0=0, dt=1, name="demo")
    out = ts.tail(3)
    assert np.allclose(out.value, [3.0, 4.0, 5.0])


def test_timeseries_append_non_inplace():
    ts = TimeSeries([1.0, 2.0, 3.0], t0=0, dt=1, name="demo")
    out = ts.append(np.array([4.0, 5.0]), inplace=False)
    assert isinstance(out, TimeSeries)
    assert len(ts) == 3
    assert len(out) == 5
    assert np.allclose(out.value, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_timeseries_find_peaks_basic():
    pytest.importorskip("scipy")
    ts = TimeSeries([0.0, 1.0, 0.0, 2.0, 0.0], t0=0, dt=1, unit=u.V, name="sig")
    peaks, props = ts.find_peaks(height=1.5 * u.V)
    assert isinstance(props, dict)
    assert len(peaks) == 1
    assert np.allclose(peaks.value, [2.0])
    t0 = peaks.times[0]
    t0_val = t0.to_value(u.s) if hasattr(t0, "to_value") else float(t0)
    assert t0_val == pytest.approx(3.0)


def test_timeseries_find_peaks_empty_returns_empty_series():
    pytest.importorskip("scipy")
    ts = TimeSeries([0.0, 0.0, 0.0], t0=0, dt=1, unit=u.V, name="flat")
    peaks, props = ts.find_peaks(height=1.0 * u.V)
    assert isinstance(props, dict)
    assert len(peaks) == 0


def test_timeseriescore_tail_crop_append_roundtrip():
    ts = TimeSeriesCore(np.arange(6.0), t0=0, dt=1, name="core")
    tail = ts.tail(2)
    assert isinstance(tail, TimeSeriesCore)
    assert np.allclose(tail.value, [4.0, 5.0])

    cropped = ts.crop(1, 4)
    assert isinstance(cropped, TimeSeriesCore)
    assert np.allclose(cropped.value, [1.0, 2.0, 3.0])

    appended = ts.append(np.array([6.0, 7.0]), inplace=False)
    assert isinstance(appended, TimeSeriesCore)
    assert len(appended) == 8


def test_timeseries_crop_numeric_bounds():
    ts = TimeSeries(np.arange(10.0), t0=0, dt=1, name="demo")
    cropped = ts.crop(2, 5)

    assert isinstance(cropped, TimeSeries)
    assert len(cropped) < len(ts)
    t0 = cropped.t0
    if hasattr(t0, "to_value"):
        unit = getattr(t0, "unit", None)
        t0_val = t0.to_value(u.s) if unit is not None else t0.to_value()
    else:
        t0_val = float(t0)
    assert t0_val == pytest.approx(2.0)


def test_timeseries_to_from_pandas_roundtrip():
    pytest.importorskip("pandas")

    ts = TimeSeries(np.arange(3.0), t0=0, dt=1, name="demo")
    series = ts.to_pandas(index="seconds")

    restored = TimeSeries.from_pandas(series, t0=ts.t0, dt=ts.dt, unit=ts.unit)
    assert isinstance(restored, TimeSeries)
    assert len(restored) == len(ts)
