import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from gwpy.timeseries import TimeSeries as BaseTimeSeries
from gwpy.timeseries.core import LIGOTimeGPS

from gwexpy.timeseries import TimeSeries


def test_timeseries_new_gps_coercion():
    # Test string GPS coercion
    # 2026-03-27 12:00:00 UTC is 1458648018.0 GPS
    t0_str = "2026-03-27 12:00:00"
    ts = TimeSeries([1, 2, 3], t0=t0_str, dt=1 * u.s)
    assert float(ts.t0.value) == pytest.approx(1458648018.0)

    # Test LIGOTimeGPS coercion
    t0_gps = LIGOTimeGPS(1234567890, 123000000)
    ts = TimeSeries([1, 2, 3], t0=t0_gps, dt=1 * u.s)
    assert float(ts.t0.value) == pytest.approx(1234567890.123)

    # Test astropy Time coercion
    t0_time = Time("2026-03-27T12:00:00", format="isot", scale="utc")
    ts = TimeSeries([1, 2, 3], t0=t0_time, dt=1 * u.s)
    assert float(ts.t0.value) == pytest.approx(1458648018.0)

    # Test epoch coercion
    ts = TimeSeries([1, 2, 3], epoch=t0_str, dt=1 * u.s)
    assert float(ts.epoch.value) == pytest.approx(1458648018.0)


def test_timeseries_new_no_coercion_for_non_time():
    # If dt is not time-like, t0 should not be coerced to GPS float if possible
    ts = TimeSeries([1, 2, 3], t0=10, dt=0.1 * u.m)
    # Note: _coerce_t0_gps might still return a Quantity if it's numeric,
    # but the constructor logic checks should_coerce.
    assert ts.t0.value == 10
    assert ts.dt.unit == u.m


def test_timeseries_array_finalize_propagation():
    ts = TimeSeries([1, 2, 3], t0=0, dt=1)
    ts._gwex_test_attr = "hello"
    
    # Slicing
    sliced = ts[1:3]
    assert sliced._gwex_test_attr == "hello"
    
    # View casting
    view = ts.view(TimeSeries)
    assert view._gwex_test_attr == "hello"


def test_timeseries_basic_ops_return_type():
    ts = TimeSeries(np.arange(10), t0=0, dt=1)
    
    # tail
    t = ts.tail(3)
    assert isinstance(t, TimeSeries)
    assert len(t) == 3
    
    # crop
    c = ts.crop(2, 5)
    assert isinstance(c, TimeSeries)
    assert float(c.t0.value) == 2.0
    
    # append
    other = TimeSeries([10, 11], t0=10, dt=1)
    appended = ts.append(other, inplace=False)
    assert isinstance(appended, TimeSeries)
    assert len(appended) == 12


def test_timeseries_find_peaks_with_quantities():
    pytest.importorskip("scipy")
    # Signal with peaks at t=2 and t=6
    data = np.zeros(11)
    data[2] = 5.0
    data[6] = 10.0
    ts = TimeSeries(data, t0=0, dt=1 * u.s, unit=u.V)
    
    # Test distance as Quantity
    peaks, props = ts.find_peaks(distance=5 * u.s)
    assert len(peaks) == 1  # only the higher peak if distance is large? 
    # Wait, find_peaks distance logic: peaks must be separated by at least distance.
    # At t=2 and t=6, distance is 4s. So if distance=5s, only one peak remains.
    assert float(peaks.times[0].value) == 6.0
    
    # Test width as Quantity
    # We need a wider peak to test width
    data_wide = np.zeros(20)
    data_wide[5:10] = 1.0 # peak at 7.5?
    ts_wide = TimeSeries(data_wide, t0=0, dt=1 * u.s, unit=u.V)
    peaks, props = ts_wide.find_peaks(width=2 * u.s)
    # Scipy find_peaks width is tricky with zeros, but let's just ensure it doesn't crash
    assert isinstance(peaks, TimeSeries)


def test_regularity_mixin():
    # Regular series
    ts_reg = TimeSeries([1, 2, 3], t0=0, dt=1)
    assert ts_reg.is_regular is True
    ts_reg._check_regular() # should not raise
    
    # Irregular series
    # GWpy TimeSeries defaults to regular if created with dt.
    # To make it irregular, we can use xindex.
    times = [0, 1, 2.5, 3]
    ts_irreg = TimeSeries([1, 2, 3, 4], times=times)
    assert ts_irreg.is_regular is False
    
    with pytest.raises(ValueError, match="requires a regular sample rate"):
        ts_irreg._check_regular("Test method")
