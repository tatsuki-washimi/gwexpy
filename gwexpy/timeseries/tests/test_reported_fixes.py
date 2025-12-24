
import pytest
import numpy as np
from astropy import units as u
from datetime import datetime
from gwexpy.timeseries import TimeSeries
from gwexpy.timeseries.preprocess import align_timeseries_collection
from gwexpy.types.series_creator import as_series
from gwexpy.time import to_gps

def test_align_mixed_dimensionless_unit():
    # Test case: one series is dimensionless, another is in seconds
    ts1 = TimeSeries([1, 2, 3], dt=0.5, t0=0) # dimensionless
    ts2 = TimeSeries([10, 20, 30], dt=0.5*u.s, t0=0*u.s) # seconds

    # This should now work without UnitConversionError
    values, times, meta = align_timeseries_collection([ts1, ts2])

    assert meta['dt'].unit == u.s
    assert times.unit == u.s
    assert values.shape == (3, 2)
    assert np.all(values[:, 0] == [1, 2, 3])
    assert np.all(values[:, 1] == [10, 20, 30])

def test_align_converts_to_gps_seconds():
    # Test case: inputs are in minutes, output aligns to GPS seconds
    min_unit = u.min
    ts1 = TimeSeries([1, 2, 3], dt=1*min_unit, t0=0*min_unit)
    ts2 = TimeSeries([10, 20, 30], dt=1*min_unit, t0=1*min_unit)

    values, times, meta = align_timeseries_collection([ts1, ts2])

    # GPS seconds should be used for t0/time axis
    assert meta['dt'].unit == u.s
    assert times.unit == u.s
    assert meta['t0'].unit == u.s

def test_crop_with_array_to_gps():
    # Use valid datetime objects away from epoch to avoid edge cases
    t_start = [datetime(2020, 1, 1, 0, 0, 10)]
    t_end = [datetime(2020, 1, 1, 0, 0, 20)]

    start_gps = to_gps(t_start[0])
    ts = TimeSeries(np.arange(100), dt=1, t0=start_gps - 5)
    tsc = ts.crop(start=t_start, end=t_end)

    assert tsc.t0.value == pytest.approx(start_gps)
    assert len(tsc) == 10

def test_as_series_with_datetime():
    times = [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 0, 0, 1)]

    # Should work now
    ts = as_series(times)

    assert isinstance(ts, TimeSeries)
    assert ts.t0.value == pytest.approx(to_gps(times[0]))
    assert ts.dt.value == pytest.approx(1.0)
    assert ts.unit == u.s
    # Values represent the axis values (GPS seconds here)
    assert ts.value == pytest.approx([to_gps(times[0]), to_gps(times[1])])
