import numpy as np
from astropy import units as u


def test_as_series_time_index_identity_sets_t0_and_converts_values():
    from gwexpy.timeseries import TimeSeries
    from gwexpy.types import as_series

    ts = TimeSeries(np.arange(10), dt=1.0, t0=1419724818)
    times = ts.times

    ts_from_times = as_series(times, unit="h")

    assert ts_from_times.t0 == ts.t0
    assert ts_from_times.dt == ts.dt
    assert ts_from_times.unit == u.Unit("h")

    expected = u.Quantity(np.asarray(times), times.unit).to("h").value
    assert np.allclose(ts_from_times.value, expected)


def test_as_series_frequency_index_identity_sets_axis_and_converts_values():
    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.types import as_series

    fs = FrequencySeries(np.arange(5), f0=0, df=1, unit="V")
    freqs = fs.frequencies

    fs_from_freqs = as_series(freqs, unit="mHz")

    assert fs_from_freqs.unit == u.Unit("mHz")
    expected = u.Quantity(np.asarray(freqs), freqs.unit).to("mHz").value
    assert np.allclose(fs_from_freqs.value, expected)


def test_as_series_angular_frequency_quantity_uses_hz_axis():
    from gwexpy.types import as_series

    ang = (2 * np.pi * np.arange(5)) * (u.rad / u.s)
    fs = as_series(ang, unit="Hz")

    assert fs.unit == u.Unit("Hz")
    assert np.allclose(fs.value, np.arange(5))


def test_as_series_datetime_array():
    from gwexpy.types import as_series
    from datetime import datetime
    
    times = [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 0, 0, 1)]
    ts = as_series(times)
    
    assert ts.t0.value == 1261872018.0 # GPS
    assert ts.dt.value == 1.0
    assert ts.unit == u.s
    assert np.allclose(ts.value, [1261872018.0, 1261872019.0])
