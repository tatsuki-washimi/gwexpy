import numpy as np
from astropy import units as u
from gwexpy.timeseries import TimeSeries

def test_timeseries_smooth_mixin():
    """Test that SignalAnalysisMixin works for TimeSeries too."""
    data = np.zeros(10)
    data[5] = 3.0
    # [0, 0, 0, 0, 0, 3, 0, ...]
    ts = TimeSeries(data, dt=1.0)

    # Smooth width 3
    # at 4: (0+0+3)/3 = 1
    # at 5: (0+3+0)/3 = 1
    # at 6: (3+0+0)/3 = 1

    smoothed = ts.smooth(3)

    # Verify type preservation
    assert isinstance(smoothed, TimeSeries)
    assert smoothed.dt == ts.dt
    assert smoothed.t0 == ts.t0

    np.testing.assert_allclose(smoothed.value[4:7], 1.0)

def test_timeseries_find_peaks_mixin():
    """Test find_peaks for TimeSeries."""
    data = np.array([0, 1, 0, 2, 0])
    ts = TimeSeries(data, dt=0.5, unit='V')

    peaks, props = ts.find_peaks(threshold=0.5)

    # Should find 1 and 2
    assert len(peaks) == 2
    assert isinstance(peaks, TimeSeries)
    # peaks should be [1, 2]
    np.testing.assert_array_equal(peaks.value, [1, 2])

    # Check that peaks preserved metadata
    assert peaks.unit == 'V'
    # Check times: indices 1 and 3 -> 0.5s and 1.5s
    np.testing.assert_array_equal(peaks.times.value, [0.5, 1.5])

def test_timeseries_find_peaks_conversion():
    """Test unit handling in find_peaks kwargs for TimeSeries."""
    data = np.zeros(100)
    data[50] = 10.0
    # Use explicit units for dt to avoid any ambiguity
    ts = TimeSeries(data, dt=0.1*u.s) # 10Hz sample rate

    # distance=1.0s -> 10 samples
    # We use a smaller threshold to be safe
    peaks, _ = ts.find_peaks(distance=1.0*u.s, threshold=5)
    assert len(peaks) == 1

    # width=0.1s -> 1 sample (a single sample spike is roughly 1 sample wide)
    peaks2, _ = ts.find_peaks(width=0.01*u.s, threshold=5)
    assert len(peaks2) == 1
