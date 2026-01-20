
import numpy as np
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries


def test_smooth_amplitude():
    # Simple step
    data = np.zeros(10)
    data[5:] = 1.0
    fs = FrequencySeries(data, df=1.0)

    # Width 3, amplitude smoothing
    # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # smooth at 4: (0+0+1)/3 = 0.33...
    # smooth at 5: (0+1+1)/3 = 0.66...
    smoothed = fs.smooth(3, method='amplitude', ignore_nan=True)

    assert isinstance(smoothed, FrequencySeries)
    assert smoothed.unit == fs.unit

    # Default is pandas rolling which handles edges with min_periods=1
    # Check middle values
    expected_mid = np.convolve(data, np.ones(3)/3, mode='same')
    # Pandas rolling center=True behaves slightly differently at edges depending on version/impl,
    # but let's check core region
    np.testing.assert_allclose(smoothed.value[2:-2], expected_mid[2:-2])

def test_smooth_power():
    data = np.array([1.0, 2.0, 1.0])
    fs = FrequencySeries(data, df=1.0)
    # Power: 1, 4, 1
    # Smooth width 3: (1+4+1)/3 = 2
    smoothed = fs.smooth(3, method='power')

    assert smoothed.unit == fs.unit**2
    assert np.isclose(smoothed.value[1], 2.0)

def test_smooth_db():
    # 1.0 -> 0dB
    # 10.0 -> 20dB
    data = np.array([1.0, 10.0, 1.0])
    fs = FrequencySeries(data, df=1.0)

    # Smooth dB: 0, 20, 0 -> avg = 6.666...
    smoothed = fs.smooth(3, method='db')

    assert smoothed.unit == u.Unit('dB')
    np.testing.assert_allclose(smoothed.value[1], 20.0/3.0)

def test_smooth_complex():
    data = np.array([1j, 1+1j, 1])
    fs = FrequencySeries(data, df=1.0)

    # Real: 0, 1, 1
    # Imag: 1, 1, 0

    smoothed = fs.smooth(3, method='complex')
    assert smoothed.dtype.kind == 'c'

    # Middle point
    expected_re = (0+1+1)/3
    expected_im = (1+1+0)/3

    np.testing.assert_allclose(smoothed.value[1].real, expected_re)
    np.testing.assert_allclose(smoothed.value[1].imag, expected_im)

def test_smooth_nan_handling():
    data = np.array([1.0, np.nan, 3.0])
    fs = FrequencySeries(data, df=1.0)

    # ignore_nan=True (default)
    # At index 1: (1 + 3) / 2 = 2.0 (skips NaN)
    s1 = fs.smooth(3, ignore_nan=True)
    assert np.isclose(s1.value[1], 2.0)

    # ignore_nan=False
    # Should propagate NaN (or result in NaN depending on backend)
    s2 = fs.smooth(3, ignore_nan=False)
    assert np.isnan(s2.value[1])
