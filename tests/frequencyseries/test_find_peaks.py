
import astropy.units as u
import numpy as np

from gwexpy.frequencyseries import FrequencySeries


def test_find_peaks_basic():
    data = np.zeros(100)
    data[10] = 5.0
    data[50] = 10.0
    spec = FrequencySeries(data, df=1.0)

    peaks, props = spec.find_peaks(threshold=3.0)
    assert len(peaks) == 2
    # df=1.0, so index 10 is 10Hz, index 50 is 50Hz
    assert 10.0 in peaks.frequencies.value
    assert 50.0 in peaks.frequencies.value

def test_find_peaks_units():
    data = np.zeros(100)
    data[10] = 5.0
    data[50] = 10.0
    spec = FrequencySeries(data, df=1.0, unit='m')

    # Test threshold with units
    peaks, props = spec.find_peaks(threshold=6.0 * u.m)
    assert len(peaks) == 1
    assert peaks.frequencies[0].value == 50.0

    # Test different unit conversion
    peaks, props = spec.find_peaks(threshold=600.0 * u.cm)
    assert len(peaks) == 1
    assert peaks.frequencies[0].value == 50.0

def test_find_peaks_method():
    data = np.array([1.0, 10.0, 1.0])
    spec = FrequencySeries(data, df=1.0)

    # Test db method
    peaks, props = spec.find_peaks(method='db', threshold=10) # 20*log10(10)=20 > 10
    assert len(peaks) == 1

    # Test power method
    peaks, props = spec.find_peaks(method='power', threshold=50) # 10^2=100 > 50
    assert len(peaks) == 1

def test_find_peaks_kwargs():
    data = np.zeros(100)
    data[10] = 10.0
    data[15] = 10.0 # Close to 10
    spec = FrequencySeries(data, df=1.0)

    # Use distance to exclude close peaks
    peaks, props = spec.find_peaks(threshold=5.0, distance=10)
    assert len(peaks) == 1
