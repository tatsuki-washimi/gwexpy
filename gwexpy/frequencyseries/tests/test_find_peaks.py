
import numpy as np
from gwexpy.frequencyseries import FrequencySeries
import astropy.units as u

def test_find_peaks_basic():
    data = np.zeros(100)
    data[10] = 5.0
    data[50] = 10.0
    spec = FrequencySeries(data, df=1.0)
    
    indices, props = spec.find_peaks(threshold=3.0)
    assert len(indices) == 2
    assert 10 in indices
    assert 50 in indices

def test_find_peaks_units():
    data = np.zeros(100)
    data[10] = 5.0
    data[50] = 10.0
    spec = FrequencySeries(data, df=1.0, unit='m')
    
    # Test threshold with units
    indices, props = spec.find_peaks(threshold=6.0 * u.m)
    assert len(indices) == 1
    assert indices[0] == 50
    
    # Test different unit conversion
    indices, props = spec.find_peaks(threshold=600.0 * u.cm)
    assert len(indices) == 1
    assert indices[0] == 50

def test_find_peaks_method():
    data = np.array([1.0, 10.0, 1.0])
    spec = FrequencySeries(data, df=1.0)
    
    # Test db method
    indices, props = spec.find_peaks(method='db', threshold=10) # 20*log10(10)=20 > 10
    assert len(indices) == 1
    
    # Test power method
    indices, props = spec.find_peaks(method='power', threshold=50) # 10^2=100 > 50
    assert len(indices) == 1

def test_find_peaks_kwargs():
    data = np.zeros(100)
    data[10] = 10.0
    data[15] = 10.0 # Close to 10
    spec = FrequencySeries(data, df=1.0)
    
    # Use distance to exclude close peaks
    indices, props = spec.find_peaks(threshold=5.0, distance=10)
    assert len(indices) == 1
