import numpy as np
import pytest
from gwexpy.spectrogram import SpectrogramMatrix, SpectrogramList, SpectrogramDict

def test_sgm_conversion():
    """Test SpectrogramMatrix conversion methods (to_list, to_dict)."""
    times = np.linspace(0, 10, 11)
    freqs = np.linspace(0, 50, 6)
    data = np.random.rand(2, 11, 6)
    
    sgm = SpectrogramMatrix(data, times=times, frequencies=freqs, rows=['ch1', 'ch2'])
    
    # 3D: (Batch, time, freq)
    
    # to_list
    sgl = sgm.to_list()
    assert isinstance(sgl, SpectrogramList)
    assert len(sgl) == 2
    assert sgl[0].shape == (11, 6)
    assert np.allclose(sgl[0].frequencies.value, freqs)
    
    # to_dict
    sgd = sgm.to_dict()
    assert isinstance(sgd, SpectrogramDict)
    assert 'ch1' in sgd
    assert 'ch2' in sgd
    
    # 4D: (Row, Col, Time, Freq)
    data4d = np.random.rand(2, 2, 11, 6)
    sgm4d = SpectrogramMatrix(data4d, times=times, frequencies=freqs, rows=['R1', 'R2'], cols=['C1', 'C2'])
    
    sgd4d = sgm4d.to_dict()
    # If 4D and multiple columns, keys are tuples (row, col)
    assert ('R1', 'C1') in sgd4d
    assert len(sgd4d) == 4
    
    # to_series_2Dlist 
    list2d = sgm4d.to_series_2Dlist()
    assert len(list2d) == 2
    assert len(list2d[0]) == 2
    assert list2d[0][0].shape == (11, 6)
    assert np.allclose(list2d[0][0].frequencies.value, freqs)

def test_sgm_analysis():
    """Test SpectrogramMatrix analysis methods (crop, interpolate)."""
    times = np.linspace(0, 10, 11) # 11 points (0,1,..10)
    freqs = np.linspace(0, 50, 6)  # 6 points
    data = np.random.rand(2, 11, 6)
    
    sgm = SpectrogramMatrix(data, times=times, frequencies=freqs, rows=['ch1', 'ch2'])
    
    # Crop (Time axis)
    cropped = sgm.crop(start=2, end=8) # time 2..8
    assert cropped.times[0] >= 2
    assert cropped.times[-1] < 8
    # Should slice axis 1 (Time). 
    assert cropped.shape[1] < 11
    assert cropped.shape[2] == 6 # Freq should be untouched
    
    # Interpolate (Time axis)
    new_times = np.linspace(0, 10, 21)
    interped = sgm.interpolate(new_times, kind='linear')
    assert interped.shape[1] == 21
    assert interped.shape[2] == 6

if __name__ == "__main__":
    test_sgm_conversion()
    test_sgm_analysis()
