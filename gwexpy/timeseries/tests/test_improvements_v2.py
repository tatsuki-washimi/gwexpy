import numpy as np
import pytest
from astropy import units as u
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.frequencyseries import FrequencySeriesMatrix

def test_tsm_impute_vectorized():
    # Create matrix with same NaNs across channels
    data = np.random.randn(2, 1, 100)
    data[:, :, 50:60] = np.nan
    tsm = TimeSeriesMatrix(data, dt=0.1)
    
    # Impute
    tsm_imp = tsm.impute(method="linear")
    
    # Check that NaNs are filled
    assert not np.any(np.isnan(tsm_imp.value))
    # Check that it's still a TimeSeriesMatrix
    assert isinstance(tsm_imp, TimeSeriesMatrix)

def test_tsm_impute_mixed_nans():
    # Create matrix with different NaNs across channels
    data = np.random.randn(2, 1, 100)
    data[0, 0, 10:20] = np.nan
    data[1, 0, 50:60] = np.nan
    tsm = TimeSeriesMatrix(data, dt=0.1)
    
    # Impute
    tsm_imp = tsm.impute(method="linear")
    
    assert not np.any(np.isnan(tsm_imp.value))

def test_laplace_chunking():
    # Large number of frequencies to trigger chunking
    ts = TimeSeries(np.random.randn(1000), dt=1.0)
    # n=1000, n_freqs=20000 -> 20M elements > 10M limit
    freqs = np.linspace(0, 10, 20000)
    
    # Should work without memory error
    fs = ts.laplace(sigma=0.1, frequencies=freqs)
    assert len(fs) == 20000

def test_series_matrix_ufunc_vectorized():
    data1 = np.ones((2, 2, 10))
    data2 = np.ones((2, 2, 10)) * 2
    tsm1 = TimeSeriesMatrix(data1, dt=1.0)
    tsm2 = TimeSeriesMatrix(data2, dt=1.0)
    
    # Test addition
    tsm3 = tsm1 + tsm2
    assert np.all(tsm3.value == 3.0)
    assert isinstance(tsm3, TimeSeriesMatrix)

def test_tsm_resample():
    data = np.random.randn(2, 1, 1000)
    tsm = TimeSeriesMatrix(data, dt=0.01) # 100 Hz
    # Resample to 50 Hz
    tsm_res = tsm.resample(50 * u.Hz)
    assert tsm_res.shape[-1] == 500
    assert tsm_res.dt == 0.02 * u.s

def test_fsm_smooth():
    data = np.ones((2, 2, 100)) + 1j * np.ones((2, 2, 100))
    fsm = FrequencySeriesMatrix(data, df=1.0)
    fsm_s = fsm.smooth(10, method='complex')
    assert fsm_s.shape == fsm.shape
    assert np.allclose(fsm_s.value, data) # Smoothing ones gives ones
    
    # Smooth amplitude
    fsm_a = fsm.smooth(10, method='amplitude')
    assert np.all(fsm_a.units == fsm.units)

def test_is_regular():
    ts = TimeSeries([1, 2, 3], dt=1.0)
    assert ts.is_regular == True
    
    # Irregular
    ts_irr = TimeSeries([1, 2, 4], times=[0, 1, 3])
    assert ts_irr.is_regular == False
