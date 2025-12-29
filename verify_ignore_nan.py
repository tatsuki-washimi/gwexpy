
import numpy as np
import pytest
from astropy import units as u
from gwexpy.types import Series, Array, Array2D
from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectrogram import Spectrogram
from gwexpy.spectral import bootstrap_spectrogram

def test_abstract_stats():
    # Test Array
    data = np.array([1.0, 2.0, np.nan, 4.0])
    arr = Array(data)
    
    assert arr.mean(ignore_nan=True) == np.nanmean(data)
    assert np.isnan(arr.mean(ignore_nan=False))
    
    assert arr.std(ignore_nan=True) == np.nanstd(data)
    assert arr.median(ignore_nan=True) == 2.0
    
    # Test rms
    assert arr.rms(ignore_nan=True) == np.sqrt(np.nanmean(data**2))
    assert np.isnan(arr.rms(ignore_nan=False))

def test_timeseries_rolling():
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    ts = TimeSeries(data, dt=1.0)
    
    # Rolling mean with window=3
    # centered=False
    # idx 2: [1, 2, nan] -> mean should be 1.5 if ignore_nan=True, else nan
    rm_ignore = ts.rolling_mean(3, ignore_nan=True)
    rm_prop = ts.rolling_mean(3, ignore_nan=False)
    
    assert rm_ignore[2] == 1.5
    assert np.isnan(rm_prop[2])

def test_timeseries_resample():
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    ts = TimeSeries(data, dt=1.0)
    
    # Resample to 2s bins
    # bin 0: [1, 2] -> 1.5
    # bin 1: [nan, 4] -> 4.0 if ignore_nan=True
    rs_ignore = ts.resample('2s', ignore_nan=True)
    rs_prop = ts.resample('2s', ignore_nan=False)
    
    assert rs_ignore[1] == 4.0
    assert np.isnan(rs_prop[1])

def test_frequencyseries_smooth():
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    fs = FrequencySeries(data, df=1.0)
    
    # Smooth with width=3
    # idx 1: [1, nan, 3] -> mean should be 2.0 if ignore_nan=True
    sm_ignore = fs.smooth(3, ignore_nan=True)
    sm_prop = fs.smooth(3, ignore_nan=False)
    
    assert sm_ignore[1] == 2.0
    assert np.isnan(sm_prop[1])

def test_bootstrap_spectrogram():
    data = np.array([
        [1.0, 2.0],
        [np.nan, 4.0],
        [5.0, 6.0]
    ])
    spec = Spectrogram(data, dt=1.0, df=1.0)
    
    # Bootstrap with n_boot=10 to be fast
    # Result should have no NaNs if ignore_nan=True
    bs_ignore = bootstrap_spectrogram(spec, n_boot=10, ignore_nan=True)
    assert not np.isnan(bs_ignore.value).any()
    
    # If ignore_nan=False, it might have NaNs (depending on sampling)
    # But for simplicity, we just check that ignore_nan works.

if __name__ == "__main__":
    test_abstract_stats()
    test_timeseries_rolling()
    test_timeseries_resample()
    test_frequencyseries_smooth()
    test_bootstrap_spectrogram()
    print("All verification tests passed!")
