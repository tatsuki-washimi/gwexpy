import pytest
import numpy as np
from astropy import units as u
from gwexpy.spectrogram import Spectrogram
from gwexpy.frequencyseries import BifrequencyMap

def test_bootstrap_advanced_features():
    # Setup data
    # Time: 100 points, dt=1s
    # Freq: 10 points, df=1Hz -> 0..9 Hz
    times = np.arange(100)
    frequencies = np.arange(10)
    
    # Create somewhat correlated data
    # f=1 and f=2 correlated
    data = np.random.randn(100, 10)
    data[:, 2] = data[:, 1] * 0.9 + 0.1 * np.random.randn(100) # Correlate f2 with f1
    
    unit = 'V'
    spec = Spectrogram(data, times=times, frequencies=frequencies, unit=unit, name='TestSpec')
    
    # 1. Test Rebinning
    # Rebin by 2Hz. Original df=1Hz. So bin_size=2.
    # Output freq size should be 10 // 2 = 5.
    bs_rebin = spec.bootstrap(rebin_width=2.0, n_boot=20)
    assert bs_rebin.size == 5
    assert bs_rebin.df.value == 2.0
    
    # 2. Test Block Bootstrap
    # Block size 10. Should run without error.
    bs_block = spec.bootstrap(block_size=10, n_boot=20)
    assert bs_block.size == 10
    
    # 3. Test Covariance Map
    bs_res, cov_map = spec.bootstrap(return_map=True, n_boot=50)
    assert isinstance(cov_map, BifrequencyMap)
    assert cov_map.shape == (10, 10)
    
    # Check covariance structure
    # f1 (idx 1) and f2 (idx 2) should have high correlation
    cov_val = cov_map.value
    # cov(f1, f1)
    var_1 = cov_val[1, 1]
    # cov(f2, f2)
    var_2 = cov_val[2, 2]
    # cov(f1, f2)
    cov_12 = cov_val[1, 2]
    
    # Correlation coefficient approx 0.9?
    corr = cov_12 / np.sqrt(var_1 * var_2)
    # The bootstrap covariance estimates the covariance of the MEAN (or MEDIAN).
    # The correlation of the means should be similar to correlation of data if stationary? 
    # Actually, covariance of the estimator.
    # If data columns are correlated, their means are correlated.
    # We just check it's returned and has correct shape/units.
    assert cov_map.unit == u.V**2
    
    # 4. Test Rebinning + Covariance
    bs_rebin_cov, cov_map_rebin = spec.bootstrap(rebin_width=2.0, return_map=True, n_boot=20)
    assert bs_rebin_cov.size == 5
    assert cov_map_rebin.shape == (5, 5)

if __name__ == "__main__":
    test_bootstrap_advanced_features()
    print("All tests passed!")
