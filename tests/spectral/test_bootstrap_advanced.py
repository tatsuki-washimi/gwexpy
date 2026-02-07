import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import BifrequencyMap
from gwexpy.spectrogram import Spectrogram


def test_bootstrap_advanced_features():
    # Setup data
    # Time: 100 points, dt=1s
    # Freq: 10 points, df=1Hz -> 0..9 Hz
    times = np.arange(100)
    frequencies = np.arange(10)

    # Create somewhat correlated data
    # f=1 and f=2 correlated
    data = np.random.randn(100, 10)
    data[:, 2] = data[:, 1] * 0.9 + 0.1 * np.random.randn(100)  # Correlate f2 with f1

    unit = "V"
    spec = Spectrogram(
        data, times=times, frequencies=frequencies, unit=unit, name="TestSpec"
    )

    # 1. Test Rebinning
    # Rebin by 2Hz. Original df=1Hz. So bin_size=2.
    # Output freq size should be 10 // 2 = 5.
    bs_rebin = spec.bootstrap(rebin_width=2.0, n_boot=20)
    assert bs_rebin.size == 5
    assert bs_rebin.df.value == 2.0

    # 2. Test Block Bootstrap
    # Block size 10 seconds (dt=1s, so 10 samples). Should run without error.
    bs_block = spec.bootstrap(block_size=10.0, n_boot=20)
    assert bs_block.size == 10

    # 3. Test Covariance Map
    bs_res, cov_map = spec.bootstrap(return_map=True, n_boot=50)
    assert isinstance(cov_map, BifrequencyMap)
    assert cov_map.shape == (10, 10)

    # Check covariance structure (covariance map returned)
    # The bootstrap covariance estimates the covariance of the MEAN (or MEDIAN).
    # The correlation of the means should be similar to correlation of data if stationary?
    # Actually, covariance of the estimator.
    # If data columns are correlated, their means are correlated.
    # We just check it's returned and has correct shape/units.
    assert cov_map.unit == u.V**2

    # 4. Test Rebinning + Covariance
    bs_rebin_cov, cov_map_rebin = spec.bootstrap(
        rebin_width=2.0, return_map=True, n_boot=20
    )
    assert bs_rebin_cov.size == 5
    assert cov_map_rebin.shape == (5, 5)


def test_bootstrap_nfft_noverlap():
    """Test nfft/noverlap parameters (sample-based specification)."""
    from astropy import units as u

    from gwexpy.spectrogram import Spectrogram

    np.random.seed(789)

    # Create spectrogram with known dt
    n_time = 50
    frequencies = np.arange(1, 51, 1.0)  # 1 Hz resolution, 50 points
    data = np.random.randn(n_time, 50)

    # dt = 0.5 seconds
    spectrogram = Spectrogram(
        data,
        times=np.arange(n_time) * 0.5,
        frequencies=frequencies * u.Hz,
        dt=0.5 * u.s,
        df=1.0 * u.Hz,
    )

    # Test nfft/noverlap (sample-based)
    # nfft=10 samples * 0.5s/sample = 5 seconds = fftlength
    # noverlap=5 samples * 0.5s/sample = 2.5 seconds = overlap
    result_samples = spectrogram.bootstrap(
        nfft=10, noverlap=5, n_boot=30, method="median"
    )

    # Test equivalent time-based specification
    result_time = spectrogram.bootstrap(
        fftlength=5.0, overlap=2.5, n_boot=30, method="median"
    )

    # Results should be similar (same parameters)
    assert result_samples.size == result_time.size
    assert result_samples.df == result_time.df

    # Test error: cannot mix nfft with overlap
    try:
        spectrogram.bootstrap(nfft=10, overlap=2.5, n_boot=10)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Cannot use overlap (seconds) with nfft (samples)" in str(e)


if __name__ == "__main__":
    test_bootstrap_advanced_features()
    test_bootstrap_nfft_noverlap()
    print("All tests passed!")
