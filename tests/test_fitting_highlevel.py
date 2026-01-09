"""
Tests for fit_bootstrap_spectrum high-level pipeline.
"""
import numpy as np
import pytest

# Skip if optional dependencies are not available
pytest.importorskip("iminuit")


def test_fit_bootstrap_spectrum_basic():
    """Test basic usage with synthetic spectrogram."""
    from gwexpy.spectrogram import Spectrogram
    from gwexpy.fitting import fit_bootstrap_spectrum
    from astropy import units as u
    
    np.random.seed(42)
    
    # Create synthetic spectrogram data
    # Power law noise: PSD = A * f^alpha
    n_time = 50
    frequencies = np.linspace(1, 100, 30)
    
    A_true, alpha_true = 10.0, -1.5
    psd_true = A_true * frequencies**alpha_true
    
    # Add noise to each time slice
    data = np.array([psd_true * (1 + 0.1 * np.random.normal(size=len(frequencies)))
                     for _ in range(n_time)])
    
    # Create spectrogram
    spectrogram = Spectrogram(
        data,
        times=np.arange(n_time),
        frequencies=frequencies * u.Hz,
        dt=1 * u.s,
        df=(frequencies[1] - frequencies[0]) * u.Hz,
        unit=u.Unit("1/Hz"),
    )
    
    # Define model
    def power_law(f, A, alpha):
        return A * f**alpha
    
    # Run pipeline without MCMC, without plotting
    result = fit_bootstrap_spectrum(
        spectrogram,
        model_fn=power_law,
        freq_range=(5, 80),
        n_boot=100,  # Small for testing
        initial_params={"A": 5, "alpha": -1},
        run_mcmc=False,
        plot=False,
    )
    
    assert result.minuit.valid
    assert hasattr(result, 'psd')
    assert hasattr(result, 'cov')
    assert result.cov is not None
    
    # Check recovered parameters are reasonable
    assert 1 < result.params["A"] < 50
    assert -3 < result.params["alpha"] < 0


def test_fit_bootstrap_spectrum_with_mcmc():
    """Test pipeline with MCMC enabled."""
    pytest.importorskip("emcee")
    pytest.importorskip("corner")
    
    from gwexpy.spectrogram import Spectrogram
    from gwexpy.fitting import fit_bootstrap_spectrum
    from astropy import units as u
    
    np.random.seed(123)
    
    # Smaller dataset for faster test
    n_time = 30
    frequencies = np.linspace(5, 50, 15)
    
    A_true, alpha_true = 5.0, -1.0
    psd_true = A_true * frequencies**alpha_true
    
    data = np.array([psd_true * (1 + 0.05 * np.random.normal(size=len(frequencies)))
                     for _ in range(n_time)])
    
    spectrogram = Spectrogram(
        data,
        times=np.arange(n_time),
        frequencies=frequencies * u.Hz,
        dt=1 * u.s,
        df=(frequencies[1] - frequencies[0]) * u.Hz,
    )
    
    def power_law(f, A, alpha):
        return A * f**alpha
    
    result = fit_bootstrap_spectrum(
        spectrogram,
        model_fn=power_law,
        n_boot=50,
        initial_params={"A": 3, "alpha": -0.5},
        run_mcmc=True,
        mcmc_walkers=16,
        mcmc_steps=100,
        mcmc_burn_in=20,
        plot=False,
        progress=False,
    )
    
    assert result.minuit.valid
    assert result.samples is not None
    
    # Check parameter intervals are available
    intervals = result.parameter_intervals
    assert "A" in intervals
    assert "alpha" in intervals


def test_fit_bootstrap_spectrum_with_rebin():
    """Test pipeline with frequency rebinning."""
    from gwexpy.spectrogram import Spectrogram
    from gwexpy.fitting import fit_bootstrap_spectrum
    from astropy import units as u
    
    np.random.seed(456)
    
    n_time = 40
    frequencies = np.arange(1, 101, 0.5)  # 0.5 Hz resolution, 200 points
    
    A_true, alpha_true = 8.0, -1.2
    psd_true = A_true * frequencies**alpha_true
    
    data = np.array([psd_true * (1 + 0.08 * np.random.normal(size=len(frequencies)))
                     for _ in range(n_time)])
    
    spectrogram = Spectrogram(
        data,
        times=np.arange(n_time),
        frequencies=frequencies * u.Hz,
        dt=1 * u.s,
        df=0.5 * u.Hz,
    )
    
    def power_law(f, A, alpha):
        return A * f**alpha
    
    result = fit_bootstrap_spectrum(
        spectrogram,
        model_fn=power_law,
        rebin_width=2.0,  # Rebin to 2 Hz
        n_boot=50,
        initial_params={"A": 5, "alpha": -1},
        run_mcmc=False,
        plot=False,
    )
    
    assert result.minuit.valid
    # After rebinning, should have fewer frequency points
    assert len(result.x) < 200


def test_fit_bootstrap_spectrum_stores_metadata():
    """Test that pipeline stores relevant metadata in result."""
    from gwexpy.spectrogram import Spectrogram
    from gwexpy.fitting import fit_bootstrap_spectrum
    from astropy import units as u
    
    np.random.seed(789)
    
    n_time = 20
    frequencies = np.linspace(10, 50, 10)
    
    data = np.random.rand(n_time, len(frequencies)) * 10
    
    spectrogram = Spectrogram(
        data,
        times=np.arange(n_time),
        frequencies=frequencies * u.Hz,
        dt=1 * u.s,
        df=(frequencies[1] - frequencies[0]) * u.Hz,
    )
    
    def linear(f, a, b):
        return a * f + b
    
    result = fit_bootstrap_spectrum(
        spectrogram,
        model_fn=linear,
        method="mean",
        n_boot=30,
        initial_params={"a": 0.1, "b": 5},
        run_mcmc=False,
        plot=False,
    )
    
    # Check metadata is stored
    assert hasattr(result, 'psd')
    assert hasattr(result, 'cov')
    assert hasattr(result, 'bootstrap_method')
    assert result.bootstrap_method == "mean"
    
    # Check cov is BifrequencyMap
    assert hasattr(result.cov, 'inverse')


def test_fit_bootstrap_spectrum_bounds():
    """Test pipeline with parameter bounds."""
    from gwexpy.spectrogram import Spectrogram
    from gwexpy.fitting import fit_bootstrap_spectrum
    from astropy import units as u
    
    np.random.seed(101)
    
    n_time = 25
    frequencies = np.linspace(5, 30, 12)
    
    A_true, alpha_true = 10.0, -1.5
    psd_true = A_true * frequencies**alpha_true
    
    data = np.array([psd_true * (1 + 0.05 * np.random.normal(size=len(frequencies)))
                     for _ in range(n_time)])
    
    spectrogram = Spectrogram(
        data,
        times=np.arange(n_time),
        frequencies=frequencies * u.Hz,
        dt=1 * u.s,
        df=(frequencies[1] - frequencies[0]) * u.Hz,
    )
    
    def power_law(f, A, alpha):
        return A * f**alpha
    
    result = fit_bootstrap_spectrum(
        spectrogram,
        model_fn=power_law,
        n_boot=30,
        initial_params={"A": 5, "alpha": -1},
        bounds={"A": (0, 100), "alpha": (-5, 0)},
        run_mcmc=False,
        plot=False,
    )
    
    assert result.minuit.valid
    # Parameters should be within bounds
    assert 0 <= result.params["A"] <= 100
    assert -5 <= result.params["alpha"] <= 0
