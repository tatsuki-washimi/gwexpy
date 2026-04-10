import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwexpy.fitting import fit_bootstrap_spectrum

pytest.importorskip("iminuit")


def test_fit_bootstrap_spectrum_alignment_regression():
    # Synthetic data
    sample_rate = 1024
    duration = 4
    t = np.linspace(0, duration, duration * sample_rate, endpoint=False)
    # Sine wave + small noise
    data = 1.0 * np.sin(2 * np.pi * 100 * t) + 1e-3 * np.random.randn(len(t))
    series = TimeSeries(data, sample_rate=sample_rate, name="test")

    # Run fit_bootstrap_spectrum
    fftlength = 1.0
    overlap = 0.5
    fmin, fmax = 80, 120

    def dummy_model(f, a, b):
        return a * f + b

    # This should not raise frequency alignment errors
    results = fit_bootstrap_spectrum(
        series,
        model_fn=dummy_model,
        fmin=fmin,
        fmax=fmax,
        fftlength=fftlength,
        overlap=overlap,
    )

    # Basic verification
    psd = results.psd
    cov = results.cov

    # Assert same number of frequencies
    assert len(psd) == len(cov.frequency1)

    # Verify frequencies match
    np.testing.assert_allclose(psd.frequencies.value, cov.frequency1.value)
    np.testing.assert_allclose(psd.frequencies.value, cov.frequency2.value)

    # Verify covariance matrix matches PSD size
    assert cov.shape == (len(psd), len(psd))


def test_fit_bootstrap_spectrum_stride_removal():
    # Synthetic data
    data = np.random.randn(2048)
    series = TimeSeries(data, sample_rate=512)

    def dummy_model(f, a):
        return a * f

    # Should work without stride
    results = fit_bootstrap_spectrum(series, model_fn=dummy_model, fftlength=1, overlap=0.5, plot=False)
    assert hasattr(results, "psd")

    # Should raise TypeError if stride is passed (as it was causing internal errors)
    # Note: If fit_bootstrap_spectrum has **kwargs, it won't raise TypeError automatically.
    # We should decide if we want to explicitly prohibit it.
    # For now, let's just confirm it doesn't fail with an internal TypeError if it's there but ignored,
    # OR we fix the code to raise it.
    with pytest.raises(TypeError):
        fit_bootstrap_spectrum(series, model_fn=dummy_model, fftlength=1, overlap=0.5, stride=1, plot=False)
