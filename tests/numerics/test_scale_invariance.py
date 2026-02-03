"""
Verification tests for numerical hardening against scale-related instabilities.
This suite ensures that core algorithms (whitening, ICA, HHT, filtering)
behave correctly (scale-invariantly or linearly) even with very low amplitude
gravitational-wave data (O(1e-21)).
"""

import numpy as np
import pytest

from gwexpy.numerics import SAFE_FLOOR, safe_log_scale
from gwexpy.signal.preprocessing import whiten
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.timeseries.decomposition import ica_fit, ica_transform


@pytest.fixture
def check_scale_invariance():
    """
    Fixture to verify that a function is either scale-invariant or linear.
    """
    def _check(func, data, scale_factor=1e-20, mode="linear"):
        """
        Logic:
        Result 1: y1 = func(data)
        Result 2: y2 = func(data * scale)
        Validation: y2 should be y1 * scale (linear) OR y2 should be y1 (scale-invariant).
        """
        y1 = func(data)
        y2 = func(data * scale_factor)

        # Handle objects that have a .value attribute (e.g. TimeSeries, WhiteningModel.W)
        def _get_val(obj):
            if hasattr(obj, "value"):
                return obj.value
            if hasattr(obj, "W"): # For WhiteningModel
                return obj.W
            return obj

        v1 = _get_val(y1)
        v2 = _get_val(y2)

        if mode == "linear":
            # Expect y2 == y1 * scale
            np.testing.assert_allclose(v2, v1 * scale_factor, rtol=1e-5)
        elif mode == "invariant":
            # Expect y2 == y1
            np.testing.assert_allclose(v2, v1, rtol=1e-5)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return _check

def test_whitening_invariant(check_scale_invariance):
    """whiten(X) should return same whitened data for X and 10^-20 X."""
    # We test the whitened data itself, which should be normalized to unit variance
    # regardless of the input scale, provided the regularization epsilon adapts.
    def get_whitened_data(data):
        Xw, _ = whiten(data, method="pca", return_model=True)
        return Xw

    # Generate multi-feature random data O(1)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 3))

    # This is expected to FAIL if 'whiten' uses a fixed eps=1e-12.
    # For 10^-20 data, variance is 10^-40. If eps=1e-12, the whitening
    # will be dominated by eps, resulting in data scaled to ~10^-14 instead of O(1).
    check_scale_invariance(get_whitened_data, data, scale_factor=1e-20, mode="invariant")

def test_ica_source_recovery():
    """ica_fit should recover sources mixed at 10^-21 amplitude."""
    # Generate 2 distinct sources
    t = np.linspace(0, 1, 1000)
    s1 = np.sin(2 * np.pi * 7 * t)  # 7Hz sine
    rng = np.random.default_rng(42)
    s2 = rng.standard_normal(1000) # Gaussian noise
    S = np.c_[s1, s2]

    # Mix sources with a non-singular matrix
    A = np.array([[1, 0.5], [0.5, 1]])
    X = (S @ A.T) * 1e-21

    # Convert to TimeSeriesMatrix (channels, 1, samples)
    tsm = TimeSeriesMatrix(X.T[:, None, :], t0=0, dt=t[1]-t[0])

    # Fit ICA
    # If it fails due to numerical precision (e.g. tol=1e-4 on 10^-21 data),
    # or if pre-whitening destroys the signal, this will fail.
    res = ica_fit(tsm, n_components=2, random_state=42)

    # Check if sources are recovered (correlation should be high)
    sources_mat = ica_transform(res, tsm)
    recovered = np.asarray(sources_mat.value)

    # Standardize to (samples, components)
    if recovered.ndim == 3:
        recovered = np.transpose(np.squeeze(recovered, axis=1))
    elif recovered.ndim == 2 and recovered.shape[0] < recovered.shape[1]:
        recovered = recovered.T

    # Ensure S and recovered have same number of samples
    if recovered.shape[0] != S.shape[0] and recovered.shape[1] == S.shape[0]:
        recovered = recovered.T

    assert recovered.shape[0] == S.shape[0], f"Sample mismatch: {recovered.shape} vs {S.shape}"

    corr = np.abs(np.corrcoef(S.T, recovered.T))[:2, 2:]
    # Match components to sources based on max correlation
    max_corr = np.max(corr, axis=0)

    assert np.all(max_corr > 0.9), (
        f"ICA failed to recover sources at 1e-21 scale. "
        f"Max correlations: {max_corr}. This often indicates "
        "numerical swamping during pre-whitening or standardization."
    )

def test_hht_vmin():
    """hht_spectrogram should not contain nan or empty plots for small data."""
    t = np.linspace(0, 0.1, 1000)
    # Signal at 10^-21 amplitude
    data = 1e-21 * np.sin(2 * np.pi * 100 * t)
    ts = TimeSeries(data, t0=0, dt=t[1]-t[0])

    # Test TimeSeries.hht method
    try:
        res = ts.hht(output="spectrogram")
        # Ensure no NaNs were produced by internal divisions or logs
        assert not np.any(np.isnan(res.value)), "HHT spectrogram contains NaNs for 10^-21 data"
        # Ensure signal was not zeroed out by a floor that was too high
        assert np.max(np.abs(res.value)) > 0, "HHT spectrogram is all zeros for 10^-21 data"
    except Exception as e:
        pytest.fail(f"HHT computation crashed for 10^-21 data: {e}")

def test_safe_log():
    """Verify safe_log_scale allows inputs down to 10^-50 without clipping to flat -200dB."""
    # Values covering a wide range below the common 1e-20 floor
    vals = np.array([1e-25, 1e-30, 1e-45])

    # If safe_log_scale uses a 1e-20 floor, these will all be approximately the same.
    log_vals = safe_log_scale(vals)

    # Check for distinct values (avoiding clipping)
    # With 1e-20 floor, 1e-25 and 1e-30 will be within ~1e-10 of each other.
    assert not np.allclose(log_vals[0], log_vals[1], atol=1e-5), (
        f"safe_log_scale clipped 1e-25 and 1e-30 to nearly same value: {log_vals[0]}"
    )
    assert not np.allclose(log_vals[1], log_vals[2], atol=1e-5), (
        f"safe_log_scale clipped 1e-30 and 1e-45 to nearly same value: {log_vals[1]}"
    )
    # Should handle values down to SAFE_FLOOR (1e-50)
    assert np.isfinite(log_vals[2]), "safe_log_scale produced non-finite value for 1e-45"

def test_filter_stability():
    """Verify high-pass/band-pass filters don't explode on 10^-21 data."""
    t = np.linspace(0, 1, 2048)
    rng = np.random.default_rng(42)
    data = 1e-21 * rng.standard_normal(2048)
    ts = TimeSeries(data, t0=0, dt=t[1]-t[0])

    # High-pass filter at 20Hz
    # Numerical instability in filters often shows as NaNs or extreme values
    try:
        filtered = ts.highpass(20)
        assert not np.any(np.isnan(filtered.value)), "Highpass filter produced NaNs for 10^-21 data"
        assert np.all(np.isfinite(filtered.value)), "Highpass filter produced non-finite values"

        # Verify the signal scale is preserved (not zeroed or exploded)
        std_val = np.std(filtered.value)
        assert 1e-23 < std_val < 1e-20, (
            f"Filter output scale is suspect: std={std_val:.2e}. "
            "Expected O(1e-21)."
        )
    except Exception as e:
        pytest.fail(f"Filtering operation crashed for 10^-21 data: {e}")
