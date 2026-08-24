"""Tests for the public Allen et al. median-bias utility."""

import numpy as np
import pytest

from gwexpy.signal.spectral import median_bias

# Independent oracle values from the Allen et al. FINDCHIRP Appendix B
# correction for the median of exponentially distributed periodograms.
ALLEN_MEDIAN_BIAS = {
    1: 1.0,
    2: 1.0,
    3: 0.8333333333333333,
    4: 0.8333333333333333,
    5: 0.7833333333333333,
    6: 0.7833333333333333,
    7: 0.7595238095238095,
    8: 0.7595238095238095,
    9: 0.7456349206349207,
    10: 0.7456349206349207,
    50: 0.7032471605759183,
}


@pytest.mark.parametrize("n, expected", ALLEN_MEDIAN_BIAS.items())
def test_median_bias_matches_allen_findchirp_oracle(n, expected):
    """Return the published finite-sample correction for small and medium N."""
    assert median_bias(n) == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize("n", [np.int8(3), np.int64(50), np.uint8(3), np.uint64(50)])
def test_median_bias_accepts_positive_numpy_integer_input(n):
    """Accept signed and unsigned NumPy integer segment counts."""
    assert median_bias(n) == pytest.approx(
        ALLEN_MEDIAN_BIAS[int(n)], rel=1e-15, abs=1e-15
    )


@pytest.mark.parametrize("n", ALLEN_MEDIAN_BIAS)
def test_median_bias_matches_lal_oracle_when_available(n):
    """Agree with the official LAL implementation when LAL is installed."""
    lal = pytest.importorskip("lal")
    assert median_bias(n) == pytest.approx(lal.MedianBias(n), rel=1e-6, abs=1e-15)


@pytest.mark.parametrize("n", ALLEN_MEDIAN_BIAS)
def test_median_bias_matches_pycbc_oracle_when_available(n):
    """Agree with the independent PyCBC FINDCHIRP implementation when present."""
    pycbc_psd = pytest.importorskip("pycbc.psd")
    assert median_bias(n) == pytest.approx(
        pycbc_psd.median_bias(n), rel=1e-6, abs=1e-15
    )


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_median_bias_rejects_boolean_input(value):
    """Reject booleans instead of treating them as integer segment counts."""
    with pytest.raises(TypeError, match="N must be an integer"):
        median_bias(value)


@pytest.mark.parametrize("value", [1.0, "3", None, np.array(3)])
def test_median_bias_rejects_non_integer_input(value):
    """Reject values that cannot represent an integer segment count."""
    with pytest.raises(TypeError, match="N must be an integer"):
        median_bias(value)


@pytest.mark.parametrize("value", [0, -1, -50])
def test_median_bias_rejects_non_positive_input(value):
    """Reject zero and negative segment counts with a clear value error."""
    with pytest.raises(ValueError, match="N must be positive"):
        median_bias(value)
