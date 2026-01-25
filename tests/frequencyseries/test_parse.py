import gwpy.frequencyseries._fdcommon as fd
import numpy as np
import pytest


def test_parse_filter_zpk():
    """Test parse_filter with zpk (zeros, poles, gain) format."""
    filt = ([1], [1], 1)
    parsed = fd.parse_filter(filt)
    assert parsed is not None
    # Verify it returns a proper filter representation
    if hasattr(parsed, "freqresp"):
        freqs = np.array([1.0, 10.0, 100.0])
        resp = parsed.freqresp(2 * np.pi * freqs)
        assert resp is not None


def test_parse_filter_sos():
    """Test parse_filter with second-order sections format."""
    # Create a simple SOS filter
    sos = np.array([[1, 0, 0, 1, 0, 0]])  # Unity gain section
    parsed = fd.parse_filter(sos)
    assert parsed is not None


def test_parse_filter_invalid():
    """Test parse_filter with invalid input raises appropriate error."""
    with pytest.raises((ValueError, TypeError)):
        fd.parse_filter("not_a_filter")
