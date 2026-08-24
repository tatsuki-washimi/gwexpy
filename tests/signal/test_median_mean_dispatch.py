"""Tests for GWexpy's GWpy-compatible median-mean PSD extension."""

import numpy as np
import pytest
from astropy import units as u
from gwpy.signal import spectral as gwpy_spectral

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.signal.spectral import median_mean
from gwexpy.timeseries import TimeSeries


def _fixture_timeseries() -> TimeSeries:
    """Build a deterministic, exactly segmentable physical fixture."""
    samples = np.arange(60, dtype=float)
    data = np.sin(2 * np.pi * samples / 8) + 0.25 * np.cos(2 * np.pi * samples / 5)
    return TimeSeries(
        data,
        sample_rate=8,
        unit=u.V,
        t0=1234567890,
        name="median-mean-fixture",
        channel="H1:MEDIAN_MEAN",
    )


def test_median_mean_is_owned_and_registered_through_gwpy_surface():
    """Expose the GWexpy extension through GWpy's normal method lookup."""
    assert gwpy_spectral.get_method("median-mean") is median_mean


def test_median_mean_psd_and_asd_preserve_contract():
    """Preserve class, units, axes, metadata, and ASD/PSD numerical identity."""
    pytest.importorskip("lal")
    timeseries = _fixture_timeseries()
    kwargs = {"fftlength": 1, "overlap": 0.5, "window": "hann"}

    psd = timeseries.psd(method="median-mean", **kwargs)
    asd = timeseries.asd(method="median-mean", **kwargs)

    assert isinstance(psd, FrequencySeries)
    assert isinstance(asd, FrequencySeries)
    assert psd.unit == u.V**2 / u.Hz
    assert asd.unit == u.V / u.Hz**0.5
    np.testing.assert_allclose(psd.frequencies.value, np.arange(5, dtype=float))
    np.testing.assert_allclose(asd.frequencies.value, psd.frequencies.value)
    np.testing.assert_allclose(asd.value, np.sqrt(psd.value), rtol=1e-12, atol=1e-15)

    for result in (psd, asd):
        assert result.name == timeseries.name
        assert result.channel == timeseries.channel
        assert result.epoch == timeseries.epoch
