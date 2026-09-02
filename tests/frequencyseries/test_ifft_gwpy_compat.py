"""GWpy differential contracts for ``FrequencySeries.ifft`` metadata."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries as GWpyFrequencySeries
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries

LARGE_GPS_EPOCH = LIGOTimeGPS(1234567890, 123456789)


def _one_sided_spectrum(original_length):
    samples = np.arange(original_length, dtype=float)
    signal = samples + 0.25 * np.cos(2 * np.pi * samples / original_length)
    spectrum = np.fft.rfft(signal) / original_length
    spectrum[1:] *= 2
    return signal, spectrum


@pytest.mark.parametrize("sample_rate", [4, 4096, 1_000_000, 10_000_000])
@pytest.mark.parametrize("original_length", [8, 9], ids=["even", "odd"])
def test_ifft_default_preserves_parent_axis_shape_and_values(
    sample_rate,
    original_length,
):
    signal, spectrum = _one_sided_spectrum(original_length)
    df = sample_rate / original_length * u.Hz
    kwargs = {
        "df": df,
        "epoch": LARGE_GPS_EPOCH,
        "unit": u.m,
        "name": "phase-2-ifft",
        "channel": "H1:PHASE2-STRAIN",
    }

    expected = GWpyFrequencySeries(spectrum, **kwargs).ifft()
    result = FrequencySeries(spectrum, **kwargs).ifft()

    assert type(result) is TimeSeries
    assert isinstance(result, GWpyTimeSeries)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected.value)
    assert result.unit == expected.unit
    assert result.name == expected.name
    assert result.channel == expected.channel
    assert result.xunit == expected.xunit
    assert result.t0 == expected.t0
    assert result.dt == expected.dt
    assert result.span == expected.span
    assert result.times.unit == expected.times.unit
    np.testing.assert_array_equal(result.times.value, expected.times.value)
    assert result.sample_rate == expected.sample_rate
    assert np.isfinite(result.sample_rate.value)

    if original_length % 2 == 0:
        np.testing.assert_allclose(
            result.value,
            signal,
            rtol=1e-12,
            atol=1e-12,
        )
