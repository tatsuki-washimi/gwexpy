from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.analysis.bruco import FastCoherenceEngine
from gwexpy.analysis.stats import SpectralStats
from gwexpy.frequencyseries import FrequencySeries


def _sine_timeseries() -> GwpyTimeSeries:
    sample_rate = 128.0
    t = np.arange(int(sample_rate * 4.0)) / sample_rate
    data = np.sin(2.0 * np.pi * 8.0 * t)
    return GwpyTimeSeries(data, sample_rate=sample_rate * u.Hz)


def test_fast_bruco_coherence_zero_aux_returns_zero_contract():
    target = _sine_timeseries()
    aux = GwpyTimeSeries(np.zeros(len(target)), sample_rate=target.sample_rate)
    engine = FastCoherenceEngine(target, fftlength=1.0, overlap=0.5)

    coherence = engine.compute_coherence(aux)

    assert coherence.shape == engine.frequencies.shape
    assert engine.frequencies[0] == pytest.approx(0.0)
    assert np.all(np.diff(engine.frequencies) > 0.0)
    np.testing.assert_array_equal(coherence, np.zeros_like(coherence))


def test_fast_bruco_identical_input_coherence_stays_in_numeric_bounds():
    target = _sine_timeseries()
    engine = FastCoherenceEngine(target, fftlength=1.0, overlap=0.5)

    coherence = engine.compute_coherence(target)

    assert np.all(np.isfinite(coherence))
    assert np.min(coherence) >= -1e-12
    assert np.max(coherence) <= 1.0 + 1e-12
    np.testing.assert_allclose(coherence, 1.0, atol=1e-12)


def test_spectral_stats_zero_sigma_current_contract_returns_infinite_significance():
    frequencies = [10, 20] * u.Hz
    mean = FrequencySeries([1.0, 2.0], frequencies=frequencies, unit=u.m)
    sigma = FrequencySeries([0.0, 1.0], frequencies=frequencies, unit=u.m)
    injected = FrequencySeries([2.0, 4.0], frequencies=frequencies, unit=u.m)

    with pytest.warns(RuntimeWarning, match="divide by zero"):
        significance = SpectralStats(mean=mean, sigma=sigma, n_avg=2).significance(
            injected
        )

    assert significance.unit == u.dimensionless_unscaled
    np.testing.assert_allclose(significance.frequencies.to_value(u.Hz), [10.0, 20.0])
    assert np.isinf(significance.value[0])
    assert significance.value[1] == pytest.approx(2.0)
