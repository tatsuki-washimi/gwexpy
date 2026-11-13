"""GauCh on single-FFT ASD samples with the standard two-sided KS statistic (#879).

Each ASD sample is one Hann-windowed FFT segment. Interior bins are tested
against a fitted Rayleigh amplitude, DC and (even FFT length) Nyquist against a
fitted half-normal amplitude. The same statistic is used for the Monte Carlo
null. p-values use the rank formula ``(r + 1) / (N + 1)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from scipy import stats

import gwexpy.statistics.gauch as gauch_module
from gwexpy.statistics.gauch import (
    _get_lilliefors_null_distribution,
    _get_rayleigh_lilliefors_pvalue,
    _ks_statistic,
    _segment_asds,
    compute_gauch,
)
from gwexpy.timeseries import TimeSeries

FS = 64


def _noise(seed=3, duration=256.0, rate=FS, scale=1.0, **kwargs):
    rng = np.random.default_rng(seed)
    return TimeSeries(
        rng.normal(size=int(duration * rate)) * scale, sample_rate=rate, **kwargs
    )


# --- input: one Hann FFT per segment ---------------------------------------------------


@pytest.mark.parametrize("hop_fraction", [1.0, 0.5], ids=["no-overlap", "half-overlap"])
def test_asd_samples_equal_one_gwpy_asd_per_segment(hop_fraction):
    """The reference construction: ``segment.asd(fftlength, window="hann")``."""
    ts = _noise(duration=16.0)
    nfft, hop = FS, int(FS * hop_fraction)

    asds = _segment_asds(ts.value, FS, nfft, hop)

    assert asds.shape == ((len(ts) - nfft) // hop + 1, nfft // 2 + 1)
    for i in (0, 1, 5, asds.shape[0] - 1):
        segment = TimeSeries(ts.value[i * hop : i * hop + nfft], sample_rate=FS)
        expected = segment.asd(fftlength=1.0, window="hann").value
        np.testing.assert_allclose(asds[i], expected, rtol=1e-10, atol=0)


def test_interior_power_is_exponential_not_averaged():
    """Averaging periodograms halves this variance; that was the old input."""
    asds = _segment_asds(_noise(duration=4096.0).value, FS, FS, FS)
    power = asds[:, 1:-1] ** 2
    normalized = power / power.mean(axis=0)

    assert normalized.var() == pytest.approx(1.0, abs=0.05)


# --- statistic: standard two-sided KS with a fitted scale --------------------------------


def test_rayleigh_statistic_equals_scipy_kstest():
    rng = np.random.default_rng(1)
    samples = rng.rayleigh(scale=3e-21, size=(39, 6))

    d = _ks_statistic(samples, np.zeros(6, dtype=bool))

    for k in range(6):
        sigma = np.sqrt(np.mean(samples[:, k] ** 2) / 2.0)
        expected = stats.kstest(
            samples[:, k], stats.rayleigh(scale=sigma).cdf
        ).statistic
        assert d[k] == pytest.approx(expected, rel=1e-12)


def test_halfnormal_statistic_equals_scipy_kstest():
    rng = np.random.default_rng(2)
    samples = np.abs(rng.normal(scale=3e-21, size=(40, 4)))

    d = _ks_statistic(samples, np.ones(4, dtype=bool))

    for k in range(4):
        sigma = np.sqrt(np.mean(samples[:, k] ** 2))
        expected = stats.kstest(
            samples[:, k], stats.halfnorm(scale=sigma).cdf
        ).statistic
        assert d[k] == pytest.approx(expected, rel=1e-12)


def test_statistic_includes_the_left_ecdf_limit():
    """D- exceeds D+ here; a right-endpoint-only statistic reports 0.1086."""
    samples = np.array([[0.4], [0.7], [1.0], [1.5]])

    d = _ks_statistic(samples, np.zeros(1, dtype=bool))

    assert d[0] == pytest.approx(0.15134502, abs=1e-8)


def test_dead_column_gives_nan_statistic():
    samples = np.zeros((10, 2))
    samples[:, 1] = np.random.default_rng(0).rayleigh(size=10)

    d = _ks_statistic(samples, np.zeros(2, dtype=bool))

    assert np.isnan(d[0])
    assert np.isfinite(d[1])


def test_null_distribution_uses_the_same_two_sided_statistic():
    n, n_trials = 12, 30
    dist = _get_lilliefors_null_distribution(n, n_trials, seed=5)

    rng = np.random.default_rng(5)
    expected = []
    for _ in range(n_trials):
        x = np.sqrt(-2.0 * np.log(rng.random(n)))
        sigma = np.sqrt(np.mean(x**2) / 2.0)
        expected.append(stats.kstest(x, stats.rayleigh(scale=sigma).cdf).statistic)

    np.testing.assert_allclose(dist, np.sort(expected), rtol=1e-12)


def test_n39_null_agrees_with_recorded_kagra_reference_quantiles():
    """Quantiles recorded from the KAGRA ``n39.npy`` reference (10,000 trials).

    The numbers come from a saved record of that file, not from the file
    itself, which is not in this repository. Tolerances are about three
    standard errors of a 10,000-trial quantile. The former right-endpoint
    statistic has a 95% point near 0.159 and fails this.
    """
    dist = _get_lilliefors_null_distribution(39, 20000, seed=39)

    assert np.quantile(dist, 0.50) == pytest.approx(0.109035, abs=0.002)
    assert np.quantile(dist, 0.95) == pytest.approx(0.169010, abs=0.004)
    assert np.quantile(dist, 0.99) == pytest.approx(0.200701, abs=0.008)


def test_halfnormal_null_is_a_calibrated_reference():
    """Half-normal samples scored against the half-normal null are uniform."""
    n, n_trials = 20, 4000
    dist = gauch_module._get_halfnormal_null_distribution(n, n_trials, seed=1)
    rng = np.random.default_rng(99)
    samples = np.abs(rng.normal(size=(n, 4000)))

    d = _ks_statistic(samples, np.ones(4000, dtype=bool))
    rate = np.mean(d >= np.quantile(dist, 0.95))

    assert rate == pytest.approx(0.05, abs=0.015)


# --- compute_gauch: bin classes, axes, parameters ------------------------------------------


def _expected_statistic(ts, nfft, hop, window, row, col, model):
    x = _segment_asds(ts.value, FS, nfft, hop)[row : row + window, col]
    if model == "rayleigh":
        dist = stats.rayleigh(scale=np.sqrt(np.mean(x**2) / 2.0))
    else:
        dist = stats.halfnorm(scale=np.sqrt(np.mean(x**2)))
    return stats.kstest(x, dist.cdf).statistic


def test_even_fft_length_uses_halfnormal_for_dc_and_nyquist():
    ts = _noise()
    res = compute_gauch(ts, fftlength=1.0, window=20, n_monte_carlo=50, seed=1)
    stat = res.statistic_map.value
    last = stat.shape[1] - 1

    for row in (0, 7):
        assert stat[row, 0] == pytest.approx(
            _expected_statistic(ts, FS, FS, 20, row, 0, "halfnormal"), rel=1e-10
        )
        assert stat[row, last] == pytest.approx(
            _expected_statistic(ts, FS, FS, 20, row, last, "halfnormal"), rel=1e-10
        )
        assert stat[row, 9] == pytest.approx(
            _expected_statistic(ts, FS, FS, 20, row, 9, "rayleigh"), rel=1e-10
        )


def test_odd_fft_length_keeps_the_last_bin_interior():
    rate = 63
    ts = _noise(rate=rate)
    res = compute_gauch(ts, fftlength=1.0, window=20, n_monte_carlo=50, seed=1)
    stat = res.statistic_map.value
    last = stat.shape[1] - 1
    x = _segment_asds(ts.value, rate, rate, rate)[:20]

    assert stat.shape[1] == rate // 2 + 1
    expected_last = stats.kstest(
        x[:, last], stats.rayleigh(scale=np.sqrt(np.mean(x[:, last] ** 2) / 2.0)).cdf
    ).statistic
    expected_dc = stats.kstest(
        x[:, 0], stats.halfnorm(scale=np.sqrt(np.mean(x[:, 0] ** 2))).cdf
    ).statistic
    assert stat[0, last] == pytest.approx(expected_last, rel=1e-10)
    assert stat[0, 0] == pytest.approx(expected_dc, rel=1e-10)


def test_time_and_frequency_axes():
    ts = _noise(duration=64.0, t0=1000.0)
    window = 10

    res = compute_gauch(ts, fftlength=1.0, window=window, n_monte_carlo=20, seed=1)

    for result in (res.pvalue_map, res.statistic_map):
        assert result.shape == (64 - window + 1, FS // 2 + 1)
        np.testing.assert_allclose(
            result.frequencies.value, np.fft.rfftfreq(FS, 1 / FS)
        )
        assert result.frequencies.unit == u.Hz
        np.testing.assert_allclose(
            result.times.value, 1000.0 + window // 2 + np.arange(64 - window + 1)
        )
        assert result.times.unit == u.s
    assert res.n_samples == window
    assert res.metadata["stride"] == 1.0


def test_axes_match_the_previous_spectrogram_based_layout():
    ts = _noise(duration=64.0, t0=1000.0)
    reference = ts.spectrogram(fftlength=1.0, stride=1.0)

    res = compute_gauch(ts, fftlength=1.0, window=10, n_monte_carlo=20, seed=1)

    np.testing.assert_allclose(
        res.pvalue_map.times.value, reference.times.value[5 : 5 + 55]
    )
    np.testing.assert_allclose(
        res.pvalue_map.frequencies.value, reference.frequencies.value
    )


def test_half_overlap_gives_the_reference_segment_count():
    """KAGRA configuration: interval / fftlength * 2 - 1 samples per interval."""
    ts = _noise(duration=20.0)

    res = compute_gauch(
        ts, fftlength=1.0, overlap=0.5, window=39, n_monte_carlo=20, seed=1
    )

    assert res.pvalue_map.shape[0] == 1  # 39 half-overlapping 1 s segments span 20 s
    assert res.metadata["stride"] == 0.5

    longer = compute_gauch(
        _noise(duration=30.0),
        fftlength=1.0,
        overlap=0.5,
        window=39,
        n_monte_carlo=20,
        seed=1,
    )
    assert longer.pvalue_map.shape[0] == 21
    assert longer.pvalue_map.dt.value == pytest.approx(0.5)


@pytest.mark.parametrize("hop", [1, 3])
def test_short_even_fft_rejects_non_half_overlap(hop):
    """With 4 FFT samples, only hop=2 represents 50% overlap."""
    with pytest.raises(ValueError, match="overlap"):
        gauch_module._resolve_hop(4.0, float(hop), None, 1.0)


@pytest.mark.parametrize("hop", [1, 2, 3])
def test_three_sample_fft_rejects_uncalibrated_interior(hop):
    """The short Hann/detrend marginal fails even with disjoint segments."""
    with pytest.raises(ValueError, match="three-sample.*non-Rayleigh"):
        compute_gauch(_noise(duration=8, rate=3), fftlength=1, stride=hop / 3, window=3)


@pytest.mark.parametrize("hop", [2, 4, 5])
def test_short_even_fft_accepts_half_or_disjoint_hops(hop):
    stride, nfft, resolved_hop = gauch_module._resolve_hop(4.0, float(hop), None, 1.0)
    assert (stride, nfft, resolved_hop) == (float(hop), 4, hop)


@pytest.mark.parametrize("hop", [2, 3])
def test_odd_fft_allows_nearest_half_sample_hops(hop):
    stride, nfft, resolved_hop = gauch_module._resolve_hop(5.0, float(hop), None, 1.0)
    assert (stride, nfft, resolved_hop) == (float(hop), 5, hop)


def test_gapped_segments_are_accepted():
    res = compute_gauch(
        _noise(), fftlength=1.0, stride=2.0, window=10, n_monte_carlo=20, seed=1
    )

    assert res.pvalue_map.dt.value == pytest.approx(2.0)
    assert np.isfinite(res.pvalue_map.value).all()


@pytest.mark.parametrize("overlap", [0.25, 0.75, 0.9])
def test_uncalibrated_overlap_is_rejected(overlap):
    with pytest.raises(ValueError, match="overlap"):
        compute_gauch(
            _noise(), fftlength=1.0, overlap=overlap, window=10, n_monte_carlo=20
        )


def test_too_few_segments_raises():
    with pytest.raises(ValueError, match="Too few segments"):
        compute_gauch(_noise(duration=8.0), fftlength=1.0, window=40, n_monte_carlo=20)


def test_statistic_is_scale_invariant_at_strain_amplitudes():
    big = compute_gauch(
        _noise(scale=1.0), fftlength=1.0, window=20, n_monte_carlo=50, seed=1
    )
    tiny = compute_gauch(
        _noise(scale=1e-21), fftlength=1.0, window=20, n_monte_carlo=50, seed=1
    )

    np.testing.assert_allclose(
        tiny.statistic_map.value, big.statistic_map.value, rtol=1e-9
    )
    np.testing.assert_array_equal(tiny.pvalue_map.value, big.pvalue_map.value)


def test_input_series_is_not_modified_and_units_are_accepted():
    ts = _noise(unit="m")
    before = ts.value.copy()

    res = compute_gauch(ts, fftlength=1.0, window=20, n_monte_carlo=50, seed=1)

    np.testing.assert_array_equal(ts.value, before)
    assert ts.unit == u.m
    assert res.pvalue_map.unit == u.dimensionless_unscaled
    assert res.statistic_map.unit == u.dimensionless_unscaled


def test_dead_series_gives_nan_everywhere():
    res = compute_gauch(
        TimeSeries(np.zeros(2048), sample_rate=256),
        fftlength=0.25,
        window=10,
        n_monte_carlo=50,
    )

    assert np.isnan(res.pvalue_map.value).all()
    assert np.isnan(res.statistic_map.value).all()


# --- p-values: rank formula, reproducibility --------------------------------------------


def test_pvalues_lie_on_the_rank_lattice():
    n_trials = 60
    res = compute_gauch(
        _noise(), fftlength=1.0, window=10, n_monte_carlo=n_trials, seed=11
    )
    p = res.pvalue_map.value

    steps = p * (n_trials + 1)
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    assert p.min() >= 1 / (n_trials + 1)
    assert p.max() <= 1.0


def test_interior_pvalue_matches_the_helper():
    n_trials, window = 60, 10
    res = compute_gauch(
        _noise(), fftlength=1.0, window=window, n_monte_carlo=n_trials, seed=11
    )

    for index in [(0, 3), (5, 20)]:
        expected = _get_rayleigh_lilliefors_pvalue(
            float(res.statistic_map.value[index]), n=window, n_trials=n_trials, seed=11
        )
        assert res.pvalue_map.value[index] == pytest.approx(expected, rel=1e-15)


def test_helper_pvalue_bounds():
    assert _get_rayleigh_lilliefors_pvalue(1e6, n=40, n_trials=10) == pytest.approx(
        1 / 11
    )
    assert _get_rayleigh_lilliefors_pvalue(-1.0, n=40, n_trials=10) == 1.0


def test_seeded_result_is_reproducible_including_real_only_bins():
    kwargs = dict(fftlength=1.0, window=10, n_monte_carlo=60, seed=5)

    first = compute_gauch(_noise(), **kwargs)
    second = compute_gauch(_noise(), **kwargs)

    np.testing.assert_array_equal(first.pvalue_map.value, second.pvalue_map.value)
    assert first.pvalue_map.provenance == second.pvalue_map.provenance


def test_rng_path_is_reproducible_and_bypasses_caches():
    gauch_module._LILLIEFORS_CACHE.clear()
    gauch_module._HALFNORMAL_CACHE.clear()
    kwargs = dict(fftlength=1.0, window=10, n_monte_carlo=60)

    first = compute_gauch(_noise(), rng=np.random.default_rng(8), **kwargs)
    second = compute_gauch(_noise(), rng=np.random.default_rng(8), **kwargs)

    np.testing.assert_array_equal(first.pvalue_map.value, second.pvalue_map.value)
    assert not gauch_module._LILLIEFORS_CACHE
    assert not gauch_module._HALFNORMAL_CACHE


def test_default_path_caches_both_null_models():
    gauch_module._LILLIEFORS_CACHE.clear()
    gauch_module._HALFNORMAL_CACHE.clear()

    compute_gauch(_noise(), fftlength=1.0, window=10, n_monte_carlo=33)

    assert (10, 33) in gauch_module._LILLIEFORS_CACHE
    assert (10, 33) in gauch_module._HALFNORMAL_CACHE


# --- calibration on Gaussian noise ---------------------------------------------------------


def _disjoint_pvalues(overlap, window, n_real=8, duration=4096.0, n_trials=5000):
    interior, dc, nyquist = [], [], []
    for m in range(n_real):
        res = compute_gauch(
            _noise(seed=8790 + m, duration=duration),
            fftlength=1.0,
            overlap=overlap,
            window=window,
            n_monte_carlo=n_trials,
            seed=100 + m,
        )
        p = res.pvalue_map.value[::window]
        interior.append(p[:, 1:-1].ravel())
        dc.append(p[:, 0])
        nyquist.append(p[:, -1])
    return np.concatenate(interior), np.concatenate(dc), np.concatenate(nyquist)


@pytest.mark.long
@pytest.mark.parametrize(
    ("overlap", "window"),
    [(None, 40), (0.5, 39)],
    ids=["no-overlap-n40", "half-overlap-n39"],
)
def test_white_noise_rejection_rate_matches_nominal(overlap, window):
    """Disjoint windows of independent series; the old input rejected 57% at 0.05.

    Interior tolerance is about four standard errors: binomial error of the
    pooled tests plus the Monte Carlo error of each realization's null.
    """
    interior, dc, nyquist = _disjoint_pvalues(overlap, window)

    assert np.mean(interior <= 0.05) == pytest.approx(0.05, abs=0.012)
    assert np.mean(interior <= 0.01) == pytest.approx(0.01, abs=0.006)
    for real_only in (dc, nyquist):
        assert 0.015 < np.mean(real_only <= 0.05) < 0.09
