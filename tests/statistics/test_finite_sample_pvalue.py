"""Finite-sample Monte Carlo p-values for Rayleigh and GauCh (#507).

With ``N`` simulated null statistics, the observed statistic is ranked among
``N + 1`` values. The one-sided p-value is ``(r + 1) / (N + 1)`` where ``r``
counts simulated statistics at least as extreme as the observed one. The
Rayleigh test keeps its "double the smaller tail" convention.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from astropy import units as u

import gwexpy.statistics.gauch as gauch_module
import gwexpy.statistics.rayleigh_test as rayleigh_module
from gwexpy.spectrogram import Spectrogram
from gwexpy.statistics.dq_flag import to_segments
from gwexpy.statistics.gauch import _get_rayleigh_lilliefors_pvalue, compute_gauch
from gwexpy.statistics.rayleigh_test import rayleigh_pvalue
from gwexpy.timeseries import TimeSeries

NULL = np.array([0.6, 0.8, 1.0, 1.3, 1.7])  # N = 5, sorted


def _spectrogram(values):
    values = np.atleast_2d(np.asarray(values, dtype=float))
    return Spectrogram(values, t0=0 * u.s, dt=1 * u.s, f0=10 * u.Hz, df=1 * u.Hz)


def _fixed_null(monkeypatch, module, name, null):
    def fake(n, n_trials, *, rng=None, seed=None):
        return np.asarray(null, dtype=float)

    monkeypatch.setattr(module, name, fake)


def _rayleigh(monkeypatch, observed, null=NULL):
    _fixed_null(
        monkeypatch, rayleigh_module, "_get_rayleigh_stat_null_distribution", null
    )
    spec = _spectrogram(observed)
    return rayleigh_pvalue(spec, n_samples=16, n_monte_carlo=len(null)).value.ravel()


# --- Rayleigh: p = min(1, 2 * (min(r_upper, r_lower) + 1) / (N + 1)) ----------------


@pytest.mark.parametrize(
    ("observed", "r_upper", "r_lower"),
    [
        (0.1, 5, 0),  # below every simulated value
        (9.9, 0, 5),  # above every simulated value
        (0.7, 4, 1),
        (1.5, 1, 4),
        (1.1, 2, 3),
        (0.9, 3, 2),
        (0.8, 4, 2),  # tie: counted in both tails (>= and <=)
        (1.0, 3, 3),  # tie at the median
        (1.7, 1, 5),  # tie at the largest simulated value
        (0.6, 5, 1),  # tie at the smallest simulated value
    ],
)
def test_rayleigh_rank_formula(monkeypatch, observed, r_upper, r_lower):
    n = len(NULL)
    assert np.count_nonzero(NULL >= observed) == r_upper
    assert np.count_nonzero(NULL <= observed) == r_lower
    expected = min(Fraction(1), Fraction(2 * (min(r_upper, r_lower) + 1), n + 1))

    (p,) = _rayleigh(monkeypatch, [observed])

    assert p == pytest.approx(float(expected), rel=1e-15)


@pytest.mark.parametrize("observed", [0.0, 1e-300, 1e300])
def test_rayleigh_pvalue_is_never_zero(monkeypatch, observed):
    (p,) = _rayleigh(monkeypatch, [observed])

    assert p == pytest.approx(2 / 6, rel=1e-15)
    assert p > 0.0


def test_rayleigh_pvalues_lie_on_the_rank_lattice(monkeypatch):
    rng = np.random.default_rng(507)
    null = np.sort(rng.exponential(size=199))
    observed = rng.exponential(size=(4, 50)) * 3.0

    p = _rayleigh(monkeypatch, observed, null=null)

    steps = p * (len(null) + 1) / 2
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    assert p.min() >= 2 / (len(null) + 1)
    assert p.max() <= 1.0


def test_rayleigh_pvalue_clips_at_one(monkeypatch):
    # N even and the observed value at the centre: 2 * (N/2 + 1) / (N + 1) > 1.
    (p,) = _rayleigh(monkeypatch, [1.15], null=[0.6, 0.8, 1.3, 1.7])

    assert p == 1.0


def test_rayleigh_size_is_controlled_under_exchangeability():
    """P(p <= alpha) <= alpha for every alpha, by enumerating all N+1 ranks."""
    n = 200
    null = np.arange(1, n + 1, dtype=float)
    observed = np.arange(0, n + 1) + 0.5  # one value per rank gap
    upper = n - np.searchsorted(null, observed, side="left")
    lower = np.searchsorted(null, observed, side="right")
    old = np.clip(2.0 * np.minimum(upper, lower) / n, 0.0, 1.0)
    new = np.minimum(1.0, 2.0 * (np.minimum(upper, lower) + 1) / (n + 1))

    for alpha in (0.001, 0.01, 0.05, 0.1, 0.5):
        assert np.mean(new <= alpha) <= alpha
    # The uncorrected estimator is anticonservative at these levels.
    assert np.mean(old <= 0.05) > 0.05
    assert np.mean(old <= 0.01) > 0.01


def test_rayleigh_nonfinite_statistic_is_still_nan(monkeypatch):
    with pytest.warns(RuntimeWarning, match="non-finite"):
        p = _rayleigh(monkeypatch, [np.nan, np.inf, 1.1])

    assert np.isnan(p[0])
    assert np.isnan(p[1])
    assert p[2] == pytest.approx(1.0)


def test_rayleigh_dc_bin_is_still_nan(monkeypatch):
    _fixed_null(
        monkeypatch, rayleigh_module, "_get_rayleigh_stat_null_distribution", NULL
    )
    spec = Spectrogram(
        np.array([[9.9, 9.9, 9.9]]), t0=0 * u.s, dt=1 * u.s, f0=0 * u.Hz, df=1 * u.Hz
    )

    with pytest.warns(RuntimeWarning, match="DC"):
        p = rayleigh_pvalue(spec, n_samples=16, n_monte_carlo=5).value

    assert np.isnan(p[0, 0])
    np.testing.assert_allclose(p[0, 1:], 2 / 6)


def test_rayleigh_seeded_result_is_reproducible_and_keeps_provenance():
    spec = _spectrogram(np.linspace(0.2, 3.0, 12).reshape(3, 4))

    first = rayleigh_pvalue(spec, n_samples=16, n_monte_carlo=99, seed=7)
    second = rayleigh_pvalue(spec, n_samples=16, n_monte_carlo=99, seed=7)

    np.testing.assert_array_equal(first.value, second.value)
    assert first.seed == 7
    assert first.n_monte_carlo == 99
    assert first.provenance == second.provenance
    assert first.value.min() >= 2 / 100


# --- strict ``p < alpha`` decisions in to_segments -----------------------------------


@pytest.mark.parametrize(
    ("n_trials", "vetoed"),
    [(1000, False), (1999, False), (2000, True)],
    ids=["N1000", "N1999-boundary", "N2000"],
)
def test_most_extreme_rayleigh_bin_at_alpha_0p001(monkeypatch, n_trials, vetoed):
    """The smallest Rayleigh p-value is 2/(N+1): below 0.001 needs N >= 2000."""
    null = np.linspace(0.5, 1.5, n_trials)
    _fixed_null(
        monkeypatch, rayleigh_module, "_get_rayleigh_stat_null_distribution", null
    )
    # Two time steps: a single-step spectrogram has no time spacing to span.
    spec = _spectrogram([[100.0], [100.0]])
    p = rayleigh_pvalue(spec, n_samples=16, n_monte_carlo=n_trials)

    flag = to_segments(p, alpha=0.001)

    assert (len(flag.active) > 0) is vetoed


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_rayleigh_decisions_at_common_alpha_are_unchanged_for_n1000(alpha):
    """At N=1000 the strict ``p < alpha`` rank set is the same before and after."""
    n = 1000
    k = np.arange(0, n // 2 + 1)  # min(r_upper, r_lower)
    old = np.minimum(1.0, 2.0 * k / n)
    new = np.minimum(1.0, 2.0 * (k + 1) / (n + 1))

    np.testing.assert_array_equal(old < alpha, new < alpha)


# --- GauCh: p = (r + 1) / (N + 1), r = #{D_sim >= D_obs} ---------------------------


@pytest.mark.parametrize(
    ("observed", "r"),
    [(9.9, 0), (0.1, 5), (1.5, 1), (1.1, 2), (1.3, 2), (1.7, 1), (0.6, 5)],
)
def test_gauch_helper_rank_formula(monkeypatch, observed, r):
    _fixed_null(monkeypatch, gauch_module, "_get_lilliefors_null_distribution", NULL)
    assert np.count_nonzero(NULL >= observed) == r

    p = _get_rayleigh_lilliefors_pvalue(observed, n=40, n_trials=5)

    assert p == pytest.approx((r + 1) / 6, rel=1e-15)


def test_gauch_helper_minimum_is_one_over_n_plus_one():
    p = _get_rayleigh_lilliefors_pvalue(1e6, n=40, n_trials=10)

    assert p == pytest.approx(1 / 11, rel=1e-15)


def test_gauch_helper_maximum_is_one():
    p = _get_rayleigh_lilliefors_pvalue(-1.0, n=40, n_trials=10)

    assert p == 1.0


def _noise(seed=3, duration=64.0, rate=128):
    rng = np.random.default_rng(seed)
    return TimeSeries(rng.normal(size=int(duration * rate)) * 1e-21, sample_rate=rate)


def test_compute_gauch_matches_the_helper_per_bin():
    """Both GauCh code paths must apply the same formula to the same null."""
    n_trials, window = 60, 10
    res = compute_gauch(
        _noise(), fftlength=1.0, window=window, n_monte_carlo=n_trials, seed=11
    )
    stat = res.statistic_map.value
    p = res.pvalue_map.value

    for index in [(0, 3), (5, 20), (stat.shape[0] - 1, stat.shape[1] - 2)]:
        expected = _get_rayleigh_lilliefors_pvalue(
            float(stat[index]), n=window, n_trials=n_trials, seed=11
        )
        assert p[index] == pytest.approx(expected, rel=1e-15)


def test_compute_gauch_pvalues_lie_on_the_rank_lattice():
    n_trials = 60
    res = compute_gauch(
        _noise(), fftlength=1.0, window=10, n_monte_carlo=n_trials, seed=11
    )
    p = res.pvalue_map.value
    p = p[np.isfinite(p)]

    steps = p * (n_trials + 1)
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    assert p.min() >= 1 / (n_trials + 1)
    assert p.max() <= 1.0
    assert res.metadata["n_monte_carlo"] == n_trials
    assert res.metadata["seed"] == 11


def test_compute_gauch_extreme_statistic_gets_the_minimum(monkeypatch):
    n_trials = 25
    _fixed_null(
        monkeypatch,
        gauch_module,
        "_get_lilliefors_null_distribution",
        np.full(n_trials, -1.0),  # every observed D exceeds the null
    )
    below = compute_gauch(_noise(), fftlength=1.0, window=10, n_monte_carlo=n_trials)
    _fixed_null(
        monkeypatch,
        gauch_module,
        "_get_lilliefors_null_distribution",
        np.full(n_trials, 2.0),  # D <= 1 never reaches the null
    )
    above = compute_gauch(_noise(), fftlength=1.0, window=10, n_monte_carlo=n_trials)

    np.testing.assert_allclose(below.pvalue_map.value, 1 / (n_trials + 1))
    np.testing.assert_allclose(above.pvalue_map.value, 1.0)


def test_compute_gauch_dead_bin_is_still_nan():
    ts = TimeSeries(np.zeros(2048), sample_rate=256)

    res = compute_gauch(ts, fftlength=0.25, window=10, n_monte_carlo=50)

    assert np.isnan(res.pvalue_map.value).all()


def test_compute_gauch_seeded_result_is_reproducible():
    kwargs = dict(fftlength=1.0, window=10, n_monte_carlo=60, seed=5)

    first = compute_gauch(_noise(), **kwargs)
    second = compute_gauch(_noise(), **kwargs)

    np.testing.assert_array_equal(first.pvalue_map.value, second.pvalue_map.value)
    np.testing.assert_array_equal(first.statistic_map.value, second.statistic_map.value)
    assert first.pvalue_map.provenance == second.pvalue_map.provenance


def test_gauch_size_is_controlled_under_exchangeability():
    n = 200
    r = np.arange(0, n + 1)  # every rank of the observed statistic
    old = np.maximum(r / n, 1 / n)
    new = (r + 1) / (n + 1)

    for alpha in (0.005, 0.01, 0.05, 0.1, 0.5):
        assert np.mean(new <= alpha) <= alpha
    assert np.mean(old <= 0.005) > 0.005
