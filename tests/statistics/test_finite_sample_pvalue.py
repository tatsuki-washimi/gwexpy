"""Finite-sample Monte Carlo p-values for the Rayleigh test (#507).

With ``N`` simulated null statistics, the observed statistic is ranked among
``N + 1`` values. The one-sided p-value is ``(r + 1) / (N + 1)`` where ``r``
counts simulated statistics at least as extreme as the observed one. The
Rayleigh test keeps its "double the smaller tail" convention. The same
correction for GauCh is deferred to its recalibration (#879).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from astropy import units as u

import gwexpy.statistics.rayleigh_test as rayleigh_module
from gwexpy.spectrogram import Spectrogram
from gwexpy.statistics.dq_flag import to_segments
from gwexpy.statistics.gauch import _get_rayleigh_lilliefors_pvalue
from gwexpy.statistics.rayleigh_test import rayleigh_pvalue

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


# --- GauCh is deliberately not changed here ----------------------------------------


def test_gauch_pvalue_keeps_its_floor_until_the_null_is_recalibrated():
    """GauCh keeps ``max(r/N, 1/N)``; its rank correction waits for #879.

    Moving the GauCh minimum from ``1/N`` to ``1/(N+1)`` would make bins
    beyond the whole simulated null satisfy a strict ``p < 0.001`` at
    ``N=1000``. The current GauCh statistic and null are not calibrated
    (#879), so that correction is applied together with the recalibration.
    """
    p = _get_rayleigh_lilliefors_pvalue(1e6, n=40, n_trials=10)

    assert p == pytest.approx(1 / 10, rel=1e-15)
