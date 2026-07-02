"""Degenerate-input regression tests for gwexpy.statistics (issue #459 + S6).

Covers the 9 findings in #459 plus the supplement S6 finding: degenerate /
non-finite inputs must not silently become a wrong-but-plausible numeric result
(all-NaN, p=0 false veto, Inf-corrupted null distribution, IndexError, or a
mis-sized segment).
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from scipy import stats

from gwexpy.spectrogram import Spectrogram
from gwexpy.statistics.dq_flag import to_segments
from gwexpy.statistics.gauch import (
    _get_rayleigh_lilliefors_pvalue,
    compute_gauch,
)
from gwexpy.statistics.rayleigh_test import (
    _get_rayleigh_stat_null_distribution,
    rayleigh_pvalue,
)
from gwexpy.statistics.student_t_indicator import compute_student_t_nu
from gwexpy.timeseries import TimeSeries


def _spec(values):
    return Spectrogram(
        np.asarray(values, dtype=float),
        t0=0 * u.s,
        dt=1 * u.s,
        f0=10 * u.Hz,
        df=1 * u.Hz,
    )


# --- rayleigh_test: entry guards (#2, #3) -----------------------------------


def test_null_distribution_rejects_zero_samples():
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        _get_rayleigh_stat_null_distribution(0, 100)


def test_null_distribution_rejects_zero_trials():
    with pytest.raises(ValueError, match="n_monte_carlo must be >= 1"):
        _get_rayleigh_stat_null_distribution(10, 0)


def test_rayleigh_pvalue_rejects_zero_monte_carlo():
    spec = _spec(np.ones((4, 8)))
    with pytest.raises(ValueError, match="n_monte_carlo must be >= 1"):
        rayleigh_pvalue(spec, n_samples=10, n_monte_carlo=0)


def test_rayleigh_pvalue_rejects_zero_samples():
    spec = _spec(np.ones((4, 8)))
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        rayleigh_pvalue(spec, n_samples=0, n_monte_carlo=100)


# --- rayleigh_test: non-finite statistic -> NaN, not p=0 (S6) ---------------


def test_rayleigh_pvalue_nan_statistic_becomes_nan_not_zero():
    spec = _spec([[1.0, np.nan], [0.9, 1.1]])
    with pytest.warns(RuntimeWarning, match="non-finite"):
        result = rayleigh_pvalue(spec, n_samples=10, n_monte_carlo=200)
    # The NaN-statistic bin must be NaN (excluded from veto), never p=0.
    assert np.isnan(result.value[0, 1])
    assert result.value[0, 1] != 0.0


# --- rayleigh_test: rand()=0 must not inject Inf (#8) ------------------------


def test_null_distribution_floors_uniform_draw():
    with patch("numpy.random.rand", return_value=np.zeros(10)):
        dist = _get_rayleigh_stat_null_distribution(10, 5)
    assert np.all(np.isfinite(dist))


# --- dq_flag: empty spectrogram (#4) ----------------------------------------


def test_to_segments_empty_spectrogram_returns_empty_flag():
    sp = Spectrogram(
        np.empty((0, 4)), t0=0 * u.s, dt=1 * u.s, f0=1 * u.Hz, df=1 * u.Hz
    )
    flag = to_segments(sp)  # must not raise IndexError
    assert len(flag.active) == 0
    assert len(flag.known) == 0


# --- dq_flag: single time step uses metadata dt (#5) ------------------------


def test_to_segments_single_step_uses_metadata_dt():
    sp = Spectrogram(
        np.array([[0.01]]),
        t0=100 * u.s,
        dt=0.25 * u.s,
        f0=10 * u.Hz,
        df=1 * u.Hz,
    )
    flag = to_segments(sp, alpha=0.05)
    known = flag.known[0]
    # width must reflect dt=0.25 (one-bin span), not the old hardcoded 1.0.
    assert (known[1] - known[0]) == pytest.approx(0.25)


# --- gauch: dead bin -> NaN p, not silent 0 (#1) ----------------------------


def test_compute_gauch_dead_bin_yields_nan_pvalue():
    ts = TimeSeries(np.zeros(2048), sample_rate=256)
    res = compute_gauch(ts, fftlength=0.25, window=10, n_monte_carlo=50)
    # an all-zero series has sigma2==0 everywhere -> p must be NaN, never 0.
    assert np.isnan(res.pvalue_map.value).any()
    assert not np.any(res.pvalue_map.value == 0.0)


# --- gauch: Monte-Carlo p-value floored at 1/n_trials (#7) ------------------


def test_lilliefors_pvalue_floored_at_one_over_n_trials():
    p = _get_rayleigh_lilliefors_pvalue(1e6, n=40, n_trials=10)
    assert p == pytest.approx(0.1)
    assert p > 0.0


# --- gauch: entry guards mirroring rayleigh_test (#2/#3 analogue) -----------


def test_lilliefors_rejects_zero_trials():
    with pytest.raises(ValueError, match="n_monte_carlo must be >= 1"):
        _get_rayleigh_lilliefors_pvalue(0.3, n=40, n_trials=0)


def test_lilliefors_rejects_zero_window():
    with pytest.raises(ValueError, match="must be >= 1"):
        _get_rayleigh_lilliefors_pvalue(0.3, n=0, n_trials=10)


# --- gauch: cache keyed by (n, n_trials) (#9) -------------------------------


def test_lilliefors_cache_keyed_by_n_trials():
    from gwexpy.statistics import gauch

    gauch._LILLIEFORS_CACHE.clear()
    _get_rayleigh_lilliefors_pvalue(0.3, n=37, n_trials=10)
    _get_rayleigh_lilliefors_pvalue(0.3, n=37, n_trials=64)
    assert (37, 10) in gauch._LILLIEFORS_CACHE
    assert (37, 64) in gauch._LILLIEFORS_CACHE
    assert len(gauch._LILLIEFORS_CACHE[(37, 64)]) == 64


# --- student_t: fit failure warns instead of silent NaN (#6) ----------------


def test_compute_student_t_nu_warns_on_fit_failure():
    ts = TimeSeries(np.random.default_rng(0).standard_normal(2048), sample_rate=256)
    with patch.object(stats.t, "fit", side_effect=RuntimeError("fit failed")):
        with pytest.warns(RuntimeWarning, match="stats.t.fit failed"):
            res = compute_student_t_nu(ts, fftlength=0.25, window=10)
    assert np.isnan(res.value).all()


def test_compute_student_t_nu_no_warning_on_success():
    ts = TimeSeries(np.random.default_rng(1).standard_normal(2048), sample_rate=256)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = compute_student_t_nu(ts, fftlength=0.25, window=10)
    assert np.isfinite(res.value).any()
