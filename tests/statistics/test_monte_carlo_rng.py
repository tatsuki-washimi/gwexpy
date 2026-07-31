"""Tests for the rng/seed reproducibility contract added to the Monte Carlo
based statistics (#464): gwexpy.statistics.rayleigh_test and
gwexpy.statistics.gauch.
"""

from __future__ import annotations

import threading

import astropy.units as u
import numpy as np
import pytest

from gwexpy.spectrogram import Spectrogram
from gwexpy.statistics import gauch as gauch_module
from gwexpy.statistics import rayleigh_test as rayleigh_module
from gwexpy.statistics.gauch import (
    _get_lilliefors_null_distribution,
    _get_rayleigh_lilliefors_pvalue,
    compute_gauch,
)
from gwexpy.statistics.rayleigh_test import (
    _get_rayleigh_stat_null_distribution,
    rayleigh_pvalue,
)
from gwexpy.timeseries import TimeSeries


def _spec(values):
    return Spectrogram(
        np.asarray(values, dtype=float),
        t0=0 * u.s,
        dt=1 * u.s,
        f0=10 * u.Hz,
        df=1 * u.Hz,
    )


class TestRayleighNullDistributionReproducibility:
    def test_same_seed_gives_identical_distribution(self):
        d1 = _get_rayleigh_stat_null_distribution(20, 50, seed=42)
        d2 = _get_rayleigh_stat_null_distribution(20, 50, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seed_gives_different_distribution(self):
        d1 = _get_rayleigh_stat_null_distribution(20, 50, seed=1)
        d2 = _get_rayleigh_stat_null_distribution(20, 50, seed=2)
        assert not np.array_equal(d1, d2)

    def test_rng_state_advances_across_calls(self):
        """Reusing the same Generator instance must not reset its state --
        two calls with it must draw different samples."""
        rng = np.random.default_rng(7)
        d1 = _get_rayleigh_stat_null_distribution(20, 50, rng=rng)
        d2 = _get_rayleigh_stat_null_distribution(20, 50, rng=rng)
        assert not np.array_equal(d1, d2)

    def test_rng_takes_priority_over_seed(self):
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        d_rng = _get_rayleigh_stat_null_distribution(20, 50, rng=rng_a, seed=999)
        d_seed_only = _get_rayleigh_stat_null_distribution(20, 50, rng=rng_b)
        # rng=... reproduces the same draws as an equivalently-seeded rng
        # passed alone; the seed=999 kwarg must have no effect.
        np.testing.assert_array_equal(d_rng, d_seed_only)

    def test_seed_or_rng_bypasses_shared_cache(self):
        rayleigh_module._RAYLEIGH_STAT_CACHE.clear()
        _get_rayleigh_stat_null_distribution(21, 51, seed=5)
        assert (21, 51) not in rayleigh_module._RAYLEIGH_STAT_CACHE

    def test_no_args_still_populates_shared_cache(self):
        rayleigh_module._RAYLEIGH_STAT_CACHE.clear()
        _get_rayleigh_stat_null_distribution(22, 52)
        assert (22, 52) in rayleigh_module._RAYLEIGH_STAT_CACHE

    def test_no_args_path_still_controlled_by_legacy_global_seed(self):
        """#464 review: the no-args path must keep drawing from the legacy
        global numpy.random state (not a fresh, unseeded default_rng()) so
        that a pre-existing numpy.random.seed(...) call still determines
        the result, exactly as it did before rng=/seed= were added."""
        rayleigh_module._RAYLEIGH_STAT_CACHE.clear()
        np.random.seed(123)
        d1 = _get_rayleigh_stat_null_distribution(23, 53)

        rayleigh_module._RAYLEIGH_STAT_CACHE.clear()
        np.random.seed(123)
        d2 = _get_rayleigh_stat_null_distribution(23, 53)

        np.testing.assert_array_equal(d1, d2)

    def test_default_path_is_thread_safe(self):
        """#464: concurrent callers on the same (n, n_trials) key must all
        observe one consistently-populated cache entry, not a race that
        double-computes or hands back a partially-written array."""
        rayleigh_module._RAYLEIGH_STAT_CACHE.clear()
        results: list[np.ndarray] = []
        errors: list[Exception] = []

        def worker():
            try:
                results.append(_get_rayleigh_stat_null_distribution(30, 200))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert list(rayleigh_module._RAYLEIGH_STAT_CACHE.keys()) == [(30, 200)]
        assert all(np.array_equal(r, results[0]) for r in results)


class TestRayleighPvalueReproducibility:
    def test_same_seed_gives_identical_pvalues(self):
        spec = _spec(np.linspace(0.5, 1.5, 8).reshape(2, 4))
        r1 = rayleigh_pvalue(spec, n_samples=10, n_monte_carlo=50, seed=42)
        r2 = rayleigh_pvalue(spec, n_samples=10, n_monte_carlo=50, seed=42)
        np.testing.assert_array_equal(r1.value, r2.value)

    def test_provenance_attributes_recorded(self):
        spec = _spec(np.linspace(0.5, 1.5, 8).reshape(2, 4))

        default_result = rayleigh_pvalue(spec, n_samples=10, n_monte_carlo=30)
        assert default_result.n_monte_carlo == 30
        assert not hasattr(default_result, "seed")
        assert not hasattr(default_result, "rng_provided")

        seeded_result = rayleigh_pvalue(
            spec, n_samples=10, n_monte_carlo=30, seed=7
        )
        assert seeded_result.seed == 7
        assert not hasattr(seeded_result, "rng_provided")

        rng_result = rayleigh_pvalue(
            spec, n_samples=10, n_monte_carlo=30, rng=np.random.default_rng(7)
        )
        assert rng_result.rng_provided is True
        assert not hasattr(rng_result, "seed")

        with pytest.warns(UserWarning, match="seed is ignored"):
            both_result = rayleigh_pvalue(
                spec, n_samples=10, n_monte_carlo=30,
                rng=np.random.default_rng(7), seed=999,
            )
        assert both_result.rng_provided is True
        assert both_result.seed_unused is True


class TestLilliefersNullDistributionReproducibility:
    def test_same_seed_gives_identical_distribution(self):
        d1 = _get_lilliefors_null_distribution(15, 40, seed=42)
        d2 = _get_lilliefors_null_distribution(15, 40, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_seed_or_rng_bypasses_shared_cache(self):
        gauch_module._LILLIEFORS_CACHE.clear()
        _get_lilliefors_null_distribution(16, 41, seed=5)
        assert (16, 41) not in gauch_module._LILLIEFORS_CACHE

    def test_pvalue_reproducible_with_seed(self):
        p1 = _get_rayleigh_lilliefors_pvalue(0.3, n=17, n_trials=42, seed=1)
        p2 = _get_rayleigh_lilliefors_pvalue(0.3, n=17, n_trials=42, seed=1)
        assert p1 == p2

    def test_no_args_path_still_controlled_by_legacy_global_seed(self):
        """#464 review: mirrors the rayleigh_test analogue -- the no-args
        path must stay on the legacy global numpy.random state."""
        gauch_module._LILLIEFORS_CACHE.clear()
        np.random.seed(123)
        d1 = _get_lilliefors_null_distribution(18, 43)

        gauch_module._LILLIEFORS_CACHE.clear()
        np.random.seed(123)
        d2 = _get_lilliefors_null_distribution(18, 43)

        np.testing.assert_array_equal(d1, d2)


class TestComputeGauchRng:
    def test_rng_path_does_not_raise_keyerror(self):
        """Regression: compute_gauch used to pre-warm the shared cache and
        then read _LILLIEFORS_CACHE[(window, n_monte_carlo)] directly, which
        KeyErrors (or returns a stale unrelated distribution) once rng/seed
        bypass that cache."""
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(2048), sample_rate=256
        )
        res = compute_gauch(
            ts, fftlength=0.25, window=10, n_monte_carlo=50, seed=42
        )
        assert np.isfinite(res.pvalue_map.value).any()

    def test_same_seed_gives_identical_result(self):
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(2048), sample_rate=256
        )
        r1 = compute_gauch(ts, fftlength=0.25, window=10, n_monte_carlo=50, seed=42)
        r2 = compute_gauch(ts, fftlength=0.25, window=10, n_monte_carlo=50, seed=42)
        np.testing.assert_array_equal(r1.pvalue_map.value, r2.pvalue_map.value)

    def test_metadata_recorded(self):
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(2048), sample_rate=256
        )
        default_res = compute_gauch(ts, fftlength=0.25, window=10, n_monte_carlo=50)
        assert default_res.metadata["n_monte_carlo"] == 50
        assert "seed" not in default_res.metadata

        seeded_res = compute_gauch(
            ts, fftlength=0.25, window=10, n_monte_carlo=50, seed=7
        )
        assert seeded_res.metadata["seed"] == 7

        with pytest.warns(UserWarning, match="seed is ignored"):
            both_res = compute_gauch(
                ts, fftlength=0.25, window=10, n_monte_carlo=50,
                rng=np.random.default_rng(7), seed=999,
            )
        assert both_res.metadata["rng_provided"] is True
        assert both_res.metadata["seed_unused"] is True

    def test_default_path_still_uses_shared_cache(self):
        gauch_module._LILLIEFORS_CACHE.clear()
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(2048), sample_rate=256
        )
        compute_gauch(ts, fftlength=0.25, window=10, n_monte_carlo=33)
        assert (10, 33) in gauch_module._LILLIEFORS_CACHE


class TestClipFloorPreserved:
    """#464: the np.clip(..., tiny, 1.0) floor from #459 must survive the
    RNG refactor for both Monte Carlo sites.
    """

    class _ZeroGenerator:
        def random(self, size):
            return np.zeros(size)

    def test_rayleigh_null_distribution_floor(self):
        dist = _get_rayleigh_stat_null_distribution(10, 5, rng=self._ZeroGenerator())
        assert np.all(np.isfinite(dist))

    def test_lilliefors_null_distribution_floor(self):
        dist = _get_lilliefors_null_distribution(10, 5, rng=self._ZeroGenerator())
        assert np.all(np.isfinite(dist))


class TestDegenerateInputGuardsWithRngKwargs:
    """Entry-point guards (#459) must still fire when rng/seed are passed."""

    def test_rayleigh_null_distribution_rejects_zero_samples_with_seed(self):
        with pytest.raises(ValueError, match="n_samples must be >= 2"):
            _get_rayleigh_stat_null_distribution(0, 100, seed=1)

    def test_lilliefors_null_distribution_rejects_zero_trials_with_seed(self):
        with pytest.raises(ValueError, match="n_monte_carlo must be >= 1"):
            _get_lilliefors_null_distribution(10, 0, seed=1)
