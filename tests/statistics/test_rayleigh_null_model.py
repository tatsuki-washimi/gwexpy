"""Tests for the Rayleigh-statistic null distribution model (#506).

Through v0.1.11 `rayleigh_pvalue()` simulated its null from Rayleigh
*amplitude* samples while the statistic it scored came from exponential
*power* segments, and `TimeSeries.rayleigh_test()` passed a hard-coded
``n_samples=39`` unrelated to the data. These tests pin down the corrected
model, the segment-count derivation, and the DC/Nyquist handling.

Every test that builds a null distribution passes ``seed=`` so that
`_get_rayleigh_stat_null_distribution` bypasses the process-global
`_RAYLEIGH_STAT_CACHE`. Polluting that cache from here would break
`test_monte_carlo_rng.py::...test_default_path_is_thread_safe`, which
compares the whole cache key set exactly and is therefore
collection-order-sensitive.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gwpy.signal.window import recommended_overlap

from gwexpy.statistics.dq_flag import to_segments
from gwexpy.statistics.rayleigh_test import (
    _get_rayleigh_stat_null_distribution,
    _simulate_rayleigh_null,
    rayleigh_pvalue,
)
from gwexpy.timeseries import TimeSeries

FS = 128


def _noise(seed, duration, sample_rate=FS):
    return TimeSeries(
        np.random.default_rng(seed).normal(size=int(sample_rate * duration)),
        sample_rate=sample_rate,
    )


N_MONTE_CARLO = 1000
N_REPEATS = 6
N_COLUMNS = 200


def _rayleigh_test_spectrogram(noise_seed, null_seed):
    ts = _noise(noise_seed, 8.0 * N_COLUMNS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ts.rayleigh_test(
            fftlength=1.0, stride=8.0, n_monte_carlo=N_MONTE_CARLO, seed=null_seed
        )


@pytest.fixture(scope="module")
def pvalue_pool():
    """Interior-bin p-values from `N_REPEATS` independent noise/null pairs.

    Module-scoped because building it costs ~18 s and several tests score
    the same pool at different `alpha`. Independent *null* seeds are what
    the tolerance band needs; see
    `TestCalibration.test_end_to_end_false_positive_rate`.
    """
    pools = []
    for m in range(N_REPEATS):
        spec = _rayleigh_test_spectrogram(900_000 + m, 700_000 + m)
        # Drop the masked DC/Nyquist columns; keep the interior bins.
        pools.append(spec.value[:, 1:-1].ravel())
    pool = np.concatenate(pools)
    assert np.isfinite(pool).all()
    return pool


@pytest.fixture(scope="module")
def single_spectrogram():
    return _rayleigh_test_spectrogram(900_000, 700_000)


def _target_size(alpha, n_monte_carlo):
    """Achievable two-sided size of a Monte Carlo p-value.

    The observed statistic and the ``N`` null draws are exchangeable, so
    ``P(p <= 2k/N) = 2(k+1)/(N+1)`` with ``k = floor(alpha * N / 2)``. This
    is *not* ``alpha``: at ``N=1000`` the nominal 0.05 is really 0.05195.
    Centring a tolerance band on ``alpha`` would reject a correct
    implementation.
    """
    k = int(alpha * n_monte_carlo / 2)
    return 2 * (k + 1) / (n_monte_carlo + 1)


class TestDegenerateSegmentCountGuard:
    """``n_samples < 2`` must raise at every entry point, not degrade silently.

    The statistic is ``std(P)/mean(P)`` over ``n`` segments. At ``n == 1``
    every trial has ``std == 0``, giving an all-zero null against which any
    observed statistic scores ``p == 0``; at ``n <= 0`` numpy returns NaN,
    giving an all-NaN null with the same effect. Neither raises on its own,
    so each of the three entry points is guarded and tested separately --
    `_get_rayleigh_stat_null_distribution` also memoises into a shared
    cache, so a guard only at the lowest layer would still let a degenerate
    entry be stored and handed back.
    """

    @pytest.mark.parametrize("n_samples", [0, 1])
    def test_rayleigh_pvalue_rejects(self, n_samples):
        spec = _noise(1024, 32.0).rayleigh_spectrogram(8.0, 1.0)
        with pytest.raises(ValueError, match="n_samples must be >= 2"):
            rayleigh_pvalue(spec, n_samples=n_samples, n_monte_carlo=100)

    @pytest.mark.parametrize("n", [0, 1])
    def test_null_distribution_rejects(self, n):
        with pytest.raises(ValueError, match="n_samples must be >= 2"):
            _get_rayleigh_stat_null_distribution(n, 100, seed=1)

    @pytest.mark.parametrize("n", [0, 1])
    def test_simulate_null_rejects(self, n):
        with pytest.raises(ValueError, match="n_samples must be >= 2"):
            _simulate_rayleigh_null(n, 100, np.random.default_rng(1).random)


class TestNullDistributionModel:
    """The null must be the sample CV of Exp(1) power, not Rayleigh amplitude."""

    @pytest.mark.parametrize("n", [8, 39, 64, 128])
    def test_second_moment_matches_exact_value(self, n):
        """``E[R^2] = (n-1)/(n+1)`` exactly.

        The sample CV of ``n`` i.i.d. Exp(1) values reduces to Greenwood's
        statistic: ``R^2 = n * sum(p_i^2) - 1`` for
        ``p ~ Dirichlet(1, ..., 1)``, whose moments are closed-form. This is
        the sharpest available check -- the pre-#506 amplitude null misses it
        by 14% at n=8 -- and it needs no reference implementation.
        """
        n_trials = 20000
        dist = _get_rayleigh_stat_null_distribution(n, n_trials, seed=12345)
        sq = dist**2
        exact = (n - 1) / (n + 1)
        # Monte Carlo error only; no fudge factor.
        sem = np.std(sq) / np.sqrt(n_trials)
        assert abs(np.mean(sq) - exact) < 4 * sem

    def test_amplitude_null_would_fail_the_moment_check(self):
        """Guard the guard: the pre-#506 model must be detectable.

        A moment check that both models pass would be worthless as a
        regression test, so assert the old one is actually rejected.
        """
        n, n_trials = 8, 20000
        rng = np.random.default_rng(1)
        const = np.sqrt((4.0 - np.pi) / np.pi)
        old = np.array(
            [
                np.std(s) / (np.mean(s) * const)
                for s in np.sqrt(
                    -2.0
                    * np.log(
                        np.clip(rng.random((n_trials, n)), np.finfo(float).tiny, 1.0)
                    )
                )
            ]
        )
        sq = old**2
        exact = (n - 1) / (n + 1)
        sem = np.std(sq) / np.sqrt(n_trials)
        assert abs(np.mean(sq) - exact) > 20 * sem

    def test_statistic_is_scale_invariant(self):
        """The CV is scale-free, so ``-log(u)`` and ``-2 log(u)`` must agree.

        The factor 2 is a power of two, so it leaves the mantissa untouched
        and the results are bit-identical. Generalising to a non-dyadic
        factor requires `assert_allclose(rtol=1e-12)` instead.
        """
        n, n_trials = 30, 20000
        a = _simulate_rayleigh_null(n, n_trials, np.random.default_rng(3).random)
        b = np.sort(
            np.array(
                [
                    np.std(x) / np.mean(x)
                    for x in -2.0
                    * np.log(
                        np.clip(
                            np.random.default_rng(3).random((n_trials, n)),
                            np.finfo(float).tiny,
                            1.0,
                        )
                    )
                ]
            )
        )
        np.testing.assert_array_equal(a, b)


class TestSegmentCountDerivation:
    """``n_samples`` must equal the segments GWpy actually averaged."""

    @pytest.mark.parametrize(
        ("stride", "fftlength", "overlap", "sample_rate"),
        [
            (32, 1.0, 0, FS),
            (32, 1.0, None, FS),
            (10, 1.0, None, FS),
            (16, 0.5, None, FS),
            (32, 2.0, None, FS),
            (4, 0.5, 0, FS),
            # For odd FFT lengths GWpy's recommended overlap is 65, so the
            # 64-sample hop must not be confused with the overlap itself.
            (2, 1.0, None, 129),
            # Explicit fftlength / 2 rounds to 64 samples: the hop is 65.
            (2, 1.0, 0.5, 129),
        ],
    )
    def test_derived_n_samples_matches_welch_call_count(
        self, stride, fftlength, overlap, sample_rate, monkeypatch
    ):
        """Ground truth by counting `welch` calls.

        Nothing reconstructible from the output spectrogram is reliable
        here: ``dt * df`` gives the no-overlap count (32 instead of 64 for
        the default overlap), and assuming GWpy chunks the series into
        ``nstride`` samples is off by one -- `_chunk_timeseries` actually
        cuts ``nstride + noverlap``. Counting the calls is the only
        measurement that cannot drift from GWpy's behaviour silently.
        """
        import gwpy.signal.spectral._scipy as gwpy_scipy

        original = gwpy_scipy.welch
        calls = {"n": 0, "lengths": set()}

        def counting_welch(timeseries, segmentlength, **kwargs):
            calls["n"] += 1
            calls["lengths"].add(int(timeseries.size))
            return original(timeseries, segmentlength, **kwargs)

        monkeypatch.setattr(gwpy_scipy, "welch", counting_welch)

        # `rayleigh_test` imports `rayleigh_pvalue` from the module on every
        # call, so patching the module attribute intercepts the real one.
        captured = {}
        import gwexpy.statistics.rayleigh_test as rt_module

        original_pvalue = rt_module.rayleigh_pvalue

        def capturing_pvalue(rayleigh_spec, n_samples, *args, **kwargs):
            captured["n_samples"] = n_samples
            return original_pvalue(rayleigh_spec, n_samples, *args, **kwargs)

        monkeypatch.setattr(rt_module, "rayleigh_pvalue", capturing_pvalue)

        ts = _noise(0, 128, sample_rate=sample_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spec = ts.rayleigh_test(
                fftlength=fftlength,
                stride=stride,
                overlap=overlap,
                n_monte_carlo=50,
                seed=1,
            )

        n_columns = spec.shape[0]
        true_n = calls["n"] // n_columns
        assert calls["n"] == true_n * n_columns, "ragged segment count across columns"
        # Every averaged slice is a full FFT length: GWpy either raises or
        # drops data, it never feeds a short segment through.
        assert calls["lengths"] == {int(round(fftlength * sample_rate))}
        assert captured["n_samples"] == true_n

    @pytest.mark.parametrize("overlap", [None, 0])
    def test_rejects_stride_with_no_complete_fft_segment(self, overlap):
        ts = _noise(0, 8)
        with pytest.raises(ValueError, match="not enough samples for one complete FFT"):
            ts.rayleigh_test(
                fftlength=1.0,
                stride=0.25,
                overlap=overlap,
                n_monte_carlo=50,
                seed=1,
            )

    @pytest.mark.parametrize(
        ("stride", "overlap"),
        [(0.5, None), (1.0, 0)],
    )
    def test_rejects_single_spectral_segment(self, stride, overlap):
        ts = _noise(0, 8)
        with pytest.raises(ValueError, match="requires at least two spectral segments"):
            ts.rayleigh_test(
                fftlength=1.0,
                stride=stride,
                overlap=overlap,
                n_monte_carlo=50,
                seed=1,
            )

    @pytest.mark.parametrize(
        ("stride", "overlap"),
        [(1.0, None), (2.0, 0)],
    )
    def test_accepts_exactly_two_spectral_segments(self, stride, overlap):
        ts = _noise(0, 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = ts.rayleigh_test(
                fftlength=1.0,
                stride=stride,
                overlap=overlap,
                n_monte_carlo=50,
                seed=1,
            )
        assert np.isfinite(result.value[:, 1:-1]).all()

    def test_default_overlap_is_not_dt_times_df(self):
        """Pin the specific wrong answer #506 shipped.

        At the default overlap the true count is twice ``dt * df``. Without
        this, a future refactor could reintroduce the ``dt * df`` shortcut
        and every test above would still pass at ``overlap=0``.
        """
        ts = _noise(0, 128)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spec = ts.rayleigh_test(
                fftlength=1.0, stride=32, overlap=None, n_monte_carlo=50, seed=1
            )
        dt_df = round(float(spec.dt.value) * float(spec.df.value))
        assert dt_df == 32  # the value the old code would have used

    def test_explicit_mismatched_n_samples_warns(self):
        ts = _noise(0, 128)
        with pytest.warns(UserWarning, match="disagrees with"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("always", UserWarning)
                ts.rayleigh_test(
                    fftlength=1.0, stride=32, n_samples=39, n_monte_carlo=50, seed=1
                )

    def test_explicit_matching_n_samples_does_not_warn(self):
        """Backward compatibility: passing the right value stays silent."""
        ts = _noise(0, 128)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            ts.rayleigh_test(
                fftlength=1.0, stride=32, n_samples=64, n_monte_carlo=50, seed=1
            )


class TestOverlapRestriction:
    """Only overlaps for which the i.i.d. Exp model holds are accepted."""

    @pytest.mark.parametrize("fraction", [0.25, 0.75])
    def test_unsupported_overlap_raises(self, fraction):
        """75% is the dangerous one: GWpy returns without error or warning.

        It uses ~36% of the data *and* the per-segment powers stop being
        approximately i.i.d. exponential (two-sample KS D=0.076 against the
        best-fitting segment count, p=0), so correcting ``n_samples`` cannot
        rescue it. 25% merely raises inside scipy.
        """
        ts = _noise(0, 128)
        with pytest.raises(ValueError, match="i.i.d"):
            ts.rayleigh_test(
                fftlength=1.0, stride=32, overlap=fraction, n_monte_carlo=50, seed=1
            )

    @pytest.mark.parametrize("overlap", [0, None, 0.5])
    def test_supported_overlaps_accepted(self, overlap):
        ts = _noise(0, 128)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ts.rayleigh_test(
                fftlength=1.0, stride=32, overlap=overlap, n_monte_carlo=50, seed=1
            )

    def test_recommended_overlap_differs_from_half_for_odd_nfft(self):
        """Why the default must be resolved through GWpy, not computed.

        ``recommended_overlap`` is not ``nfft // 2`` for an odd FFT length,
        and using the wrong one shifts the derived segment count.
        """
        assert recommended_overlap("hann", 128) == 64
        assert recommended_overlap("hann", 129) == 65 != 129 // 2
        assert recommended_overlap("hann", 127) == 64 != 127 // 2


class TestRealOnlyBins:
    """DC and Nyquist power is chi2_1, not Exp(1), so those bins are NaN."""

    def test_even_nfft_masks_dc_and_nyquist(self):
        ts = _noise(0, 64)
        with pytest.warns(RuntimeWarning, match="DC and Nyquist"):
            spec = ts.rayleigh_test(
                fftlength=1.0, stride=8.0, n_monte_carlo=100, seed=1
            )
        assert np.isnan(spec.value[:, 0]).all()
        assert np.isnan(spec.value[:, -1]).all()
        assert np.isfinite(spec.value[:, 1:-1]).all()

    def test_odd_nfft_masks_dc_only(self):
        """An odd FFT length has no exact Nyquist bin.

        ``rfftfreq(129)[-1] = 0.4961``, short of the 0.5 Nyquist frequency,
        so the last bin is an ordinary complex interior bin. Masking it
        would discard a valid measurement.
        """
        sample_rate = 129
        assert not np.isclose(np.fft.rfftfreq(sample_rate, d=1.0)[-1], 0.5)
        ts = _noise(0, 64, sample_rate=sample_rate)
        with pytest.warns(RuntimeWarning, match="DC bin"):
            spec = ts.rayleigh_test(
                fftlength=1.0, stride=8.0, n_monte_carlo=100, seed=1
            )
        assert np.isnan(spec.value[:, 0]).all()
        assert np.isfinite(spec.value[:, -1]).all()


class TestCalibration:
    """False-positive rate must hit the achievable size, not ``alpha``."""

    @pytest.mark.long
    @pytest.mark.parametrize("alpha", [0.05, 0.01])
    def test_end_to_end_false_positive_rate(self, alpha, pvalue_pool):
        """Through the real GWpy path, on real white noise.

        Exercising `_get_rayleigh_stat_null_distribution` alone cannot catch
        this class of bug: #506 was a mismatch between the null and the
        statistic, so any test that does not compute the statistic the way
        production does is blind to it. This also covers the ``n_samples``
        derivation and the ``overlap`` forwarding end to end.

        The band comes from
        ``sd = sqrt(alpha(1-alpha)/n_obs + alpha(1-alpha/2)/(M*N_mc))``.
        The second term is irreducible: all bins and times in one repeat
        share a single null draw, so it does not shrink with more observed
        trials -- only with more independent null seeds (``M``).
        """
        pool = pvalue_pool
        measured = np.count_nonzero(pool <= alpha) / pool.size
        target = _target_size(alpha, N_MONTE_CARLO)
        sd = np.sqrt(
            alpha * (1 - alpha) / pool.size
            + alpha * (1 - alpha / 2) / (N_REPEATS * N_MONTE_CARLO)
        )
        assert abs(measured - target) < 4 * sd, (
            f"alpha={alpha}: measured {measured:.5f}, target {target:.5f}, "
            f"4 sd = {4 * sd:.5f}"
        )

    @pytest.mark.long
    def test_pre_fix_model_would_fail_this(self):
        """The pre-#506 pipeline misses the target size by ~100 sd.

        Reproduced locally rather than by reverting the source, so a
        refactor that reintroduces either defect is caught. Both defects are
        shown separately because fixing only the null distribution still
        leaves the rate roughly 4x the target.
        """
        n_trials = N_MONTE_CARLO
        alpha = 0.05
        target = _target_size(alpha, n_trials)
        stale_n = 39  # the removed hard-coded default; the true count is 16

        def score(null, observed):
            upper = len(null) - np.searchsorted(null, observed, side="left")
            lower = np.searchsorted(null, observed, side="right")
            return np.clip(2.0 * np.minimum(upper, lower) / len(null), 0.0, 1.0)

        def amplitude_null(n, trials, rng):
            const = np.sqrt((4.0 - np.pi) / np.pi)
            samples = np.sqrt(
                -2.0
                * np.log(np.clip(rng.random((trials, n)), np.finfo(float).tiny, 1.0))
            )
            return np.sort(samples.std(axis=1) / (samples.mean(axis=1) * const))

        amplitude_rates, stale_n_rates = [], []
        for m in range(N_REPEATS):
            ts = _noise(900_000 + m, 8.0 * N_COLUMNS)
            observed = (
                ts.rayleigh_spectrogram(stride=8.0, fftlength=1.0, overlap=None)
                .value[:, 1:-1]
                .ravel()
            )
            amplitude_rates.append(
                score(
                    amplitude_null(
                        stale_n, n_trials, np.random.default_rng(700_000 + m)
                    ),
                    observed,
                )
            )
            stale_n_rates.append(
                score(
                    _get_rayleigh_stat_null_distribution(
                        stale_n, n_trials, seed=700_000 + m
                    ),
                    observed,
                )
            )

        amplitude = np.count_nonzero(np.concatenate(amplitude_rates) <= alpha) / sum(
            r.size for r in amplitude_rates
        )
        stale = np.count_nonzero(np.concatenate(stale_n_rates) <= alpha) / sum(
            r.size for r in stale_n_rates
        )
        # Pre-fix: ~0.30. Exponential null but stale n_samples: ~0.20.
        assert amplitude > 4 * target
        assert stale > 3 * target


class TestKnownRemainingLimitations:
    """Pin states that are deliberately *not* fixed here, so they are visible."""

    @pytest.mark.long
    def test_p_equals_zero_still_occurs(self, single_spectrogram):
        """No finite-Monte-Carlo floor, unlike `compute_gauch`.

        `gauch` clamps its p-values at ``1/N`` while this function can
        report exactly 0, which downstream code may read as infinite
        significance. Fixing the asymmetry is tracked separately (#507);
        this test exists so removing the floor asymmetry is a visible,
        deliberate change rather than a silent one.
        """
        pool = single_spectrogram.value[:, 1:-1].ravel()
        assert np.count_nonzero(pool == 0.0) > 0

    @pytest.mark.long
    def test_to_segments_still_vetoes_almost_every_time(self, single_spectrogram):
        """Per-bin calibration does not fix the frequency multiplicity.

        `to_segments` flags a time when *any* bin has ``p < alpha``, with no
        multiple-comparison correction, so 63 perfectly calibrated bins veto
        ``1 - 0.95**63 ~= 96%`` of times at the default ``alpha=0.05``. This
        is why "#506 caused the spurious vetoes" is the wrong attribution
        and why the fix must not be described as repairing them.
        """
        spec = single_spectrogram
        # The NaN DC/Nyquist columns need no special handling: `NaN < alpha`
        # is False, so they are already excluded from the veto.
        flag = to_segments(spec, alpha=0.05)
        vetoed = sum(float(seg[1] - seg[0]) for seg in flag.active)
        total = float(spec.times.value[-1] - spec.times.value[0])
        assert vetoed / total > 0.9
