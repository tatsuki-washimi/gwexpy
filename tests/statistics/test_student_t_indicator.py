"""Tests for gwexpy.statistics.student_t_indicator (#465: GPS time axis
reconstruction, input validation, and DC/Nyquist bin fit bias).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from gwexpy.statistics.student_t_indicator import compute_student_t_nu
from gwexpy.timeseries import TimeSeries


def _ts(n=512, sample_rate=32, t0=0):
    return TimeSeries(
        np.random.default_rng(0).standard_normal(n), sample_rate=sample_rate, t0=t0
    )


# fftlength=0.5s @ 32 Hz -> nfft=16 (9 one-sided bins); window=5 keeps the
# nested stats.t.fit loop cheap (this module fits per time/freq bin in a
# plain Python loop, so keep test signals small -- see #465).
_FFTLENGTH = 0.5
_WINDOW = 5


class TestGpsTimeAxis:
    def test_t0_zero_times_are_unshifted(self):
        ts = _ts(t0=0)
        res = compute_student_t_nu(ts, fftlength=_FFTLENGTH, window=_WINDOW)
        assert res.times.value[0] >= 0

    def test_t0_nonzero_shifts_times_onto_gps_axis(self):
        t0 = 1_000_000_000
        ts_at_zero = _ts(t0=0)
        ts_shifted = _ts(t0=t0)
        res0 = compute_student_t_nu(ts_at_zero, fftlength=_FFTLENGTH, window=_WINDOW)
        res_shifted = compute_student_t_nu(
            ts_shifted, fftlength=_FFTLENGTH, window=_WINDOW
        )
        np.testing.assert_allclose(
            res_shifted.times.value, res0.times.value + t0, rtol=0, atol=1e-9
        )


class TestInputValidation:
    def test_stride_greater_than_fftlength_raises(self):
        ts = _ts()
        with pytest.raises(ValueError, match="must not exceed fftlength"):
            compute_student_t_nu(ts, fftlength=_FFTLENGTH, stride=1.0, window=_WINDOW)

    def test_stride_times_fs_below_one_raises(self):
        ts = _ts(sample_rate=32)
        # stride so small that int(stride * fs) == 0
        with pytest.raises(ValueError, match="stride \\* sample_rate"):
            compute_student_t_nu(ts, fftlength=_FFTLENGTH, stride=0.001, window=_WINDOW)

    def test_fftlength_non_positive_raises(self):
        ts = _ts()
        with pytest.raises(ValueError, match="fftlength must be finite and positive"):
            compute_student_t_nu(ts, fftlength=0.0, window=_WINDOW)

    def test_fftlength_non_finite_raises(self):
        ts = _ts()
        with pytest.raises(ValueError, match="fftlength must be finite and positive"):
            compute_student_t_nu(ts, fftlength=float("nan"), window=_WINDOW)

    def test_overlap_negative_raises(self):
        ts = _ts()
        with pytest.raises(ValueError, match="overlap must be finite"):
            compute_student_t_nu(ts, fftlength=_FFTLENGTH, overlap=-0.1, window=_WINDOW)

    def test_overlap_equal_to_fftlength_raises(self):
        """overlap == fftlength collapses stride to 0, which must be rejected."""
        ts = _ts()
        with pytest.raises(ValueError, match="stride must be finite and positive"):
            compute_student_t_nu(
                ts, fftlength=_FFTLENGTH, overlap=_FFTLENGTH, window=_WINDOW
            )

    def test_window_non_positive_raises(self):
        ts = _ts()
        with pytest.raises(ValueError, match="window must be a positive integer"):
            compute_student_t_nu(ts, fftlength=_FFTLENGTH, window=0)

    def test_window_non_integer_raises(self):
        ts = _ts()
        with pytest.raises(ValueError, match="window must be a positive integer"):
            compute_student_t_nu(ts, fftlength=_FFTLENGTH, window=10.5)

    def test_frange_low_greater_than_high_raises(self):
        ts = _ts()
        with pytest.raises(ValueError, match="frange low"):
            compute_student_t_nu(
                ts, fftlength=_FFTLENGTH, window=_WINDOW, frange=(10.0, 1.0)
            )

    def test_frange_non_finite_raises(self):
        ts = _ts()
        with pytest.raises(ValueError, match="frange must be finite"):
            compute_student_t_nu(
                ts, fftlength=_FFTLENGTH, window=_WINDOW, frange=(0.0, float("inf"))
            )

    def test_valid_inputs_still_succeed(self):
        """Regression: the validation additions must not reject valid calls."""
        ts = _ts()
        res = compute_student_t_nu(
            ts,
            fftlength=_FFTLENGTH,
            stride=_FFTLENGTH / 2,
            window=_WINDOW,
            overlap=None,
            frange=(1.0, 10.0),
        )
        assert res.value.size > 0

    def test_complex_input_raises(self):
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(512).astype(complex),
            sample_rate=32,
            t0=0,
        )
        with pytest.raises(ValueError, match="real-valued"):
            compute_student_t_nu(ts, fftlength=_FFTLENGTH, window=_WINDOW)


class TestDcNyquistRealOnlyFit:
    """#465: DC/Nyquist bins of a real input are purely real, so their fit
    must use only ``window`` real samples, not ``2 * window`` re+im
    samples -- identified structurally (index 0 / the last one-sided bin
    for even nfft), not via a floating-point frequency comparison.
    """

    def _fit_sample_lengths(self, ts, **kwargs):
        """Run compute_student_t_nu with stats.t.fit patched to record the
        length of the samples array passed for each (i, j) call, in the
        deterministic i-then-j iteration order used by the implementation.
        """
        lengths = []
        orig_fit = stats.t.fit

        def fake_fit(samples):
            lengths.append(len(samples))
            return orig_fit(samples)

        with patch.object(stats.t, "fit", side_effect=fake_fit):
            res = compute_student_t_nu(ts, **kwargs)
        return res, lengths

    def test_even_nfft_dc_and_nyquist_use_window_samples(self):
        # fs=16, fftlength=0.5 -> nfft=8 (even) -> f = [0, 2, 4, 6, 8] Hz;
        # Nyquist (8 Hz) is an exact bin. window=9 -> n_times=9 -> n_out=1,
        # so the 5 stats.t.fit calls correspond 1:1 to f[0..4] in order.
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(64), sample_rate=16, t0=0
        )
        res, lengths = self._fit_sample_lengths(ts, fftlength=0.5, window=9)
        np.testing.assert_allclose(res.frequencies.value, [0, 2, 4, 6, 8])
        assert lengths == [9, 18, 18, 18, 9]

    def test_odd_nfft_top_bin_is_not_nyquist_and_stays_two_sided(self):
        # fs=16, fftlength=7/16 -> nfft=7 (odd) -> no exact Nyquist bin;
        # f = [0, 2.286, 4.571, 6.857] Hz, none equal to fs/2=8. window=10
        # -> n_times=10 -> n_out=1, so the 4 calls map 1:1 to f[0..3].
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(64), sample_rate=16, t0=0
        )
        res, lengths = self._fit_sample_lengths(ts, fftlength=7 / 16, window=10)
        assert len(res.frequencies.value) == 4
        assert not np.isclose(res.frequencies.value[-1], 8.0)
        # DC (index 0) is real-only; the rest (including the odd-nfft top
        # bin, which is not Nyquist) stay two-sided.
        assert lengths == [10, 20, 20, 20]

    def test_frange_excluding_dc_and_nyquist_keeps_bins_two_sided(self):
        # Same even-nfft setup as above, but frange excludes both f=0 and
        # f=8 -- the remaining bins must not be real-only-fitted.
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(64), sample_rate=16, t0=0
        )
        res, lengths = self._fit_sample_lengths(
            ts, fftlength=0.5, window=9, frange=(1.0, 7.0)
        )
        np.testing.assert_allclose(res.frequencies.value, [2, 4, 6])
        assert lengths == [18, 18, 18]

    def test_fine_frequency_resolution_does_not_over_match_dc(self):
        # Regression: a floating-point np.isclose(f, 0.0) comparison (the
        # original approach) misclassifies several near-DC bins as DC once
        # the bin spacing (fs/nfft) drops below its default atol=1e-8 --
        # e.g. fs=1e-7, nfft=32 gives df=3.125e-9 and 4 bins (indices 0-3)
        # all satisfy np.isclose(f, 0.0). The structural (index-based) fix
        # must mark only index 0 as DC regardless of frequency resolution.
        fs = 1e-7
        nfft = 32
        assert fs / nfft < 1e-8  # sanity: finer than the old atol
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(64), sample_rate=fs, t0=0
        )
        res, lengths = self._fit_sample_lengths(ts, fftlength=nfft / fs, window=3)
        # n_freqs = nfft//2 + 1 = 17; nfft even -> index 0 (DC) and index
        # -1 (Nyquist) are real-only, everything else stays two-sided.
        assert len(res.frequencies.value) == 17
        assert lengths[0] == 3  # DC
        assert lengths[1:-1] == [6] * 15  # all other bins two-sided
        assert lengths[-1] == 3  # Nyquist

    def test_requested_even_nfft_shrunk_to_odd_by_short_input(self):
        # scipy.signal.stft silently shrinks nperseg to len(ts) when nfft
        # exceeds it. Requested nfft=32 (even) but input length=31 ->
        # effective segment length is 31 (odd), so the last bin must NOT
        # be treated as Nyquist even though the *requested* nfft is even.
        fs = 16.0
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(31), sample_rate=fs, t0=0
        )
        with pytest.warns(UserWarning, match="nperseg"):
            res, lengths = self._fit_sample_lengths(ts, fftlength=32 / fs, window=2)
        assert len(res.frequencies.value) == 16  # (effective 31 // 2) + 1
        assert not np.isclose(res.frequencies.value[-1], fs / 2)
        assert lengths[0] == 2  # DC
        assert lengths[1:] == [4] * 15  # no Nyquist bin; all stay two-sided

    def test_requested_odd_nfft_shrunk_to_even_by_short_input(self):
        # Mirror case: requested nfft=31 (odd) but input length=30 ->
        # effective segment length is 30 (even), so the last bin IS the
        # exact Nyquist bin even though the *requested* nfft is odd.
        fs = 16.0
        ts = TimeSeries(
            np.random.default_rng(0).standard_normal(30), sample_rate=fs, t0=0
        )
        with pytest.warns(UserWarning, match="nperseg"):
            res, lengths = self._fit_sample_lengths(ts, fftlength=31 / fs, window=2)
        assert len(res.frequencies.value) == 16  # (effective 30 // 2) + 1
        np.testing.assert_allclose(res.frequencies.value[-1], fs / 2)
        assert lengths[0] == 2  # DC
        assert lengths[1:-1] == [4] * 14  # two-sided
        assert lengths[-1] == 2  # Nyquist


class TestDcNyquistNumericRegression:
    """#465: numeric regression pinning the DC-bin fix.

    Provenance (recorded per the v0.1.11 plan's review-before-merge gate):
    seed=12345, scipy==1.17.1, sample_rate=64 Hz, fftlength=0.25s, window=30,
    non-overlapping stride (stride=fftlength), signal length=5314 samples,
    bins evaluated: DC (f=0 Hz) vs one AC bin (f=4 Hz), over ~305 independent
    fit windows (frange=(0, 5) keeps the loop to 2 frequency bins).

    Before the DC real-only fix (re+im concatenated at DC too):
      median(nu_dc)=0.308, frac(nu_dc<10)=1.000 vs median(nu_ac)=4.47e8,
      frac(nu_ac<10)=0.075 -> log10 diff=9.16, frac diff=0.925 (both fail).
    After the fix: median(nu_dc)=3.02e9, frac(nu_dc<10)=0.157 vs
      median(nu_ac)=4.47e8, frac(nu_ac<10)=0.075 -> log10 diff=0.83,
      frac diff=0.082 (both comfortably pass the thresholds below).

    Contrast-based, not per-bin absolute pins, since stats.t.fit's nu is
    itself heavy-tailed even for pure Gaussian input (see #465 plan notes).
    """

    def test_dc_bin_matches_ac_bin_order_of_magnitude(self):
        fs = 64.0
        fftlength = 0.25
        window = 30
        nstep = int(fftlength * fs)
        n_out_target = 300
        n_times_needed = n_out_target + window - 1
        n_samples = int(fftlength * fs) + (n_times_needed - 1) * nstep + 50

        ts = TimeSeries(
            np.random.default_rng(12345).standard_normal(n_samples),
            sample_rate=fs,
            t0=0,
        )
        res = compute_student_t_nu(
            ts, fftlength=fftlength, window=window, frange=(0, 5)
        )
        np.testing.assert_allclose(res.frequencies.value, [0, 4])

        dc = res.value[:, 0]
        ac = res.value[:, 1]
        dc = dc[np.isfinite(dc)]
        ac = ac[np.isfinite(ac)]
        assert len(dc) > 100 and len(ac) > 100

        log10_diff = abs(np.log10(np.median(dc)) - np.log10(np.median(ac)))
        frac_diff = abs(np.mean(dc < 10) - np.mean(ac < 10))

        assert log10_diff <= 1.5, f"DC/AC median nu differ by {log10_diff} decades"
        assert frac_diff <= 0.20, f"DC/AC frac(nu<10) differ by {frac_diff}"
