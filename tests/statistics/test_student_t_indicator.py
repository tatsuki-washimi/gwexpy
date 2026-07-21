"""Tests for gwexpy.statistics.student_t_indicator (#465 part 1: GPS time
axis reconstruction and input validation).
"""

from __future__ import annotations

import numpy as np
import pytest

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
            compute_student_t_nu(
                ts, fftlength=_FFTLENGTH, stride=1.0, window=_WINDOW
            )

    def test_stride_times_fs_below_one_raises(self):
        ts = _ts(sample_rate=32)
        # stride so small that int(stride * fs) == 0
        with pytest.raises(ValueError, match="stride \\* sample_rate"):
            compute_student_t_nu(
                ts, fftlength=_FFTLENGTH, stride=0.001, window=_WINDOW
            )

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
            compute_student_t_nu(
                ts, fftlength=_FFTLENGTH, overlap=-0.1, window=_WINDOW
            )

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
            ts, fftlength=_FFTLENGTH, stride=_FFTLENGTH / 2, window=_WINDOW,
            overlap=None, frange=(1.0, 10.0),
        )
        assert res.value.size > 0
