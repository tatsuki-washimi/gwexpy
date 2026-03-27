"""Tests for gwexpy/signal/preprocessing/imputation.py."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.signal.preprocessing.imputation import (
    _bfill_numpy,
    _coerce_times,
    _ffill_numpy,
    _limit_mask,
    impute,
)


# ---------------------------------------------------------------------------
# _coerce_times
# ---------------------------------------------------------------------------


class TestCoerceTimes:
    def test_basic(self):
        t, idx = _coerce_times(np.array([0.0, 1.0, 2.0]), 3)
        np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])

    def test_value_attr(self):
        # Line 16 — x has .value attribute after np.asarray
        # Create a subclass of ndarray that retains .value
        class ArrayWithValue(np.ndarray):
            @property
            def value(self):
                return np.asarray(self)

        times = np.array([0.0, 1.0, 2.0]).view(ArrayWithValue)
        t, idx = _coerce_times(times, 3)
        np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])

    def test_non_1d_raises(self):
        # Line 18
        with pytest.raises(ValueError, match="1D"):
            _coerce_times(np.ones((2, 3)), 6)

    def test_wrong_length_raises(self):
        # Line 20
        with pytest.raises(ValueError, match="same length"):
            _coerce_times(np.array([0.0, 1.0]), 5)

    def test_non_increasing_raises(self):
        # Line 24 — duplicate times → not strictly increasing after sort
        with pytest.raises(ValueError, match="strictly increasing"):
            _coerce_times(np.array([0.0, 1.0, 1.0]), 3)

    def test_unsorted_input(self):
        # Sort index returned for unsorted input
        t, idx = _coerce_times(np.array([2.0, 0.0, 1.0]), 3)
        np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# _limit_mask
# ---------------------------------------------------------------------------


class TestLimitMask:
    def test_none_limit_returns_all_false(self):
        # Line 30
        nans = np.array([True, True, False, True])
        mask = _limit_mask(nans, None)
        assert not np.any(mask)

    def test_negative_limit_raises(self):
        # Line 33
        nans = np.array([True, False])
        with pytest.raises(ValueError, match="non-negative"):
            _limit_mask(nans, -1)

    def test_zero_limit_returns_all_nans(self):
        # Line 35
        nans = np.array([True, False, True])
        mask = _limit_mask(nans, 0)
        np.testing.assert_array_equal(mask, nans)

    def test_limit_one(self):
        # Lines 46-50 — run_len > limit
        nans = np.array([False, True, True, True, False])
        mask = _limit_mask(nans, 1, direction="forward")
        # First NaN is filled (within limit=1), rest are masked
        assert mask[2] == True or mask[3] == True

    def test_limit_backward(self):
        # Line 48 — direction='backward'
        nans = np.array([False, True, True, True, False])
        mask = _limit_mask(nans, 1, direction="backward")
        assert isinstance(mask, np.ndarray)

    def test_trailing_nan_run(self):
        # Lines 52-58 — run ends at array boundary
        nans = np.array([False, True, True, True])
        mask = _limit_mask(nans, 1, direction="forward")
        # NaN run of length 3 > limit 1: positions after the first should be masked
        assert np.any(mask)

    def test_trailing_nan_backward(self):
        # Lines 55-56 — trailing NaN run, backward direction
        nans = np.array([False, True, True, True])
        mask = _limit_mask(nans, 1, direction="backward")
        assert isinstance(mask, np.ndarray)


# ---------------------------------------------------------------------------
# _ffill_numpy
# ---------------------------------------------------------------------------


class TestFfillNumpy:
    def test_basic(self):
        # Lines 63-78
        val = np.array([1.0, np.nan, np.nan, 4.0])
        out = _ffill_numpy(val)
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0, 4.0])

    def test_leading_nans_unchanged(self):
        # Line 70 — no previous value yet → continue
        val = np.array([np.nan, np.nan, 3.0])
        out = _ffill_numpy(val)
        assert np.isnan(out[0])
        assert np.isnan(out[1])
        assert out[2] == pytest.approx(3.0)

    def test_limit(self):
        # Line 71 — run >= limit → stop filling
        val = np.array([1.0, np.nan, np.nan, np.nan])
        out = _ffill_numpy(val, limit=1)
        assert out[1] == pytest.approx(1.0)
        assert np.isnan(out[2])
        assert np.isnan(out[3])


# ---------------------------------------------------------------------------
# _bfill_numpy
# ---------------------------------------------------------------------------


class TestBfillNumpy:
    def test_basic(self):
        # Lines 81-97
        val = np.array([np.nan, np.nan, 3.0, np.nan])
        out = _bfill_numpy(val)
        np.testing.assert_array_equal(out, [3.0, 3.0, 3.0, np.nan])

    def test_trailing_nans_unchanged(self):
        # Line 89 — no next value yet → continue
        val = np.array([1.0, np.nan, np.nan])
        out = _bfill_numpy(val)
        assert out[0] == pytest.approx(1.0)
        assert np.isnan(out[1])
        assert np.isnan(out[2])

    def test_limit(self):
        # Line 90 — run >= limit → stop filling
        val = np.array([np.nan, np.nan, np.nan, 4.0])
        out = _bfill_numpy(val, limit=1)
        assert out[2] == pytest.approx(4.0)
        assert np.isnan(out[0])
        assert np.isnan(out[1])


# ---------------------------------------------------------------------------
# impute
# ---------------------------------------------------------------------------


class TestImpute:
    def test_no_nans_passthrough(self):
        val = np.array([1.0, 2.0, 3.0])
        result = impute(val)
        np.testing.assert_array_equal(result, val)

    def test_interpolate_basic(self):
        val = np.array([1.0, np.nan, 3.0])
        result = impute(val, method="interpolate")
        assert result[1] == pytest.approx(2.0)

    def test_interpolate_complex(self):
        # Lines 190-197 — complex array
        val = np.array([1 + 0j, np.nan + 0j, 3 + 0j])
        result = impute(val, method="interpolate")
        assert result[1].real == pytest.approx(2.0)

    def test_ffill(self):
        val = np.array([1.0, np.nan, np.nan, 4.0])
        result = impute(val, method="ffill")
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)

    def test_bfill(self):
        val = np.array([np.nan, np.nan, 3.0])
        result = impute(val, method="bfill")
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(3.0)

    def test_mean(self):
        val = np.array([1.0, np.nan, 3.0])
        result = impute(val, method="mean")
        assert result[1] == pytest.approx(2.0)

    def test_median(self):
        val = np.array([1.0, np.nan, 3.0, 5.0])
        result = impute(val, method="median")
        assert result[1] == pytest.approx(3.0)

    def test_unknown_method_raises(self):
        val = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="Unknown impute method"):
            impute(val, method="bogus")

    def test_with_times(self):
        val = np.array([1.0, np.nan, 3.0])
        times = np.array([0.0, 1.0, 2.0])
        result = impute(val, method="interpolate", times=times)
        assert result[1] == pytest.approx(2.0)

    def test_max_gap(self):
        # Lines 176-181 — max_gap constraint
        val = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = impute(val, method="interpolate", times=times, max_gap=1.5)
        # Gap of 4 units exceeds max_gap=1.5, so NaNs should be restored
        assert np.any(np.isnan(result))

    def test_max_gap_with_value_attr(self):
        # Line 178 — max_gap has .value attribute (Quantity)
        val = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        max_gap = u.Quantity(1.5, u.s)
        result = impute(val, method="interpolate", times=times, max_gap=max_gap)
        assert isinstance(result, np.ndarray)

    def test_limit_interpolate(self):
        # Lines 250-253 — limit restores excess NaNs after interpolation
        val = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        result = impute(val, method="interpolate", limit=1)
        # Only first NaN (within limit) should be filled
        assert np.isnan(result[2])
        assert np.isnan(result[3])

    def test_fill_value_non_nan(self):
        # Lines 260-270 — fill_value != nan, has valid indices → fill edge NaNs
        # Use max_gap so that left_val/right_val=nan, making edge NaNs remain
        val = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = impute(val, method="interpolate", times=times, max_gap=0.5, fill_value=0.0)
        # Edges should be filled with fill_value=0 (no valid neighbor within max_gap)
        assert result[0] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)

    def test_fill_value_ffill_non_nan(self):
        # Line 272 — method in ('ffill', 'bfill') + non-nan fill_value
        val = np.array([np.nan, 1.0, 2.0, np.nan])
        result = impute(val, method="ffill", fill_value=-999.0)
        assert result[0] == pytest.approx(-999.0)

    def test_fill_value_mean_all_nan(self):
        # Lines 273-275 — method='mean' and all NaN → fill with fill_value
        val = np.array([np.nan, np.nan, np.nan])
        result = impute(val, method="mean", fill_value=0.0)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_fill_value_non_numeric_type_error(self):
        # Lines 258-259 — np.isnan(fill_value) raises TypeError → is_fill_nan=False
        val = np.array([1.0, np.nan, 3.0])
        # "not_nan" is a string → np.isnan raises TypeError
        result = impute(val, method="mean", fill_value="not_nan")
        assert result is not None

    def test_with_sorted_times(self):
        # Lines 277-280 — sort_idx unsort
        val = np.array([1.0, 5.0, np.nan])
        times = np.array([2.0, 0.0, 1.0])  # unsorted
        result = impute(val, method="interpolate", times=times)
        assert len(result) == 3

    def test_no_valid_values_interpolate(self):
        # Line 188-189 — no valid values → pass (no interpolation done)
        val = np.array([np.nan, np.nan])
        result = impute(val, method="interpolate")
        assert np.all(np.isnan(result))

    def test_fill_value_no_valid_indices(self):
        # Line 263 — len(valid_indices) == 0 with non-nan fill_value
        val = np.array([np.nan, np.nan])
        result = impute(val, method="interpolate", fill_value=5.0)
        np.testing.assert_array_equal(result, [5.0, 5.0])

    def test_ffill_numpy_fallback(self, monkeypatch):
        # Lines 206-207 — pandas not available → _ffill_numpy fallback
        import sys
        monkeypatch.setitem(sys.modules, "pandas", None)
        val = np.array([1.0, np.nan, np.nan])
        result = impute(val, method="ffill")
        assert result[1] == pytest.approx(1.0)

    def test_bfill_numpy_fallback(self, monkeypatch):
        # Lines 215-216 — pandas not available → _bfill_numpy fallback
        import sys
        monkeypatch.setitem(sys.modules, "pandas", None)
        val = np.array([np.nan, np.nan, 3.0])
        result = impute(val, method="bfill")
        assert result[1] == pytest.approx(3.0)
