"""Tests for gwexpy/types/series_matrix_validation_mixin.py."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeriesMatrix


def _make_tsm(n_rows=2, n_cols=2, n_t=10, t0=0.0, dt=0.01):
    data = np.random.default_rng(42).normal(size=(n_rows, n_cols, n_t))
    return TimeSeriesMatrix(data, t0=t0, dt=dt)


# ---------------------------------------------------------------------------
# is_contiguous
# ---------------------------------------------------------------------------

class TestIsContiguous:
    def test_contiguous_forward(self):
        tsm1 = _make_tsm(t0=0.0, dt=1.0, n_t=10)
        tsm2 = _make_tsm(t0=10.0, dt=1.0, n_t=10)
        assert tsm1.is_contiguous(tsm2) == 1

    def test_contiguous_backward(self):
        tsm1 = _make_tsm(t0=10.0, dt=1.0, n_t=10)
        tsm2 = _make_tsm(t0=0.0, dt=1.0, n_t=10)
        assert tsm1.is_contiguous(tsm2) == -1

    def test_not_contiguous(self):
        tsm1 = _make_tsm(t0=0.0, dt=1.0, n_t=10)
        tsm2 = _make_tsm(t0=50.0, dt=1.0, n_t=10)
        assert tsm1.is_contiguous(tsm2) == 0


# ---------------------------------------------------------------------------
# is_contiguous_exact
# ---------------------------------------------------------------------------

class TestIsContiguousExact:
    def test_contiguous_exact_forward(self):
        tsm1 = _make_tsm(t0=0.0, dt=1.0, n_t=10)
        tsm2 = _make_tsm(t0=10.0, dt=1.0, n_t=10)
        assert tsm1.is_contiguous_exact(tsm2) == 1

    def test_contiguous_exact_backward(self):
        tsm1 = _make_tsm(t0=10.0, dt=1.0, n_t=10)
        tsm2 = _make_tsm(t0=0.0, dt=1.0, n_t=10)
        assert tsm1.is_contiguous_exact(tsm2) == -1

    def test_contiguous_exact_not(self):
        tsm1 = _make_tsm(t0=0.0, dt=1.0, n_t=10)
        tsm2 = _make_tsm(t0=100.0, dt=1.0, n_t=10)
        assert tsm1.is_contiguous_exact(tsm2) == 0

    def test_contiguous_exact_shape_mismatch_raises(self):
        # Lines 64-67
        tsm1 = _make_tsm(n_t=10)
        tsm2 = _make_tsm(n_t=20)
        with pytest.raises(ValueError, match="shape does not match"):
            tsm1.is_contiguous_exact(tsm2)


# ---------------------------------------------------------------------------
# is_compatible_exact
# ---------------------------------------------------------------------------

class TestIsCompatibleExact:
    def test_compatible_exact_same(self):
        tsm1 = _make_tsm()
        tsm2 = tsm1.copy()
        assert tsm1.is_compatible_exact(tsm2) is True

    def test_compatible_exact_xindex_mismatch_raises(self):
        # Line 115 — different xindex
        tsm1 = _make_tsm(t0=0.0)
        tsm2 = _make_tsm(t0=1.0)
        with pytest.raises(ValueError, match="xindex does not match"):
            tsm1.is_compatible_exact(tsm2)

    def test_compatible_exact_shape_mismatch_raises(self):
        # Lines 117-120 — same xindex but different shape
        tsm1 = _make_tsm(n_rows=2, n_cols=2, n_t=10)
        tsm2 = _make_tsm(n_rows=3, n_cols=2, n_t=10)
        with pytest.raises(ValueError):
            tsm1.is_compatible_exact(tsm2)

    def test_compatible_exact_row_keys_mismatch_raises(self):
        # Line 123 — different row keys
        tsm1 = _make_tsm()
        tsm2 = tsm1.copy()
        # Manually change row keys
        import collections
        from gwexpy.types.metadata import MetaDataDict, MetaData
        old_items = list(tsm2.rows.items())
        new_dict = collections.OrderedDict()
        new_dict["newrow0"] = old_items[0][1]
        new_dict["newrow1"] = old_items[1][1]
        tsm2.rows = MetaDataDict(new_dict, expected_size=2, key_prefix="row")
        with pytest.raises(ValueError, match="row keys"):
            tsm1.is_compatible_exact(tsm2)


# ---------------------------------------------------------------------------
# is_compatible
# ---------------------------------------------------------------------------

class TestIsCompatible:
    def test_compatible_with_ndarray(self):
        # Lines 143-149 — non-SeriesMatrix input
        tsm = _make_tsm()
        arr = np.ones(tsm.shape)
        assert tsm.is_compatible(arr) is True

    def test_compatible_ndarray_shape_mismatch_raises(self):
        tsm = _make_tsm()
        arr = np.ones((3, 3, 10))
        with pytest.raises(ValueError, match="shape does not match"):
            tsm.is_compatible(arr)

    def test_compatible_with_same_matrix(self):
        tsm1 = _make_tsm()
        tsm2 = tsm1.copy()
        assert tsm1.is_compatible(tsm2) is True

    def test_compatible_matrix_shape_mismatch_raises(self):
        # Line 152 — matrix rows/cols don't match
        tsm1 = _make_tsm(n_rows=2, n_cols=2)
        tsm2 = _make_tsm(n_rows=3, n_cols=2)
        with pytest.raises(ValueError, match="matrix shape does not match"):
            tsm1.is_compatible(tsm2)

    def test_compatible_unit_mismatch_raises(self):
        # Line 185 — unit mismatch at element
        tsm1 = _make_tsm()
        tsm2 = tsm1.copy()
        # Modify unit of one element in tsm2
        tsm2.meta[0, 0].unit = u.m
        tsm1.meta[0, 0].unit = u.s
        with pytest.raises(ValueError, match="unit does not match"):
            tsm1.is_compatible(tsm2)

    def test_compatible_dx_mismatch_raises(self):
        # Lines 172-173 — dx mismatch leads to ValueError
        tsm1 = _make_tsm(dt=0.01)
        tsm2 = _make_tsm(dt=0.02)
        with pytest.raises(ValueError):
            tsm1.is_compatible(tsm2)

    def test_compatible_xindex_fallback_mismatch_raises(self):
        # Lines 174-178 — xindex fallback when dx comparison fails
        tsm1 = _make_tsm()
        tsm2 = _make_tsm(t0=1.0)
        # Both have same dt so dx check will fail on value
        with pytest.raises(ValueError):
            tsm1.is_compatible(tsm2)
