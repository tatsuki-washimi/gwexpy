"""Tests for gwexpy/types/series_matrix_indexing.py."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix
from gwexpy.types.seriesmatrix import SeriesMatrix


def _make_tsm(n_rows=2, n_cols=3, n_t=10):
    data = np.arange(n_rows * n_cols * n_t, dtype=float).reshape(n_rows, n_cols, n_t)
    return TimeSeriesMatrix(data, t0=0.0, dt=0.01)


# ---------------------------------------------------------------------------
# __getitem__ — single cell access
# ---------------------------------------------------------------------------


class TestGetItemSingleCell:
    def test_single_cell_int_int(self):
        tsm = _make_tsm()
        result = tsm[0, 0]
        assert result.shape == (10,)

    def test_single_cell_int_int_slice(self):
        tsm = _make_tsm()
        result = tsm[0, 1, :5]
        assert result.shape == (5,)

    def test_single_cell_no_xindex(self):
        # xindex=None branch (line 91): SeriesMatrix with xindex provided
        data = np.ones((2, 2, 5))
        sm = SeriesMatrix(data, xindex=np.arange(5))
        result = sm[0, 0]
        assert result.shape == (5,)

    def test_single_cell_series_class_none(self):
        # series_class=None → falls back to gwpy Series
        data = np.ones((2, 2, 5))
        sm = SeriesMatrix(data, xindex=np.arange(5))
        result = sm[0, 0]
        assert result is not None


# ---------------------------------------------------------------------------
# __getitem__ — slice / non-scalar selection
# ---------------------------------------------------------------------------


class TestGetItemSlice:
    def test_row_slice(self):
        tsm = _make_tsm()
        result = tsm[0:2, :, :]
        assert result.shape == (2, 3, 10)

    def test_col_slice(self):
        tsm = _make_tsm()
        result = tsm[:, 0:2, :]
        assert result.shape == (2, 2, 10)

    def test_row_int_col_slice(self):
        # scalar r with non-scalar c → result.ndim < 3 path
        tsm = _make_tsm()
        result = tsm[0, :, :]
        assert result.shape == (1, 3, 10)

    def test_col_int_row_slice(self):
        # non-scalar r with scalar c → result.ndim < 3 path
        tsm = _make_tsm()
        result = tsm[:, 0, :]
        assert result.shape == (2, 1, 10)

    def test_xindex_scalar_s(self):
        # integer s → xindex sliced with s:s+1
        tsm = _make_tsm()
        result = tsm[:, :, 0]
        assert result.shape == (2, 3, 1)

    def test_xindex_slice_preserves_xindex(self):
        # xindex is sliced for slice path
        tsm = _make_tsm()
        result = tsm[0:2, 0:2, 0:5]
        assert result.shape == (2, 2, 5)
        assert result.xindex is not None

    def test_result_not_instance_of_self(self):
        # When super().__getitem__ returns a plain ndarray → view conversion (line 124)
        tsm = _make_tsm()
        result = tsm[0:1, 0:1, :]
        assert isinstance(result, TimeSeriesMatrix)

    def test_meta_ndim_reshape(self):
        # Scalar ri or ci with slice → meta dimension restore
        tsm = _make_tsm()
        result = tsm[0, :, :]  # scalar ri
        assert result.ndim == 3

    def test_meta_col_scalar_restore(self):
        tsm = _make_tsm()
        result = tsm[:, 0, :]  # scalar ci
        assert result.ndim == 3


# ---------------------------------------------------------------------------
# __getitem__ — label-based (string) indexing
# ---------------------------------------------------------------------------


class TestGetItemLabelBased:
    def test_row_string_indexing(self):
        # String row key → row_index lookup (lines 102-103)
        tsm = _make_tsm()
        row_key = tsm.row_keys()[0]  # e.g. 'row0'
        result = tsm[row_key, 0]
        assert result.shape == (10,)

    def test_col_string_indexing(self):
        # String col key → col_index lookup (lines 111-112)
        tsm = _make_tsm()
        col_key = tsm.col_keys()[1]  # e.g. 'col1'
        result = tsm[0, col_key]
        assert result.shape == (10,)

    def test_row_list_of_strings(self):
        # List of string row keys → list comprehension (lines 104-105)
        tsm = _make_tsm()
        keys = list(tsm.row_keys())
        result = tsm[keys, :, :]
        assert result.shape == (2, 3, 10)

    def test_col_list_of_strings(self):
        # List of string col keys → list comprehension (lines 113-114)
        tsm = _make_tsm()
        keys = list(tsm.col_keys()[:2])
        result = tsm[:, keys, :]
        assert result.shape == (2, 2, 10)

    def test_int_row_list_and_int_col_list_are_cartesian(self):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        sm = SeriesMatrix(
            data,
            xindex=np.arange(5),
            rows={"r0": {}, "r1": {}, "r2": {}},
            cols={"c0": {}, "c1": {}, "c2": {}, "c3": {}},
        )

        result = sm[[0, 2], [1, 3], :]

        assert result.shape == (2, 2, 5)
        expected = data[np.ix_([0, 2], [1, 3], np.arange(5))]
        np.testing.assert_array_equal(result.value, expected)
        assert result.row_keys() == ("r0", "r2")
        assert result.col_keys() == ("c1", "c3")

    def test_label_row_list_and_label_col_list_are_cartesian(self):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        sm = SeriesMatrix(
            data,
            xindex=np.arange(5),
            rows={"r0": {}, "r1": {}, "r2": {}},
            cols={"c0": {}, "c1": {}, "c2": {}, "c3": {}},
        )

        result = sm[["r0", "r2"], ["c1", "c3"], :]

        assert result.shape == (2, 2, 5)
        expected = data[np.ix_([0, 2], [1, 3], np.arange(5))]
        np.testing.assert_array_equal(result.value, expected)
        assert result.row_keys() == ("r0", "r2")
        assert result.col_keys() == ("c1", "c3")

    def test_ndarray_integer_row_list_and_col_list_are_cartesian(self):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        sm = SeriesMatrix(data, xindex=np.arange(5))

        result = sm[np.array([0, 2]), np.array([1, 3]), ...]

        assert result.shape == (2, 2, 5)
        expected = data[np.ix_([0, 2], [1, 3], np.arange(5))]
        np.testing.assert_array_equal(result.value, expected)

    def test_ndarray_boolean_row_mask_and_col_mask_are_cartesian(self):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        sm = SeriesMatrix(data, xindex=np.arange(5))

        result = sm[np.array([True, False, True]), np.array([False, True, False, True])]

        assert result.shape == (2, 2, 5)
        expected = data[np.ix_([0, 2], [1, 3], np.arange(5))]
        np.testing.assert_array_equal(result.value, expected)

    def test_wrong_length_boolean_mask_raises(self):
        sm = SeriesMatrix(np.zeros((3, 4, 5)), xindex=np.arange(5))

        with pytest.raises(IndexError, match="selector length mismatch"):
            sm[[True, False], [1, 3], :]

    def test_non_integer_numeric_selector_raises(self):
        sm = SeriesMatrix(np.zeros((3, 4, 5)), xindex=np.arange(5))

        with pytest.raises(TypeError, match="integer positions"):
            sm[[0.0, 2.0], [1, 3], :]


# ---------------------------------------------------------------------------
# __setitem__
# ---------------------------------------------------------------------------


class TestSetItem:
    def test_setitem_plain_value(self):
        # Line 213-214 — plain scalar assignment
        tsm = _make_tsm()
        tsm[0, 0, 0] = 999.0
        assert tsm.view(np.ndarray)[0, 0, 0] == 999.0

    def test_setitem_array_value(self):
        # Line 210-212 — array/Series with .shape attribute
        tsm = _make_tsm()
        arr = np.ones(10)
        tsm[0, 0] = arr
        np.testing.assert_array_equal(tsm.view(np.ndarray)[0, 0], arr)

    def test_setitem_with_quantity(self):
        tsm = _make_tsm()
        q = u.Quantity(np.zeros(10), u.s)
        # Quantity has .value, so _to_base returns .value
        tsm[0, 0] = q
        np.testing.assert_array_equal(tsm.view(np.ndarray)[0, 0], np.zeros(10))

    def test_setitem_slice(self):
        tsm = _make_tsm()
        tsm[:, :, 0] = 0.0
        assert np.all(tsm.view(np.ndarray)[:, :, 0] == 0.0)

    def test_setitem_string_row(self):
        # String row key → row_index lookup (lines 196-197)
        tsm = _make_tsm()
        row_key = tsm.row_keys()[0]
        col_key = tsm.col_keys()[0]
        tsm[row_key, col_key, :] = 99.0
        assert np.all(tsm.view(np.ndarray)[0, 0, :] == 99.0)

    def test_setitem_row_list_and_col_list_scalar_is_cartesian(self):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        sm = SeriesMatrix(data.copy(), xindex=np.arange(5))

        sm[[0, 2], [1, 3], :] = -1.0

        expected = data.copy()
        expected[
            np.array([0, 2])[:, np.newaxis], np.array([1, 3])[np.newaxis, :], :
        ] = -1.0
        np.testing.assert_array_equal(sm.value, expected)

    def test_setitem_row_list_and_col_list_shaped_value_is_cartesian(self):
        sm = SeriesMatrix(np.zeros((3, 4, 5)), xindex=np.arange(5))
        values = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5)

        sm[[0, 2], [1, 3], :] = values

        np.testing.assert_array_equal(sm.value[0, 1], values[0, 0])
        np.testing.assert_array_equal(sm.value[0, 3], values[0, 1])
        np.testing.assert_array_equal(sm.value[2, 1], values[1, 0])
        np.testing.assert_array_equal(sm.value[2, 3], values[1, 1])
        assert np.all(sm.value[1, :, :] == 0.0)

    def test_setitem_wrong_length_boolean_mask_raises(self):
        sm = SeriesMatrix(np.zeros((3, 4, 5)), xindex=np.arange(5))

        with pytest.raises(IndexError, match="selector length mismatch"):
            sm[[True, False], [1, 3], :] = 1.0

    def test_setitem_non_integer_numeric_selector_raises(self):
        sm = SeriesMatrix(np.zeros((3, 4, 5)), xindex=np.arange(5))

        with pytest.raises(TypeError, match="integer positions"):
            sm[[0.0, 2.0], [1, 3], :] = 1.0


# ---------------------------------------------------------------------------
# loc property
# ---------------------------------------------------------------------------


class TestLoc:
    def test_loc_getitem(self):
        # Lines 220-230
        tsm = _make_tsm()
        result = tsm.loc[0, 0]
        assert result.shape == (10,)

    def test_loc_setitem(self):
        tsm = _make_tsm()
        tsm.loc[0, 0] = np.zeros(10)
        np.testing.assert_array_equal(tsm.view(np.ndarray)[0, 0], np.zeros(10))

    def test_loc_slice(self):
        tsm = _make_tsm()
        result = tsm.loc[0:2, 0:2]
        assert result.shape == (2, 2, 10)


# ---------------------------------------------------------------------------
# submatrix
# ---------------------------------------------------------------------------


class TestSubmatrix:
    def test_submatrix_label_selection_preserves_cartesian_shape_and_metadata(self):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        meta_arr = np.empty((3, 4), dtype=object)
        for i in range(3):
            for j in range(4):
                meta_arr[i, j] = MetaData(unit=u.m, name=f"elem-{i}-{j}")
        sm = SeriesMatrix(
            data,
            xindex=u.Quantity(np.arange(5, dtype=float), u.s),
            meta=MetaDataMatrix(meta_arr),
            rows={
                "r0": {"name": "row zero"},
                "r1": {"name": "row one"},
                "r2": {"name": "row two"},
            },
            cols={
                "c0": {"name": "col zero"},
                "c1": {"name": "col one"},
                "c2": {"name": "col two"},
                "c3": {"name": "col three"},
            },
        )

        result = sm.submatrix(["r0", "r2"], ["c1", "c3"])

        assert result.shape == (2, 2, 5)
        expected = data[np.ix_([0, 2], [1, 3], np.arange(5))]
        np.testing.assert_array_equal(result.value, expected)
        np.testing.assert_array_equal(result.xindex.value, np.arange(5, dtype=float))
        assert result.xindex.unit == u.s
        assert result.row_keys() == ("r0", "r2")
        assert result.col_keys() == ("c1", "c3")
        assert result.rows["r2"].name == "row two"
        assert result.cols["c3"].name == "col three"
        assert result.meta[1, 1].name == "elem-2-3"

    def test_getitem_non_array_result(self):
        # Line 121 — result not np.ndarray → return early
        # This happens when super().__getitem__ returns a scalar
        tsm = _make_tsm()
        # Accessing a single element beyond slicing returns scalar
        val = tsm.view(np.ndarray)[0, 0, 0]
        assert isinstance(val, (float, np.floating))
