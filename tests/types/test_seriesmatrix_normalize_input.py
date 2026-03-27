"""Tests for seriesmatrix_validation._normalize_input and related handlers."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwpy.types.series import Series

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.seriesmatrix_validation import (
    _broadcast_attr,
    _detect_input_type,
    _expand_key,
    _infer_length_from_items,
    _normalize_input,
    _resolve_scalar_shape,
    _slice_metadata_dict,
    check_labels_unique,
    to_series,
)
from gwexpy.types.metadata import MetaData, MetaDataDict


# ---------------------------------------------------------------------------
# _detect_input_type
# ---------------------------------------------------------------------------

def test_detect_none():
    assert _detect_input_type(None) == "none"


def test_detect_scalar_int():
    assert _detect_input_type(3) == "scalar"


def test_detect_scalar_float():
    assert _detect_input_type(3.14) == "scalar"


def test_detect_scalar_quantity():
    assert _detect_input_type(1.0 * u.m) == "scalar_quantity"


def test_detect_series():
    s = Series(np.ones(5), xindex=np.arange(5))
    assert _detect_input_type(s) == "series"


def test_detect_ndarray_1d():
    assert _detect_input_type(np.ones(5)) == "ndarray_1d_2d"


def test_detect_ndarray_2d():
    assert _detect_input_type(np.ones((3, 4))) == "ndarray_1d_2d"


def test_detect_ndarray_3d():
    assert _detect_input_type(np.ones((2, 3, 4))) == "ndarray_3d"


def test_detect_dict():
    assert _detect_input_type({"a": [1, 2]}) == "dict"


def test_detect_list():
    assert _detect_input_type([1, 2, 3]) == "list"


def test_detect_unknown():
    assert _detect_input_type(object()) == "unknown"


def test_detect_quantity_3d():
    q = u.Quantity(np.ones((2, 3, 4)), u.m)
    assert _detect_input_type(q) == "ndarray_3d"


# ---------------------------------------------------------------------------
# _normalize_input — none
# ---------------------------------------------------------------------------

def test_normalize_none():
    arr, attrs, xi = _normalize_input(None)
    assert arr.shape == (0, 0, 0)


# ---------------------------------------------------------------------------
# _normalize_input — scalar
# ---------------------------------------------------------------------------

def test_normalize_scalar():
    arr, attrs, xi = _normalize_input(3.0)
    assert arr.shape == (1, 1, 1)
    assert arr[0, 0, 0] == 3.0


def test_normalize_scalar_with_xindex():
    xindex = np.arange(5)
    arr, attrs, xi = _normalize_input(2.0, xindex=xindex)
    assert arr.shape == (1, 1, 5)
    assert np.all(arr == 2.0)


def test_normalize_scalar_with_shape():
    arr, attrs, xi = _normalize_input(1.0, shape=(2, 3, 4))
    assert arr.shape == (2, 3, 4)


def test_normalize_scalar_with_shape_2d():
    arr, attrs, xi = _normalize_input(1.0, shape=(2, 3))
    assert arr.shape == (2, 3, 1)


def test_normalize_scalar_with_shape_invalid():
    with pytest.raises(ValueError, match="shape must be 2D or 3D"):
        _normalize_input(1.0, shape=(1,))


# ---------------------------------------------------------------------------
# _normalize_input — scalar_quantity
# ---------------------------------------------------------------------------

def test_normalize_scalar_quantity():
    arr, attrs, xi = _normalize_input(2.0 * u.m)
    assert arr.shape == (1, 1, 1)
    assert arr[0, 0, 0] == pytest.approx(2.0)


def test_normalize_scalar_quantity_with_unit_conversion():
    arr, attrs, xi = _normalize_input(1.0 * u.km, units=u.m)
    assert arr[0, 0, 0] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# _normalize_input — Series
# ---------------------------------------------------------------------------

def test_normalize_series():
    s = Series(np.array([1.0, 2.0, 3.0]), xindex=np.arange(3), unit=u.m)
    arr, attrs, xi = _normalize_input(s)
    assert arr.shape == (1, 1, 3)
    np.testing.assert_allclose(arr[0, 0], [1.0, 2.0, 3.0])


def test_normalize_series_with_unit_conversion():
    s = Series(np.array([1.0, 2.0]), xindex=np.arange(2), unit=u.km)
    arr, attrs, xi = _normalize_input(s, units=u.m)
    np.testing.assert_allclose(arr[0, 0], [1000.0, 2000.0])


# ---------------------------------------------------------------------------
# _normalize_input — ndarray 1d/2d
# ---------------------------------------------------------------------------

def test_normalize_ndarray_1d():
    arr, attrs, xi = _normalize_input(np.array([1.0, 2.0, 3.0]))
    assert arr.shape == (1, 1, 3)


def test_normalize_ndarray_2d():
    data = np.ones((3, 5))
    arr, attrs, xi = _normalize_input(data)
    assert arr.shape == (3, 1, 5)


def test_normalize_quantity_1d():
    q = u.Quantity([1.0, 2.0], u.m)
    arr, attrs, xi = _normalize_input(q, units=u.cm)
    np.testing.assert_allclose(arr[0, 0], [100.0, 200.0])


# ---------------------------------------------------------------------------
# _normalize_input — ndarray 3d
# ---------------------------------------------------------------------------

def test_normalize_ndarray_3d():
    data = np.ones((2, 3, 5))
    arr, attrs, xi = _normalize_input(data)
    assert arr.shape == (2, 3, 5)


def test_normalize_quantity_3d_with_unit():
    q = u.Quantity(np.ones((1, 1, 4)), u.km)
    arr, attrs, xi = _normalize_input(q, units=u.m)
    np.testing.assert_allclose(arr[0, 0], [1000.0] * 4)


# ---------------------------------------------------------------------------
# _normalize_input — list
# ---------------------------------------------------------------------------

def test_normalize_empty_list():
    arr, attrs, xi = _normalize_input([])
    assert arr.shape == (0, 0, 0)


def test_normalize_list_1d():
    xindex = np.arange(5, dtype=float)
    arr, attrs, xi = _normalize_input([1.0, 2.0, 3.0], xindex=xindex)
    assert arr.shape == (3, 1, 5)


def test_normalize_list_2d():
    xindex = np.arange(5, dtype=float)
    arr, attrs, xi = _normalize_input([[1.0, 2.0], [3.0, 4.0]], xindex=xindex)
    assert arr.shape == (2, 2, 5)


def test_normalize_list_ragged_raises():
    with pytest.raises(ValueError, match="Ragged"):
        _normalize_input([[1.0, 2.0], [3.0]])


def test_normalize_list_of_series():
    s1 = Series(np.array([1.0, 2.0]), xindex=np.arange(2))
    s2 = Series(np.array([3.0, 4.0]), xindex=np.arange(2))
    arr, attrs, xi = _normalize_input([[s1, s2]])
    assert arr.shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# _normalize_input — dict
# ---------------------------------------------------------------------------

def test_normalize_dict():
    data = {"row0": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
    arr, attrs, xi = _normalize_input(data)
    assert arr.shape == (1, 2, 2)


def test_normalize_dict_with_dx_x0():
    data = {"row0": [np.array([1.0, 2.0, 3.0])]}
    arr, attrs, xi = _normalize_input(data, dx=0.1, x0=0.0)
    assert arr.shape == (1, 1, 3)


# ---------------------------------------------------------------------------
# _normalize_input — seriesmatrix
# ---------------------------------------------------------------------------

def test_normalize_seriesmatrix():
    sm = SeriesMatrix(np.ones((2, 2, 5)), xindex=np.arange(5))
    arr, attrs, xi = _normalize_input(sm)
    assert arr.shape == (2, 2, 5)


# ---------------------------------------------------------------------------
# _normalize_input — unknown type raises
# ---------------------------------------------------------------------------

def test_normalize_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported data type"):
        _normalize_input(object())


# ---------------------------------------------------------------------------
# _resolve_scalar_shape
# ---------------------------------------------------------------------------

def test_resolve_scalar_shape_none_none():
    assert _resolve_scalar_shape(None, None) == (1, 1, 1)


def test_resolve_scalar_shape_with_xindex():
    xindex = np.arange(10)
    assert _resolve_scalar_shape(None, xindex) == (1, 1, 10)


def test_resolve_scalar_shape_2d():
    assert _resolve_scalar_shape((3, 4), None) == (3, 4, 1)


def test_resolve_scalar_shape_3d():
    assert _resolve_scalar_shape((3, 4, 5), None) == (3, 4, 5)


# ---------------------------------------------------------------------------
# _broadcast_attr
# ---------------------------------------------------------------------------

def test_broadcast_attr_none():
    assert _broadcast_attr(None, (2, 2), "units") is None


def test_broadcast_attr_scalar():
    result = _broadcast_attr(u.m, (2, 3), "units")
    assert result.shape == (2, 3)


def test_broadcast_attr_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        _broadcast_attr(np.array([u.m, u.s, u.kg]), (2, 2), "units")


# ---------------------------------------------------------------------------
# _expand_key
# ---------------------------------------------------------------------------

def test_expand_key_single_int():
    result = _expand_key(0, 3)
    assert result == (0, slice(None), slice(None))


def test_expand_key_tuple():
    result = _expand_key((1, 2), 3)
    assert result == (1, 2, slice(None))


def test_expand_key_ellipsis():
    result = _expand_key((Ellipsis, 2), 3)
    assert result[2] == 2


# ---------------------------------------------------------------------------
# _infer_length_from_items
# ---------------------------------------------------------------------------

def test_infer_length_from_series():
    s = Series(np.ones(7), xindex=np.arange(7))
    assert _infer_length_from_items([s], None) == 7


def test_infer_length_from_quantity():
    q = u.Quantity(np.ones(5), u.m)
    assert _infer_length_from_items([q], None) == 5


def test_infer_length_from_ndarray():
    arr = np.ones(4)
    assert _infer_length_from_items([arr], None) == 4


def test_infer_length_from_shape():
    assert _infer_length_from_items([1.0], (2, 3, 8)) == 8


def test_infer_length_scalar_no_shape():
    assert _infer_length_from_items([1.0], None) == 1


# ---------------------------------------------------------------------------
# _slice_metadata_dict
# ---------------------------------------------------------------------------

def test_slice_metadata_dict_by_slice():
    meta = MetaDataDict(
        {"a": MetaData(), "b": MetaData(), "c": MetaData()},
        expected_size=3,
        key_prefix="row",
    )
    result = _slice_metadata_dict(meta, slice(0, 2), "row")
    assert len(result) == 2


def test_slice_metadata_dict_by_list():
    meta = MetaDataDict(
        {"a": MetaData(), "b": MetaData(), "c": MetaData()},
        expected_size=3,
        key_prefix="row",
    )
    result = _slice_metadata_dict(meta, [0, 2], "row")
    assert len(result) == 2


def test_slice_metadata_dict_by_int():
    meta = MetaDataDict(
        {"a": MetaData(), "b": MetaData()},
        expected_size=2,
        key_prefix="row",
    )
    result = _slice_metadata_dict(meta, 1, "row")
    assert len(result) == 1


def test_slice_metadata_dict_fallback():
    meta = MetaDataDict(
        {"a": MetaData()},
        expected_size=1,
        key_prefix="row",
    )
    # Passing an unsupported key type → fallback returns original
    result = _slice_metadata_dict(meta, "unsupported", "row")
    assert result is meta


# ---------------------------------------------------------------------------
# to_series — additional branches
# ---------------------------------------------------------------------------

def test_to_series_scalar_quantity():
    xindex = np.arange(5, dtype=float)
    s = to_series(2.0 * u.m, xindex=xindex)
    assert len(s) == 5
    np.testing.assert_allclose(s.value, 2.0)


def test_to_series_scalar_quantity_no_xindex_raises():
    with pytest.raises(ValueError, match="without xindex"):
        to_series(2.0 * u.m, xindex=None)


def test_to_series_quantity_array():
    q = u.Quantity([1.0, 2.0, 3.0], u.m)
    s = to_series(q, xindex=np.arange(3))
    assert len(s) == 3


def test_to_series_scalar_no_xindex_raises():
    with pytest.raises(ValueError, match="without xindex"):
        to_series(3.14, xindex=None)


def test_to_series_scalar_with_xindex():
    s = to_series(5.0, xindex=np.arange(4))
    assert len(s) == 4
    np.testing.assert_allclose(s.value, 5.0)


def test_to_series_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported element type"):
        to_series({"a": 1}, xindex=np.arange(3))


