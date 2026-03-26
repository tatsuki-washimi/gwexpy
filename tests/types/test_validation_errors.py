from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.metadata import MetaData, MetaDataMatrix
from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.seriesmatrix_validation import (
    build_index_if_needed,
    check_add_sub_compatibility,
    check_epoch_and_sampling,
    check_labels_unique,
    check_no_nan_inf,
    check_shape_xindex_compatibility,
    check_unit_dimension_compatibility,
    check_xindex_monotonic,
    infer_xindex_from_items,
    to_series,
)


def _make_matrix(*, unit=u.m, xindex=None, data=None, meta=None):
    if data is None:
        data = np.zeros((2, 2, 3))
    if xindex is None:
        xindex = np.arange(data.shape[2])
    return SeriesMatrix(data, unit=unit, xindex=xindex, meta=meta)


def test_check_add_sub_compatibility_raises_on_unit_mismatch():
    sm1 = _make_matrix(unit=u.m)
    sm2 = _make_matrix(unit=u.s)
    with pytest.raises(u.UnitConversionError):
        check_add_sub_compatibility(sm1, sm2)


def test_check_shape_xindex_compatibility_raises_on_xindex():
    sm1 = _make_matrix(xindex=np.array([0, 1, 2]))
    sm2 = _make_matrix(xindex=np.array([0, 1, 3]))
    with pytest.raises(ValueError, match="xindex mismatch"):
        check_shape_xindex_compatibility(sm1, sm2)


def test_check_xindex_monotonic_raises():
    sm = _make_matrix(xindex=np.array([0, 2, 1]))
    with pytest.raises(ValueError, match="monotonic"):
        check_xindex_monotonic(sm)


def test_check_labels_unique_raises_on_duplicate_rows():
    class _Dummy:
        def row_keys(self):
            return ("row0", "row0")

        def col_keys(self):
            return ("col0",)

        meta = MetaDataMatrix([[MetaData()]])

    with pytest.raises(ValueError, match="Duplicate row labels"):
        check_labels_unique(_Dummy())


def test_check_no_nan_inf_raises():
    data = np.zeros((2, 2, 3))
    data[0, 0, 0] = np.nan
    sm = _make_matrix(data=data)
    with pytest.raises(ValueError, match="NaN"):
        check_no_nan_inf(sm)

    data = np.zeros((2, 2, 3))
    data[0, 0, 0] = np.inf
    sm = _make_matrix(data=data)
    with pytest.raises(ValueError, match="Inf"):
        check_no_nan_inf(sm)


# --- check_unit_dimension_compatibility ---

def test_check_unit_dimension_compatibility_ok():
    sm1 = _make_matrix(unit=u.m)
    sm2 = _make_matrix(unit=u.km)  # same dimension (length)
    assert check_unit_dimension_compatibility(sm1, sm2) is True


def test_check_unit_dimension_compatibility_raises():
    sm1 = _make_matrix(unit=u.m)
    sm2 = _make_matrix(unit=u.s)  # length vs time
    with pytest.raises(u.UnitConversionError):
        check_unit_dimension_compatibility(sm1, sm2)


def test_check_unit_dimension_compatibility_expected_dim():
    sm1 = _make_matrix(unit=u.m)
    with pytest.raises(u.UnitConversionError):
        check_unit_dimension_compatibility(sm1, expected_dim="time")


# --- check_epoch_and_sampling ---

def test_check_epoch_and_sampling_ok():
    class _FakeSM:
        def __init__(self, epoch, dx):
            self.epoch = epoch
            self.dx = dx

    assert check_epoch_and_sampling(_FakeSM(0.0, 1.0), _FakeSM(0.0, 1.0)) is True


def test_check_epoch_and_sampling_epoch_mismatch():
    class _FakeSM:
        def __init__(self, epoch, dx):
            self.epoch = epoch
            self.dx = dx

    with pytest.raises(ValueError, match="Epoch"):
        check_epoch_and_sampling(_FakeSM(0.0, 1.0), _FakeSM(1.0, 1.0))


def test_check_epoch_and_sampling_dx_mismatch():
    class _FakeSM:
        def __init__(self, epoch, dx):
            self.epoch = epoch
            self.dx = dx

    with pytest.raises(ValueError, match="dx"):
        check_epoch_and_sampling(_FakeSM(0.0, 1.0), _FakeSM(0.0, 2.0))


# --- check_xindex_monotonic: decreasing is ok ---

def test_check_xindex_monotonic_decreasing_ok():
    sm = _make_matrix(xindex=np.array([3, 2, 1]))
    assert check_xindex_monotonic(sm) is True


# --- check_add_sub_compatibility: shape mismatch ---

def test_check_add_sub_compatibility_raises_on_shape_mismatch():
    sm1 = SeriesMatrix(np.zeros((2, 1, 3)), unit=u.m, xindex=np.arange(3))
    sm2 = SeriesMatrix(np.zeros((3, 1, 3)), unit=u.m, xindex=np.arange(3))
    with pytest.raises(ValueError, match="Shape mismatch"):
        check_add_sub_compatibility(sm1, sm2)


# --- check_no_nan_inf: clean data returns True ---

def test_check_no_nan_inf_ok():
    sm = _make_matrix()
    assert check_no_nan_inf(sm) is True


# --- to_series ---

def test_to_series_from_ndarray():
    arr = np.array([1.0, 2.0, 3.0])
    xindex = np.array([0.0, 1.0, 2.0])
    result = to_series(arr, xindex)
    np.testing.assert_allclose(result.value, arr)


def test_to_series_from_scalar():
    xindex = np.array([0.0, 1.0, 2.0])
    result = to_series(5.0, xindex)
    assert np.all(result.value == 5.0)


def test_to_series_from_quantity_scalar():
    xindex = np.array([0.0, 1.0, 2.0])
    result = to_series(5.0 * u.m, xindex)
    assert np.all(result.value == 5.0)
    assert result.unit == u.m


def test_to_series_from_quantity_array():
    xindex = np.array([0.0, 1.0, 2.0])
    result = to_series(np.array([1.0, 2.0, 3.0]) * u.m, xindex)
    np.testing.assert_allclose(result.value, [1.0, 2.0, 3.0])


def test_to_series_scalar_no_xindex_raises():
    with pytest.raises(ValueError):
        to_series(5.0, None)


def test_to_series_unsupported_type_raises():
    with pytest.raises(TypeError):
        to_series({"a": 1}, np.array([0.0, 1.0]))


# --- build_index_if_needed ---

def test_build_index_if_needed_from_dx_x0():
    xindex = build_index_if_needed(None, dx=0.1, x0=0.0, xunit=None, length=10)
    assert len(xindex) == 10


def test_build_index_if_needed_existing_xindex():
    existing = np.arange(5)
    result = build_index_if_needed(existing, dx=None, x0=None, xunit=None, length=5)
    assert result is existing


def test_build_index_if_needed_no_args_raises():
    with pytest.raises(ValueError):
        build_index_if_needed(None, dx=None, x0=None, xunit=None, length=5)


# --- infer_xindex_from_items ---

def test_infer_xindex_from_items_none():
    result = infer_xindex_from_items([1, 2, 3])
    assert result is None
