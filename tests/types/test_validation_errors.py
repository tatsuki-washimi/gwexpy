from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.metadata import MetaData, MetaDataMatrix
from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.seriesmatrix_validation import (
    check_add_sub_compatibility,
    check_labels_unique,
    check_no_nan_inf,
    check_shape_xindex_compatibility,
    check_xindex_monotonic,
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
