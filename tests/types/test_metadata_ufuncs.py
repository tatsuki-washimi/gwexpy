from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix


def test_metadata_add_same_unit():
    m1 = MetaData(name="a", unit=u.m)
    m2 = MetaData(name="b", unit=u.m)
    out = m1 + m2
    assert isinstance(out, MetaData)
    assert out.unit == u.m
    assert out.name == "a"


def test_metadata_add_unit_mismatch_raises():
    m1 = MetaData(name="a", unit=u.m)
    m2 = MetaData(name="b", unit=u.s)
    with pytest.raises(u.UnitConversionError):
        _ = m1 + m2


def test_metadata_multiply_units():
    m1 = MetaData(name="a", unit=u.m)
    m2 = MetaData(name="b", unit=u.s)
    out = m1 * m2
    assert out.unit == u.m * u.s


def test_metadata_power_dimensionless_exponent():
    m = MetaData(name="a", unit=u.m)
    out = m**2
    assert out.unit == u.m**2

    out_q = m ** (2 * u.dimensionless_unscaled)
    assert out_q.unit == u.m**2


def test_metadata_transcendental_requires_dimensionless():
    m = MetaData(name="a", unit=u.m)
    with pytest.raises(u.UnitConversionError):
        _ = np.exp(m)

    dim = MetaData(name="b", unit=u.dimensionless_unscaled)
    out = np.exp(dim)
    assert out.unit.is_equivalent(u.dimensionless_unscaled)


def test_metadata_dict_ufunc_multiply_scalar():
    mdict = MetaDataDict({"a": MetaData(unit=u.m), "b": MetaData(unit=u.s)})
    out = np.multiply(mdict, 2)
    assert isinstance(out, MetaDataDict)
    assert out["a"].unit == u.m
    assert out["b"].unit == u.s


def test_metadata_matrix_ufunc_multiply_scalar():
    mat = MetaDataMatrix([[MetaData(unit=u.m)], [MetaData(unit=u.s)]])
    out = np.multiply(mat, 3)
    assert isinstance(out, MetaDataMatrix)
    assert out[0, 0].unit == u.m
    assert out[1, 0].unit == u.s
