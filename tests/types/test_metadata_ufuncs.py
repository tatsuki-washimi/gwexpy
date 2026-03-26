from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix, get_unit


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


# --- MetaData: repr / str ---

def test_metadata_repr():
    m = MetaData(name="ch1", unit=u.m)
    r = repr(m)
    assert "ch1" in r
    assert "m" in r


def test_metadata_str():
    m = MetaData(name="sig", unit=u.Hz)
    s = str(m)
    assert "sig" in s


def test_metadata_repr_html():
    m = MetaData(name="x", unit=u.s)
    h = m._repr_html_()
    assert "<table>" in h


# --- MetaData: unary operators ---

def test_metadata_abs():
    m = MetaData(name="v", unit=u.m)
    out = abs(m)
    assert isinstance(out, MetaData)
    assert out.unit == u.m


def test_metadata_neg():
    m = MetaData(name="v", unit=u.m)
    out = -m
    assert isinstance(out, MetaData)
    assert out.unit == u.m


def test_metadata_pos():
    m = MetaData(name="v", unit=u.m)
    out = +m
    assert isinstance(out, MetaData)
    assert out.unit == u.m


# --- MetaData: sub / div ---

def test_metadata_sub_same_unit():
    m1 = MetaData(name="a", unit=u.m)
    m2 = MetaData(name="b", unit=u.m)
    out = m1 - m2
    assert isinstance(out, MetaData)
    assert out.unit == u.m


def test_metadata_truediv():
    m1 = MetaData(name="a", unit=u.m)
    m2 = MetaData(name="b", unit=u.s)
    out = m1 / m2
    assert isinstance(out, MetaData)
    assert out.unit == u.m / u.s


def test_metadata_rsub():
    m = MetaData(name="a", unit=u.dimensionless_unscaled)
    out = 5 - m
    assert isinstance(out, MetaData)


def test_metadata_rtruediv():
    m = MetaData(name="a", unit=u.m)
    out = (1 * u.s) / m
    assert isinstance(out, MetaData)
    assert out.unit == u.s / u.m


# --- MetaData: numpy ufuncs (sqrt, square, comparison) ---

def test_metadata_sqrt():
    m = MetaData(name="a", unit=u.m**2)
    out = np.sqrt(m)
    assert isinstance(out, MetaData)
    assert out.unit.is_equivalent(u.m)


def test_metadata_square():
    m = MetaData(name="a", unit=u.m)
    out = np.square(m)
    assert isinstance(out, MetaData)
    assert out.unit.is_equivalent(u.m**2)


def test_metadata_comparison_ufunc():
    m1 = MetaData(name="a", unit=u.m)
    m2 = MetaData(name="b", unit=u.km)  # same dimension
    out = np.less(m1, m2)
    assert isinstance(out, MetaData)
    assert out.unit.is_equivalent(u.dimensionless_unscaled)


# --- MetaData: from_series / as_meta ---

def test_metadata_from_series():
    class _FakeSeries:
        name = "ch"
        channel = "L1:DARM"
        unit = u.m

    m = MetaData.from_series(_FakeSeries())
    assert m.name == "ch"
    assert m.unit == u.m


def test_metadata_as_meta_passthrough():
    m = MetaData(name="a", unit=u.m)
    out = m.as_meta(m)
    assert out is m


def test_metadata_as_meta_from_quantity():
    m = MetaData(name="a", unit=u.m)
    out = m.as_meta(3.0 * u.s)
    assert isinstance(out, MetaData)
    assert out.unit == u.s


# --- MetaData: setters ---

def test_metadata_unit_setter_string():
    m = MetaData()
    m.unit = "m"
    assert m.unit == u.m


def test_metadata_unit_setter_invalid_falls_back():
    m = MetaData()
    m.unit = "NOT_A_UNIT_XYZ"
    assert m.unit == u.dimensionless_unscaled


def test_metadata_channel_setter_invalid_falls_back():
    m = MetaData()
    m.channel = None
    # Should not raise


# --- MetaDataDict: construction ---

def test_metadatadict_from_list_of_strings():
    mdd = MetaDataDict(["ch0", "ch1", "ch2"])
    assert list(mdd.keys()) == ["ch0", "ch1", "ch2"]
    assert all(isinstance(v, MetaData) for v in mdd.values())


def test_metadatadict_from_list_of_dicts():
    mdd = MetaDataDict([{"name": "a", "unit": u.m}, {"name": "b", "unit": u.s}])
    assert mdd["key0"].name == "a"
    assert mdd["key1"].unit == u.s


def test_metadatadict_from_list_duplicate_keys_raises():
    with pytest.raises(ValueError, match="Duplicate"):
        MetaDataDict(["ch0", "ch0"])


def test_metadatadict_from_metadatadict():
    orig = MetaDataDict({"a": MetaData(unit=u.m)})
    copy = MetaDataDict(orig)
    assert copy["a"].unit == u.m


def test_metadatadict_invalid_type_raises():
    with pytest.raises(TypeError):
        MetaDataDict(42)


def test_metadatadict_expected_size_mismatch_raises():
    with pytest.raises(ValueError):
        MetaDataDict(["a", "b"], expected_size=3)


# --- MetaDataDict: properties ---

def test_metadatadict_names_channels_units():
    mdd = MetaDataDict({"x": MetaData(name="x", unit=u.m), "y": MetaData(name="y", unit=u.s)})
    assert mdd.names == ["x", "y"]
    assert mdd.units == [u.m, u.s]


def test_metadatadict_to_dataframe():
    pd = pytest.importorskip("pandas")
    mdd = MetaDataDict({"a": MetaData(name="a", unit=u.m)})
    df = mdd.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "a" in df.index


def test_metadatadict_repr():
    mdd = MetaDataDict({"a": MetaData()})
    r = repr(mdd)
    assert "MetaDataDict" in r


def test_metadatadict_repr_html():
    pytest.importorskip("pandas")
    mdd = MetaDataDict({"a": MetaData(unit=u.m)})
    h = mdd._repr_html_()
    assert "<table" in h


# --- MetaDataDict: arithmetic ---

def test_metadatadict_sub():
    mdd1 = MetaDataDict({"a": MetaData(unit=u.m)})
    mdd2 = MetaDataDict({"a": MetaData(unit=u.m)})
    out = mdd1 - mdd2
    assert isinstance(out, MetaDataDict)


def test_metadatadict_mul_scalar():
    mdd = MetaDataDict({"a": MetaData(unit=u.m)})
    out = mdd * MetaData(unit=u.s)
    assert isinstance(out, MetaDataDict)
    assert out["a"].unit == u.m * u.s


def test_metadatadict_truediv():
    mdd = MetaDataDict({"a": MetaData(unit=u.m)})
    out = mdd / MetaData(unit=u.s)
    assert isinstance(out, MetaDataDict)


def test_metadatadict_pow():
    mdd = MetaDataDict({"a": MetaData(unit=u.m)})
    out = mdd**2
    assert isinstance(out, MetaDataDict)
    assert out["a"].unit.is_equivalent(u.m**2)


def test_metadatadict_key_mismatch_raises():
    mdd1 = MetaDataDict({"a": MetaData(unit=u.m)})
    mdd2 = MetaDataDict({"b": MetaData(unit=u.m)})
    with pytest.raises(ValueError, match="Keys"):
        _ = mdd1 + mdd2


def test_metadatadict_from_series():
    class _FakeSeries:
        name = "ch"
        channel = ""
        unit = u.m

    mdd = MetaDataDict.from_series({"x": _FakeSeries(), "y": _FakeSeries()})
    assert isinstance(mdd, MetaDataDict)
    assert mdd["x"].unit == u.m


def test_metadatadict_from_series_list():
    class _FakeSeries:
        name = "ch"
        channel = ""
        unit = u.s

    mdd = MetaDataDict.from_series([_FakeSeries()])
    assert mdd["key0"].unit == u.s


def test_metadatadict_from_series_invalid_type_raises():
    with pytest.raises(TypeError):
        MetaDataDict.from_series(42)


# --- get_unit ---

def test_get_unit_metadata():
    m = MetaData(unit=u.m)
    assert get_unit(m) == u.m


def test_get_unit_quantity():
    assert get_unit(3.0 * u.s) == u.s


def test_get_unit_plain_number():
    assert get_unit(5.0) == u.dimensionless_unscaled
