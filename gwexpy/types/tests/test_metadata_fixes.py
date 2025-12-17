import pytest
import numpy as np
from astropy import units as u

from gwexpy.types.metadata import MetaData


def test_metadata_right_hand_operations_dimensionless():
    """右側演算で MetaData の name/channel を保持し例外にならないこと"""
    m = MetaData(name="a", unit=u.dimensionless_unscaled)

    out1 = 2 + m
    out2 = 2 * m
    out3 = m + 2
    out4 = m * 2

    for out in (out1, out2, out3, out4):
        assert isinstance(out, MetaData)
        assert out.name == "a"
        # channel は Channel("") だが比較は文字列で十分
        assert str(out.channel) == ""
        assert out.unit.is_equivalent(u.dimensionless_unscaled)


def test_metadata_right_hand_unit_mismatch_raises():
    """非互換単位の加算は UnitConversionError を送出"""
    m = MetaData(name="a", unit=u.m)
    with pytest.raises(u.UnitConversionError):
        _ = 2 + m


def test_metadata_right_hand_quantity_left():
    """Quantity 左側でも MetaData 情報を保持して演算できる"""
    m = MetaData(name="a", unit=u.dimensionless_unscaled)
    out = (3 * u.m) * m
    assert isinstance(out, MetaData)
    assert out.unit == u.m
    assert out.name == "a"
