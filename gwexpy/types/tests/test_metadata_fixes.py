import pytest
from astropy import units as u

from gwexpy.types.metadata import MetaData


def test_metadata_right_hand_operations_dimensionless():
    """Ensure name/channel are preserved and no exception occurs during right-hand operations."""
    m = MetaData(name="a", unit=u.dimensionless_unscaled)

    out1 = 2 + m
    out2 = 2 * m
    out3 = m + 2
    out4 = m * 2

    for out in (out1, out2, out3, out4):
        assert isinstance(out, MetaData)
        assert out.name == "a"
        # channel is Channel(""), but string comparison is sufficient.
        assert str(out.channel) == ""
        assert out.unit.is_equivalent(u.dimensionless_unscaled)


def test_metadata_right_hand_unit_mismatch_raises():
    """Addition with incompatible units should raise UnitConversionError."""
    m = MetaData(name="a", unit=u.m)
    with pytest.raises(u.UnitConversionError):
        _ = 2 + m


def test_metadata_right_hand_quantity_left():
    """MetaData information is preserved even when a Quantity is on the left-hand side."""
    m = MetaData(name="a", unit=u.dimensionless_unscaled)
    out = (3 * u.m) * m
    assert isinstance(out, MetaData)
    assert out.unit == u.m
    assert out.name == "a"
