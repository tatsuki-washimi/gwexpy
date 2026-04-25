import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix


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


# ---------------------------------------------------------------------------
# MetaDataMatrix setters: shape mismatch must raise ValueError
# ---------------------------------------------------------------------------

def test_metadatamatrix_names_setter_size_mismatch_raises():
    mat = MetaDataMatrix(shape=(2, 3))
    with pytest.raises(ValueError, match="names"):
        mat.names = np.array(["a", "b"])  # 2 != 6


def test_metadatamatrix_units_setter_size_mismatch_raises():
    mat = MetaDataMatrix(shape=(2, 3))
    with pytest.raises(ValueError, match="units"):
        mat.units = np.array([u.m, u.s])  # 2 != 6


def test_metadatamatrix_channels_setter_size_mismatch_raises():
    mat = MetaDataMatrix(shape=(2, 3))
    with pytest.raises(ValueError, match="channels"):
        mat.channels = np.array(["H1:X"])  # 1 != 6


def test_metadatamatrix_names_setter_flat_array_reshapes():
    """A flat array whose size matches is accepted and applied in row-major order."""
    mat = MetaDataMatrix(shape=(2, 3))
    mat.names = np.array(["a", "b", "c", "d", "e", "f"])
    assert mat.names[0, 0] == "a"
    assert mat.names[0, 2] == "c"
    assert mat.names[1, 2] == "f"


def test_metadatamatrix_units_setter_flat_array_reshapes():
    mat = MetaDataMatrix(shape=(2, 2))
    mat.units = np.array([u.m, u.s, u.Hz, u.kg])
    assert mat[0, 0].unit == u.m
    assert mat[1, 1].unit == u.kg


def test_metadatamatrix_channels_setter_flat_array_reshapes():
    mat = MetaDataMatrix(shape=(1, 2))
    mat.channels = np.array(["H1:A", "L1:B"])
    assert str(mat[0, 0].channel) == "H1:A"
    assert str(mat[0, 1].channel) == "L1:B"


def test_metadatamatrix_names_setter_exact_shape():
    mat = MetaDataMatrix(shape=(2, 2))
    mat.names = np.array([["x", "y"], ["z", "w"]])
    assert mat.names[0, 1] == "y"
    assert mat.names[1, 0] == "z"


# ---------------------------------------------------------------------------
# MetaDataMatrix ufunc: shape mismatch between two matrices raises ValueError
# ---------------------------------------------------------------------------

def test_metadatamatrix_ufunc_shape_mismatch_raises():
    m1 = MetaDataMatrix(shape=(2, 2))
    m2 = MetaDataMatrix(shape=(2, 3))
    with pytest.raises(ValueError, match="[Ss]hape"):
        np.multiply(m1, m2)


# ---------------------------------------------------------------------------
# floor_divide unit propagation
# ---------------------------------------------------------------------------

def test_metadata_floor_divide_unit():
    m1 = MetaData(name="a", unit=u.m)
    m2 = MetaData(name="b", unit=u.s)
    out = np.floor_divide(m1, m2)
    assert isinstance(out, MetaData)
    assert out.unit.is_equivalent(u.m / u.s)


# ---------------------------------------------------------------------------
# CSV round-trip: MetaDataMatrix
# ---------------------------------------------------------------------------

def test_metadatamatrix_csv_roundtrip(tmp_path):
    mat = MetaDataMatrix([
        [MetaData(name="a", unit=u.m, channel="H1:A"),
         MetaData(name="b", unit=u.s, channel="L1:B")],
        [MetaData(name="c", unit=u.Hz, channel="H1:C"),
         MetaData(name="d", unit=u.dimensionless_unscaled, channel="")],
    ])
    path = tmp_path / "mat.csv"
    mat.write(str(path))
    mat2 = MetaDataMatrix.read(str(path))

    assert mat2.shape == (2, 2)
    assert mat2[0, 0].name == "a"
    assert mat2[0, 0].unit.is_equivalent(u.m)
    assert mat2[0, 1].unit.is_equivalent(u.s)
    assert mat2[1, 0].unit.is_equivalent(u.Hz)
    assert mat2[1, 1].name == "d"


# ---------------------------------------------------------------------------
# CSV round-trip: MetaDataDict
# ---------------------------------------------------------------------------

def test_metadatadict_csv_roundtrip(tmp_path):
    pytest.importorskip("pandas")
    mdd = MetaDataDict({
        "x": MetaData(name="x", unit=u.m, channel="H1:X"),
        "y": MetaData(name="y", unit=u.s, channel="L1:Y"),
    })
    path = tmp_path / "mdd.csv"
    mdd.write(str(path))
    mdd2 = MetaDataDict.read(str(path))

    assert list(mdd2.keys()) == ["x", "y"]
    assert mdd2["x"].unit.is_equivalent(u.m)
    assert mdd2["x"].name == "x"
    assert mdd2["y"].unit.is_equivalent(u.s)
    assert mdd2["y"].name == "y"
