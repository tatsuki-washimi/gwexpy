"""SeriesMatrix HDF5 ``attrs`` serialization must not drop entries silently (#777)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.timeseries import TimeSeriesMatrix


def _make_tsm(attrs=None):
    data = np.arange(2 * 2 * 8, dtype=float).reshape(2, 2, 8) * 1e-21
    return TimeSeriesMatrix(data, t0=0.0, dt=1.0, attrs=attrs)


SUPPORTED_ATTRS = {
    "run": "O4c",
    "segment": 7,
    "scale": 1.5e-21,
    "flag": True,
    "missing": None,
    "channels": ["K1:A", "K1:B"],
    "nested": {"window": "hann", "overlap": 0.5},
}


def test_supported_attrs_roundtrip_unchanged(tmp_path):
    target = tmp_path / "supported.h5"
    tsm = _make_tsm(attrs=dict(SUPPORTED_ATTRS))

    tsm.write(target, format="hdf5")
    back = TimeSeriesMatrix.read(target, format="hdf5")

    assert back.attrs == SUPPORTED_ATTRS
    np.testing.assert_array_equal(back.value, tsm.value)
    np.testing.assert_array_equal(back.xindex.value, tsm.xindex.value)
    assert back.xindex.unit == tsm.xindex.unit


@pytest.mark.parametrize("attrs", [None, {}], ids=["none", "empty"])
def test_empty_attrs_still_write(tmp_path, attrs):
    target = tmp_path / "empty.h5"
    tsm = _make_tsm(attrs=attrs)

    tsm.write(target, format="hdf5")
    back = TimeSeriesMatrix.read(target, format="hdf5")

    assert not back.attrs
    np.testing.assert_array_equal(back.value, tsm.value)


@pytest.mark.parametrize(
    "bad",
    [
        u.Quantity(5, "m"),
        np.arange(3),
        {1, 2},
        object(),
    ],
    ids=["quantity", "ndarray", "set", "object"],
)
def test_unsupported_attr_raises_instead_of_dropping(tmp_path, bad):
    target = tmp_path / "unsupported.h5"
    tsm = _make_tsm(attrs={"run": "O4c", "cal_ref": bad})

    with pytest.raises(TypeError, match="cal_ref"):
        tsm.write(target, format="hdf5")

    assert not target.exists()


def test_unsupported_attr_to_hdf5_direct_call_raises(tmp_path):
    target = tmp_path / "direct.h5"
    tsm = _make_tsm(attrs={"cal_ref": u.Quantity(5, "m")})

    with pytest.raises(TypeError, match="attrs"):
        tsm.to_hdf5(target)

    assert not target.exists()


def test_unsupported_attr_leaves_existing_file_intact(tmp_path):
    target = tmp_path / "existing.h5"
    original = _make_tsm(attrs={"run": "O4c"})
    original.write(target, format="hdf5")
    before = target.read_bytes()

    with pytest.raises(TypeError):
        _make_tsm(attrs={"cal_ref": u.Quantity(5, "m")}).write(target, format="hdf5")

    assert target.read_bytes() == before
    back = TimeSeriesMatrix.read(target, format="hdf5")
    assert back.attrs == {"run": "O4c"}


def test_circular_attrs_raise_instead_of_dropping(tmp_path):
    target = tmp_path / "circular.h5"
    loop: dict = {}
    loop["self"] = loop

    with pytest.raises(TypeError, match="attrs"):
        _make_tsm(attrs=loop).write(target, format="hdf5")

    assert not target.exists()


def test_frequencyseriesmatrix_shares_the_fail_closed_contract(tmp_path):
    target = tmp_path / "fsm.h5"
    data = np.ones((1, 2, 4), dtype=complex)
    fsm = FrequencySeriesMatrix(
        data, f0=0.0, df=1.0, attrs={"cal_ref": u.Quantity(5, "m")}
    )

    with pytest.raises(TypeError, match="cal_ref"):
        fsm.write(target, format="hdf5")

    assert not target.exists()


def test_existing_layout_and_metadata_unchanged(tmp_path):
    target = tmp_path / "layout.h5"
    tsm = _make_tsm(attrs={"run": "O4c"})

    tsm.write(target, format="hdf5")

    with h5py.File(target, "r") as h5f:
        assert set(h5f) == {"data", "xindex", "meta", "rows", "cols"}
        assert h5f.attrs["attrs_json"] == '{"run": "O4c"}'
        assert h5f["meta/units"].shape == (2, 2)
