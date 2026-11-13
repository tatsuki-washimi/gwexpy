"""HDF5 round-trip of ``FrequencySeries`` with an irregular frequency axis (#881)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)

N = 64
FREQS = np.geomspace(1.0, 10.0, N)
REAL = np.arange(N, dtype=np.float64) * 1e-21
COMPLEX = (np.arange(N, dtype=np.float64) + 1j * np.linspace(-1, 1, N)) * 1e-21


def _irregular(data, name="probe"):
    fs = FrequencySeries(data, frequencies=FREQS, unit="m/s2", name=name)
    assert not fs.xindex.regular
    return fs


def _assert_same(back, original):
    assert type(back) is type(original)
    assert back.dtype == original.dtype
    np.testing.assert_array_equal(back.value, original.value)
    np.testing.assert_array_equal(back.frequencies.value, original.frequencies.value)
    assert back.frequencies.unit == original.frequencies.unit
    assert back.unit == original.unit
    assert back.name == original.name
    assert not back.xindex.regular


@pytest.mark.parametrize("data", [REAL, COMPLEX], ids=["float64", "complex128"])
def test_irregular_frequencyseries_hdf5_roundtrip(tmp_path, data):
    fs = _irregular(data)
    target = tmp_path / "irregular.h5"

    fs.write(target, format="hdf5")

    _assert_same(FrequencySeries.read(target, format="hdf5"), fs)


def test_irregular_frequencyseries_hdf5_non_hz_axis_unit(tmp_path):
    fs = FrequencySeries(REAL, frequencies=FREQS * u.kHz, unit="m/s2", name="probe")
    target = tmp_path / "irregular_khz.h5"

    fs.write(target, format="hdf5")
    back = FrequencySeries.read(target, format="hdf5")

    assert back.frequencies.unit == u.kHz
    np.testing.assert_array_equal(back.frequencies.value, FREQS)


def test_irregular_frequencyseries_hdf5_is_readable_by_gwpy(tmp_path):
    fs = _irregular(REAL)
    target = tmp_path / "irregular.h5"

    fs.write(target, format="hdf5")
    back = GwpyFrequencySeries.read(target, format="hdf5")

    np.testing.assert_array_equal(back.frequencies.value, FREQS)
    np.testing.assert_array_equal(back.value, REAL)


def test_irregular_frequencyseries_hdf5_layout_has_no_dx(tmp_path):
    fs = _irregular(REAL)
    target = tmp_path / "irregular.h5"

    fs.write(target, format="hdf5")

    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["probe"].attrs)
    assert "dx" not in attrs
    assert attrs["xunit"] == "Hz"
    np.testing.assert_array_equal(attrs["xindex"], FREQS)


def test_regular_frequencyseries_hdf5_layout_unchanged(tmp_path):
    fs = FrequencySeries(REAL, f0=2.0, df=0.5, unit="m/s2", name="probe")
    target = tmp_path / "regular.h5"

    fs.write(target, format="hdf5")

    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["probe"].attrs)
        libver = h5f.libver
    assert attrs["x0"] == 2.0
    assert attrs["dx"] == 0.5
    assert attrs["xunit"] == "Hz"
    assert "xindex" not in attrs
    assert libver == ("earliest", "v114") or libver[0] == "earliest"
    back = FrequencySeries.read(target, format="hdf5")
    assert back.xindex.regular
    np.testing.assert_array_equal(back.frequencies.value, fs.frequencies.value)


def test_irregular_frequencyseries_hdf5_explicit_attrs_are_respected(tmp_path):
    fs = _irregular(REAL)
    target = tmp_path / "irregular.h5"

    fs.write(target, format="hdf5", attrs={"note": "caller"})

    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["probe"].attrs)
    assert attrs["note"] == "caller"
    assert "xunit" not in attrs


def test_irregular_frequencyseries_hdf5_existing_file_requires_overwrite(tmp_path):
    fs = _irregular(REAL)
    target = tmp_path / "irregular.h5"
    fs.write(target, format="hdf5")

    with pytest.raises(OSError, match="File exists"):
        fs.write(target, format="hdf5")

    fs.write(target, format="hdf5", overwrite=True)
    _assert_same(FrequencySeries.read(target, format="hdf5"), fs)


def test_irregular_frequencyseries_hdf5_append_keeps_other_datasets(tmp_path):
    target = tmp_path / "irregular.h5"
    first = _irregular(REAL, name="first")
    second = _irregular(COMPLEX, name="second")

    first.write(target, format="hdf5")
    second.write(target, format="hdf5", append=True)

    _assert_same(FrequencySeries.read(target, format="hdf5", path="first"), first)
    _assert_same(FrequencySeries.read(target, format="hdf5", path="second"), second)


def test_irregular_frequencyseries_hdf5_long_axis_roundtrip(tmp_path):
    # 20000 float64 samples exceed the 64 KiB compact-attribute limit of the
    # default (earliest) HDF5 file format.
    n = 20000
    freqs = np.geomspace(1.0, 1.0e4, n)
    fs = FrequencySeries(np.arange(n, dtype=np.float64), frequencies=freqs, name="p")
    assert not fs.xindex.regular
    target = tmp_path / "long.h5"

    fs.write(target, format="hdf5")
    back = FrequencySeries.read(target, format="hdf5")

    np.testing.assert_array_equal(back.frequencies.value, freqs)
    np.testing.assert_array_equal(back.value, fs.value)


def test_irregular_frequencyseries_hdf5_long_axis_in_legacy_group_is_diagnosed(
    tmp_path,
):
    n = 20000
    fs = FrequencySeries(
        np.arange(n, dtype=np.float64),
        frequencies=np.geomspace(1.0, 1.0e4, n),
        name="p",
    )
    with h5py.File(tmp_path / "legacy.h5", "w") as h5f:
        with pytest.raises(OSError, match="irregular"):
            fs.write(h5f, format="hdf5")


def test_irregular_frequencyseriesdict_hdf5_roundtrip(tmp_path):
    fsd = FrequencySeriesDict(
        {
            "H1:ASD": _irregular(REAL, name="H1:ASD"),
            "L1:TF": _irregular(COMPLEX, name="L1:TF"),
        }
    )
    target = tmp_path / "dict.h5"

    fsd.write(target, format="hdf5")
    back = FrequencySeriesDict.read(target, format="hdf5")

    assert list(back) == list(fsd)
    for key, original in fsd.items():
        _assert_same(back[key], original)


def test_irregular_frequencyserieslist_hdf5_roundtrip(tmp_path):
    fsl = FrequencySeriesList(
        [_irregular(REAL, name="a"), _irregular(COMPLEX, name="b")]
    )
    target = tmp_path / "list.h5"

    fsl.write(target, format="hdf5")
    back = FrequencySeriesList.read(target, format="hdf5")

    assert len(back) == 2
    for restored, original in zip(back, fsl):
        _assert_same(restored, original)
