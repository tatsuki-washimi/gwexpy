"""HDF5 round-trip of ``Spectrogram`` with irregular time/frequency axes (#887 §2.5)."""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest
from astropy import units as u
from gwpy.spectrogram import Spectrogram as GwpySpectrogram

from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.time._reference import (
    TIME_REFERENCE_RELATIVE,
    get_time_reference,
)

IRR_TIMES = np.array([0.0, 0.5, 1.5, 3.5])
REG_TIMES = np.array([10.0, 10.5, 11.0, 11.5])
IRR_FREQS = np.geomspace(1.0, 16.0, 5)
REG_FREQS = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
DATA = np.arange(20.0, dtype=np.float64).reshape(4, 5) * 1e-21


def _assert_roundtrip_same(back: Spectrogram, orig: Spectrogram) -> None:
    assert type(back) is type(orig)
    assert back.dtype == orig.dtype
    np.testing.assert_array_equal(back.value, orig.value)
    np.testing.assert_array_equal(back.times.value, orig.times.value)
    np.testing.assert_array_equal(back.frequencies.value, orig.frequencies.value)
    assert back.xunit == orig.xunit
    assert back.yunit == orig.yunit
    assert back.unit == orig.unit
    assert back.name == orig.name


def test_irregular_time_and_regular_freq_roundtrip(tmp_path) -> None:
    sg = Spectrogram(DATA, times=IRR_TIMES, frequencies=REG_FREQS, unit="m", name="tx")
    target = tmp_path / "irr_t.h5"

    sg.write(target, format="hdf5")
    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["tx"].attrs)
    assert "xindex" in attrs
    assert "dx" not in attrs
    assert "x0" not in attrs
    assert "yindex" not in attrs
    assert attrs["dy"] == 5.0
    assert attrs["y0"] == 5.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = Spectrogram.read(target, format="hdf5")
    assert not any("xindex was given" in str(w.message) for w in caught)
    _assert_roundtrip_same(back, sg)


def test_regular_time_and_irregular_freq_roundtrip(tmp_path) -> None:
    sg = Spectrogram(DATA, times=REG_TIMES, frequencies=IRR_FREQS, unit="m", name="fy")
    target = tmp_path / "irr_f.h5"

    sg.write(target, format="hdf5")
    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["fy"].attrs)
    assert "yindex" in attrs
    assert "dy" not in attrs
    assert "y0" not in attrs
    assert "xindex" not in attrs
    assert attrs["dx"] == 0.5
    assert attrs["x0"] == 10.0

    back = Spectrogram.read(target, format="hdf5")
    _assert_roundtrip_same(back, sg)


def test_both_axes_irregular_small_scale_roundtrip(tmp_path) -> None:
    """Small-scale (1e-20) irregular axes must not be linearized by atol=1e-8."""
    times = np.array([0.0, 1e-20, 3e-20, 6e-20])
    freqs = np.array([1e-20, 2e-20, 4e-20, 8e-20, 16e-20])
    sg = Spectrogram(DATA, times=times, frequencies=freqs, unit="V", name="tiny")
    assert sg._xindex.regular  # GWpy Index.regular misclassifies < 1e-8
    assert sg._yindex.regular

    target = tmp_path / "tiny_irr.h5"
    sg.write(target, format="hdf5")
    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["tiny"].attrs)
    assert "xindex" in attrs
    assert "yindex" in attrs
    assert "dx" not in attrs
    assert "dy" not in attrs

    back = Spectrogram.read(target, format="hdf5")
    _assert_roundtrip_same(back, sg)

    gw = GwpySpectrogram.read(target, format="hdf5")
    np.testing.assert_array_equal(gw.times.value, times)
    np.testing.assert_array_equal(gw.frequencies.value, freqs)


def test_nearly_regular_geomspace_axes_preserve_exact_coords(tmp_path) -> None:
    times = np.geomspace(1.0, 2.0, 20000)
    freqs = np.geomspace(10.0, 20.0, 100)
    values = np.ones((times.size, freqs.size), dtype=np.float64)
    sg = Spectrogram(values, times=times, frequencies=freqs, name="near")
    assert sg._xindex.regular  # GWpy's absolute tolerance misclassifies this axis.
    assert not sg._yindex.regular or not np.array_equal(
        np.linspace(freqs[0], freqs[-1], freqs.size), freqs
    )

    target = tmp_path / "near.h5"
    sg.write(target, format="hdf5")
    back = Spectrogram.read(target, format="hdf5")
    _assert_roundtrip_same(back, sg)


def test_regular_spectrogram_hdf5_layout_unchanged(tmp_path) -> None:
    sg = Spectrogram(DATA, t0=10.0, dt=0.5, f0=5.0, df=5.0, unit="m", name="reg")
    target = tmp_path / "regular.h5"

    sg.write(target, format="hdf5")
    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["reg"].attrs)
        libver = h5f.libver
    assert attrs["x0"] == 10.0
    assert attrs["dx"] == 0.5
    assert attrs["y0"] == 5.0
    assert attrs["dy"] == 5.0
    assert "xindex" not in attrs
    assert "yindex" not in attrs
    assert libver[0] == "earliest"

    back = Spectrogram.read(target, format="hdf5")
    _assert_roundtrip_same(back, sg)


def test_non_default_axis_units_and_relative_time_reference(tmp_path) -> None:
    sg = Spectrogram(
        DATA,
        times=IRR_TIMES * u.ms,
        frequencies=IRR_FREQS * u.kHz,
        epoch=None,
        unit="V",
        name="rel_irr",
    )
    assert get_time_reference(sg) == TIME_REFERENCE_RELATIVE

    target = tmp_path / "rel_irr.h5"
    sg.write(target, format="hdf5")
    back = Spectrogram.read(target, format="hdf5")
    _assert_roundtrip_same(back, sg)
    assert get_time_reference(back) == TIME_REFERENCE_RELATIVE


def test_long_irregular_axis_uses_dense_libver_and_diagnoses_earliest(tmp_path) -> None:
    n = 10_000
    times = np.geomspace(1.0, 100.0, n)
    freqs = np.array([1.0, 2.0])
    sg = Spectrogram(np.ones((n, 2)), times=times, frequencies=freqs, name="long")

    target = tmp_path / "long.h5"
    sg.write(target, format="hdf5")
    back = Spectrogram.read(target, format="hdf5")
    _assert_roundtrip_same(back, sg)

    with h5py.File(tmp_path / "earliest.h5", "w") as h5f:
        with pytest.raises(OSError, match=r"time \(x\) axis, 10000 samples"):
            sg.write(h5f, format="hdf5")
        assert list(h5f) == []


def test_long_frequency_axis_diagnostic_names_y_axis(tmp_path) -> None:
    freqs = np.geomspace(1.0, 100.0, 9000)
    sg = Spectrogram(np.zeros((2, freqs.size)), t0=0, dt=1, frequencies=freqs, name="y")
    with h5py.File(tmp_path / "earliest.h5", "w") as h5f:
        with pytest.raises(OSError, match=r"frequency \(y\) axis, 9000 samples"):
            sg.write(h5f, format="hdf5")
        assert list(h5f) == []


def test_conflicting_explicit_axis_attrs_rejected(tmp_path) -> None:
    sg = Spectrogram(DATA, times=IRR_TIMES, frequencies=IRR_FREQS, name="bad")
    with pytest.raises(ValueError, match=r"attrs\['dx'\]"):
        sg.write(tmp_path / "bad_dx.h5", format="hdf5", attrs={"dx": 1.0})
    with pytest.raises(ValueError, match=r"attrs\['dy'\]"):
        sg.write(tmp_path / "bad_dy.h5", format="hdf5", attrs={"dy": 1.0})
    with pytest.raises(ValueError, match=r"attrs\['yindex'\]"):
        sg.write(tmp_path / "bad_yi.h5", format="hdf5", attrs={"yindex": IRR_FREQS + 1})


def test_spectrogram_collections_preserve_irregular_axes(tmp_path) -> None:
    sg1 = Spectrogram(DATA, times=IRR_TIMES, frequencies=IRR_FREQS, unit="m", name="s1")
    sg2 = Spectrogram(
        DATA * 2, times=IRR_TIMES, frequencies=IRR_FREQS, unit="m", name="s2"
    )

    dict_path = tmp_path / "sgd.h5"
    SpectrogramDict({"H1:S1": sg1, "L1:S2": sg2}).write(dict_path, format="hdf5")
    sgd_back = SpectrogramDict().read(dict_path, format="hdf5")
    _assert_roundtrip_same(sgd_back["H1:S1"], sg1)
    _assert_roundtrip_same(sgd_back["L1:S2"], sg2)

    list_path = tmp_path / "sgl.h5"
    SpectrogramList([sg1, sg2]).write(list_path, format="hdf5")
    sgl_back = SpectrogramList().read(list_path, format="hdf5")
    _assert_roundtrip_same(sgl_back[0], sg1)
    _assert_roundtrip_same(sgl_back[1], sg2)


def test_887_jitter_on_both_axes_is_not_linearized(tmp_path) -> None:
    """Reproduce #887 inventory §2.5: GWpy-regular, non-affine axes."""
    times = np.arange(100) * 1e-6 + np.where(np.arange(100) % 2, 1e-9, 0.0)
    freqs = np.linspace(0.0, 100.0, 10) + np.where(np.arange(10) % 2, 1e-9, 0.0)
    sg = Spectrogram(
        np.ones((times.size, freqs.size)), times=times, frequencies=freqs, name="jit"
    )
    assert sg._xindex.regular and sg._yindex.regular

    target = tmp_path / "jitter.h5"
    sg.write(target, format="hdf5")
    with h5py.File(target, "r") as h5f:
        attrs = dict(h5f["jit"].attrs)
    assert {"xindex", "yindex"} <= set(attrs)
    assert not {"dx", "dy", "x0", "y0"} & set(attrs)

    back = Spectrogram.read(target, format="hdf5")
    _assert_roundtrip_same(back, sg)
    gw = GwpySpectrogram.read(target, format="hdf5")
    np.testing.assert_array_equal(gw.times.value, times)
    np.testing.assert_array_equal(gw.frequencies.value, freqs)
