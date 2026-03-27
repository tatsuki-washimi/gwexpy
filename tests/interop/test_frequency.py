"""Tests for gwexpy/interop/frequency.py."""
from __future__ import annotations

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.interop.frequency import (
    from_hdf5_frequencyseries,
    from_pandas_frequencyseries,
    from_xarray_frequencyseries,
    to_hdf5_frequencyseries,
    to_pandas_frequencyseries,
    to_xarray_frequencyseries,
)


def _make_fs(n=10, unit="1/Hz", name="ch"):
    freqs = np.linspace(1.0, 10.0, n)
    vals = np.random.default_rng(0).uniform(0.1, 1.0, n)
    return FrequencySeries(vals, frequencies=freqs, unit=unit, name=name)


# ---------------------------------------------------------------------------
# pandas round-trip
# ---------------------------------------------------------------------------


class TestToPandasFrequencySeries:
    def test_basic(self):
        pd = pytest.importorskip("pandas")
        fs = _make_fs()
        series = to_pandas_frequencyseries(fs)
        assert len(series) == len(fs)
        assert series.index.name == "frequency"

    def test_copy_flag(self):
        pytest.importorskip("pandas")
        fs = _make_fs()
        s_no_copy = to_pandas_frequencyseries(fs, copy=False)
        s_copy = to_pandas_frequencyseries(fs, copy=True)
        np.testing.assert_array_equal(s_no_copy.values, s_copy.values)

    def test_invalid_index_raises(self):
        pytest.importorskip("pandas")
        fs = _make_fs()
        with pytest.raises(ValueError, match="index must be"):
            to_pandas_frequencyseries(fs, index="time")

    def test_custom_name(self):
        pytest.importorskip("pandas")
        fs = _make_fs()
        s = to_pandas_frequencyseries(fs, name="myname")
        assert s.name == "myname"


class TestFromPandasFrequencySeries:
    def test_basic_roundtrip(self):
        pd = pytest.importorskip("pandas")
        fs = _make_fs()
        series = to_pandas_frequencyseries(fs)
        fs2 = from_pandas_frequencyseries(FrequencySeries, series)
        np.testing.assert_array_almost_equal(fs2.value, fs.value)

    def test_with_explicit_frequencies(self):
        pd = pytest.importorskip("pandas")
        fs = _make_fs()
        series = to_pandas_frequencyseries(fs)
        freqs = np.linspace(1, 10, 10)
        fs2 = from_pandas_frequencyseries(FrequencySeries, series, frequencies=freqs)
        np.testing.assert_array_equal(fs2.frequencies.value, freqs)

    def test_with_df_f0(self):
        pd = pytest.importorskip("pandas")
        fs = _make_fs()
        series = to_pandas_frequencyseries(fs)
        fs2 = from_pandas_frequencyseries(FrequencySeries, series, df=1.0, f0=1.0)
        assert fs2.value is not None

    def test_with_unit(self):
        pd = pytest.importorskip("pandas")
        fs = _make_fs()
        series = to_pandas_frequencyseries(fs)
        fs2 = from_pandas_frequencyseries(FrequencySeries, series, unit="m")
        assert str(fs2.unit) == "m"


# ---------------------------------------------------------------------------
# xarray round-trip
# ---------------------------------------------------------------------------


class TestToXarrayFrequencySeries:
    def test_basic(self):
        pytest.importorskip("xarray")
        fs = _make_fs()
        da = to_xarray_frequencyseries(fs)
        assert da.dims == ("frequency",)
        assert len(da) == len(fs)

    def test_freq_coord_hz(self):
        pytest.importorskip("xarray")
        fs = _make_fs()
        da = to_xarray_frequencyseries(fs, freq_coord="Hz")
        assert "frequency" in da.coords

    def test_freq_coord_index(self):
        pytest.importorskip("xarray")
        fs = _make_fs()
        da = to_xarray_frequencyseries(fs, freq_coord="index")
        # index mode → arange
        assert da.coords["frequency"].values[0] == 0

    def test_attrs_contain_unit(self):
        pytest.importorskip("xarray")
        fs = _make_fs(unit="1/Hz")
        da = to_xarray_frequencyseries(fs)
        assert "unit" in da.attrs


class TestFromXarrayFrequencySeries:
    def test_roundtrip(self):
        pytest.importorskip("xarray")
        fs = _make_fs()
        da = to_xarray_frequencyseries(fs)
        fs2 = from_xarray_frequencyseries(FrequencySeries, da)
        np.testing.assert_array_almost_equal(fs2.value, fs.value)

    def test_with_explicit_unit(self):
        pytest.importorskip("xarray")
        fs = _make_fs()
        da = to_xarray_frequencyseries(fs)
        fs2 = from_xarray_frequencyseries(FrequencySeries, da, unit="m")
        assert str(fs2.unit) == "m"


# ---------------------------------------------------------------------------
# HDF5 round-trip
# ---------------------------------------------------------------------------


class TestHdf5FrequencySeries:
    def test_roundtrip(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        fs = _make_fs()
        fp = tmp_path / "fs.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5_frequencyseries(fs, h5f, "channel")
            fs2 = from_hdf5_frequencyseries(FrequencySeries, h5f, "channel")
        np.testing.assert_array_almost_equal(fs2.value, fs.value)

    def test_overwrite_false_raises(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        fs = _make_fs()
        fp = tmp_path / "fs.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5_frequencyseries(fs, h5f, "ch")
            with pytest.raises(OSError, match="exists"):
                to_hdf5_frequencyseries(fs, h5f, "ch", overwrite=False)

    def test_overwrite_true_replaces(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        fs = _make_fs()
        fp = tmp_path / "fs.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5_frequencyseries(fs, h5f, "ch")
            to_hdf5_frequencyseries(fs, h5f, "ch", overwrite=True)  # should not raise
            fs2 = from_hdf5_frequencyseries(FrequencySeries, h5f, "ch")
        np.testing.assert_array_almost_equal(fs2.value, fs.value)

    def test_hdf5_from_df_f0_when_no_frequencies(self, tmp_path):
        """If frequencies attr absent but df/f0 present, reconstruct."""
        h5py = pytest.importorskip("h5py")
        fs = _make_fs(n=5)
        fp = tmp_path / "fs.h5"
        with h5py.File(fp, "w") as h5f:
            to_hdf5_frequencyseries(fs, h5f, "ch")
            # Remove frequencies attr to test fallback
            del h5f["ch"].attrs["frequencies"]
            h5f["ch"].attrs["df"] = 1.0
            h5f["ch"].attrs["f0"] = 0.0
            fs2 = from_hdf5_frequencyseries(FrequencySeries, h5f, "ch")
        assert fs2 is not None
        assert len(fs2) == 5
