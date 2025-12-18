import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries


def make_fs():
    freqs = np.linspace(0, 10, 6) * u.Hz
    data = np.arange(6, dtype=float)
    return FrequencySeries(data, frequencies=freqs, unit="1", name="fs")


def test_frequencyseries_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    fs = make_fs()
    s = fs.to_pandas()
    fs2 = FrequencySeries.from_pandas(s, unit="1")
    np.testing.assert_allclose(fs2.value, fs.value)
    np.testing.assert_allclose(fs2.frequencies.value, fs.frequencies.value)


def test_frequencyseries_xarray_roundtrip():
    pytest.importorskip("xarray")
    fs = make_fs()
    da = fs.to_xarray()
    fs2 = FrequencySeries.from_xarray(da, unit="1")
    np.testing.assert_allclose(fs2.value, fs.value)
    np.testing.assert_allclose(fs2.frequencies.value, fs.frequencies.value)


def test_frequencyseries_hdf5_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    fs = make_fs()
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        fs.to_hdf5_dataset(f, "freq")
    with h5py.File(path, "r") as f:
        fs2 = FrequencySeries.from_hdf5_dataset(f, "freq")
    np.testing.assert_allclose(fs2.value, fs.value)
    np.testing.assert_allclose(fs2.frequencies, fs.frequencies)
