"""Tests for FrequencySeries I/O: HDF5 roundtrip, dttxml error handling, CSV."""

import numpy as np
import pytest

import gwexpy.frequencyseries.io.dttxml  # trigger registration

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)
from gwexpy.frequencyseries.io.dttxml import read_frequencyseriesdict_dttxml


class TestFrequencySeriesHdf5:
    def test_roundtrip(self, tmp_path):
        fs = FrequencySeries(
            np.array([1.0, 2.0, 3.0, 4.0]),
            f0=0, df=10, unit="1/Hz", name="test_psd",
        )
        path = tmp_path / "fs.hdf5"
        fs.write(str(path), format="hdf5")
        fs2 = FrequencySeries.read(str(path), format="hdf5")
        np.testing.assert_allclose(fs2.value, fs.value)
        assert np.isclose(fs2.df.value, fs.df.value)

    def test_dict_roundtrip(self, tmp_path):
        fsd = FrequencySeriesDict({
            "H1:ASD": FrequencySeries(
                np.arange(5.0), frequencies=np.arange(5.0), unit="1", name="H1:ASD",
            ),
            "L1:ASD": FrequencySeries(
                np.arange(5.0) * 2, frequencies=np.arange(5.0), unit="1", name="L1:ASD",
            ),
        })
        path = tmp_path / "fsd.hdf5"
        fsd.write(str(path), format="hdf5")
        fsd2 = FrequencySeriesDict.read(str(path), format="hdf5")
        assert set(fsd2.keys()) == {"H1:ASD", "L1:ASD"}
        np.testing.assert_allclose(fsd2["H1:ASD"].value, fsd["H1:ASD"].value)

    def test_list_roundtrip(self, tmp_path):
        fsl = FrequencySeriesList(
            FrequencySeries(np.arange(3.0), frequencies=np.arange(3.0), unit="1"),
            FrequencySeries(np.arange(3.0) * 2, frequencies=np.arange(3.0), unit="1"),
        )
        path = tmp_path / "fsl.hdf5"
        fsl.write(str(path), format="hdf5")
        fsl2 = FrequencySeriesList.read(str(path), format="hdf5")
        assert len(fsl2) == 2
        np.testing.assert_allclose(fsl2[0].value, fsl[0].value)

    def test_complex_data_roundtrip(self, tmp_path):
        data = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        fs = FrequencySeries(data, f0=0, df=1, name="cplx")
        path = tmp_path / "complex.hdf5"
        fs.write(str(path), format="hdf5")
        fs2 = FrequencySeries.read(str(path), format="hdf5")
        np.testing.assert_allclose(fs2.value, data)


class TestFrequencySeriesDttxml:
    def test_requires_products(self, tmp_path):
        dummy = tmp_path / "dummy.xml"
        dummy.write_text("<dttxml></dttxml>")
        with pytest.raises(ValueError, match="products"):
            read_frequencyseriesdict_dttxml(str(dummy))

    def test_invalid_product_type(self, tmp_path):
        dummy = tmp_path / "dummy.xml"
        dummy.write_text("<dttxml></dttxml>")
        with pytest.raises(ValueError):
            read_frequencyseriesdict_dttxml(str(dummy), products="TIMESERIES")


class TestFrequencySeriesCsv:
    def test_roundtrip_with_metadata(self, tmp_path):
        fs = FrequencySeries(
            np.array([10.0, 20.0, 30.0]),
            f0=0, df=5, unit="m/Hz", name="csv_test",
        )
        path = tmp_path / "fs.csv"
        fs.write(str(path), format="csv")
        fs2 = FrequencySeries.read(str(path), format="csv")
        np.testing.assert_allclose(fs2.value, fs.value)
