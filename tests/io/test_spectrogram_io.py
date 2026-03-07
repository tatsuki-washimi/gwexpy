"""Tests for Spectrogram I/O: HDF5 roundtrip for Dict, List, single."""

import numpy as np
import pytest

from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList


class TestSpectrogramHdf5:
    def test_roundtrip(self, tmp_path):
        sg = Spectrogram(
            np.arange(12.0).reshape(3, 4),
            times=np.arange(3.0),
            frequencies=np.arange(4.0),
            unit="m",
            name="test_sg",
        )
        path = tmp_path / "sg.hdf5"
        sg.write(str(path), format="hdf5")
        sg2 = Spectrogram.read(str(path), format="hdf5")
        np.testing.assert_allclose(sg2.value, sg.value)
        assert str(sg2.unit) == "m"

    def test_dict_roundtrip(self, tmp_path):
        sgd = SpectrogramDict({
            "H1:SPEC": Spectrogram(
                np.ones((2, 3)),
                times=np.arange(2.0),
                frequencies=np.arange(3.0),
                unit="m",
                name="H1:SPEC",
            ),
            "L1:SPEC": Spectrogram(
                np.zeros((2, 3)),
                times=np.arange(2.0),
                frequencies=np.arange(3.0),
                unit="m",
                name="L1:SPEC",
            ),
        })
        path = tmp_path / "sgd.hdf5"
        sgd.write(str(path), format="hdf5")
        sgd2 = SpectrogramDict().read(str(path), format="hdf5")
        assert set(sgd2.keys()) == {"H1:SPEC", "L1:SPEC"}
        np.testing.assert_allclose(sgd2["H1:SPEC"].value, np.ones((2, 3)))

    def test_list_roundtrip(self, tmp_path):
        sgl = SpectrogramList([
            Spectrogram(
                np.arange(6.0).reshape(2, 3),
                times=np.arange(2.0),
                frequencies=np.arange(3.0),
                unit="m",
                name="sg_0",
            ),
        ])
        path = tmp_path / "sgl.hdf5"
        sgl.write(str(path), format="hdf5")
        sgl2 = SpectrogramList().read(str(path), format="hdf5")
        assert len(sgl2) == 1
        np.testing.assert_allclose(sgl2[0].value, sgl[0].value)

    def test_shape_and_metadata_preserved(self, tmp_path):
        data = np.random.default_rng(7).random((5, 10))
        sg = Spectrogram(
            data,
            times=np.linspace(0, 4, 5),
            frequencies=np.linspace(0, 100, 10),
            unit="strain",
            name="meta_sg",
        )
        path = tmp_path / "meta.hdf5"
        sg.write(str(path), format="hdf5")
        sg2 = Spectrogram.read(str(path), format="hdf5")
        assert sg2.shape == (5, 10)
        assert str(sg2.unit) == "strain"
