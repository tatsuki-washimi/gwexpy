from __future__ import annotations

import h5py
import numpy as np
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.spectrogram import Spectrogram as GwpySpectrogram
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)
from gwexpy.io.hdf5_collection import read_hdf5_keymap, read_hdf5_order
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


def test_gwpy_reads_timeseriesdict_hdf5(tmp_path):
    ts1 = TimeSeries(np.arange(4.0), sample_rate=2.0, t0=1.0, unit="m", name="A")
    ts2 = TimeSeries(
        np.arange(4.0) * 2, sample_rate=2.0, t0=1.0, unit="m", name="B"
    )
    tsd = TimeSeriesDict({"H1:TEST": ts1, "L1:TEST": ts2})

    outfile = tmp_path / "tsd_gwpy.h5"
    tsd.write(outfile, format="hdf5", layout="dataset")

    gwpy_tsd = GwpyTimeSeriesDict.read(outfile, format="hdf5")
    for gwpy_ts, expected in zip(gwpy_tsd.values(), tsd.values()):
        np.testing.assert_allclose(gwpy_ts.value, expected.value)
        assert str(gwpy_ts.unit) == str(expected.unit)


def test_gwpy_reads_timeserieslist_hdf5(tmp_path):
    ts1 = TimeSeries(np.arange(3.0), sample_rate=1.0, t0=0, unit="m")
    ts2 = TimeSeries(np.arange(3.0) * 2, sample_rate=1.0, t0=0, unit="m")
    tsl = TimeSeriesList(ts1, ts2)

    outfile = tmp_path / "tsl_gwpy.h5"
    tsl.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for idx, ds_name in enumerate(order):
            gwpy_ts = GwpyTimeSeries.read(h5f, format="hdf5", path=ds_name)
            expected = tsl[idx]
            np.testing.assert_allclose(gwpy_ts.value, expected.value)
            assert str(gwpy_ts.unit) == str(expected.unit)


def test_gwpy_reads_frequencyseriesdict_hdf5(tmp_path):
    fs = FrequencySeries(np.arange(3.0), frequencies=np.arange(3.0), unit="1")
    fsd = FrequencySeriesDict({"H1:ASD": fs})

    outfile = tmp_path / "fsd_gwpy.h5"
    fsd.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        keymap = read_hdf5_keymap(h5f)
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for ds_name in order:
            gwpy_fs = GwpyFrequencySeries.read(h5f, format="hdf5", path=ds_name)
            orig_key = keymap.get(ds_name, ds_name)
            expected = fsd[orig_key]
            np.testing.assert_allclose(gwpy_fs.value, expected.value)
            assert str(gwpy_fs.unit) == str(expected.unit)


def test_gwpy_reads_frequencyserieslist_hdf5(tmp_path):
    fsl = FrequencySeriesList(
        FrequencySeries(np.arange(3.0), frequencies=np.arange(3.0), unit="1"),
        FrequencySeries(np.arange(3.0) * 2, frequencies=np.arange(3.0), unit="1"),
    )

    outfile = tmp_path / "fsl_gwpy.h5"
    fsl.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for idx, ds_name in enumerate(order):
            gwpy_fs = GwpyFrequencySeries.read(h5f, format="hdf5", path=ds_name)
            expected = fsl[idx]
            np.testing.assert_allclose(gwpy_fs.value, expected.value)
            assert str(gwpy_fs.unit) == str(expected.unit)


def test_gwpy_reads_spectrogramdict_hdf5(tmp_path):
    sg = Spectrogram(
        np.arange(6.0).reshape(2, 3),
        times=np.arange(2.0),
        frequencies=np.arange(3.0),
        unit="m",
    )
    sgd = SpectrogramDict({"H1:SPEC": sg})

    outfile = tmp_path / "sgd_gwpy.h5"
    sgd.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        keymap = read_hdf5_keymap(h5f)
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for ds_name in order:
            gwpy_sg = GwpySpectrogram.read(h5f, format="hdf5", path=ds_name)
            orig_key = keymap.get(ds_name, ds_name)
            expected = sgd[orig_key]
            np.testing.assert_allclose(gwpy_sg.value, expected.value)
            assert str(gwpy_sg.unit) == str(expected.unit)


def test_gwpy_reads_spectrogramlist_hdf5(tmp_path):
    sgl = SpectrogramList(
        [
            Spectrogram(
                np.arange(6.0).reshape(2, 3),
                times=np.arange(2.0),
                frequencies=np.arange(3.0),
                unit="m",
            )
        ]
    )

    outfile = tmp_path / "sgl_gwpy.h5"
    sgl.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for idx, ds_name in enumerate(order):
            gwpy_sg = GwpySpectrogram.read(h5f, format="hdf5", path=ds_name)
            expected = sgl[idx]
            np.testing.assert_allclose(gwpy_sg.value, expected.value)
            assert str(gwpy_sg.unit) == str(expected.unit)
