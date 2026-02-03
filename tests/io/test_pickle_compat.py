from __future__ import annotations

import pickle
import shelve

import numpy as np
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.spectrogram import Spectrogram as GwpySpectrogram
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict
from gwpy.timeseries import TimeSeriesList as GwpyTimeSeriesList

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


def test_pickle_series_to_gwpy_types():
    ts = TimeSeries(np.arange(3.0), sample_rate=1.0, t0=0, unit="m")
    fs = FrequencySeries(np.arange(3.0), frequencies=np.arange(3.0), unit="1")
    sg = Spectrogram(
        np.arange(6.0).reshape(2, 3),
        times=np.arange(2.0),
        frequencies=np.arange(3.0),
        unit="m",
    )
    ts._gwex_test = "x"  # ensure gwexpy-only attrs are not preserved

    ts2 = pickle.loads(pickle.dumps(ts))
    fs2 = pickle.loads(pickle.dumps(fs))
    sg2 = pickle.loads(pickle.dumps(sg))

    assert isinstance(ts2, GwpyTimeSeries)
    assert isinstance(fs2, GwpyFrequencySeries)
    assert isinstance(sg2, GwpySpectrogram)
    assert not hasattr(ts2, "_gwex_test")


def test_pickle_collections_to_gwpy_or_builtin():
    ts = TimeSeries(np.arange(3.0), sample_rate=1.0, t0=0, unit="m")
    tsd = TimeSeriesDict({"H1:TEST": ts})
    tsl = TimeSeriesList(ts)
    fsd = FrequencySeriesDict(
        {
            "H1:ASD": FrequencySeries(
                np.arange(3.0), frequencies=np.arange(3.0), unit="1"
            )
        }
    )
    fsl = FrequencySeriesList(
        [FrequencySeries(np.arange(3.0), frequencies=np.arange(3.0), unit="1")]
    )
    sgd = SpectrogramDict(
        {
            "H1:SPEC": Spectrogram(
                np.arange(6.0).reshape(2, 3),
                times=np.arange(2.0),
                frequencies=np.arange(3.0),
                unit="m",
            )
        }
    )
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

    assert isinstance(pickle.loads(pickle.dumps(tsd)), GwpyTimeSeriesDict)
    assert isinstance(pickle.loads(pickle.dumps(tsl)), GwpyTimeSeriesList)
    assert isinstance(pickle.loads(pickle.dumps(fsd)), dict)
    assert isinstance(pickle.loads(pickle.dumps(fsl)), list)
    assert isinstance(pickle.loads(pickle.dumps(sgd)), dict)
    assert isinstance(pickle.loads(pickle.dumps(sgl)), list)

    fsd2 = pickle.loads(pickle.dumps(fsd))
    fsl2 = pickle.loads(pickle.dumps(fsl))
    sgd2 = pickle.loads(pickle.dumps(sgd))
    sgl2 = pickle.loads(pickle.dumps(sgl))
    assert isinstance(next(iter(fsd2.values())), GwpyFrequencySeries)
    assert isinstance(fsl2[0], GwpyFrequencySeries)
    assert isinstance(next(iter(sgd2.values())), GwpySpectrogram)
    assert isinstance(sgl2[0], GwpySpectrogram)


def test_shelve_roundtrip_to_gwpy(tmp_path):
    ts = TimeSeries(
        np.arange(3.0), sample_rate=1.0, t0=0, unit="m", name="n", channel="C1"
    )
    fs = FrequencySeries(
        np.arange(3.0), frequencies=np.arange(3.0), unit="1", name="f", channel="C2"
    )
    sg = Spectrogram(
        np.arange(6.0).reshape(2, 3),
        times=np.arange(2.0),
        frequencies=np.arange(3.0),
        unit="m",
        name="s",
        channel="C3",
    )
    tsd = TimeSeriesDict({"H1:TEST": ts})
    tsl = TimeSeriesList(ts)
    fsd = FrequencySeriesDict(
        {
            "H1:ASD": FrequencySeries(
                np.arange(3.0), frequencies=np.arange(3.0), unit="1"
            )
        }
    )
    fsl = FrequencySeriesList(
        [FrequencySeries(np.arange(3.0), frequencies=np.arange(3.0), unit="1")]
    )
    sgd = SpectrogramDict(
        {
            "H1:SPEC": Spectrogram(
                np.arange(6.0).reshape(2, 3),
                times=np.arange(2.0),
                frequencies=np.arange(3.0),
                unit="m",
            )
        }
    )
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
    path = tmp_path / "test_shelve.db"
    with shelve.open(str(path)) as db:
        db["ts"] = ts
        db["fs"] = fs
        db["sg"] = sg
        db["tsd"] = tsd
        db["tsl"] = tsl
        db["fsd"] = fsd
        db["fsl"] = fsl
        db["sgd"] = sgd
        db["sgl"] = sgl
    with shelve.open(str(path)) as db:
        obj_ts = db["ts"]
        obj_fs = db["fs"]
        obj_sg = db["sg"]
        obj_tsd = db["tsd"]
        obj_tsl = db["tsl"]
        obj_fsd = db["fsd"]
        obj_fsl = db["fsl"]
        obj_sgd = db["sgd"]
        obj_sgl = db["sgl"]
    assert isinstance(obj_ts, GwpyTimeSeries)
    assert isinstance(obj_fs, GwpyFrequencySeries)
    assert isinstance(obj_sg, GwpySpectrogram)
    assert isinstance(obj_tsd, GwpyTimeSeriesDict)
    assert isinstance(obj_tsl, GwpyTimeSeriesList)
    assert isinstance(obj_fsd, dict)
    assert isinstance(next(iter(obj_fsd.values())), GwpyFrequencySeries)
    assert isinstance(obj_fsl, list)
    assert isinstance(obj_fsl[0], GwpyFrequencySeries)
    assert isinstance(obj_sgd, dict)
    assert isinstance(next(iter(obj_sgd.values())), GwpySpectrogram)
    assert isinstance(obj_sgl, list)
    assert isinstance(obj_sgl[0], GwpySpectrogram)
    assert obj_fsd["H1:ASD"].unit == fsd["H1:ASD"].unit
    assert obj_sgd["H1:SPEC"].unit == sgd["H1:SPEC"].unit
    assert str(obj_ts.unit) == "m"
    assert obj_ts.name == "n"
    assert obj_fs.unit == fs.unit
    assert obj_fs.name == "f"
    assert obj_sg.name == "s"
