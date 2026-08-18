from __future__ import annotations

import pickle
import shelve
from types import SimpleNamespace

import numpy as np
import pytest
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


def _builder_kwargs(t0_ns, precision):
    return {
        "dt": 1.0,
        "t0": 0.0,
        "_gwex_t0_gps_ns": t0_ns,
        "_gwex_t0_gps_precision": precision,
    }


class _TimeseriesBuilderPayload:
    def __init__(self, data, kwargs):
        self.data = data
        self.kwargs = kwargs

    def __reduce__(self):
        from gwexpy.io.pickle_compat import _build_gwpy_timeseries

        return _build_gwpy_timeseries, (self.data, self.kwargs)


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


def test_pickle_preserves_only_gps_nanosecond_state_on_gwpy_timeseries():
    ts = TimeSeries(
        np.arange(3.0),
        t0_ns=1_234_567_890_123_456_789,
        dt=1.0,
        unit="m",
    )
    ts._gwex_arbitrary = "must not cross the compatibility boundary"

    restored = pickle.loads(pickle.dumps(ts))

    assert isinstance(restored, GwpyTimeSeries)
    assert restored._gwex_t0_gps_ns == 1_234_567_890_123_456_789
    assert restored._gwex_t0_gps_precision == "exact"
    assert not hasattr(restored, "_gwex_arbitrary")


@pytest.mark.parametrize(
    ("t0_ns", "precision", "error"),
    [
        (True, "exact", TypeError),
        ("1", "exact", TypeError),
        (-1, "exact", ValueError),
        (2**63, "exact", ValueError),
        (1, "invalid", ValueError),
    ],
)
def test_pickle_dumps_rejects_malformed_private_gps_state(t0_ns, precision, error):
    source = TimeSeries(np.arange(2.0), t0=0.0, dt=1.0)
    source._gwex_t0_gps_ns = t0_ns
    source._gwex_t0_gps_precision = precision

    with pytest.raises(error):
        pickle.dumps(source)

    assert source._gwex_t0_gps_ns is t0_ns or source._gwex_t0_gps_ns == t0_ns
    assert source._gwex_t0_gps_precision == precision


@pytest.mark.parametrize(
    ("t0_ns", "precision", "error"),
    [
        (True, "exact", TypeError),
        ("1", "exact", TypeError),
        (-1, "exact", ValueError),
        (2**63, "exact", ValueError),
        (1, "arbitrary", ValueError),
        (None, "exact", ValueError),
        (1, None, ValueError),
    ],
)
def test_timeseries_pickle_builder_rejects_malformed_gps_state(t0_ns, precision, error):
    from gwexpy.io.pickle_compat import _build_gwpy_timeseries

    with pytest.raises(error):
        _build_gwpy_timeseries(np.arange(2.0), _builder_kwargs(t0_ns, precision))
    with pytest.raises(error):
        pickle.loads(
            pickle.dumps(
                _TimeseriesBuilderPayload(
                    np.arange(2.0), _builder_kwargs(t0_ns, precision)
                )
            )
        )


@pytest.mark.parametrize(
    ("t0", "expected_ns", "expected_precision"),
    [
        (123, 123_000_000_000, "exact"),
        (10.0000000005, 10_000_000_000, "quantized"),
    ],
)
def test_timeseries_pickle_builder_preserves_exact_and_quantized_state(
    t0, expected_ns, expected_precision
):
    source = TimeSeries(np.arange(2.0), t0=t0, dt=1.0)
    restored = pickle.loads(pickle.dumps(source))

    assert isinstance(restored, GwpyTimeSeries)
    assert restored._gwex_t0_gps_ns == expected_ns
    assert restored._gwex_t0_gps_precision == expected_precision


def test_legacy_timeseries_pickle_roundtrip_without_gps_state_remains_valid():
    source = TimeSeries(np.arange(2.0), t0=0.0, dt=1.0)
    source._gwex_t0_gps_ns = None
    source._gwex_t0_gps_precision = None

    restored = pickle.loads(pickle.dumps(source))

    assert isinstance(restored, GwpyTimeSeries)
    assert not hasattr(restored, "_gwex_t0_gps_ns")
    assert not hasattr(restored, "_gwex_t0_gps_precision")


def test_legacy_timeseries_pickle_payload_without_gps_state_remains_valid():
    from gwexpy.io.pickle_compat import _build_gwpy_timeseries

    payload = pickle.dumps(
        _TimeseriesBuilderPayload(np.arange(2.0), {"dt": 1.0, "t0": 0.0})
    )

    restored = pickle.loads(payload)

    assert isinstance(restored, GwpyTimeSeries)
    assert not hasattr(restored, "_gwex_t0_gps_ns")
    assert not hasattr(restored, "_gwex_t0_gps_precision")


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


def test_timeseries_reduce_args_fallback_t0_dt():
    """timeseries_reduce_args falls back to t0/dt when times is None."""
    from gwexpy.io.pickle_compat import timeseries_reduce_args

    # Create mock with no times attribute but t0/dt
    obj = SimpleNamespace(
        value=np.ones(3),
        unit=None,
        name=None,
        channel=None,
        epoch=None,
        t0=0.0,
        dt=1.0,
        # no times attribute
    )
    fn, (data, kwargs) = timeseries_reduce_args(obj)
    assert "t0" in kwargs
    assert "dt" in kwargs
    assert "times" not in kwargs


def test_frequencyseries_reduce_args_fallback_f0_df():
    """frequencyseries_reduce_args falls back to f0/df when frequencies is None."""
    from gwexpy.io.pickle_compat import frequencyseries_reduce_args

    obj = SimpleNamespace(
        value=np.ones(3),
        unit=None,
        name=None,
        channel=None,
        epoch=None,
        f0=0.0,
        df=1.0,
        # no frequencies attribute
    )
    fn, (data, kwargs) = frequencyseries_reduce_args(obj)
    assert "f0" in kwargs
    assert "df" in kwargs
    assert "frequencies" not in kwargs
