"""Public contract tests for seismic and geophysical direct I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def _make_single_series():
    return TimeSeries(np.arange(8, dtype=float), sample_rate=4, name="SIG")


@pytest.mark.parametrize("fmt", ["mseed", "miniseed"])
def test_mseed_public_dict_roundtrip_and_alias(tmp_path, fmt):
    pytest.importorskip("obspy")

    tsd = TimeSeriesDict({"SIG": _make_single_series()})
    path = tmp_path / f"sample.{fmt if fmt != 'miniseed' else 'mseed'}"

    tsd.write(path, format=fmt)
    back = TimeSeriesDict.read(path, format=fmt)

    assert len(back) == 1
    assert len(next(iter(back.values()))) == len(tsd["SIG"])


@pytest.mark.parametrize("fmt", ["sac", "gse2"])
def test_obsby_backed_public_dict_roundtrip(tmp_path, fmt):
    pytest.importorskip("obspy")

    tsd = TimeSeriesDict({"SIG": _make_single_series()})
    path = tmp_path / f"sample.{fmt}"

    tsd.write(path, format=fmt)
    back = TimeSeriesDict.read(path, format=fmt)

    assert len(back) == 1
    assert len(next(iter(back.values()))) == len(tsd["SIG"])


def test_knet_public_surface_is_dict_first(monkeypatch):
    pytest.importorskip("obspy")
    from gwexpy.timeseries.io import seismic as seismic_io

    stream = seismic_io._import_obspy().Stream()
    trace = seismic_io._import_obspy().Trace(data=np.arange(4, dtype=np.int32))
    trace.stats.starttime = seismic_io._import_obspy().UTCDateTime(2024, 1, 1)
    trace.stats.delta = 0.01
    trace.stats.channel = "EW"
    trace.stats.station = "STAT"
    stream.append(trace)

    monkeypatch.setattr(seismic_io, "_read_obspy_stream", lambda *a, **k: stream)
    out = TimeSeriesDict.read("dummy.knet", format="knet")
    assert len(out) == 1
    assert str(next(iter(out.keys()))).endswith("EW")


def test_win_alias_family_public_surface_is_dict_first(monkeypatch):
    pytest.importorskip("obspy")
    from gwexpy.timeseries.io import win as win_io

    tr = win_io.Trace(data=np.array([100, 101, 102], dtype=np.int32))
    tr.stats.channel = "A1_01"
    tr.stats.sampling_rate = 100.0
    tr.stats.starttime = win_io.UTCDateTime(2023, 1, 1, 0, 0, 0)
    stream = win_io.Stream(traces=[tr])

    monkeypatch.setattr(win_io, "_read_win_fixed", lambda *a, **k: stream)

    assert str(next(iter(TimeSeriesDict.read("dummy.win", format="win").keys()))).endswith(
        "A1_01"
    )
    assert str(
        next(iter(TimeSeriesDict.read("dummy.cnt", format="win32").keys()))
    ).endswith("A1_01")


def test_ats_public_single_and_dict_entrypoints_use_fixture():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "test.ats"
    if not fixture.exists():
        pytest.skip("sample ATS file is missing")

    ts = TimeSeries.read(fixture, format="ats")
    tsd = TimeSeriesDict.read(fixture, format="ats")

    assert len(ts) > 0
    assert len(tsd) == 1


def test_ats_mth5_missing_dependency_raises_clean_importerror():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "test.ats"
    if not fixture.exists():
        pytest.skip("sample ATS file is missing")

    with pytest.raises(ImportError):
        TimeSeries.read(fixture, format="ats.mth5")
