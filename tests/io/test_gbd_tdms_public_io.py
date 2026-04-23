"""Public contract tests for GBD and TDMS direct I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

GBD_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "test.gbd"
TDMS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "test.tdms"


@pytest.mark.skipif(not GBD_FIXTURE.exists(), reason="test.gbd fixture not found")
def test_gbd_public_entrypoints_require_timezone_and_roundtrip():
    with pytest.raises(ValueError, match="timezone is required"):
        TimeSeriesDict.read(GBD_FIXTURE, format="gbd")

    tsd = TimeSeriesDict.read(GBD_FIXTURE, format="gbd", timezone="UTC")
    assert sorted(tsd.keys()) == ["Alarm", "AlarmOut", "CH1"]
    assert set(np.unique(tsd["Alarm"].value)).issubset({0.0, 1.0})

    ts = TimeSeries.read(GBD_FIXTURE, format="gbd", timezone="UTC", channels=["CH1"])
    assert ts.name == "CH1"

    matrix = TimeSeriesMatrix.read(GBD_FIXTURE, format="gbd", timezone="UTC")
    assert matrix.shape == (3, 1, 100)


@pytest.mark.skipif(not TDMS_FIXTURE.exists(), reason="test.tdms fixture not found")
def test_tdms_missing_optional_dependency_raises_clean_importerror(monkeypatch):
    from gwexpy.timeseries.io import tdms as tdms_io

    def _boom():
        raise ImportError("npTDMS is required for reading TDMS files. Install with `pip install nptdms`.")

    monkeypatch.setattr(tdms_io, "_import_nptdms", _boom)

    with pytest.raises(ImportError, match="npTDMS is required"):
        TimeSeriesDict.read(TDMS_FIXTURE, format="tdms")


def test_tdms_public_entrypoints_when_dependency_available(tmp_path):
    pytest.importorskip("nptdms")
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

    path = tmp_path / "public.tdms"
    data = np.arange(8, dtype=np.float64)
    channel = ChannelObject("Group", "Signal", data, properties={"wf_increment": 0.25})

    with TdmsWriter(str(path)) as writer:
        writer.write_segment([RootObject(), GroupObject("Group"), channel])

    tsd = TimeSeriesDict.read(path, format="tdms")
    assert sorted(tsd.keys()) == ["Group/Signal"]
    assert np.isclose(tsd["Group/Signal"].dt.value, 0.25)

    ts = TimeSeries.read(path, format="tdms")
    assert len(ts) == len(data)

    matrix = TimeSeriesMatrix.read(path, format="tdms")
    assert matrix.shape == (1, 1, len(data))
