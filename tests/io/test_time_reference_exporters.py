"""Regression tests for absolute-time-only exporters (#725)."""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io.ndscope_hdf5 import write_timeseriesdict_ndscope_hdf5


def test_ndscope_hdf5_rejects_relative_timeseries(tmp_path):
    rel = TimeSeries(np.arange(8.0), dt=0.25, t0=None, name="K1:TEST")
    tsd = TimeSeriesDict({"K1:TEST.raw": rel})
    target = tmp_path / "ndscope.h5"

    with pytest.raises(ValueError, match="relative-time"):
        write_timeseriesdict_ndscope_hdf5(tsd, target)

    assert not target.exists()

    anchored = TimeSeriesDict({"K1:TEST.raw": rel.with_epoch(1_400_000_000.0)})
    write_timeseriesdict_ndscope_hdf5(anchored, target)
    assert target.exists()


@pytest.mark.parametrize("fmt", ["gwf", "frame", "framecpp", None])
def test_gwf_write_rejects_relative_timeseries(tmp_path, fmt):
    rel = TimeSeries(np.arange(8.0), dt=0.25, t0=None, name="K1:TEST")
    target = tmp_path / ("out.gwf" if fmt is None else "out.dat")
    kwargs = {"format": fmt} if fmt is not None else {}

    with pytest.raises(ValueError, match="relative-time"):
        rel.write(target, **kwargs)

    tsd = TimeSeriesDict({"K1:TEST": rel})
    with pytest.raises(ValueError, match="relative-time"):
        tsd.write(target, **kwargs)


def test_obspy_export_rejects_relative_series_and_spectrogram():
    pytest.importorskip("obspy")
    rel_ts = TimeSeries(np.arange(8.0), dt=0.25, t0=None, name="TEST")
    with pytest.raises(ValueError, match="relative-time"):
        rel_ts.to_obspy()

    trace = rel_ts.with_epoch(1_400_000_000.0).to_obspy()
    assert trace.stats.npts == 8

    rel_sg = Spectrogram(
        np.arange(12.0).reshape(4, 3), dt=0.25, t0=None, f0=1.0, df=0.5
    )
    with pytest.raises(ValueError, match="relative-time"):
        rel_sg.to_obspy()
