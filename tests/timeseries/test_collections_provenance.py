"""Regression coverage for collection-level I/O provenance."""

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


class _RequestedTimeSeriesDict(TimeSeriesDict):
    """Concrete subclass used to assert public reader result typing."""


def test_direct_public_reader_rewraps_provenance_for_requested_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct reader keeps independent collection provenance when rewrapped."""
    source = TimeSeriesDict({"signal": TimeSeries([1, 2], sample_rate=1)})
    source._gwexpy_io = {"format": "xml.diaggui", "nested": {"kept": True}}

    from gwexpy.timeseries.io import dttxml

    monkeypatch.setattr(
        dttxml,
        "read_timeseriesdict_dttxml",
        lambda *args, **kwargs: source,
    )

    result = _RequestedTimeSeriesDict.read("reader-input.xml", format="dttxml")

    assert type(result) is _RequestedTimeSeriesDict
    assert list(result) == ["signal"]
    assert result["signal"] is source["signal"]
    assert result._gwexpy_io == source._gwexpy_io
    assert result._gwexpy_io is not source._gwexpy_io
    result._gwexpy_io["request"] = "changed"
    assert "request" not in source._gwexpy_io


def test_netcdf_public_roundtrip_retains_reader_provenance(tmp_path) -> None:
    """The public NetCDF path retains the reader's collection provenance."""
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    path = tmp_path / "provenance.nc"
    source = TimeSeriesDict(
        {"signal": TimeSeries(np.arange(4), t0=10, dt=0.25, name="signal")}
    )
    source.write(path, format="nc")

    result = _RequestedTimeSeriesDict.read(path, format="nc")

    assert type(result) is _RequestedTimeSeriesDict
    assert list(result) == ["signal"]
    assert result._gwexpy_io == {
        "format": "nc",
        "time_coord": "sample",
        "channels": ["signal"],
        "unit_source": "file",
    }


def test_crop_copies_collection_provenance_for_requested_subclass() -> None:
    """Cropping a collection keeps independent top-level provenance."""
    source = _RequestedTimeSeriesDict({"signal": TimeSeries(np.arange(6), t0=0, dt=1)})
    source._gwexpy_io = {"format": "source", "nested": {"kept": True}}

    result = source.crop(1, 4)

    assert type(result) is _RequestedTimeSeriesDict
    assert list(result) == ["signal"]
    np.testing.assert_array_equal(result["signal"].value, [1, 2, 3])
    assert result._gwexpy_io == source._gwexpy_io
    assert result._gwexpy_io is not source._gwexpy_io
    result._gwexpy_io["crop"] = "changed"
    assert "crop" not in source._gwexpy_io
