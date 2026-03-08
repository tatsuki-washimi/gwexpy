import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io import zarr_


class _FakeArray:
    def __init__(self, shape, dtype):
        self.data = np.empty(shape, dtype=dtype)
        self.attrs = {}

    def __setitem__(self, key, value):
        self.data[key] = value


class _FakeGroupV3:
    def __init__(self):
        self.created = {}

    def create_array(self, name, *, shape=None, dtype=None, data=None, overwrite=False):
        if data is not None and shape is not None:
            raise ValueError(
                "Either use the data parameter, or the shape parameter, but not both."
            )
        if shape is None:
            shape = data.shape
            dtype = data.dtype
        arr = _FakeArray(shape, dtype)
        if data is not None:
            arr[...] = data
        self.created[name] = arr
        return arr


class _FakeZarrV3:
    def __init__(self, group):
        self._group = group

    def open_group(self, target, mode="r", **kwargs):
        return self._group


def test_write_timeseriesdict_zarr_uses_shape_for_zarr_v3(monkeypatch):
    group = _FakeGroupV3()
    fake_zarr = _FakeZarrV3(group)
    monkeypatch.setattr(zarr_, "_import_zarr", lambda: fake_zarr)

    ts = TimeSeries(np.arange(8, dtype=float), dt=0.25, t0=100.0, unit="m")
    tsd = TimeSeriesDict({"chan": ts})

    zarr_.write_timeseriesdict_zarr(tsd, "unused-target")

    assert "chan" in group.created
    arr = group.created["chan"]
    assert np.allclose(arr.data, ts.value)
    assert arr.attrs["sample_rate"] == float(ts.sample_rate.value)
    assert arr.attrs["t0"] == float(ts.t0.value)
    assert arr.attrs["dt"] == float(ts.dt.value)
    assert arr.attrs["unit"] == str(ts.unit)


class _FakeGroupShapeError:
    def create_array(self, name, *, shape=None, dtype=None, data=None, overwrite=False):
        raise ValueError("shape mismatch")


class _FakeZarrShapeError:
    def open_group(self, target, mode="r", **kwargs):
        return _FakeGroupShapeError()


def test_write_timeseriesdict_zarr_does_not_swallow_value_error(monkeypatch):
    monkeypatch.setattr(zarr_, "_import_zarr", lambda: _FakeZarrShapeError())

    ts = TimeSeries(np.arange(4, dtype=float), dt=1.0, t0=0.0)
    tsd = TimeSeriesDict({"chan": ts})

    with pytest.raises(ValueError, match="shape mismatch"):
        zarr_.write_timeseriesdict_zarr(tsd, "unused-target")
