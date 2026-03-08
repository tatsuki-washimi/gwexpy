import numpy as np

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io import zarr_


class _FakeArray:
    def __init__(self, data):
        self.data = data
        self.attrs = {}


class _FakeGroupV3:
    def __init__(self):
        self.created = {}

    def create_array(self, name, *, data, shape, overwrite=False):
        if shape != data.shape:
            raise ValueError("shape mismatch")
        arr = _FakeArray(data)
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
