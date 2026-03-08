import numpy as np

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io import zarr_ as zarr_io


class _FakeArray:
    def __init__(self, data, attrs):
        self._data = np.asarray(data)
        self.attrs = attrs
        self.shape = self._data.shape

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._data
        return self._data[item]


class _FakeReadStore(dict):
    pass


class _FakeWriteStore:
    def __init__(self):
        self.arrays = {}

    def create_array(self, key, data, overwrite=True):
        arr = _FakeArray(data, {})
        self.arrays[key] = {
            "data": np.asarray(data),
            "overwrite": overwrite,
            "arr": arr,
        }
        return arr


class _FakeZarr:
    def __init__(self):
        self.calls = []
        self.read_store = _FakeReadStore(
            {
                "ch0": _FakeArray(
                    [1.0, 2.0, 3.0],
                    {"sample_rate": 10.0, "t0": 123.0, "unit": "m"},
                )
            }
        )
        self.write_store = _FakeWriteStore()

    def open_group(self, source, mode="r", **kwargs):
        self.calls.append({"source": source, "mode": mode, "kwargs": kwargs})
        if mode == "r":
            return self.read_store
        return self.write_store


def test_high_level_zarr_read_strips_gwpy_kwargs(monkeypatch):
    fake_zarr = _FakeZarr()
    monkeypatch.setattr(zarr_io, "_import_zarr", lambda: fake_zarr)

    out = TimeSeriesDict.read(
        object(),
        format="zarr",
        start=1,
        end=2,
        pad=np.nan,
        gap="pad",
        nproc=2,
        scaled=True,
        foo="bar",
    )

    assert "ch0" in out
    assert fake_zarr.calls[0]["mode"] == "r"
    assert fake_zarr.calls[0]["kwargs"] == {"foo": "bar"}


def test_write_timeseriesdict_zarr_strips_gwpy_kwargs(monkeypatch):
    fake_zarr = _FakeZarr()
    monkeypatch.setattr(zarr_io, "_import_zarr", lambda: fake_zarr)

    tsd = TimeSeriesDict(
        {"ch0": TimeSeries([1.0, 2.0], t0=0, sample_rate=4, unit="m")}
    )
    zarr_io.write_timeseriesdict_zarr(
        tsd,
        object(),
        start=1,
        end=2,
        pad=np.nan,
        gap="pad",
        nproc=2,
        scaled=True,
        compressor="zstd",
    )

    assert fake_zarr.calls[0]["mode"] == "w"
    assert fake_zarr.calls[0]["kwargs"] == {"compressor": "zstd"}
