"""Tests for TDMS reader."""

import numpy as np
import pytest

nptdms = pytest.importorskip("nptdms")

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


class TestTdmsReader:
    def _write_tdms(self, path, group_name="Group", channel_name="Signal",
                    data=None, dt=0.001, wf_start_time=None):
        """Write a minimal TDMS file using nptdms."""
        from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

        if data is None:
            data = np.arange(100, dtype=np.float64)

        root = RootObject()
        group = GroupObject(group_name)
        props = {"wf_increment": dt}
        if wf_start_time is not None:
            props["wf_start_time"] = wf_start_time
        channel = ChannelObject(group_name, channel_name, data, properties=props)

        with TdmsWriter(str(path)) as writer:
            writer.write_segment([root, group, channel])

    def test_read_single_channel(self, tmp_path):
        path = tmp_path / "single.tdms"
        data = np.arange(50, dtype=np.float64)
        self._write_tdms(path, data=data, dt=0.01)

        tsd = TimeSeriesDict.read(str(path), format="tdms")
        assert len(tsd) == 1
        key = next(iter(tsd))
        assert "Group/Signal" in key
        np.testing.assert_allclose(tsd[key].value, data)
        assert np.isclose(tsd[key].dt.value, 0.01)

    def test_read_timeseries(self, tmp_path):
        path = tmp_path / "ts.tdms"
        self._write_tdms(path)

        ts = TimeSeries.read(str(path), format="tdms")
        assert len(ts) == 100

    def test_channel_naming(self, tmp_path):
        path = tmp_path / "naming.tdms"
        self._write_tdms(path, group_name="Grp", channel_name="Ch1")

        tsd = TimeSeriesDict.read(str(path), format="tdms")
        assert "Grp/Ch1" in tsd
