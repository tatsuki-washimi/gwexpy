"""Tests for list/tuple source handling across TimeSeries readers (issue #441).

Every reader must either merge a list of paths into a single result
(when multi-file semantics are well defined) or raise a clear
``ValueError`` instead of an opaque backend ``TypeError``.
"""

import sqlite3
import struct

import numpy as np
import pytest
from scipy.io import wavfile

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io._multi import (
    expand_multi_source,
    read_multi_dict,
    reject_multi_source,
)

# ---------------------------------------------------------------------------
# Helper-level tests (no optional dependencies required)
# ---------------------------------------------------------------------------


class TestExpandMultiSource:
    def test_list(self):
        assert expand_multi_source(["a", "b"]) == ["a", "b"]

    def test_tuple(self):
        assert expand_multi_source(("a", "b")) == ["a", "b"]

    def test_single_path_returns_none(self):
        assert expand_multi_source("a.wav") is None

    def test_empty_list(self):
        assert expand_multi_source([]) == []


class TestRejectMultiSource:
    def test_list_raises(self):
        with pytest.raises(
            ValueError,
            match=r"format 'wav' does not support reading multiple files; "
            r"got 3 paths",
        ):
            reject_multi_source(["a", "b", "c"], "wav")

    def test_tuple_raises(self):
        with pytest.raises(ValueError, match="got 2 paths"):
            reject_multi_source(("a", "b"), "wav")

    def test_single_path_passes(self):
        reject_multi_source("a.wav", "wav")  # no error


class TestReadMultiDict:
    @staticmethod
    def _fake_reader(files):
        """Build a reader that maps source name -> prepared dict."""

        def reader(source, **kwargs):
            return files[source]

        return reader

    def test_distinct_channels_are_merged(self):
        files = {
            "f1": TimeSeriesDict(
                {"ch1": TimeSeries(np.ones(10), t0=0, dt=1, name="ch1")}
            ),
            "f2": TimeSeriesDict(
                {"ch2": TimeSeries(np.zeros(10), t0=0, dt=1, name="ch2")}
            ),
        }
        out = read_multi_dict(self._fake_reader(files), ["f1", "f2"], "test")
        assert set(out) == {"ch1", "ch2"}

    def test_contiguous_segments_are_concatenated(self):
        files = {
            "f1": TimeSeriesDict(
                {"ch": TimeSeries(np.arange(10.0), t0=0, dt=1, name="ch")}
            ),
            "f2": TimeSeriesDict(
                {"ch": TimeSeries(np.arange(10.0, 20.0), t0=10, dt=1, name="ch")}
            ),
        }
        out = read_multi_dict(self._fake_reader(files), ["f1", "f2"], "test")
        assert len(out["ch"]) == 20
        np.testing.assert_allclose(out["ch"].value, np.arange(20.0))

    def test_segments_are_sorted_by_start_time(self):
        files = {
            "f1": TimeSeriesDict(
                {"ch": TimeSeries(np.arange(10.0), t0=0, dt=1, name="ch")}
            ),
            "f2": TimeSeriesDict(
                {"ch": TimeSeries(np.arange(10.0, 20.0), t0=10, dt=1, name="ch")}
            ),
        }
        # pass the later file first
        out = read_multi_dict(self._fake_reader(files), ["f2", "f1"], "test")
        assert float(out["ch"].t0.value) == 0.0
        np.testing.assert_allclose(out["ch"].value, np.arange(20.0))

    def test_gap_is_padded_with_nan(self):
        files = {
            "f1": TimeSeriesDict(
                {"ch": TimeSeries(np.ones(10), t0=0, dt=1, name="ch")}
            ),
            "f2": TimeSeriesDict(
                {"ch": TimeSeries(np.ones(10), t0=15, dt=1, name="ch")}
            ),
        }
        out = read_multi_dict(self._fake_reader(files), ["f1", "f2"], "test")
        assert len(out["ch"]) == 25
        assert np.all(np.isnan(out["ch"].value[10:15]))

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="no test files provided"):
            read_multi_dict(self._fake_reader({}), [], "test")

    def test_overlap_raises_clear_error(self):
        files = {
            "f1": TimeSeriesDict(
                {"ch": TimeSeries(np.ones(10), t0=0, dt=1, name="ch")}
            ),
            "f2": TimeSeriesDict(
                {"ch": TimeSeries(np.ones(10), t0=0, dt=1, name="ch")}
            ),
        }
        with pytest.raises(ValueError, match="failed to merge channel 'ch'"):
            read_multi_dict(self._fake_reader(files), ["f1", "f2"], "test")


# ---------------------------------------------------------------------------
# WAV: multiple files are not meaningful -> clear error
# ---------------------------------------------------------------------------


class TestWavMultiSource:
    def _write_wav(self, path):
        wavfile.write(str(path), 1000, np.zeros(10, dtype=np.int16))

    @pytest.mark.parametrize("container", (list, tuple))
    def test_dict_reader_rejects_multiple_paths(self, tmp_path, container):
        from gwexpy.timeseries.io.wav import read_timeseriesdict_wav

        paths = []
        for i in range(2):
            p = tmp_path / f"f{i}.wav"
            self._write_wav(p)
            paths.append(str(p))
        with pytest.raises(
            ValueError,
            match=r"format 'wav' does not support reading multiple files; "
            r"got 2 paths",
        ):
            read_timeseriesdict_wav(container(paths))

    def test_single_reader_rejects_multiple_paths(self, tmp_path):
        from gwexpy.timeseries.io.wav import read_timeseries_wav

        p = tmp_path / "f.wav"
        self._write_wav(p)
        with pytest.raises(ValueError, match="does not support reading multiple"):
            read_timeseries_wav([str(p), str(p)])


# ---------------------------------------------------------------------------
# Audio (pydub): multiple files are not meaningful -> clear error
# (the error is raised before pydub is imported, so no skip needed)
# ---------------------------------------------------------------------------


class TestAudioMultiSource:
    @pytest.mark.parametrize("fmt", ("mp3", "flac", "ogg", "m4a"))
    def test_dict_reader_rejects_multiple_paths(self, fmt):
        from gwexpy.timeseries.io.audio import read_timeseriesdict_audio

        with pytest.raises(
            ValueError,
            match=rf"format '{fmt}' does not support reading multiple files; "
            r"got 2 paths",
        ):
            read_timeseriesdict_audio([f"a.{fmt}", f"b.{fmt}"], format_hint=fmt)

    def test_default_format_name(self):
        from gwexpy.timeseries.io.audio import read_timeseriesdict_audio

        with pytest.raises(ValueError, match="format 'audio' does not support"):
            read_timeseriesdict_audio(["a.mp3", "b.mp3"])


# ---------------------------------------------------------------------------
# TDMS: list of files merges channels / concatenates segments
# ---------------------------------------------------------------------------


class TestTdmsMultiSource:
    @staticmethod
    def _write_tdms(path, channel_name, data, dt, start):
        nptdms = pytest.importorskip("nptdms")
        from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

        del nptdms
        root = RootObject()
        group = GroupObject("Group")
        props = {"wf_increment": dt, "wf_start_time": start}
        channel = ChannelObject("Group", channel_name, data, properties=props)
        with TdmsWriter(str(path)) as writer:
            writer.write_segment([root, group, channel])

    def test_list_concatenates_same_channel(self, tmp_path):
        pytest.importorskip("nptdms")
        dt = 0.1
        p1 = tmp_path / "seg1.tdms"
        p2 = tmp_path / "seg2.tdms"
        self._write_tdms(
            p1, "Sig", np.arange(10.0), dt, np.datetime64("2024-01-01T00:00:00")
        )
        self._write_tdms(
            p2, "Sig", np.arange(10.0, 20.0), dt, np.datetime64("2024-01-01T00:00:01")
        )

        tsd = TimeSeriesDict.read([str(p1), str(p2)], format="tdms")
        assert len(tsd) == 1
        ts = tsd["Group/Sig"]
        assert len(ts) == 20
        np.testing.assert_allclose(ts.value, np.arange(20.0))

    def test_list_merges_distinct_channels(self, tmp_path):
        pytest.importorskip("nptdms")
        start = np.datetime64("2024-01-01T00:00:00")
        p1 = tmp_path / "a.tdms"
        p2 = tmp_path / "b.tdms"
        self._write_tdms(p1, "A", np.ones(5), 0.5, start)
        self._write_tdms(p2, "B", np.zeros(5), 0.5, start)

        tsd = TimeSeriesDict.read([str(p1), str(p2)], format="tdms")
        assert set(tsd) == {"Group/A", "Group/B"}

    def test_empty_list_raises(self):
        pytest.importorskip("nptdms")
        from gwexpy.timeseries.io.tdms import read_timeseriesdict_tdms

        with pytest.raises(ValueError, match="no tdms files provided"):
            read_timeseriesdict_tdms([])


# ---------------------------------------------------------------------------
# ATS: one channel per file; lists merge channels / concatenate segments
# ---------------------------------------------------------------------------


def _write_ats(
    path, *, n_samples=8, sample_freq=8.0, start_unix=1704067200, values=None
):
    """Write a minimal Metronix ATS file (header version 80)."""
    header_size = 1024
    header = bytearray(header_size)
    struct.pack_into("<H", header, 0x00, header_size)
    struct.pack_into("<h", header, 0x02, 80)  # header version
    struct.pack_into("<I", header, 0x04, n_samples)
    struct.pack_into("<f", header, 0x08, sample_freq)
    struct.pack_into("<I", header, 0x0C, start_unix)
    struct.pack_into("<d", header, 0x10, 1000.0)  # lsb_mV -> 1 V/count
    struct.pack_into("<h", header, 0xAA, 0)  # 32-bit data

    if values is None:
        values = np.arange(n_samples)
    data = np.asarray(values, dtype="<i4")

    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


class TestAtsMultiSource:
    def test_list_concatenates_segments(self, tmp_path):
        p1 = tmp_path / "seg1.ats"
        p2 = tmp_path / "seg2.ats"
        # 8 samples at 8 Hz = 1 second per file, contiguous
        _write_ats(p1, start_unix=1704067200, values=np.arange(8))
        _write_ats(p2, start_unix=1704067201, values=np.arange(8, 16))

        tsd = TimeSeriesDict.read([str(p1), str(p2)], format="ats")
        assert len(tsd) == 1
        ts = next(iter(tsd.values()))
        assert len(ts) == 16
        np.testing.assert_allclose(ts.value, np.arange(16.0))

    def test_timeseries_read_list(self, tmp_path):
        p1 = tmp_path / "seg1.ats"
        p2 = tmp_path / "seg2.ats"
        _write_ats(p1, start_unix=1704067200, values=np.arange(8))
        _write_ats(p2, start_unix=1704067201, values=np.arange(8, 16))

        ts = TimeSeries.read([str(p1), str(p2)], format="ats")
        assert len(ts) == 16

    def test_empty_list_raises(self):
        from gwexpy.timeseries.io.ats import read_timeseriesdict_ats

        with pytest.raises(ValueError, match="no ats files provided"):
            read_timeseriesdict_ats([])


# ---------------------------------------------------------------------------
# CSV: lists concatenate segments along the time axis
# ---------------------------------------------------------------------------


class TestCsvMultiSource:
    @staticmethod
    def _write_csv(path, t0, n=10, dt=0.25):
        # dt=0.25 is exactly representable in binary, so the sample
        # spacing inferred from each file matches exactly.
        lines = [f"{t0 + i * dt},{t0 + i * dt}" for i in range(n)]
        path.write_text("\n".join(lines) + "\n")

    def test_dict_reader_list(self, tmp_path):
        from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

        p1 = tmp_path / "seg1.csv"
        p2 = tmp_path / "seg2.csv"
        self._write_csv(p1, 0.0)
        self._write_csv(p2, 2.5)

        tsd = read_timeseriesdict_csv([str(p1), str(p2)])
        assert len(tsd) == 1
        ts = next(iter(tsd.values()))
        assert len(ts) == 20
        np.testing.assert_allclose(ts.value, np.arange(20) * 0.25, atol=1e-9)

    def test_timeseries_read_list(self, tmp_path):
        p1 = tmp_path / "seg1.csv"
        p2 = tmp_path / "seg2.csv"
        self._write_csv(p1, 0.0)
        self._write_csv(p2, 2.5)

        ts = TimeSeries.read([str(p1), str(p2)], format="csv")
        assert len(ts) == 20

    def test_empty_list_raises(self):
        from gwexpy.timeseries.io.csv_enhanced import read_timeseriesdict_csv

        with pytest.raises(ValueError, match="no csv files provided"):
            read_timeseriesdict_csv([])


# ---------------------------------------------------------------------------
# NetCDF4: lists merge variables / concatenate segments
# ---------------------------------------------------------------------------


class TestNetcdf4MultiSource:
    def test_list_concatenates_and_merges(self, tmp_path):
        pytest.importorskip("xarray")
        pytest.importorskip("netCDF4")
        from gwexpy.timeseries.io.netcdf4_ import (
            read_timeseriesdict_netcdf4,
            write_timeseriesdict_netcdf4,
        )

        p1 = tmp_path / "seg1.nc"
        p2 = tmp_path / "seg2.nc"
        t0 = 1000000000
        write_timeseriesdict_netcdf4(
            TimeSeriesDict({"a": TimeSeries(np.arange(10.0), t0=t0, dt=1.0, name="a")}),
            str(p1),
        )
        write_timeseriesdict_netcdf4(
            TimeSeriesDict(
                {
                    "a": TimeSeries(
                        np.arange(10.0, 20.0), t0=t0 + 10, dt=1.0, name="a"
                    ),
                    "b": TimeSeries(np.ones(10), t0=t0 + 10, dt=1.0, name="b"),
                }
            ),
            str(p2),
        )

        tsd = read_timeseriesdict_netcdf4([str(p1), str(p2)])
        assert set(tsd) == {"a", "b"}
        assert len(tsd["a"]) == 20
        np.testing.assert_allclose(tsd["a"].value, np.arange(20.0))

    def test_empty_list_raises(self):
        from gwexpy.timeseries.io.netcdf4_ import read_timeseriesdict_netcdf4

        with pytest.raises(ValueError, match="no nc files provided"):
            read_timeseriesdict_netcdf4([])


# ---------------------------------------------------------------------------
# GBD: lists concatenate segments along the time axis
# ---------------------------------------------------------------------------


def _write_gbd(path, start_str, values):
    """Write a minimal single-channel GRAPHTEC GBD file."""
    size = len(values)
    header_lines = [
        "HeaderSiz=0",
        f"$$Time Start={start_str}",
        "$$Data Sample=1",
        "$$Data Type=Little,Int16",
        "$$Data Order=CH1",
        f"$$Data Counts={size}",
    ]
    while True:
        header = "\r\n".join(header_lines) + "\r\n"
        sz = len(header.encode("ascii"))
        new_line = f"HeaderSiz={sz}"
        if header_lines[0] == new_line:
            break
        header_lines[0] = new_line

    data = np.asarray(values, dtype="<i2")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


class TestGbdMultiSource:
    def test_list_concatenates_segments(self, tmp_path):
        from gwexpy.timeseries.io.gbd import read_timeseriesdict_gbd

        p1 = tmp_path / "seg1.gbd"
        p2 = tmp_path / "seg2.gbd"
        _write_gbd(p1, "2024/01/01 00:00:00", np.arange(10))
        _write_gbd(p2, "2024/01/01 00:00:10", np.arange(10, 20))

        tsd = read_timeseriesdict_gbd([str(p1), str(p2)], timezone="UTC")
        assert "CH1" in tsd
        assert len(tsd["CH1"]) == 20
        np.testing.assert_allclose(tsd["CH1"].value, np.arange(20.0))

    def test_dict_read_dispatch(self, tmp_path):
        p1 = tmp_path / "seg1.gbd"
        p2 = tmp_path / "seg2.gbd"
        _write_gbd(p1, "2024/01/01 00:00:00", np.arange(10))
        _write_gbd(p2, "2024/01/01 00:00:10", np.arange(10, 20))

        tsd = TimeSeriesDict.read([str(p1), str(p2)], format="gbd", timezone="UTC")
        assert len(tsd["CH1"]) == 20

    def test_empty_list_raises(self):
        from gwexpy.timeseries.io.gbd import read_timeseriesdict_gbd

        with pytest.raises(ValueError, match="no gbd files provided"):
            read_timeseriesdict_gbd([], timezone="UTC")


# ---------------------------------------------------------------------------
# ndscope HDF5: lists merge channels / concatenate segments
# ---------------------------------------------------------------------------


class TestNdscopeHdf5MultiSource:
    def test_list_concatenates_segments(self, tmp_path):
        pytest.importorskip("h5py")
        from gwexpy.timeseries.io.ndscope_hdf5 import (
            read_timeseriesdict_ndscope_hdf5,
            write_timeseriesdict_ndscope_hdf5,
        )

        p1 = tmp_path / "seg1.hdf5"
        p2 = tmp_path / "seg2.hdf5"
        t0 = 1300000000
        write_timeseriesdict_ndscope_hdf5(
            TimeSeriesDict({"X1:TEST": TimeSeries(np.arange(10.0), t0=t0, dt=1.0)}),
            str(p1),
        )
        write_timeseriesdict_ndscope_hdf5(
            TimeSeriesDict(
                {"X1:TEST": TimeSeries(np.arange(10.0, 20.0), t0=t0 + 10, dt=1.0)}
            ),
            str(p2),
        )

        tsd = read_timeseriesdict_ndscope_hdf5([str(p1), str(p2)])
        assert len(tsd["X1:TEST"]) == 20
        np.testing.assert_allclose(tsd["X1:TEST"].value, np.arange(20.0))

    def test_empty_list_raises(self):
        from gwexpy.timeseries.io.ndscope_hdf5 import (
            read_timeseriesdict_ndscope_hdf5,
        )

        with pytest.raises(ValueError, match="no hdf.ndscope files provided"):
            read_timeseriesdict_ndscope_hdf5([])


# ---------------------------------------------------------------------------
# Zarr: lists merge channels / concatenate segments
# ---------------------------------------------------------------------------


class TestZarrMultiSource:
    def test_list_concatenates_segments(self, tmp_path):
        pytest.importorskip("zarr")
        from gwexpy.timeseries.io.zarr_ import (
            read_timeseriesdict_zarr,
            write_timeseriesdict_zarr,
        )

        p1 = tmp_path / "seg1.zarr"
        p2 = tmp_path / "seg2.zarr"
        write_timeseriesdict_zarr(
            TimeSeriesDict(
                {"ch": TimeSeries(np.arange(10.0), t0=0, dt=1.0, name="ch")}
            ),
            str(p1),
        )
        write_timeseriesdict_zarr(
            TimeSeriesDict(
                {"ch": TimeSeries(np.arange(10.0, 20.0), t0=10, dt=1.0, name="ch")}
            ),
            str(p2),
        )

        tsd = read_timeseriesdict_zarr([str(p1), str(p2)])
        assert len(tsd["ch"]) == 20
        np.testing.assert_allclose(tsd["ch"].value, np.arange(20.0))

    def test_empty_list_raises(self):
        pytest.importorskip("zarr")
        from gwexpy.timeseries.io.zarr_ import read_timeseriesdict_zarr

        with pytest.raises(ValueError, match="no zarr files provided"):
            read_timeseriesdict_zarr([])


# ---------------------------------------------------------------------------
# SDB: lists concatenate archive databases along the time axis
# ---------------------------------------------------------------------------


def _write_sdb(path, start_unix, n_records=10, interval=1, offset=0.0):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE archive (dateTime INTEGER, barometer REAL)")
    for i in range(n_records):
        conn.execute(
            "INSERT INTO archive VALUES (?, ?)",
            (start_unix + i * interval, 29.92 + offset),
        )
    conn.commit()
    conn.close()


class TestSdbMultiSource:
    def test_list_concatenates_segments(self, tmp_path):
        from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

        p1 = tmp_path / "seg1.sdb"
        p2 = tmp_path / "seg2.sdb"
        _write_sdb(p1, 1700000000)
        _write_sdb(p2, 1700000010, offset=1.0)

        tsd = read_timeseriesdict_sdb([str(p1), str(p2)])
        assert "barometer" in tsd
        assert len(tsd["barometer"]) == 20

    def test_empty_list_raises(self):
        from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

        with pytest.raises(ValueError, match="no sdb files provided"):
            read_timeseriesdict_sdb([])


# ---------------------------------------------------------------------------
# WIN: list handling goes through the shared helper (requires obspy)
# ---------------------------------------------------------------------------


class TestWinMultiSource:
    def test_empty_list_raises(self):
        pytest.importorskip("obspy")
        from gwexpy.timeseries.io.win import read_win_file

        with pytest.raises(ValueError, match="no win files provided"):
            read_win_file([])
