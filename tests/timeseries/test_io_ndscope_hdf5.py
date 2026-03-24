"""Tests for ndscope HDF5 I/O (gwexpy.timeseries.io.ndscope_hdf5)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io.ndscope_hdf5 import (
    identify_ndscope_hdf5,
    read_timeseriesdict_ndscope_hdf5,
)

# ---------------------------------------------------------------------------
# Helpers to create synthetic ndscope HDF5 files
# ---------------------------------------------------------------------------


def _make_ndscope_raw(path, channels=None):
    """Create a minimal ndscope HDF5 file with raw data."""
    if channels is None:
        channels = {"K1:TEST-CHANNEL": np.arange(256, dtype=np.float64)}
    with h5py.File(str(path), "w") as f:
        for ch_name, data in channels.items():
            grp = f.create_group(ch_name)
            grp.create_dataset("raw", data=data)
            grp.attrs["rate_hz"] = 256.0
            grp.attrs["gps_start"] = 1_000_000_000.0
            grp.attrs["unit"] = "m"


def _make_ndscope_trend(path):
    """Create an ndscope HDF5 file with trend data (mean/min/max)."""
    n = 100
    with h5py.File(str(path), "w") as f:
        grp = f.create_group("K1:TREND-CHANNEL")
        grp.create_dataset("mean", data=np.ones(n))
        grp.create_dataset("min", data=np.zeros(n))
        grp.create_dataset("max", data=np.ones(n) * 2)
        grp.attrs["rate_hz"] = 1.0
        grp.attrs["gps_start"] = 1_000_000_000.0
        grp.attrs["unit"] = "ct"


def _make_gwpy_hdf5(path):
    """Create a gwpy-native HDF5 file (flat datasets, not groups)."""
    from astropy import units as u

    ts = TimeSeries(np.arange(10.0), sample_rate=1.0, t0=0, unit="m", name="test")
    ts.write(str(path), format="hdf5")


# ---------------------------------------------------------------------------
# Identifier tests
# ---------------------------------------------------------------------------


class TestIdentify:
    def test_identifies_ndscope_raw(self, tmp_path):
        p = tmp_path / "ndscope_raw.hdf5"
        _make_ndscope_raw(p)
        assert identify_ndscope_hdf5(TimeSeriesDict, str(p), None) is True

    def test_identifies_ndscope_h5_extension(self, tmp_path):
        p = tmp_path / "ndscope_raw.h5"
        _make_ndscope_raw(p)
        assert identify_ndscope_hdf5(TimeSeriesDict, str(p), None) is True

    def test_rejects_gwpy_hdf5(self, tmp_path):
        p = tmp_path / "gwpy_native.hdf5"
        _make_gwpy_hdf5(p)
        assert identify_ndscope_hdf5(TimeSeriesDict, str(p), None) is False

    def test_rejects_non_hdf5(self, tmp_path):
        p = tmp_path / "not_hdf5.txt"
        p.write_text("hello")
        assert identify_ndscope_hdf5(TimeSeriesDict, str(p), None) is False

    def test_rejects_none_filepath(self):
        assert identify_ndscope_hdf5(TimeSeriesDict, None, None) is False

    def test_rejects_wrong_extension(self, tmp_path):
        p = tmp_path / "ndscope.dat"
        _make_ndscope_raw(tmp_path / "temp.hdf5")
        # copy with wrong extension
        import shutil

        shutil.copy(tmp_path / "temp.hdf5", p)
        assert identify_ndscope_hdf5(TimeSeriesDict, str(p), None) is False


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


class TestRead:
    def test_read_raw(self, tmp_path):
        p = tmp_path / "raw.hdf5"
        data = np.arange(256, dtype=np.float64)
        _make_ndscope_raw(p, {"K1:TEST-CHANNEL": data})

        tsd = read_timeseriesdict_ndscope_hdf5(p)
        assert "K1:TEST-CHANNEL" in tsd
        ts = tsd["K1:TEST-CHANNEL"]
        np.testing.assert_allclose(ts.value, data)
        assert float(ts.sample_rate.value) == 256.0
        assert float(ts.t0.value) == 1_000_000_000.0

    def test_read_trend(self, tmp_path):
        p = tmp_path / "trend.hdf5"
        _make_ndscope_trend(p)

        tsd = read_timeseriesdict_ndscope_hdf5(p)
        assert "K1:TREND-CHANNEL.max" in tsd
        assert "K1:TREND-CHANNEL.mean" in tsd
        assert "K1:TREND-CHANNEL.min" in tsd
        np.testing.assert_allclose(tsd["K1:TREND-CHANNEL.min"].value, 0.0)
        np.testing.assert_allclose(tsd["K1:TREND-CHANNEL.max"].value, 2.0)

    def test_read_multi_channel(self, tmp_path):
        p = tmp_path / "multi.hdf5"
        _make_ndscope_raw(
            p,
            {
                "K1:CH_A": np.ones(100),
                "K1:CH_B": np.zeros(100),
            },
        )
        tsd = read_timeseriesdict_ndscope_hdf5(p)
        assert len(tsd) == 2
        assert "K1:CH_A" in tsd
        assert "K1:CH_B" in tsd

    def test_read_channel_filter(self, tmp_path):
        p = tmp_path / "filter.hdf5"
        _make_ndscope_raw(
            p,
            {
                "K1:CH_A": np.ones(100),
                "K1:CH_B": np.zeros(100),
            },
        )
        tsd = read_timeseriesdict_ndscope_hdf5(p, channels=["K1:CH_A"])
        assert list(tsd.keys()) == ["K1:CH_A"]

    def test_read_with_crop(self, tmp_path):
        p = tmp_path / "crop.hdf5"
        n = 1024
        with h5py.File(str(p), "w") as f:
            grp = f.create_group("K1:TEST")
            grp.create_dataset("raw", data=np.arange(n, dtype=np.float64))
            grp.attrs["rate_hz"] = 256.0
            grp.attrs["gps_start"] = 1_000_000_000.0
            grp.attrs["unit"] = "m"

        # Full span: 1e9 to 1e9 + 1024/256 = 1e9 + 4.0
        tsd = read_timeseriesdict_ndscope_hdf5(
            p, start=1_000_000_001.0, end=1_000_000_003.0
        )
        ts = tsd["K1:TEST"]
        assert float(ts.t0.value) == pytest.approx(1_000_000_001.0)
        duration = float(ts.duration.value)
        assert duration == pytest.approx(2.0)

    def test_read_via_timeseriesdict(self, tmp_path):
        """Test that TimeSeriesDict.read auto-detects ndscope format."""
        p = tmp_path / "autodetect.hdf5"
        _make_ndscope_raw(p)
        tsd = TimeSeriesDict.read(str(p), format="ndscope-hdf5")
        assert "K1:TEST-CHANNEL" in tsd

    def test_read_via_timeseries(self, tmp_path):
        """Test that TimeSeries.read works with ndscope format."""
        p = tmp_path / "single.hdf5"
        _make_ndscope_raw(p)
        ts = TimeSeries.read(str(p), format="ndscope-hdf5")
        assert ts.name == "K1:TEST-CHANNEL"
        assert len(ts) == 256


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------


class TestWrite:
    def test_write_basic(self, tmp_path):
        p = tmp_path / "out.hdf5"
        ts = TimeSeries(
            np.arange(100, dtype=np.float64),
            sample_rate=100.0,
            t0=1_000_000_000.0,
            name="K1:WRITE-TEST",
            unit="m",
        )
        tsd = TimeSeriesDict({"K1:WRITE-TEST": ts})
        tsd.write(str(p), format="ndscope-hdf5")

        # Verify HDF5 structure
        with h5py.File(str(p), "r") as f:
            assert "K1:WRITE-TEST" in f
            grp = f["K1:WRITE-TEST"]
            assert isinstance(grp, h5py.Group)
            assert "raw" in grp
            assert float(grp.attrs["rate_hz"]) == 100.0
            assert float(grp.attrs["gps_start"]) == 1_000_000_000.0
            assert grp.attrs["unit"] == "m"
            np.testing.assert_allclose(grp["raw"][:], np.arange(100, dtype=np.float64))

    def test_roundtrip(self, tmp_path):
        p = tmp_path / "roundtrip.hdf5"
        data = np.random.default_rng(42).standard_normal(512)
        ts = TimeSeries(
            data,
            sample_rate=256.0,
            t0=1_000_000_000.0,
            name="K1:ROUNDTRIP",
            unit="V",
        )
        tsd_orig = TimeSeriesDict({"K1:ROUNDTRIP": ts})
        tsd_orig.write(str(p), format="ndscope-hdf5")

        tsd_read = TimeSeriesDict.read(str(p), format="ndscope-hdf5")
        assert "K1:ROUNDTRIP" in tsd_read
        np.testing.assert_allclose(tsd_read["K1:ROUNDTRIP"].value, data)
        assert float(tsd_read["K1:ROUNDTRIP"].sample_rate.value) == 256.0
        assert float(tsd_read["K1:ROUNDTRIP"].t0.value) == 1_000_000_000.0

    def test_write_trend_roundtrip(self, tmp_path):
        """Trend-style keys (ch.mean, ch.min, ch.max) roundtrip correctly."""
        p = tmp_path / "trend_rt.hdf5"
        n = 60
        tsd = TimeSeriesDict(
            {
                "K1:CH.mean": TimeSeries(
                    np.ones(n), sample_rate=1.0, t0=1e9, name="K1:CH.mean", unit="ct"
                ),
                "K1:CH.min": TimeSeries(
                    np.zeros(n), sample_rate=1.0, t0=1e9, name="K1:CH.min", unit="ct"
                ),
                "K1:CH.max": TimeSeries(
                    np.ones(n) * 2,
                    sample_rate=1.0,
                    t0=1e9,
                    name="K1:CH.max",
                    unit="ct",
                ),
            }
        )
        tsd.write(str(p), format="ndscope-hdf5")

        # Verify internal structure: single group with 3 datasets
        with h5py.File(str(p), "r") as f:
            assert "K1:CH" in f
            grp = f["K1:CH"]
            assert set(grp.keys()) == {"mean", "min", "max"}

        tsd2 = TimeSeriesDict.read(str(p), format="ndscope-hdf5")
        assert set(tsd2.keys()) == {"K1:CH.max", "K1:CH.mean", "K1:CH.min"}
        np.testing.assert_allclose(tsd2["K1:CH.min"].value, 0.0)

    def test_write_inconsistent_rate_raises(self, tmp_path):
        """Writer must reject series with mismatched sample_rate in same group."""
        p = tmp_path / "bad_rate.hdf5"
        tsd = TimeSeriesDict(
            {
                "K1:CH.mean": TimeSeries(
                    np.ones(60), sample_rate=1.0, t0=1e9, name="K1:CH.mean", unit="ct"
                ),
                "K1:CH.min": TimeSeries(
                    np.zeros(120),
                    sample_rate=2.0,  # different!
                    t0=1e9,
                    name="K1:CH.min",
                    unit="ct",
                ),
            }
        )
        with pytest.raises(ValueError, match="sample_rate"):
            tsd.write(str(p), format="ndscope-hdf5")

    def test_write_inconsistent_t0_raises(self, tmp_path):
        """Writer must reject series with mismatched gps_start in same group."""
        p = tmp_path / "bad_t0.hdf5"
        tsd = TimeSeriesDict(
            {
                "K1:CH.mean": TimeSeries(
                    np.ones(60), sample_rate=1.0, t0=1e9, name="K1:CH.mean", unit="ct"
                ),
                "K1:CH.min": TimeSeries(
                    np.zeros(60),
                    sample_rate=1.0,
                    t0=1e9 + 1,  # different!
                    name="K1:CH.min",
                    unit="ct",
                ),
            }
        )
        with pytest.raises(ValueError, match="gps_start"):
            tsd.write(str(p), format="ndscope-hdf5")


# ---------------------------------------------------------------------------
# Regression tests (covering PR review findings)
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_raw_always_uses_bare_channel_name(self, tmp_path):
        """raw dataset key must always be grp_name, even when mean/min/max co-exist."""
        p = tmp_path / "mixed.hdf5"
        n = 64
        with h5py.File(str(p), "w") as f:
            grp = f.create_group("K1:CH")
            grp.create_dataset("raw", data=np.ones(n))
            grp.create_dataset("mean", data=np.ones(n) * 0.5)
            grp.create_dataset("min", data=np.zeros(n))
            grp.create_dataset("max", data=np.ones(n) * 2)
            grp.attrs["rate_hz"] = 16.0
            grp.attrs["gps_start"] = 1e9
            grp.attrs["unit"] = "ct"

        tsd = read_timeseriesdict_ndscope_hdf5(p)

        # raw must always be accessible as the bare channel name
        assert "K1:CH" in tsd, "raw dataset must map to bare grp_name 'K1:CH'"
        # trend datasets must have suffixes
        assert "K1:CH.mean" in tsd
        assert "K1:CH.min" in tsd
        assert "K1:CH.max" in tsd
        # raw must NOT appear with a .raw suffix
        assert "K1:CH.raw" not in tsd, "raw must not be renamed to 'K1:CH.raw'"
