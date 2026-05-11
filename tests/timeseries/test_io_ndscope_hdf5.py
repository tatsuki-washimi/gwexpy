"""Tests for ndscope HDF5 I/O (gwexpy.timeseries.io.ndscope_hdf5)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.timeseries.io.ndscope_hdf5 import (
    identify_ndscope_hdf5,
    read_timeseriesdict_ndscope_hdf5,
)

SAMPLE_HDF5 = Path(__file__).parent.parent / "fixtures" / "data" / "ndscope.h5"
_NDSCOPE_ACCEPTED_FORMATS = (
    "hdf.ndscope",
    "ndscope-hdf5",
    "ndscope_hdf5",
    "ndscopehdf5",
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
    ts = TimeSeries(np.arange(10.0), sample_rate=1.0, t0=0, unit="m", name="test")
    ts.write(str(path), format="hdf5")


def _make_timeseriesmatrix_hdf5(path):
    """Create a native TimeSeriesMatrix HDF5 file, not an ndscope file."""
    matrix = TimeSeriesMatrix(
        np.arange(24.0).reshape(2, 3, 4),
        t0=10.0,
        dt=0.5,
    )
    matrix.write(path, format="hdf5")
    return matrix


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
        tsd = TimeSeriesDict.read(str(p), format="hdf.ndscope")
        assert "K1:TEST-CHANNEL" in tsd

    @pytest.mark.parametrize("fmt", _NDSCOPE_ACCEPTED_FORMATS)
    def test_read_via_timeseriesdict_accepts_registered_aliases(self, tmp_path, fmt):
        """Accepted ndscope aliases should resolve to the same reader behavior."""
        p = tmp_path / f"{fmt}.hdf5"
        _make_ndscope_raw(p)

        tsd = TimeSeriesDict.read(str(p), format=fmt)

        assert list(tsd.keys()) == ["K1:TEST-CHANNEL"]
        np.testing.assert_allclose(
            tsd["K1:TEST-CHANNEL"].value, np.arange(256, dtype=np.float64)
        )

    @pytest.mark.parametrize("suffix", [".h5", ".hdf5"])
    def test_read_matrix_auto_detect_matches_explicit(self, tmp_path, suffix):
        p = tmp_path / f"auto_matrix{suffix}"
        channels = {
            "K1:TEST-CHANNEL-A": np.arange(256, dtype=np.float64),
            "K1:TEST-CHANNEL-B": np.arange(256, dtype=np.float64) + 1000.0,
        }
        _make_ndscope_raw(p, channels=channels)

        matrix_auto = TimeSeriesMatrix.read(str(p))
        matrix_format_none = TimeSeriesMatrix.read(str(p), format=None)
        matrix_explicit = TimeSeriesMatrix.read(str(p), format="hdf.ndscope")
        matrix_positional = TimeSeriesMatrix.read(str(p), "hdf.ndscope")
        matrix_from_dict = TimeSeriesDict.read(
            str(p), format="hdf.ndscope"
        ).to_matrix()

        assert matrix_auto.shape == matrix_explicit.shape == (2, 1, 256)
        np.testing.assert_allclose(matrix_auto.value, matrix_explicit.value)
        np.testing.assert_allclose(matrix_format_none.value, matrix_explicit.value)
        np.testing.assert_allclose(matrix_positional.value, matrix_explicit.value)
        np.testing.assert_allclose(matrix_from_dict.value, matrix_explicit.value)
        assert float(matrix_auto.sample_rate.value) == pytest.approx(
            float(matrix_explicit.sample_rate.value)
        )
        assert float(matrix_auto.t0.value) == pytest.approx(
            float(matrix_explicit.t0.value)
        )
        assert list(matrix_auto.channel_names) == [
            "K1:TEST-CHANNEL-A",
            "K1:TEST-CHANNEL-B",
        ]

    def test_read_matrix_auto_detect_channels_and_crop(self, tmp_path):
        p = tmp_path / "auto_matrix_channel.hdf5"
        channels = {
            "K1:CH_A": np.arange(400, dtype=np.float64),
            "K1:CH_B": np.arange(400, dtype=np.float64) + 2000.0,
        }
        _make_ndscope_raw(p, channels=channels)

        auto = TimeSeriesMatrix.read(
            str(p), channels=["K1:CH_B"], start=1_000_000_000.0, end=1_000_000_001.0
        )
        explicit = TimeSeriesMatrix.read(
            str(p),
            format="hdf.ndscope",
            channels=["K1:CH_B"],
            start=1_000_000_000.0,
            end=1_000_000_001.0,
        )

        assert auto.shape[0] == 1
        assert list(auto.channel_names) == ["K1:CH_B"]
        np.testing.assert_allclose(auto.value, explicit.value)
        assert float(auto.t0.value) == pytest.approx(float(explicit.t0.value))

    def test_read_matrix_non_ndscope_hdf5_does_not_auto_detect_ndscope(self, tmp_path):
        p = tmp_path / "native_matrix.hdf5"
        expected = _make_timeseriesmatrix_hdf5(p)

        assert identify_ndscope_hdf5(TimeSeriesDict, str(p), None) is False

        auto = TimeSeriesMatrix.read(str(p), format=None)
        explicit = TimeSeriesMatrix.read(str(p), format="hdf5")

        assert auto.shape == expected.shape == explicit.shape
        np.testing.assert_allclose(auto.value, explicit.value)
        np.testing.assert_allclose(auto.value, expected.value)

        with pytest.raises(ValueError):
            TimeSeriesMatrix.read(str(p), format="hdf.ndscope")

    def test_read_matrix_explicit_hdf5_does_not_invoke_ndscope_reader(self, tmp_path):
        p = tmp_path / "explicit_hdf5_ndscope.hdf5"
        _make_ndscope_raw(p)

        ndscope = TimeSeriesMatrix.read(str(p), format="hdf.ndscope")
        assert ndscope.shape == (1, 1, 256)

        with pytest.raises(KeyError, match="data"):
            TimeSeriesMatrix.read(str(p), format="hdf5")

    def test_read_via_timeseries(self, tmp_path):
        """Test that TimeSeries.read works with ndscope format."""
        p = tmp_path / "single.hdf5"
        _make_ndscope_raw(p)
        ts = TimeSeries.read(str(p), format="hdf.ndscope")
        assert ts.name == "K1:TEST-CHANNEL"
        assert len(ts) == 256

    @pytest.mark.parametrize("fmt", _NDSCOPE_ACCEPTED_FORMATS)
    def test_read_via_timeseries_accepts_registered_aliases(self, tmp_path, fmt):
        """Single-series reads should accept every registry alias exposed by Worker 1."""
        p = tmp_path / f"single_{fmt}.hdf5"
        _make_ndscope_raw(p)

        ts = TimeSeries.read(str(p), format=fmt)

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
        tsd.write(str(p), format="hdf.ndscope")

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

    @pytest.mark.parametrize("fmt", _NDSCOPE_ACCEPTED_FORMATS)
    def test_write_accepts_registered_aliases(self, tmp_path, fmt):
        """Writes should work through every accepted ndscope alias without new canonicals."""
        p = tmp_path / f"out_{fmt}.hdf5"
        ts = TimeSeries(
            np.arange(16, dtype=np.float64),
            sample_rate=8.0,
            t0=1_000_000_000.0,
            name="K1:WRITE-ALIAS",
            unit="m",
        )

        TimeSeriesDict({"K1:WRITE-ALIAS": ts}).write(str(p), format=fmt)

        reread = TimeSeriesDict.read(str(p), format="hdf.ndscope")
        assert list(reread.keys()) == ["K1:WRITE-ALIAS"]
        np.testing.assert_allclose(
            reread["K1:WRITE-ALIAS"].value, np.arange(16, dtype=np.float64)
        )

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
        tsd_orig.write(str(p), format="hdf.ndscope")

        tsd_read = TimeSeriesDict.read(str(p), format="hdf.ndscope")
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
        tsd.write(str(p), format="hdf.ndscope")

        # Verify internal structure: single group with 3 datasets
        with h5py.File(str(p), "r") as f:
            assert "K1:CH" in f
            grp = f["K1:CH"]
            assert set(grp.keys()) == {"mean", "min", "max"}

        tsd2 = TimeSeriesDict.read(str(p), format="hdf.ndscope")
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
            tsd.write(str(p), format="hdf.ndscope")

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
            tsd.write(str(p), format="hdf.ndscope")


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


# ---------------------------------------------------------------------------
# Real-data tests using tests/sample-data/ndscope.hdf5
# ---------------------------------------------------------------------------

_REAL_CHANNELS = [
    "K1:PEM-SEIS_BS_GND_EW_IN1_DQ",
    "K1:PEM-SEIS_BS_GND_NS_IN1_DQ",
    "K1:PEM-SEIS_BS_GND_UD_IN1_DQ",
]
_REAL_RATE_HZ = 512.0
_REAL_GPS_START = 1457936560.2988281
_REAL_UNIT = "ct"
_REAL_N_SAMPLES = 48677


@pytest.mark.skipif(not SAMPLE_HDF5.exists(), reason="sample data not found")
class TestRealData:
    def test_identify_real_file(self):
        assert identify_ndscope_hdf5(TimeSeriesDict, str(SAMPLE_HDF5), None) is True

    def test_read_all_channels(self):
        tsd = read_timeseriesdict_ndscope_hdf5(SAMPLE_HDF5)
        assert len(tsd) == 3

    def test_matrix_auto_detect_real_file_matches_explicit(self):
        auto = TimeSeriesMatrix.read(SAMPLE_HDF5)
        positional_none = TimeSeriesMatrix.read(SAMPLE_HDF5, None)
        explicit = TimeSeriesMatrix.read(SAMPLE_HDF5, format="hdf.ndscope")
        from_dict = TimeSeriesDict.read(SAMPLE_HDF5, format="hdf.ndscope").to_matrix()

        assert auto.shape == explicit.shape == from_dict.shape == (3, 1, _REAL_N_SAMPLES)
        assert list(auto.channel_names) == _REAL_CHANNELS
        assert float(auto.sample_rate.value) == pytest.approx(_REAL_RATE_HZ)
        assert float(auto.t0.value) == pytest.approx(_REAL_GPS_START, rel=1e-9)
        np.testing.assert_allclose(auto.value, explicit.value)
        np.testing.assert_allclose(positional_none.value, explicit.value)
        np.testing.assert_allclose(auto.value, from_dict.value)

    def test_channel_names(self):
        tsd = read_timeseriesdict_ndscope_hdf5(SAMPLE_HDF5)
        assert set(tsd.keys()) == set(_REAL_CHANNELS)

    def test_metadata(self):
        tsd = read_timeseriesdict_ndscope_hdf5(SAMPLE_HDF5)
        for ch in _REAL_CHANNELS:
            ts = tsd[ch]
            assert float(ts.sample_rate.value) == pytest.approx(_REAL_RATE_HZ)
            assert float(ts.t0.value) == pytest.approx(_REAL_GPS_START, rel=1e-9)
            assert str(ts.unit) == _REAL_UNIT

    def test_data_length(self):
        tsd = read_timeseriesdict_ndscope_hdf5(SAMPLE_HDF5)
        for ch in _REAL_CHANNELS:
            assert len(tsd[ch]) == _REAL_N_SAMPLES

    def test_channel_filter(self):
        target = _REAL_CHANNELS[0]
        tsd = read_timeseriesdict_ndscope_hdf5(SAMPLE_HDF5, channels=[target])
        assert list(tsd.keys()) == [target]

    def test_time_crop(self):
        start = _REAL_GPS_START + 10.0
        end = _REAL_GPS_START + 20.0
        tsd = read_timeseriesdict_ndscope_hdf5(SAMPLE_HDF5, start=start, end=end)
        for ch in _REAL_CHANNELS:
            ts = tsd[ch]
            assert float(ts.t0.value) == pytest.approx(start)
            assert float(ts.duration.value) == pytest.approx(10.0)
