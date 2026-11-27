"""Tests for PyCBC interoperability.

These tests inject mock pycbc modules into ``sys.modules`` so they run
without requiring PyCBC to be installed.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.interop.pycbc_ import (
    from_pycbc_frequencyseries,
    from_pycbc_timeseries,
    to_pycbc_frequencyseries,
    to_pycbc_timeseries,
)
from gwexpy.timeseries import TimeSeries

# ---------------------------------------------------------------------------
# sys.modules injection helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def pycbc_modules_mock():
    """Inject fake pycbc and pycbc.types into sys.modules."""
    captured = {}

    class FakePyCBCTimeSeries:
        def __init__(self, data, delta_t, epoch):
            captured["ts"] = {
                "data": np.asarray(data),
                "delta_t": delta_t,
                "epoch": epoch,
            }

    class FakePyCBCFrequencySeries:
        def __init__(self, data, delta_f, epoch):
            captured["fs"] = {
                "data": np.asarray(data),
                "delta_f": delta_f,
                "epoch": epoch,
            }

    pycbc_types_mock = MagicMock()
    pycbc_types_mock.TimeSeries = FakePyCBCTimeSeries
    pycbc_types_mock.FrequencySeries = FakePyCBCFrequencySeries

    pycbc_mock = MagicMock()

    originals = {
        "pycbc": sys.modules.get("pycbc"),
        "pycbc.types": sys.modules.get("pycbc.types"),
    }
    sys.modules["pycbc"] = pycbc_mock
    sys.modules["pycbc.types"] = pycbc_types_mock
    yield pycbc_types_mock, captured
    for key, val in originals.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_pycbc_timeseries(data, start_time=1e9, delta_t=1 / 1024.0):
    pycbc_ts = MagicMock()
    arr = np.asarray(data, dtype=np.float64)
    pycbc_ts.numpy.return_value = arr
    pycbc_ts.start_time = start_time
    pycbc_ts.delta_t = delta_t
    pycbc_ts._unit = None
    return pycbc_ts


def _make_pycbc_frequencyseries(data, epoch=1e9, delta_f=1.0):
    pycbc_fs = MagicMock()
    arr = np.asarray(data)
    pycbc_fs.numpy.return_value = arr
    pycbc_fs.epoch = epoch
    pycbc_fs.delta_f = delta_f
    pycbc_fs._unit = None
    return pycbc_fs


# ---------------------------------------------------------------------------
# from_pycbc_timeseries
# ---------------------------------------------------------------------------


class TestFromPycbcTimeSeries:
    def test_basic_data_preserved(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        pycbc_ts = _make_pycbc_timeseries(data)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            ts = from_pycbc_timeseries(TimeSeries, pycbc_ts)

        np.testing.assert_allclose(ts.value, data)

    def test_start_time_preserved(self):
        data = np.zeros(256)
        pycbc_ts = _make_pycbc_timeseries(data, start_time=1234567890.5)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            ts = from_pycbc_timeseries(TimeSeries, pycbc_ts)

        assert float(ts.t0.value) == pytest.approx(1234567890.5)

    def test_delta_t_preserved(self):
        data = np.zeros(512)
        pycbc_ts = _make_pycbc_timeseries(data, delta_t=1 / 512.0)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            ts = from_pycbc_timeseries(TimeSeries, pycbc_ts)

        assert float(ts.dt.value) == pytest.approx(1 / 512.0)

    def test_returns_gwexpy_type(self):
        data = np.zeros(64)
        pycbc_ts = _make_pycbc_timeseries(data)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            ts = from_pycbc_timeseries(TimeSeries, pycbc_ts)

        assert isinstance(ts, TimeSeries)

    def test_sample_count(self):
        data = np.arange(1024, dtype=float)
        pycbc_ts = _make_pycbc_timeseries(data)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            ts = from_pycbc_timeseries(TimeSeries, pycbc_ts)

        assert len(ts) == 1024


# ---------------------------------------------------------------------------
# from_pycbc_frequencyseries
# ---------------------------------------------------------------------------


class TestFromPycbcFrequencySeries:
    def test_basic_data_preserved(self):
        data = np.array([1.0 + 0j, 2.0 + 1j, 3.0 - 1j])
        pycbc_fs = _make_pycbc_frequencyseries(data, delta_f=2.0)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            fs = from_pycbc_frequencyseries(FrequencySeries, pycbc_fs)

        np.testing.assert_allclose(fs.value, data)

    def test_frequencies_constructed(self):
        data = np.zeros(10, dtype=complex)
        pycbc_fs = _make_pycbc_frequencyseries(data, delta_f=2.0)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            fs = from_pycbc_frequencyseries(FrequencySeries, pycbc_fs)

        expected = np.arange(10) * 2.0
        np.testing.assert_allclose(fs.frequencies.value, expected)

    def test_epoch_preserved(self):
        data = np.zeros(64, dtype=complex)
        pycbc_fs = _make_pycbc_frequencyseries(data, epoch=1234567890.0)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            fs = from_pycbc_frequencyseries(FrequencySeries, pycbc_fs)

        assert float(fs.epoch.value) == pytest.approx(1234567890.0)

    def test_returns_gwexpy_type(self):
        data = np.zeros(32, dtype=complex)
        pycbc_fs = _make_pycbc_frequencyseries(data)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            fs = from_pycbc_frequencyseries(FrequencySeries, pycbc_fs)

        assert isinstance(fs, FrequencySeries)


# ---------------------------------------------------------------------------
# to_pycbc_timeseries
# ---------------------------------------------------------------------------


class TestToPycbcTimeSeries:
    def test_data_and_dt_passed(self, pycbc_modules_mock):
        _, captured = pycbc_modules_mock
        data = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries(data, t0=1e9, dt=1 / 1024.0)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            to_pycbc_timeseries(ts)

        np.testing.assert_allclose(captured["ts"]["data"], data)
        assert captured["ts"]["delta_t"] == pytest.approx(1 / 1024.0)
        assert captured["ts"]["epoch"] == pytest.approx(1e9)


# ---------------------------------------------------------------------------
# to_pycbc_frequencyseries
# ---------------------------------------------------------------------------


class TestToPycbcFrequencySeries:
    def test_df_and_epoch_passed(self, pycbc_modules_mock):
        _, captured = pycbc_modules_mock
        data = np.zeros(64, dtype=complex)
        freqs = np.arange(64) * 2.0
        fs = FrequencySeries(data, frequencies=freqs)

        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            to_pycbc_frequencyseries(fs)

        assert captured["fs"]["delta_f"] == pytest.approx(2.0)

    @pytest.mark.parametrize(
        "freqs",
        [
            np.geomspace(1.0, 10.0, 16),
            np.array([0.0, 1e-20, 3e-20, 6e-20]),
            np.array([3.0, 2.0, 1.0]),
        ],
    )
    def test_irregular_or_non_increasing_frequency_axis_raises(
        self, pycbc_modules_mock, freqs
    ):
        fs = FrequencySeries(np.zeros(len(freqs), dtype=complex), frequencies=freqs)
        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            with pytest.raises(ValueError, match="irregular|non-increasing"):
                to_pycbc_frequencyseries(fs)

    def test_near_regular_frequency_axis_rejects_lossy_export(self, pycbc_modules_mock):
        freqs = np.array([0.0, 1.0, 2.0 + 1e-7, 3.0 + 2e-7])
        fs = FrequencySeries(np.zeros(freqs.size), frequencies=freqs)
        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            with pytest.raises(ValueError, match="irregular"):
                to_pycbc_frequencyseries(fs)

    def test_gwpy_regular_jitter_axis_rejects_lossy_export(self, pycbc_modules_mock):
        # GWpy's absolute 1e-8 tolerance calls this axis regular (#887); its
        # first interval would misplace every odd sample by 1 ns.
        freqs = np.arange(100) * 1e-6 + np.where(np.arange(100) % 2, 1e-9, 0.0)
        fs = FrequencySeries(np.zeros(freqs.size), frequencies=freqs)
        assert fs.xindex.regular
        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            with pytest.raises(ValueError, match="irregular"):
                to_pycbc_frequencyseries(fs)

    @pytest.mark.parametrize(
        "fs",
        [
            FrequencySeries(np.zeros(300), f0=0.0, df=1 / 3),
            FrequencySeries(np.zeros(1000), frequencies=np.linspace(0, 10, 1000)),
        ],
        ids=["define-1/3", "linspace"],
    )
    def test_uniform_axis_with_float_roundoff_is_exported(self, pycbc_modules_mock, fs):
        _, captured = pycbc_modules_mock
        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            to_pycbc_frequencyseries(fs)
        freqs = fs.frequencies.value
        assert captured["fs"]["delta_f"] == freqs[1] - freqs[0]

    @pytest.mark.parametrize(
        "fs",
        [
            FrequencySeries(np.zeros(4), f0=1.0, df=1.0),
            FrequencySeries(np.zeros(100), f0=10.0, df=0.25),
            FrequencySeries(np.zeros(1), f0=20.0, df=1.0),
        ],
        ids=["f0=1", "cropped-f0=10", "single-sample"],
    )
    def test_nonzero_f0_raises_instead_of_shifting_frequencies(
        self, pycbc_modules_mock, fs
    ):
        # PyCBC frequency series start at 0 Hz; exporting f0 != 0 would move
        # every sample down by f0.
        _, captured = pycbc_modules_mock
        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            with pytest.raises(ValueError, match="nonzero initial frequency"):
                to_pycbc_frequencyseries(fs)
        assert "fs" not in captured

    def test_zero_f0_regular_export_unchanged(self, pycbc_modules_mock):
        _, captured = pycbc_modules_mock
        data = np.arange(9, dtype=complex)
        fs = FrequencySeries(data, f0=0.0, df=0.5, epoch=1234567890.0)
        with patch("gwexpy.interop.pycbc_.require_optional", return_value=MagicMock()):
            to_pycbc_frequencyseries(fs)
        np.testing.assert_array_equal(captured["fs"]["data"], data)
        assert captured["fs"]["delta_f"] == 0.5
        assert captured["fs"]["epoch"] == 1234567890.0


# ---------------------------------------------------------------------------
# Missing PyCBC (import error)
# ---------------------------------------------------------------------------


class TestMissingPycbc:
    def test_from_pycbc_timeseries_raises_importerror(self):
        pycbc_ts = _make_pycbc_timeseries(np.zeros(4))
        with patch(
            "gwexpy.interop.pycbc_.require_optional",
            side_effect=ImportError("pycbc not installed"),
        ):
            with pytest.raises(ImportError, match="pycbc"):
                from_pycbc_timeseries(TimeSeries, pycbc_ts)

    def test_from_pycbc_frequencyseries_raises_importerror(self):
        pycbc_fs = _make_pycbc_frequencyseries(np.zeros(4))
        with patch(
            "gwexpy.interop.pycbc_.require_optional",
            side_effect=ImportError("pycbc not installed"),
        ):
            with pytest.raises(ImportError, match="pycbc"):
                from_pycbc_frequencyseries(FrequencySeries, pycbc_fs)
