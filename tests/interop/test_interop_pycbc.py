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
from gwexpy.timeseries import TimeSeries
from gwexpy.interop.pycbc_ import (
    from_pycbc_frequencyseries,
    from_pycbc_timeseries,
    to_pycbc_frequencyseries,
    to_pycbc_timeseries,
)

# ---------------------------------------------------------------------------
# sys.modules injection helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def pycbc_modules_mock():
    """Inject fake pycbc and pycbc.types into sys.modules."""
    captured = {}

    class FakePyCBCTimeSeries:
        def __init__(self, data, delta_t, epoch):
            captured["ts"] = {"data": np.asarray(data), "delta_t": delta_t, "epoch": epoch}

    class FakePyCBCFrequencySeries:
        def __init__(self, data, delta_f, epoch):
            captured["fs"] = {"data": np.asarray(data), "delta_f": delta_f, "epoch": epoch}

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
