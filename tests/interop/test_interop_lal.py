"""Tests for LALSuite interoperability.

These tests inject a mock ``gwexpy.utils.lal`` into ``sys.modules`` so they
run without requiring LALSuite to be installed.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import astropy.units as u

# ---------------------------------------------------------------------------
# Module-level sys.modules injection for gwexpy.utils.lal
# ---------------------------------------------------------------------------


def _make_lal_utils_mock(unit=u.m):
    """Return a fake gwexpy.utils.lal module."""
    mod = MagicMock(spec=ModuleType)
    mod.from_lal_unit = MagicMock(return_value=unit)
    mod.LAL_TYPE_FROM_NUMPY = {np.dtype("float64").type: "REAL8", np.dtype("complex128").type: "COMPLEX16"}

    created_lalts = None

    def _create_ts(name, epoch, f0_unused, dt, unit, n):
        lalts = MagicMock()
        lalts.data = SimpleNamespace(data=np.zeros(n))
        return lalts

    mod.find_typed_function = MagicMock(return_value=_create_ts)
    mod.to_lal_ligotimegps = MagicMock(return_value=MagicMock())
    mod.to_lal_unit = MagicMock(return_value=MagicMock())
    return mod


@pytest.fixture()
def lal_utils_mock():
    """Inject a fake gwexpy.utils.lal into sys.modules for the duration of the test."""
    mock = _make_lal_utils_mock()
    original = sys.modules.get("gwexpy.utils.lal")
    sys.modules["gwexpy.utils.lal"] = mock
    yield mock
    if original is None:
        sys.modules.pop("gwexpy.utils.lal", None)
    else:
        sys.modules["gwexpy.utils.lal"] = original


# ---------------------------------------------------------------------------
# Mock LAL struct helpers
# ---------------------------------------------------------------------------


def _make_lal_timeseries(data, epoch=1e9, dt=1 / 1024.0, name="test", unit_str="m"):
    lalts = SimpleNamespace()
    lalts.data = SimpleNamespace(data=np.asarray(data, dtype=np.float64))
    lalts.epoch = epoch
    lalts.deltaT = dt
    lalts.name = name
    lalts.sampleUnits = unit_str
    return lalts


def _make_lal_frequencyseries(data, epoch=1e9, f0=0.0, df=1.0, name="test", unit_str="1/sqrt(Hz)"):
    lalfs = SimpleNamespace()
    lalfs.data = SimpleNamespace(data=np.asarray(data))
    lalfs.epoch = epoch
    lalfs.f0 = f0
    lalfs.deltaF = df
    lalfs.name = name
    lalfs.sampleUnits = unit_str
    return lalfs


# ---------------------------------------------------------------------------
# Imports (after fixture setup concept; done lazily to avoid circular issues)
# ---------------------------------------------------------------------------

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries
from gwexpy.interop.lal_ import (
    from_lal_frequencyseries,
    from_lal_timeseries,
    to_lal_frequencyseries,
    to_lal_timeseries,
)

# ---------------------------------------------------------------------------
# from_lal_timeseries
# ---------------------------------------------------------------------------


class TestFromLalTimeSeries:
    def test_basic_data_preserved(self, lal_utils_mock):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        lalts = _make_lal_timeseries(data)

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            ts = from_lal_timeseries(TimeSeries, lalts)

        np.testing.assert_allclose(ts.value, data)

    def test_epoch_preserved(self, lal_utils_mock):
        data = np.zeros(256)
        lalts = _make_lal_timeseries(data, epoch=1234567890.5)

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            ts = from_lal_timeseries(TimeSeries, lalts)

        assert float(ts.t0.value) == pytest.approx(1234567890.5)

    def test_dt_preserved(self, lal_utils_mock):
        data = np.zeros(512)
        lalts = _make_lal_timeseries(data, dt=1 / 512.0)

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            ts = from_lal_timeseries(TimeSeries, lalts)

        assert float(ts.dt.value) == pytest.approx(1 / 512.0)

    def test_name_preserved(self, lal_utils_mock):
        data = np.zeros(64)
        lalts = _make_lal_timeseries(data, name="K1:TEST-CHANNEL")

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            ts = from_lal_timeseries(TimeSeries, lalts)

        assert ts.name == "K1:TEST-CHANNEL"

    def test_returns_gwexpy_type(self, lal_utils_mock):
        data = np.zeros(64)
        lalts = _make_lal_timeseries(data)

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            ts = from_lal_timeseries(TimeSeries, lalts)

        assert isinstance(ts, TimeSeries)

    def test_copy_true(self, lal_utils_mock):
        data = np.array([1.0, 2.0, 3.0])
        lalts = _make_lal_timeseries(data)

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            ts = from_lal_timeseries(TimeSeries, lalts, copy=True)

        # Modifying original should not affect ts
        lalts.data.data[0] = 999.0
        assert ts.value[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# from_lal_frequencyseries
# ---------------------------------------------------------------------------


class TestFromLalFrequencySeries:
    def test_basic_data_preserved(self, lal_utils_mock):
        data = np.array([1.0 + 0j, 2.0 + 1j, 3.0 - 1j])
        lalfs = _make_lal_frequencyseries(data, f0=0.0, df=1.0)

        lal_utils_mock.from_lal_unit.return_value = u.Unit("1/sqrt(Hz)")
        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            fs = from_lal_frequencyseries(FrequencySeries, lalfs)

        np.testing.assert_allclose(fs.value, data)

    def test_frequencies_constructed(self, lal_utils_mock):
        data = np.zeros(10)
        lalfs = _make_lal_frequencyseries(data, f0=10.0, df=2.0)

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            fs = from_lal_frequencyseries(FrequencySeries, lalfs)

        expected = 10.0 + np.arange(10) * 2.0
        np.testing.assert_allclose(fs.frequencies.value, expected)

    def test_epoch_preserved(self, lal_utils_mock):
        data = np.zeros(64)
        lalfs = _make_lal_frequencyseries(data, epoch=1234567890.0)

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            fs = from_lal_frequencyseries(FrequencySeries, lalfs)

        assert float(fs.epoch.value) == pytest.approx(1234567890.0)

    def test_returns_gwexpy_type(self, lal_utils_mock):
        data = np.zeros(32)
        lalfs = _make_lal_frequencyseries(data)

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            fs = from_lal_frequencyseries(FrequencySeries, lalfs)

        assert isinstance(fs, FrequencySeries)


# ---------------------------------------------------------------------------
# to_lal_timeseries
# ---------------------------------------------------------------------------


class TestToLalTimeSeries:
    def test_data_transferred(self, lal_utils_mock):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        ts = TimeSeries(data, t0=1e9, dt=1 / 1024.0, name="K1:TEST", unit="m")

        lalts_mock = MagicMock()
        lalts_mock.data = SimpleNamespace(data=np.zeros(len(data)))
        lal_utils_mock.find_typed_function.return_value = lambda *a, **k: lalts_mock

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            result = to_lal_timeseries(ts)

        np.testing.assert_allclose(result.data.data, data)


# ---------------------------------------------------------------------------
# to_lal_frequencyseries
# ---------------------------------------------------------------------------


class TestToLalFrequencySeries:
    def test_data_transferred(self, lal_utils_mock):
        data = np.zeros(64)
        fs = FrequencySeries(data, frequencies=np.arange(64, dtype=float))

        lalfs_mock = MagicMock()
        lalfs_mock.data = SimpleNamespace(data=np.zeros(len(data)))
        lal_utils_mock.find_typed_function.return_value = lambda *a, **k: lalfs_mock

        with patch("gwexpy.interop.lal_.require_optional", return_value=MagicMock()):
            result = to_lal_frequencyseries(fs)

        np.testing.assert_allclose(result.data.data, data)


# ---------------------------------------------------------------------------
# Missing LAL (import error)
# ---------------------------------------------------------------------------


class TestMissingLal:
    def test_from_lal_timeseries_raises_importerror(self):
        lalts = _make_lal_timeseries(np.zeros(4))
        with patch(
            "gwexpy.interop.lal_.require_optional",
            side_effect=ImportError("lal not installed"),
        ):
            with pytest.raises(ImportError, match="lal"):
                from_lal_timeseries(TimeSeries, lalts)

    def test_from_lal_frequencyseries_raises_importerror(self):
        lalfs = _make_lal_frequencyseries(np.zeros(4))
        with patch(
            "gwexpy.interop.lal_.require_optional",
            side_effect=ImportError("lal not installed"),
        ):
            with pytest.raises(ImportError, match="lal"):
                from_lal_frequencyseries(FrequencySeries, lalfs)
