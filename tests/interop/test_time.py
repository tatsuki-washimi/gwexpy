"""Tests for gwexpy/interop/_time.py."""
from __future__ import annotations

from datetime import UTC, datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from gwexpy.interop._time import (
    LeapSecondConversionError,
    datetime_utc_to_gps,
    gps_to_datetime_utc,
)


# GPS epoch: 1980-01-06 00:00:00 UTC = GPS 0
GPS_EPOCH = 630763213  # 2000-01-01 00:00:00 UTC in GPS time (approx)


class TestGpsToDatetimeUtc:
    def test_gps_zero_is_epoch(self):
        """GPS 0 corresponds to 1980-01-06 00:00:00 UTC."""
        dt = gps_to_datetime_utc(0)
        assert dt.year == 1980
        assert dt.month == 1
        assert dt.day == 6
        assert dt.tzinfo is not None

    def test_returns_utc_aware_datetime(self):
        dt = gps_to_datetime_utc(0)
        assert dt.tzinfo == UTC

    def test_known_gps_time(self):
        """GPS 1000000000 → known UTC value."""
        dt = gps_to_datetime_utc(1000000000)
        assert isinstance(dt, datetime)
        assert dt.year == 2011

    def test_float_input(self):
        dt = gps_to_datetime_utc(1000000000.5)
        assert isinstance(dt, datetime)

    def test_default_leap_policy_is_raise(self):
        """Default leap policy must be 'raise'."""
        import inspect
        sig = inspect.signature(gps_to_datetime_utc)
        assert sig.parameters["leap"].default == "raise"

    def test_invalid_leap_policy_raises_not_implemented(self):
        """Unknown leap policy raises NotImplementedError."""
        with patch("gwexpy.interop._time.Time", return_value=self._make_leap_mock("second must be in 0..59")):
            with pytest.raises(NotImplementedError, match="not implemented"):
                gps_to_datetime_utc(0, leap="unknown")

    def test_non_leap_value_error_reraises(self):
        """ValueError unrelated to leap seconds is re-raised."""
        with patch("gwexpy.interop._time.Time", return_value=self._make_leap_mock("something unrelated")):
            with pytest.raises(ValueError, match="something unrelated"):
                gps_to_datetime_utc(0, leap="raise")

    def test_leap_raise_policy(self):
        """leap='raise' raises LeapSecondConversionError on leap second."""
        with patch("gwexpy.interop._time.Time", return_value=self._make_leap_mock()):
            with pytest.raises(LeapSecondConversionError):
                gps_to_datetime_utc(0, leap="raise")

    def _make_leap_mock(self, error_msg="leap second"):
        """Helper: return a mock Time() that triggers leap ValueError."""
        from types import SimpleNamespace
        ymdhms = SimpleNamespace(year=2016, month=12, day=31, hour=23, minute=59)

        def _to_datetime(**kwargs):
            raise ValueError(error_msg)

        fake_utc = SimpleNamespace(to_datetime=_to_datetime, ymdhms=ymdhms)
        fake_t = SimpleNamespace(utc=fake_utc)
        return fake_t

    def test_leap_floor_policy(self):
        """leap='floor' clamps to second=59, microsecond=999999."""
        with patch("gwexpy.interop._time.Time", return_value=self._make_leap_mock()):
            dt = gps_to_datetime_utc(0, leap="floor")
        assert dt.second == 59
        assert dt.microsecond == 999999
        assert dt.tzinfo == UTC

    def test_leap_ceil_policy(self):
        """leap='ceil' rounds up to next minute."""
        with patch("gwexpy.interop._time.Time", return_value=self._make_leap_mock()):
            dt = gps_to_datetime_utc(0, leap="ceil")
        # 2016-12-31 23:59 + 1 min = 2017-01-01 00:00:00
        assert dt == datetime(2017, 1, 1, 0, 0, 0, tzinfo=UTC)


class TestDatetimeUtcToGps:
    def test_aware_datetime(self):
        """Aware UTC datetime converts without error."""
        dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC)
        gps = datetime_utc_to_gps(dt)
        assert float(gps) > 0

    def test_naive_treated_as_utc(self):
        """Naive datetime is treated as UTC."""
        dt_naive = datetime(2000, 1, 1, 0, 0, 0)
        dt_aware = datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC)
        gps_naive = datetime_utc_to_gps(dt_naive)
        gps_aware = datetime_utc_to_gps(dt_aware)
        assert float(gps_naive) == pytest.approx(float(gps_aware))

    def test_roundtrip(self):
        """GPS → datetime → GPS is stable."""
        gps_orig = 1000000000
        dt = gps_to_datetime_utc(gps_orig)
        gps_back = datetime_utc_to_gps(dt)
        assert float(gps_back) == pytest.approx(gps_orig, abs=1e-3)

    def test_epoch_value(self):
        """GPS epoch (1980-01-06) should map to GPS 0."""
        dt = datetime(1980, 1, 6, 0, 0, 0, tzinfo=UTC)
        gps = datetime_utc_to_gps(dt)
        assert float(gps) == pytest.approx(0.0, abs=1e-3)

    def test_returns_ligotimegps(self):
        from gwpy.time import LIGOTimeGPS
        dt = datetime(2000, 1, 1, tzinfo=UTC)
        result = datetime_utc_to_gps(dt)
        assert isinstance(result, LIGOTimeGPS)


class TestLeapSecondConversionError:
    def test_is_value_error_subclass(self):
        assert issubclass(LeapSecondConversionError, ValueError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(LeapSecondConversionError):
            raise LeapSecondConversionError("test")

    def test_message_preserved(self):
        try:
            raise LeapSecondConversionError("leap at GPS 1234")
        except LeapSecondConversionError as e:
            assert "leap" in str(e)
