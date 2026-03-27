"""Tests for gwexpy/io/utils.py."""
from __future__ import annotations

import datetime as _dt
import warnings

import numpy as np
import pytest
from astropy import units as u

from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    ensure_datetime,
    ensure_dependency,
    extract_audio_metadata,
    filter_by_channels,
    parse_timezone,
    set_provenance,
)


# ---------------------------------------------------------------------------
# parse_timezone
# ---------------------------------------------------------------------------


class TestParseTimezone:
    def test_none_raises(self):
        with pytest.raises(ValueError, match="timezone must be specified"):
            parse_timezone(None)

    def test_tzinfo_passthrough(self):
        # Line 26 — isinstance(tz, _dt.tzinfo) → return tz
        tz = _dt.timezone.utc
        result = parse_timezone(tz)
        assert result is tz

    def test_int_offset(self):
        # Line 28 — int/float → timedelta(hours=...)
        result = parse_timezone(9)
        expected = _dt.timezone(_dt.timedelta(hours=9))
        assert result == expected

    def test_float_offset(self):
        result = parse_timezone(5.5)
        expected = _dt.timezone(_dt.timedelta(hours=5.5))
        assert result == expected

    def test_iana_name(self):
        # Line 31 — ZoneInfo(tz) succeeds
        result = parse_timezone("Asia/Tokyo")
        assert result is not None

    def test_utc_string(self):
        # "UTC" is a valid IANA name → ZoneInfo("UTC") returned first
        result = parse_timezone("UTC")
        assert result is not None

    def test_gmt_string(self):
        # "GMT" is a valid IANA name → ZoneInfo("GMT") returned first
        result = parse_timezone("GMT")
        assert result is not None

    def test_positive_offset_with_colon(self):
        # Lines 37-43 — "+09:00"
        result = parse_timezone("+09:00")
        expected = _dt.timezone(_dt.timedelta(hours=9))
        assert result == expected

    def test_negative_offset_with_colon(self):
        # Lines 37-39 — sign=-1
        result = parse_timezone("-05:30")
        expected = _dt.timezone(_dt.timedelta(hours=-5, minutes=-30))
        assert result == expected

    def test_offset_without_colon(self):
        # Line 45 — no colon → hours[:2], minutes[2:] or "0"
        result = parse_timezone("+0900")
        expected = _dt.timezone(_dt.timedelta(hours=9))
        assert result == expected

    def test_offset_without_minutes(self):
        # Line 45 — cleaned[2:] == "" → "0"
        result = parse_timezone("+08")
        expected = _dt.timezone(_dt.timedelta(hours=8))
        assert result == expected

    def test_unsupported_type_raises(self):
        # Line 55
        with pytest.raises(ValueError, match="Unsupported timezone specifier"):
            parse_timezone([9, 0])


# ---------------------------------------------------------------------------
# datetime_to_gps
# ---------------------------------------------------------------------------


class TestDatetimeToGps:
    def test_aware_datetime(self):
        dt = _dt.datetime(2017, 1, 1, 0, 0, 0, tzinfo=_dt.timezone.utc)
        result = datetime_to_gps(dt)
        assert isinstance(result, float)
        assert result > 0

    def test_naive_raises(self):
        # Line 65
        dt = _dt.datetime(2017, 1, 1, 0, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            datetime_to_gps(dt)

    def test_date_object(self):
        # Lines 62-63 — isinstance(dt, _dt.date) and not isinstance(dt, _dt.datetime)
        d = _dt.date(2017, 1, 1)
        result = datetime_to_gps(d)
        assert isinstance(result, float)
        assert result > 0


# ---------------------------------------------------------------------------
# ensure_datetime
# ---------------------------------------------------------------------------


class TestEnsureDatetime:
    def test_aware_datetime_passthrough(self):
        dt = _dt.datetime(2020, 6, 1, tzinfo=_dt.timezone.utc)
        result = ensure_datetime(dt)
        assert result == dt

    def test_naive_datetime_with_tzinfo(self):
        # Lines 76-77 — value.tzinfo is None and tzinfo is not None → replace
        dt = _dt.datetime(2020, 6, 1)
        tz = _dt.timezone.utc
        result = ensure_datetime(dt, tzinfo=tz)
        assert result.tzinfo is not None

    def test_naive_datetime_without_tzinfo_raises(self):
        # Lines 78-79 — naive with no tzinfo → ValueError
        dt = _dt.datetime(2020, 6, 1)
        with pytest.raises(ValueError, match="Naive datetime requires timezone"):
            ensure_datetime(dt)

    def test_numeric_timestamp(self):
        # Lines 81-82 — int/float → fromtimestamp
        result = ensure_datetime(0)
        assert result.tzinfo is not None
        assert result.year == 1970

    def test_float_timestamp(self):
        result = ensure_datetime(1234567890.0)
        assert isinstance(result, _dt.datetime)

    def test_string_slash_format(self):
        # Lines 86 — "%Y/%m/%d %H:%M:%S.%f"
        result = ensure_datetime("2020/06/01 12:00:00.000000")
        assert result.year == 2020

    def test_string_slash_no_microsecond(self):
        # Lines 87 — "%Y/%m/%d %H:%M:%S"
        result = ensure_datetime("2020/06/01 12:00:00")
        assert result.year == 2020

    def test_string_dash_format(self):
        # Lines 88 — "%Y-%m-%d %H:%M:%S"
        result = ensure_datetime("2020-06-01 12:00:00")
        assert result.year == 2020

    def test_string_iso_format(self):
        # Lines 89 — "%Y-%m-%dT%H:%M:%S"
        result = ensure_datetime("2020-06-01T12:00:00")
        assert result.year == 2020

    def test_string_comma_format(self):
        # Lines 90 — "%Y-%m-%d,%H:%M:%S"
        result = ensure_datetime("2020-06-01,12:00:00")
        assert result.year == 2020

    def test_string_with_tzinfo(self):
        # Lines 95-96 — string with no tz → replace tzinfo if provided
        result = ensure_datetime("2020/06/01 12:00:00", tzinfo=_dt.timezone.utc)
        assert result.tzinfo is not None

    def test_unrecognized_raises(self):
        # Line 98
        with pytest.raises(ValueError, match="Unrecognised time value"):
            ensure_datetime("not-a-date-at-all-xyz")


# ---------------------------------------------------------------------------
# apply_unit
# ---------------------------------------------------------------------------


class TestApplyUnit:
    def test_none_passthrough(self):
        obj = object()
        result = apply_unit(obj, None)
        assert result is obj

    def test_empty_string_passthrough(self):
        obj = object()
        result = apply_unit(obj, "")
        assert result is obj

    def test_object_with_override_unit(self):
        # Lines 128-130 — hasattr override_unit
        class FakeSeries:
            def __init__(self):
                self._unit = u.m
            def override_unit(self, unit):
                self._unit = u.Unit(unit)
        obj = FakeSeries()
        result = apply_unit(obj, "V")
        assert result is obj

    def test_object_with_settable_unit(self):
        # Lines 131-133 — series.unit = u.Unit(unit) succeeds
        class SimpleObj:
            def __init__(self):
                self.unit = u.m
        obj = SimpleObj()
        result = apply_unit(obj, "V")
        assert result is obj
        assert obj.unit == u.V

    def test_object_without_unit_fallback_constructor(self):
        # Lines 134-149 — unit setter raises AttributeError → constructor fallback
        class ConstructableObj:
            def __init__(self, value=None, unit=None, **kwargs):
                self._value = value
                self._unit = unit

            @property
            def value(self):
                return self._value

            @property
            def unit(self):
                return self._unit

            @unit.setter
            def unit(self, v):
                raise AttributeError("immutable unit")

        obj = ConstructableObj(value=np.array([1.0, 2.0]), unit=u.m)
        result = apply_unit(obj, "V")
        assert result is not None

    def test_object_unit_setter_typeerror(self):
        # Lines 134-149 — TypeError on unit setter → constructor fallback
        class TypeErrorUnit:
            def __init__(self, value=None, unit=None, **kwargs):
                self._value = value

            @property
            def value(self):
                return self._value

            @property
            def unit(self):
                return u.m

            @unit.setter
            def unit(self, v):
                raise TypeError("cannot set unit")

        obj = TypeErrorUnit(value=np.array([1.0]))
        result = apply_unit(obj, "V")
        assert result is not None


# ---------------------------------------------------------------------------
# set_provenance
# ---------------------------------------------------------------------------


class TestSetProvenance:
    def test_with_attrs_dict(self):
        # Lines 159-161
        class ObjWithAttrs:
            attrs = {}
        obj = ObjWithAttrs()
        set_provenance(obj, {"source": "test"})
        assert obj.attrs["source"] == "test"

    def test_without_attrs_sets_gwexpy_io(self):
        # Line 165 — no attrs → setattr _gwexpy_io
        class PlainObj:
            pass
        obj = PlainObj()
        set_provenance(obj, {"source": "test"})
        assert hasattr(obj, "_gwexpy_io")
        assert obj._gwexpy_io["source"] == "test"

    def test_attrs_not_dict_falls_through(self):
        # Lines 162-163 — attrs exists but is not a dict → TypeError suppressed
        class ObjWithBadAttrs:
            attrs = "not_a_dict"
        obj = ObjWithBadAttrs()
        # Should not raise
        set_provenance(obj, {"source": "test"})

    def test_attrs_dict_subclass_update_raises(self):
        # Lines 162-163 — isinstance(obj.attrs, dict) is True but update() raises TypeError
        class BadDict(dict):
            def update(self, d=None, **kwargs):
                raise TypeError("broken update")
        obj_bad = type("Obj", (), {"attrs": BadDict()})()
        # Should not raise; _gwexpy_io fallback is used
        set_provenance(obj_bad, {"source": "test"})


# ---------------------------------------------------------------------------
# filter_by_channels
# ---------------------------------------------------------------------------


class TestFilterByChannels:
    def test_none_returns_full_mapping(self):
        mapping = {"a": 1, "b": 2}
        result = filter_by_channels(mapping, None)
        assert result is mapping

    def test_filters_by_channels(self):
        mapping = {"a": 1, "b": 2, "c": 3}
        result = filter_by_channels(mapping, ["a", "c"])
        assert set(result.keys()) == {"a", "c"}

    def test_missing_channel_excluded(self):
        mapping = {"a": 1, "b": 2}
        result = filter_by_channels(mapping, ["x", "a"])
        assert list(result.keys()) == ["a"]


# ---------------------------------------------------------------------------
# ensure_dependency
# ---------------------------------------------------------------------------


class TestEnsureDependency:
    def test_existing_module(self):
        np_mod = ensure_dependency("numpy")
        assert np_mod is not None

    def test_missing_module_raises(self):
        # Lines 225-230
        with pytest.raises(ImportError, match="not_a_real_pkg_xyz"):
            ensure_dependency("not_a_real_pkg_xyz")

    def test_extra_in_error_message(self):
        # Lines 227-228 — extra appended
        with pytest.raises(ImportError, match=r"pip install"):
            ensure_dependency("not_a_real_pkg_xyz", extra="gui")

    def test_import_name_override(self):
        # Lines 223-224 — import_name used instead of package_name
        mod = ensure_dependency("numpy_package_install_name", import_name="numpy")
        assert mod is not None


# ---------------------------------------------------------------------------
# extract_audio_metadata
# ---------------------------------------------------------------------------


class TestExtractAudioMetadata:
    def test_missing_tinytag_returns_empty(self):
        # Lines 287-293 — ImportError → warn + return {}
        import sys
        # Temporarily hide tinytag
        tinytag_mod = sys.modules.pop("tinytag", None)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = extract_audio_metadata("fake.mp3")
            assert result == {}
        finally:
            if tinytag_mod is not None:
                sys.modules["tinytag"] = tinytag_mod

    def test_nonexistent_file_returns_empty(self):
        # Lines 294-301 — general exception → warn + return {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extract_audio_metadata("/nonexistent/path/to/file.mp3")
        # Returns empty dict (either ImportError or general exception)
        assert isinstance(result, dict)
