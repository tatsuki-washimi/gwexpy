"""Additional tests for gwexpy/io/utils.py to cover missing branches."""
from __future__ import annotations

import datetime as _dt
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.io.utils import (
    apply_unit,
    ensure_datetime,
    ensure_dependency,
    maybe_pad_timeseries,
    parse_timezone,
)


# ---------------------------------------------------------------------------
# maybe_pad_timeseries
# ---------------------------------------------------------------------------

def test_maybe_pad_timeseries_passthrough():
    """Test maybe_pad_timeseries returns original if gap is not pad/raise."""
    ts = object()
    # gap="ignore" should just return ts
    result = maybe_pad_timeseries(ts, gap="ignore")
    assert result is ts


def test_maybe_pad_timeseries_delegation():
    """Test maybe_pad_timeseries delegates to _pad_series with correct error flag."""
    import gwpy
    gwpy_major = int(gwpy.__version__.split(".")[0])
    if gwpy_major >= 4:
        target = "gwpy.timeseries.connect._pad_series"
    else:
        target = "gwpy.timeseries.io.core._pad_series"
        
    ts = MagicMock()
    with patch(target) as mock_pad:
        maybe_pad_timeseries(ts, gap="raise")
        mock_pad.assert_called_with(ts, np.nan, start=None, end=None, error=True)
        
        maybe_pad_timeseries(ts, gap="pad")
        mock_pad.assert_called_with(ts, np.nan, start=None, end=None, error=False)


# ---------------------------------------------------------------------------
# apply_unit
# ---------------------------------------------------------------------------

def test_apply_unit_series_matrix_path():
    """Test apply_unit for SeriesMatrix."""
    class DummySeriesMatrix:
        def __init__(self):
            self.meta = np.array([[{"unit": u.m}]], dtype=object)
    
    obj = DummySeriesMatrix()
    with patch("gwexpy.interop._registry.ConverterRegistry.get_constructor", return_value=DummySeriesMatrix):
        result = apply_unit(obj, "V")
        assert obj.meta[0, 0]["unit"] == u.V


def test_apply_unit_registry_resolution_failure():
    """Test apply_unit when Registry lookup fails (Line 114 range)."""
    # Trigger KeyError inside the try block
    with patch("gwexpy.interop._registry.ConverterRegistry.get_constructor", side_effect=KeyError):
        class SimpleObj:
            def __init__(self):
                self.unit = u.m
        obj = SimpleObj()
        # Should NOT raise TypeError, but fall through to line 125+
        result = apply_unit(obj, "V")
        assert obj.unit == u.V


def test_apply_unit_constructor_fallback_attribute_error():
    """Test apply_unit constructor fallback on AttributeError (Line 150)."""
    class FixedUnitObj:
        def __init__(self, value, unit=None, **kwargs):
            self.value = value
            self._unit = unit
        @property
        def unit(self):
            return self._unit
        @unit.setter
        def unit(self, v):
            raise AttributeError("immutable")
            
    obj = FixedUnitObj(np.array([1.0]), unit=u.m)
    result = apply_unit(obj, "V")
    assert result is not obj
    assert result.unit == u.V


# ---------------------------------------------------------------------------
# parse_timezone
# ---------------------------------------------------------------------------

def test_parse_timezone_strip():
    """Test parse_timezone strips whitespace (Line 33)."""
    result = parse_timezone("  +09:00  ")
    expected = _dt.timezone(_dt.timedelta(hours=9))
    assert result == expected


def test_parse_timezone_manual_fallback():
    """Test parse_timezone falls back to manual parse when ZoneInfo fails."""
    # +05:30 is not a ZoneInfo name, should fall to manual parse
    result = parse_timezone("+05:30")
    assert result == _dt.timezone(_dt.timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# ensure_datetime
# ---------------------------------------------------------------------------

def test_ensure_datetime_aware_passthrough():
    """Test ensure_datetime does not replace tzinfo if already aware (Line 76)."""
    tz1 = _dt.timezone(_dt.timedelta(hours=9))
    tz2 = _dt.timezone.utc
    dt = _dt.datetime(2020, 1, 1, tzinfo=tz1)
    result = ensure_datetime(dt, tzinfo=tz2)
    assert result.tzinfo is tz1


# ---------------------------------------------------------------------------
# ensure_dependency
# ---------------------------------------------------------------------------

def test_ensure_dependency_import_name_mismatch():
    """Test ensure_dependency with import_name mismatch (Line 223)."""
    with pytest.raises(ImportError, match="fake-pkg is required"):
        ensure_dependency("fake-pkg", import_name="nonexistent_module_xyz")
