"""Tests for gwexpy/interop/base.py and gwexpy/interop/json_.py."""

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy import units as u

from gwexpy.interop.base import (
    from_plain_array,
    resolve_meta,
    resolve_timing,
    to_plain_array,
)
from gwexpy.interop.json_ import from_dict, from_json, to_dict, to_json
from gwexpy.timeseries import TimeSeries

# ---------------------------------------------------------------------------
# to_plain_array
# ---------------------------------------------------------------------------


class TestToPlainArray:
    def test_from_timeseries(self):
        ts = TimeSeries(np.arange(5.0), t0=0, dt=1.0)
        arr = to_plain_array(ts)
        np.testing.assert_array_equal(arr, np.arange(5.0))
        assert isinstance(arr, np.ndarray)

    def test_from_astropy_quantity(self):
        q = u.Quantity(np.array([1.0, 2.0, 3.0]), unit="m")
        arr = to_plain_array(q)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_from_plain_array(self):
        x = np.array([4.0, 5.0])
        arr = to_plain_array(x)
        np.testing.assert_array_equal(arr, x)

    def test_copy_false_no_copy(self):
        x = np.array([1.0, 2.0])
        arr = to_plain_array(x, copy=False)
        # Without copy, may share data
        assert arr is not None

    def test_copy_true_makes_copy(self):
        x = np.array([1.0, 2.0])
        arr = to_plain_array(x, copy=True)
        np.testing.assert_array_equal(arr, x)


# ---------------------------------------------------------------------------
# from_plain_array
# ---------------------------------------------------------------------------


class TestFromPlainArray:
    def test_basic(self):
        arr = np.arange(5.0)
        ts = from_plain_array(TimeSeries, arr, t0=0, dt=1.0)
        assert isinstance(ts, TimeSeries)
        np.testing.assert_array_equal(ts.value, arr)

    def test_with_unit(self):
        arr = np.ones(3)
        ts = from_plain_array(TimeSeries, arr, t0=0, dt=1.0, unit="m")
        assert str(ts.unit) == "m"

    def test_with_numpy_like_get(self):
        """Objects with .get() method (cupy-like) are supported."""

        class FakeArray:
            def get(self):
                return np.array([1.0, 2.0])

        ts = from_plain_array(TimeSeries, FakeArray(), t0=0, dt=1.0)
        np.testing.assert_array_equal(ts.value, [1.0, 2.0])

    def test_with_numpy_like_numpy(self):
        """Objects with .numpy() method (torch-like) are supported."""

        class FakeArray:
            def numpy(self):
                return np.array([3.0, 4.0])

        ts = from_plain_array(TimeSeries, FakeArray(), t0=0, dt=1.0)
        np.testing.assert_array_equal(ts.value, [3.0, 4.0])


# ---------------------------------------------------------------------------
# resolve_meta / resolve_timing
# ---------------------------------------------------------------------------


class TestResolveMeta:
    def test_user_wins(self):
        assert resolve_meta("user", "source") == "user"

    def test_falls_back_to_source(self):
        assert resolve_meta(None, "source") == "source"

    def test_falsy_user_value_preserved(self):
        # An explicit falsy-but-valid value (e.g. epoch=0) must survive.
        assert resolve_meta(0, "source") == 0
        assert resolve_meta("", "source") == ""


class TestResolveTiming:
    def test_user_values_win(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t0, dt = resolve_timing(5.0, 0.25, source="x")
        assert t0 == 5.0
        assert dt == 0.25

    def test_explicit_zero_no_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t0, dt = resolve_timing(0.0, 0.5, source="x")
        assert t0 == 0.0
        assert dt == 0.5

    def test_inferred_values_used(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t0, dt = resolve_timing(
                None, None, source="x", inferred_t0=3.0, inferred_dt=2.0
            )
        assert t0 == 3.0
        assert dt == 2.0

    def test_missing_warns_and_defaults(self):
        with pytest.warns(UserWarning, match="timing metadata"):
            t0, dt = resolve_timing(None, None, source="x")
        assert t0 == 0.0
        assert dt == 1.0


# ---------------------------------------------------------------------------
# to_dict / to_json
# ---------------------------------------------------------------------------


class TestToDict:
    def _make_ts(self):
        return TimeSeries(np.arange(3.0), t0=1.0, dt=0.5, unit="m", name="ch")

    def test_basic_keys(self):
        d = to_dict(self._make_ts())
        assert "t0" in d
        assert "dt" in d
        assert "unit" in d
        assert "data" in d

    def test_t0_dt_values(self):
        d = to_dict(self._make_ts())
        assert d["t0"] == pytest.approx(1.0)
        assert d["dt"] == pytest.approx(0.5)

    def test_unit_string(self):
        d = to_dict(self._make_ts())
        assert d["unit"] == "m"

    def test_data_list(self):
        d = to_dict(self._make_ts())
        assert isinstance(d["data"], list)
        np.testing.assert_allclose(d["data"], [0.0, 1.0, 2.0])

    def test_name_included(self):
        d = to_dict(self._make_ts())
        assert d["name"] == "ch"

    def test_name_none_when_no_name(self):
        ts = TimeSeries(np.ones(3), t0=0, dt=1.0)
        d = to_dict(ts)
        assert d["name"] is None


class TestToJson:
    def test_returns_string(self):
        ts = TimeSeries(np.arange(3.0), t0=0, dt=1.0, unit="m")
        s = to_json(ts)
        assert isinstance(s, str)

    def test_valid_json(self):
        ts = TimeSeries(np.arange(3.0), t0=0, dt=1.0, unit="m")
        parsed = json.loads(to_json(ts))
        assert "data" in parsed


# ---------------------------------------------------------------------------
# from_dict / from_json
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_basic(self):
        d = {"data": [1.0, 2.0, 3.0], "t0": 0.0, "dt": 1.0, "unit": "m"}
        ts = from_dict(TimeSeries, d)
        assert isinstance(ts, TimeSeries)
        np.testing.assert_array_equal(ts.value, [1.0, 2.0, 3.0])

    def test_default_t0_dt(self):
        # Missing t0/dt now fall back to 0/1 with a UserWarning (no longer
        # silent) -- the values are preserved for backward compatibility.
        d = {"data": [1.0, 2.0]}
        with pytest.warns(UserWarning, match="timing metadata"):
            ts = from_dict(TimeSeries, d)
        assert ts.t0.value == pytest.approx(0.0)
        assert ts.dt.value == pytest.approx(1.0)

    def test_explicit_t0_zero_no_warning(self):
        # Falsy-but-valid t0=0.0 must be honoured without a warning.
        import warnings

        d = {"data": [1.0, 2.0], "dt": 0.5}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ts = from_dict(TimeSeries, d, t0=0.0)
        assert ts.t0.value == pytest.approx(0.0)
        assert ts.dt.value == pytest.approx(0.5)

    def test_channel_roundtrip(self):
        ts = TimeSeries(
            np.arange(3.0), t0=1.0, dt=0.5, unit="m", channel="X1:TEST-CHAN"
        )
        d = to_dict(ts)
        assert d["channel"] == "X1:TEST-CHAN"
        ts2 = from_dict(TimeSeries, d)
        assert str(ts2.channel) == "X1:TEST-CHAN"

    def test_channel_none_when_unset(self):
        ts = TimeSeries(np.ones(3), t0=0, dt=1.0)
        d = to_dict(ts)
        assert d["channel"] is None
        ts2 = from_dict(TimeSeries, d)
        assert ts2.channel is None

    def test_user_supplied_metadata(self):
        # Source dict lacks channel/unit; the user can supply them explicitly.
        d = {"data": [1.0, 2.0, 3.0], "t0": 0.0, "dt": 1.0}
        ts = from_dict(TimeSeries, d, channel="X1:FOO", unit="m")
        assert str(ts.channel) == "X1:FOO"
        assert str(ts.unit) == "m"

    def test_user_arg_overrides_source(self):
        d = {"data": [1.0, 2.0], "t0": 0.0, "dt": 1.0, "unit": "m"}
        ts = from_dict(TimeSeries, d, unit="s")
        assert str(ts.unit) == "s"


class TestFromJson:
    def test_roundtrip(self):
        ts = TimeSeries(np.arange(4.0), t0=0.5, dt=0.25, unit="m")
        ts2 = from_json(TimeSeries, to_json(ts))
        np.testing.assert_allclose(ts2.value, ts.value)
        assert ts2.t0.value == pytest.approx(0.5)
        assert str(ts2.unit) == "m"

    def test_channel_roundtrip(self):
        ts = TimeSeries(
            np.arange(4.0), t0=0.5, dt=0.25, unit="m", channel="X1:JSON-CHAN"
        )
        ts2 = from_json(TimeSeries, to_json(ts))
        assert str(ts2.channel) == "X1:JSON-CHAN"

    def test_user_kwargs_forwarded(self):
        ts = TimeSeries(np.arange(4.0), t0=0.5, dt=0.25)
        ts2 = from_json(TimeSeries, to_json(ts), channel="X1:FOO")
        assert str(ts2.channel) == "X1:FOO"


# ---------------------------------------------------------------------------
# Public Mixin pass-through (TimeSeries.from_dict / from_json)
# ---------------------------------------------------------------------------


class TestPublicMixin:
    def test_from_dict_forwards_kwargs(self):
        d = {"data": [1.0, 2.0, 3.0], "t0": 0.0, "dt": 1.0}
        ts = TimeSeries.from_dict(d, channel="X1:FOO", unit="m")
        assert str(ts.channel) == "X1:FOO"
        assert str(ts.unit) == "m"

    def test_from_dict_channel_roundtrip(self):
        ts = TimeSeries(np.arange(3.0), t0=1.0, dt=0.5, channel="X1:MIX")
        ts2 = TimeSeries.from_dict(ts.to_dict())
        assert str(ts2.channel) == "X1:MIX"

    def test_from_json_forwards_kwargs(self):
        ts = TimeSeries(np.arange(3.0), t0=0.0, dt=1.0)
        ts2 = TimeSeries.from_json(ts.to_json(), channel="X1:BAR")
        assert str(ts2.channel) == "X1:BAR"
