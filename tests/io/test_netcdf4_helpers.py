"""Unit tests for NetCDF4 helper functions that do NOT require the netCDF4 library."""

import numpy as np
import pytest

from gwexpy.timeseries.io.netcdf4_ import (
    _decode_netcdf_key,
    _encode_netcdf_var_name,
    _to_json_native,
)


class TestToJsonNative:
    def test_int(self):
        assert _to_json_native(1) == 1
        assert isinstance(_to_json_native(1), int)

    def test_float(self):
        assert _to_json_native(3.14) == 3.14

    def test_str(self):
        assert _to_json_native("hello") == "hello"

    def test_none(self):
        assert _to_json_native(None) is None

    def test_numpy_int64(self):
        result = _to_json_native(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float64(self):
        result = _to_json_native(np.float64(1.5))
        assert result == 1.5
        assert isinstance(result, float)

    def test_numpy_bool_(self):
        result = _to_json_native(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_datetime64_falls_back_to_str(self):
        import json

        result = _to_json_native(np.datetime64("2026-04-24"))
        # Must be json-serializable — no TypeError
        json.dumps(result)
        assert isinstance(result, str)

    def test_numpy_timedelta64_falls_back_to_str(self):
        import json

        result = _to_json_native(np.timedelta64(5, "s"))
        json.dumps(result)
        assert isinstance(result, str)


class TestEncodeNetcdfVarName:
    def test_string_key(self):
        assert _encode_netcdf_var_name("signal") == "signal"

    def test_int_key(self):
        assert _encode_netcdf_var_name(42) == "42"

    def test_tuple_key_produces_legal_name(self):
        name = _encode_netcdf_var_name((0, 10))
        # NetCDF4 requires names start with a letter or underscore and contain
        # only alphanumerics, underscores, hyphens, or periods.
        assert all(c.isalnum() or c == "_" for c in name), f"Illegal chars in {name!r}"

    def test_tuple_keys_are_unique(self):
        names = {
            _encode_netcdf_var_name(k)
            for k in [(0, 1), (1, 0), ("a", "b"), ("a__b", "c"), ("a", "b__c")]
        }
        assert len(names) == 5, "Two distinct tuple keys produced the same variable name"

    def test_ambiguous_tuple_keys_do_not_collide(self):
        name1 = _encode_netcdf_var_name(("a", "b__c"))
        name2 = _encode_netcdf_var_name(("a__b", "c"))
        assert name1 != name2, "Ambiguous tuple keys must not produce the same name"

    def test_longer_tuple_produces_legal_name(self):
        name = _encode_netcdf_var_name(("x", "y", "z"))
        assert all(c.isalnum() or c == "_" for c in name)


class TestDecodeNetcdfKey:
    def test_none_returns_none(self):
        assert _decode_netcdf_key(None) is None

    def test_json_integer(self):
        result = _decode_netcdf_key("42")
        assert result == 42
        assert isinstance(result, int)

    def test_json_float(self):
        result = _decode_netcdf_key("3.14")
        assert result == pytest.approx(3.14)

    def test_json_string(self):
        assert _decode_netcdf_key('"hello"') == "hello"

    def test_json_list_becomes_tuple(self):
        result = _decode_netcdf_key("[1, 2]")
        assert result == (1, 2)
        assert isinstance(result, tuple)

    def test_json_null_returns_none(self):
        # json.loads("null") == None; _decode_netcdf_key returns None
        result = _decode_netcdf_key("null")
        assert result is None

    def test_legacy_plain_string_not_valid_json(self):
        # Strings not wrapped in quotes are invalid JSON; fallback to str
        result = _decode_netcdf_key("my_channel")
        assert result == "my_channel"

    def test_legacy_numeric_string_no_gwexpy_key_format(self):
        # "0" is valid JSON (integer 0), decoded correctly
        assert _decode_netcdf_key("0") == 0
