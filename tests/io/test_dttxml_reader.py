"""Tests for DTT XML timeseries reader."""

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeriesDict


class TestDttxmlReader:
    def test_requires_products_argument(self, tmp_path):
        dummy = tmp_path / "dummy.xml"
        dummy.write_text("<dttxml></dttxml>")
        with pytest.raises(ValueError, match="products must be specified"):
            TimeSeriesDict.read(str(dummy), format="dttxml")

    def test_unsupported_products_raises(self, tmp_path):
        dummy = tmp_path / "dummy.xml"
        dummy.write_text("<dttxml></dttxml>")
        with pytest.raises(ValueError, match="not a time-series product"):
            TimeSeriesDict.read(str(dummy), format="dttxml", products="INVALID_PRODUCT")


class TestBuildEpoch:
    def test_numeric_epoch(self):
        from gwexpy.timeseries.io.dttxml import _build_epoch

        assert _build_epoch(1234567890.0, None) == 1234567890.0

    def test_none_epoch(self):
        from gwexpy.timeseries.io.dttxml import _build_epoch

        assert _build_epoch(None, None) is None

    def test_string_epoch_utc(self):
        from gwexpy.timeseries.io.dttxml import _build_epoch

        result = _build_epoch("2020-01-01T00:00:00", "UTC")
        assert isinstance(result, float)
        assert result > 0

    def test_string_epoch_with_timezone_offset(self):
        from gwexpy.timeseries.io.dttxml import _build_epoch

        result = _build_epoch("2020-01-01T09:00:00", "+09:00")
        assert isinstance(result, float)
        assert result > 0
