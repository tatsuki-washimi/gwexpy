"""Tests for DTT XML timeseries reader."""

import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


class TestDttxmlReader:
    @pytest.mark.parametrize("fmt", ("xml.diaggui", "dttxml"))
    def test_requires_products_argument(self, tmp_path, fmt):
        dummy = tmp_path / "dummy.xml"
        dummy.write_text("<dttxml></dttxml>")
        with pytest.raises(ValueError, match="products must be specified"):
            TimeSeriesDict.read(str(dummy), format=fmt)

    @pytest.mark.parametrize("fmt", ("xml.diaggui", "dttxml"))
    def test_unsupported_products_raises(self, tmp_path, fmt):
        dummy = tmp_path / "dummy.xml"
        dummy.write_text("<dttxml></dttxml>")
        with pytest.raises(ValueError, match="not a time-series product"):
            TimeSeriesDict.read(str(dummy), format=fmt, products="INVALID_PRODUCT")

    def test_auto_detected_xml_format_is_not_ambiguous(self, tmp_path):
        dummy = tmp_path / "dummy.xml"
        dummy.write_text("<dttxml></dttxml>")
        with pytest.raises(ValueError, match="products must be specified"):
            TimeSeries.read(str(dummy))

    @pytest.mark.parametrize(
        "filename",
        ["dummy.XML", "dummy.Xml"],
    )
    def test_auto_detected_xml_format_handles_case_variants(self, tmp_path, filename):
        dummy = tmp_path / filename
        dummy.write_text("<dttxml></dttxml>")
        with pytest.raises(ValueError, match="products must be specified"):
            TimeSeries.read(str(dummy))

    @pytest.mark.parametrize(
        "filename",
        ["dummy.xml.gz", "dummy.XML.GZ"],
    )
    def test_auto_detected_xml_format_handles_gz_extensions(self, tmp_path, filename):
        import gzip

        dummy = tmp_path / filename
        with gzip.open(dummy, "wb") as fp:
            fp.write(b"<dttxml></dttxml>")

        with pytest.raises(ValueError, match="products must be specified"):
            TimeSeries.read(str(dummy))

    def test_legacy_alias_resolves_to_canonical_reader(self):
        from gwpy.io.registry import default_registry as io_registry

        canonical = io_registry.get_reader("xml.diaggui", TimeSeriesDict)
        legacy = io_registry.get_reader("dttxml", TimeSeriesDict)

        assert canonical is legacy


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
