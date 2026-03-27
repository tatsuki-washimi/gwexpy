"""Tests for gwexpy/io/dttxml_common.py."""
from __future__ import annotations

import base64
import struct
import warnings

import numpy as np
import pytest

from gwexpy.io.dttxml_common import (
    ChannelInfo,
    _decode_dtt_stream,
    extract_xml_channels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xml_with_channels(channels: list[dict], tmp_path) -> str:
    """Create a minimal DTT-style XML file with MeasChn/MeasActive params."""
    lines = ["<LIGO_LW>"]
    for i, ch in enumerate(channels):
        lines.append(f'<Param Name="MeasChn[{i}]">{ch["name"]}</Param>')
        active_val = "true" if ch.get("active", True) else "false"
        lines.append(f'<Param Name="MeasActive[{i}]">{active_val}</Param>')
    lines.append("</LIGO_LW>")
    xml_content = "\n".join(lines)
    path = tmp_path / "test.xml"
    path.write_text(xml_content)
    return str(path)


def _base64_float32(values: list[float]) -> str:
    raw = struct.pack(f"<{len(values)}f", *values)
    return base64.b64encode(raw).decode()


def _base64_float64(values: list[float]) -> str:
    raw = struct.pack(f"<{len(values)}d", *values)
    return base64.b64encode(raw).decode()


def _base64_complex64(values: list[complex]) -> str:
    flat = []
    for c in values:
        flat.extend([c.real, c.imag])
    raw = struct.pack(f"<{len(flat)}f", *flat)
    return base64.b64encode(raw).decode()


def _base64_complex128(values: list[complex]) -> str:
    flat = []
    for c in values:
        flat.extend([c.real, c.imag])
    raw = struct.pack(f"<{len(flat)}d", *flat)
    return base64.b64encode(raw).decode()


# ---------------------------------------------------------------------------
# extract_xml_channels
# ---------------------------------------------------------------------------


class TestExtractXmlChannels:
    def test_nonexistent_file_warns_and_returns_empty(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extract_xml_channels("/nonexistent/path/file.xml")
        assert result == []
        assert len(w) == 1
        assert "parsing error" in str(w[0].message).lower()

    def test_invalid_xml_warns_and_returns_empty(self, tmp_path):
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("this is not xml <unclosed")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extract_xml_channels(str(bad_xml))
        assert result == []
        assert len(w) == 1

    def test_empty_xml_returns_empty(self, tmp_path):
        xml = tmp_path / "empty.xml"
        xml.write_text("<root></root>")
        result = extract_xml_channels(str(xml))
        assert result == []

    def test_single_active_channel(self, tmp_path):
        path = _make_xml_with_channels(
            [{"name": "H1:GDS-CALIB_STRAIN", "active": True}], tmp_path
        )
        result = extract_xml_channels(path)
        assert len(result) == 1
        assert result[0]["name"] == "H1:GDS-CALIB_STRAIN"
        assert result[0]["active"] is True

    def test_inactive_channel(self, tmp_path):
        path = _make_xml_with_channels(
            [{"name": "H1:PSL_PWR", "active": False}], tmp_path
        )
        result = extract_xml_channels(path)
        assert len(result) == 1
        assert result[0]["active"] is False

    def test_multiple_channels(self, tmp_path):
        channels = [
            {"name": "H1:A", "active": True},
            {"name": "H1:B", "active": False},
            {"name": "H1:C", "active": True},
        ]
        path = _make_xml_with_channels(channels, tmp_path)
        result = extract_xml_channels(path)
        assert len(result) == 3
        assert result[0]["name"] == "H1:A"
        assert result[1]["active"] is False
        assert result[2]["name"] == "H1:C"

    def test_zero_value_active(self, tmp_path):
        """'0' should also be treated as inactive."""
        xml = tmp_path / "zero.xml"
        xml.write_text(
            '<root><Param Name="MeasChn[0]">H1:A</Param>'
            '<Param Name="MeasActive[0]">0</Param></root>'
        )
        result = extract_xml_channels(str(xml))
        assert len(result) == 1
        assert result[0]["active"] is False


# ---------------------------------------------------------------------------
# _decode_dtt_stream
# ---------------------------------------------------------------------------


class TestDecodeDttStream:
    def test_float32_little_endian(self):
        values = [1.0, 2.0, 3.0]
        encoded = _base64_float32(values)
        result = _decode_dtt_stream(encoded, "LittleEndian,base64", "float")
        assert result.dtype == np.dtype("<f4")
        np.testing.assert_allclose(result, values, rtol=1e-5)

    def test_float64_little_endian(self):
        values = [1.5, 2.5, 3.5]
        encoded = _base64_float64(values)
        result = _decode_dtt_stream(encoded, "LittleEndian,base64", "double")
        assert result.dtype == np.dtype("<f8")
        np.testing.assert_allclose(result, values)

    def test_complex64(self):
        values = [1+2j, 3+4j]
        encoded = _base64_complex64(values)
        result = _decode_dtt_stream(encoded, "LittleEndian,base64", "floatComplex")
        assert result.dtype == np.dtype("<c8")
        np.testing.assert_allclose(result.real, [1.0, 3.0], rtol=1e-5)
        np.testing.assert_allclose(result.imag, [2.0, 4.0], rtol=1e-5)

    def test_complex128(self):
        values = [1+2j, 3+4j]
        encoded = _base64_complex128(values)
        result = _decode_dtt_stream(encoded, "LittleEndian,base64", "doubleComplex")
        assert result.dtype == np.dtype("<c16")
        np.testing.assert_allclose(result.real, [1.0, 3.0])
        np.testing.assert_allclose(result.imag, [2.0, 4.0])

    def test_unknown_dtype_fallback_to_float32(self):
        values = [1.0, 2.0]
        encoded = _base64_float32(values)
        result = _decode_dtt_stream(encoded, "LittleEndian,base64", "unknown_type")
        assert result.dtype == np.dtype("<f4")

    def test_big_endian(self):
        values = [1.0, 2.0, 3.0]
        raw = struct.pack(f">3f", *values)
        encoded = base64.b64encode(raw).decode()
        result = _decode_dtt_stream(encoded, "BigEndian,base64", "float")
        assert result.dtype == np.dtype(">f4")
        np.testing.assert_allclose(result.astype(float), values, rtol=1e-5)

    def test_non_base64_encoding_raises(self):
        with pytest.raises(ValueError, match="Unsupported encoding"):
            _decode_dtt_stream("sometext", "ascii", "float")

    def test_whitespace_stripped(self):
        values = [1.0]
        encoded = "  " + _base64_float32(values) + "\n"
        result = _decode_dtt_stream(encoded, "LittleEndian,base64", "float")
        np.testing.assert_allclose(result, values, rtol=1e-5)
