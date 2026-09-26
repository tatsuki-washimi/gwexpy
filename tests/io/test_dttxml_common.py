"""Tests for gwexpy/io/dttxml_common.py."""

from __future__ import annotations

import base64
import struct
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import gwexpy.io.dttxml_common as dttxml_common
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.io.dttxml_common import (
    ChannelInfo,
    _decode_dtt_stream,
    extract_xml_channels,
    load_dttxml_products,
)
from gwexpy.timeseries import TimeSeriesDict


def test_uniform_frequency_step_uses_spacing_not_absolute_frequency():
    rounded_uniform = np.arange(100, dtype=np.float32) / 10
    # This is a possible nominal step, not proof that f0/df construction
    # reproduces every serialized float32 bin. The external-reader regression
    # verifies exact values and requires an explicit axis when it does not.
    assert dttxml_common._uniform_frequency_step(rounded_uniform) == pytest.approx(0.1)

    # A large f0 must not hide a nonuniform step in float32 storage.
    irregular = np.array(
        [1_000_000, 1_000_001, 1_000_002.5, 1_000_003.5], dtype=np.float32
    )
    assert dttxml_common._uniform_frequency_step(irregular) is None

    # A one-ULP perturbation at a large offset must not be treated as axis
    # quantization: reconstructing with df=1.0 would lose the serialized tail.
    quantized_irregular = np.array(
        [1_000_000, 1_000_001, 1_000_002.0625], dtype=np.float32
    )
    assert dttxml_common._uniform_frequency_step(quantized_irregular) is None

    increments = np.where(np.arange(99) % 2 == 0, 1.0, 1.0625).astype(np.float32)
    drifting = np.concatenate(
        (
            np.array([1_000_000.0], dtype=np.float32),
            np.float32(1_000_000.0) + np.cumsum(increments, dtype=np.float32),
        )
    )
    assert dttxml_common._uniform_frequency_step(drifting) is None


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


def _make_spectrum_with_dims(tmp_path, *, dims: tuple[int, ...], filename: str) -> str:
    """Write an ASD block whose declared dimensions can be varied independently."""
    root = ET.Element("LIGO_LW")
    result = ET.SubElement(root, "LIGO_LW", {"Name": "Result[0]", "Type": "Spectrum"})
    for name, value in {
        "Subtype": "1",
        "M": "1",
        "N": "4",
        "f0": "17.5",
        "df": "2.5",
        "ChannelA": "K1:TEST-ASD",
    }.items():
        ET.SubElement(result, "Param", {"Name": name}).text = value
    ET.SubElement(result, "Time", {"Name": "t0"}).text = "1300000000"
    array = ET.SubElement(result, "Array", {"Type": "float"})
    for dim in dims:
        ET.SubElement(array, "Dim").text = str(dim)
    stream = ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"})
    stream.text = _base64_float32([1.0, 2.0, 3.0, 4.0])

    path = tmp_path / filename
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return str(path)


class _FakeTSInfo:
    def __init__(self, timeseries, dt, gps_second):
        self.timeseries = np.asarray(timeseries)
        self.dt = dt
        self.gps_second = gps_second


class _FakeDTTXMLResults:
    def __init__(self, channel: str, info: _FakeTSInfo):
        self.TS = {channel: info}


class _FakeDiagAccess:
    def __init__(self, channel: str, timeseries, dt, gps_second):
        self.results = _FakeDTTXMLResults(
            channel,
            _FakeTSInfo(timeseries, dt=dt, gps_second=gps_second),
        )


class _FakeDTTXML:
    def __init__(self, channel: str, timeseries, dt, gps_second):
        self._channel = channel
        self._timeseries = timeseries
        self._dt = dt
        self._gps_second = gps_second

    def DiagAccess(self, source):
        del source
        return _FakeDiagAccess(
            self._channel,
            self._timeseries,
            dt=self._dt,
            gps_second=self._gps_second,
        )


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
        values = [1 + 2j, 3 + 4j]
        encoded = _base64_complex64(values)
        result = _decode_dtt_stream(encoded, "LittleEndian,base64", "floatComplex")
        assert result.dtype == np.dtype("<c8")
        np.testing.assert_allclose(result.real, [1.0, 3.0], rtol=1e-5)
        np.testing.assert_allclose(result.imag, [2.0, 4.0], rtol=1e-5)

    def test_complex128(self):
        values = [1 + 2j, 3 + 4j]
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
        raw = struct.pack(">3f", *values)
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


def _make_xml_with_string_typed_spectrum_params(tmp_path) -> str:
    """Create a minimal DTT XML file where numeric params are stored as strings."""

    n_points = 4

    def block(
        attrs: dict[str, str],
        data_b64: str,
        dtype: str,
        *,
        block_type: str,
        result_index: int,
    ) -> str:
        lines = [f'  <LIGO_LW Name="Result[{result_index}]" Type="{block_type}">']
        for key, value in attrs.items():
            lines.append(f'    <Param Name="{key}" Type="string">{value}</Param>')
        lines += [
            '    <Time Name="t0">1300000000</Time>',
            f'    <Array Type="{dtype}">',
            "      <Dim>1</Dim>",
            f"      <Dim>{n_points}</Dim>",
            f'      <Stream Encoding="LittleEndian,base64">{data_b64}</Stream>',
            "    </Array>",
            "  </LIGO_LW>",
        ]
        return "\n".join(lines)

    psd = _base64_float32([1e-10, 1e-10, 1e-10, 1e-10])
    tf = _base64_complex64([1 + 0j, 0.5 + 0.1j, 0.25 + 0.2j, 0.125 + 0.3j])

    xml = [
        "<?xml version='1.0' encoding='utf-8'?>",
        "<LIGO_LW>",
        block(
            {
                "ChannelA": "K1:SUS-ITMX_EXCITATION",
                "Subtype": "1",
                "M": "1",
                "f0": "0.0",
                "df": "1.0",
                "N": str(n_points),
            },
            psd,
            "float",
            block_type="Spectrum",
            result_index=0,
        ),
        block(
            {
                "ChannelA": "K1:SUS-ITMX_EXCITATION",
                "ChannelB[0]": "K1:SUS-ITMX_DISPLACEMENT",
                "Subtype": "0",
                "M": "1",
                "f0": "0.0",
                "df": "1.0",
                "N": str(n_points),
            },
            tf,
            "floatComplex",
            block_type="TransferFunction",
            result_index=1,
        ),
        "</LIGO_LW>",
    ]
    path = tmp_path / "string_typed_spectrum.xml"
    path.write_text("\n".join(xml))
    return str(path)


class TestLoadDttxmlProducts:
    def test_native_spectrum_dimensions_must_match_metadata(self, tmp_path):
        valid_path = _make_spectrum_with_dims(
            tmp_path, dims=(1, 4), filename="valid_dimensions.xml"
        )
        valid_products = load_dttxml_products(valid_path, native=True)
        valid = valid_products["ASD"]["K1:TEST-ASD"]
        assert valid["data"].shape == (4,)
        np.testing.assert_array_equal(valid["data"], [1.0, 2.0, 3.0, 4.0])

        malformed_path = _make_spectrum_with_dims(
            tmp_path, dims=(2, 2), filename="mismatched_dimensions.xml"
        )
        with pytest.warns(UserWarning, match="[Dd]imensions|shape"):
            malformed_products = load_dttxml_products(malformed_path, native=True)

        assert "ASD" not in malformed_products
        assert "PSD" not in malformed_products

    def test_native_spectrum_one_dimensional_size_must_match_samples(self, tmp_path):
        valid_path = _make_spectrum_with_dims(
            tmp_path, dims=(4,), filename="valid_flat_dimensions.xml"
        )
        valid_products = load_dttxml_products(valid_path, native=True)
        valid = valid_products["ASD"]["K1:TEST-ASD"]
        assert valid["data"].shape == (4,)
        np.testing.assert_array_equal(valid["data"], [1.0, 2.0, 3.0, 4.0])

        malformed_path = _make_spectrum_with_dims(
            tmp_path, dims=(999,), filename="mismatched_flat_dimensions.xml"
        )
        with pytest.warns(UserWarning, match="dimension|shape|sample|size"):
            malformed_products = load_dttxml_products(malformed_path, native=True)

        assert "ASD" not in malformed_products
        assert "PSD" not in malformed_products

    def test_native_parser_accepts_string_typed_numeric_params(self, tmp_path):
        path = _make_xml_with_string_typed_spectrum_params(tmp_path)

        products = load_dttxml_products(path, native=True)

        assert {"PSD", "ASD", "TF"} <= set(products)

        psd = products["PSD"]["K1:SUS-ITMX_EXCITATION"]
        assert psd["frequencies"].dtype.kind == "f"
        np.testing.assert_allclose(psd["frequencies"], np.arange(4, dtype=float))

        tf = products["TF"][("K1:SUS-ITMX_DISPLACEMENT", "K1:SUS-ITMX_EXCITATION")]
        assert tf["frequencies"].dtype.kind == "f"
        assert tf["data"].dtype.kind == "c"

    @pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
    def test_spectrum_subtype_three_is_coherence_not_transfer_function(
        self, tmp_path, native
    ):
        values = _base64_float32([0.1, 0.2, 0.3, 0.4])
        xml = "\n".join(
            [
                "<?xml version='1.0' encoding='utf-8'?>",
                "<LIGO_LW>",
                '  <LIGO_LW Name="Result[0]" Type="Spectrum">',
                '    <Param Name="Subtype">3</Param>',
                '    <Param Name="M">1</Param>',
                '    <Param Name="N">4</Param>',
                '    <Param Name="f0">17.5</Param>',
                '    <Param Name="df">2.5</Param>',
                '    <Param Name="ChannelA">K1:INPUT</Param>',
                '    <Param Name="ChannelB[0]">K1:OUTPUT</Param>',
                '    <Time Name="t0">1300000000</Time>',
                '    <Array Type="float">',
                "      <Dim>1</Dim>",
                "      <Dim>4</Dim>",
                '      <Stream Encoding="LittleEndian,base64">' + values + "</Stream>",
                "    </Array>",
                "  </LIGO_LW>",
                "</LIGO_LW>",
            ]
        )
        path = tmp_path / "spectrum_coherence.xml"
        path.write_text(xml)

        products = load_dttxml_products(str(path), native=native)

        assert "COH" in products
        assert "TF" not in products

    def test_native_parser_defaults_missing_t0_to_zero(self, tmp_path):
        values = _base64_float32([0.5, 1.5])
        xml = "\n".join(
            [
                "<?xml version='1.0' encoding='utf-8'?>",
                "<LIGO_LW>",
                '  <LIGO_LW Name="Result[0]" Type="Spectrum">',
                '    <Param Name="Subtype">1</Param>',
                '    <Param Name="M">1</Param>',
                '    <Param Name="N">2</Param>',
                '    <Param Name="f0">10.0</Param>',
                '    <Param Name="df">5.0</Param>',
                '    <Param Name="ChannelA">K1:TEST-ASD</Param>',
                '    <Array Type="float">',
                "      <Dim>1</Dim>",
                "      <Dim>2</Dim>",
                '      <Stream Encoding="LittleEndian,base64">' + values + "</Stream>",
                "    </Array>",
                "  </LIGO_LW>",
                "</LIGO_LW>",
            ]
        )
        path = tmp_path / "missing_t0.xml"
        path.write_text(xml)

        products = load_dttxml_products(path, native=True)

        assert products["PSD"]["K1:TEST-ASD"]["epoch"] == 0.0
        assert products["ASD"]["K1:TEST-ASD"]["epoch"] == 0.0

    @pytest.mark.skipif(
        not dttxml_common.HAS_DTTXML,
        reason="Reference parity requires the optional dttxml parser",
    )
    @pytest.mark.parametrize(
        "reference_first", [False, True], ids=["result-first", "reference-first"]
    )
    def test_reference_tf_pair_suffix_matches_external_in_either_block_order(
        self, tmp_path, reference_first
    ):
        def tf_block(name: str, values: list[complex]) -> str:
            samples = _base64_complex64(values)
            return "\n".join(
                [
                    f'  <LIGO_LW Name="{name}" Type="TransferFunction">',
                    '    <Param Name="Subtype">0</Param>',
                    '    <Param Name="M">1</Param>',
                    '    <Param Name="N">2</Param>',
                    '    <Param Name="f0">10.0</Param>',
                    '    <Param Name="df">5.0</Param>',
                    '    <Param Name="ChannelA">K1:TEST-INPUT</Param>',
                    '    <Param Name="ChannelB[0]">K1:TEST-OUTPUT</Param>',
                    '    <Time Name="t0">1300000000</Time>',
                    '    <Array Type="floatComplex">',
                    "      <Dim>1</Dim>",
                    "      <Dim>2</Dim>",
                    '      <Stream Encoding="LittleEndian,base64">'
                    + samples
                    + "</Stream>",
                    "    </Array>",
                    "  </LIGO_LW>",
                ]
            )

        result = tf_block("Result[0]", [1 + 2j, 3 + 4j])
        reference = tf_block("Reference[3]", [5 + 6j, 7 + 8j])
        blocks = [reference, result] if reference_first else [result, reference]
        path = tmp_path / "reference_tf.xml"
        path.write_text("\n".join(["<LIGO_LW>", *blocks, "</LIGO_LW>"]))

        native = load_dttxml_products(path, native=True)["TF"]
        external = load_dttxml_products(path, native=False)["TF"]
        expected_pairs = {
            ("K1:TEST-OUTPUT", "K1:TEST-INPUT"),
            ("K1:TEST-OUTPUT", "K1:TEST-INPUT(REF3)"),
        }

        assert set(native) == expected_pairs
        assert set(external) == expected_pairs
        for pair in expected_pairs:
            assert isinstance(external[pair], FrequencySeries)
            np.testing.assert_array_equal(external[pair].value, native[pair]["data"])

    def test_ts_entries_stay_dict_shaped_and_reader_consumes_them(self, monkeypatch):
        channel = "H1:CAL-TEST"
        data = np.array([1.0, -2.5, 4.25], dtype=float)
        dt = 0.25
        gps_second = 1234567890.0

        monkeypatch.setattr(
            dttxml_common,
            "dttxml",
            _FakeDTTXML(channel, data, dt=dt, gps_second=gps_second),
        )

        products = load_dttxml_products("synthetic.xml")
        payload = products["TS"][channel]

        assert isinstance(payload, dict)
        assert {"data", "dt", "epoch", "unit"} <= set(payload)
        assert payload.get("epoch") == gps_second
        assert payload["dt"] == dt
        np.testing.assert_allclose(payload["data"], data)

        from gwexpy.timeseries.io.dttxml import read_timeseriesdict_dttxml

        tsd = read_timeseriesdict_dttxml("synthetic.xml", products="TS")

        assert isinstance(tsd, TimeSeriesDict)
        assert list(tsd.keys()) == [channel]

        series = tsd[channel]
        np.testing.assert_allclose(series.value, data)
        assert np.isclose(float(series.t0.value), gps_second)
        assert np.isclose(float(series.dt.value), dt)
        assert np.isclose(float(series.sample_rate.value), 1.0 / dt)
