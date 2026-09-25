"""Source-backed regressions for DiagGUI TransferFunction subtype 6 (issue #733)."""

from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

import gwexpy.io.dttxml_common as dttxml_common
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.io.dttxml_common import HAS_DTTXML, load_dttxml_products

_EPOCH = 1_234_567_890.25
_FREQUENCIES = (17.5, 20.0, 22.5)
_SAMPLES = (1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 4.0j)
_CHANNEL_A = "K1:TEST-INPUT"
_CHANNEL_B = "K1:TEST-OUTPUT"
_PAIR = (_CHANNEL_B, _CHANNEL_A)
_TF6_REFUSAL = r"(?i)(subtype\s*6|tf\s*/\s*6|tf6)"


def _oracle_bytes(
    frequencies: tuple[float, ...] = _FREQUENCIES,
    samples: tuple[complex, ...] = _SAMPLES,
) -> bytes:
    """Pack the verified mixed stream as frequency f64 words then complex f32 pairs."""
    frequency_bytes = struct.pack(f"<{len(frequencies)}d", *frequencies)
    sample_components = tuple(
        component for z in samples for component in (z.real, z.imag)
    )
    sample_bytes = struct.pack(f"<{len(sample_components)}f", *sample_components)
    return frequency_bytes + sample_bytes


def _oracle_decode(raw: bytes, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode with explicit scalar struct formats, independent of NumPy complex views."""
    frequency_size = 8 * n
    frequencies = np.asarray(struct.unpack(f"<{n}d", raw[:frequency_size]))
    components = struct.unpack(f"<{2 * n}f", raw[frequency_size:])
    samples = np.asarray(components[::2]) + 1j * np.asarray(components[1::2])
    return frequencies, samples.astype(np.complex64)


def _write_tf6(
    tmp_path,
    *,
    name: str = "Result[6]",
    channel_a: str = _CHANNEL_A,
    channel_b: str = _CHANNEL_B,
    frequencies: tuple[float, ...] = _FREQUENCIES,
    samples: tuple[complex, ...] = _SAMPLES,
    subtype: int = 6,
    filename: str = "tf6.xml",
) -> tuple[str, np.ndarray, np.ndarray]:
    raw = _oracle_bytes(frequencies, samples)
    expected_frequencies, expected_samples = _oracle_decode(raw, len(samples))
    root = ET.Element("LIGO_LW")
    result = ET.SubElement(root, "LIGO_LW", {"Name": name, "Type": "TransferFunction"})
    params = {
        "Subtype": str(subtype),
        "M": "1",
        "N": str(len(samples)),
        "f0": "0",
        "df": "0",
        "ChannelA": channel_a,
        "ChannelB[0]": channel_b,
    }
    for param_name, value in params.items():
        ET.SubElement(result, "Param", {"Name": param_name}).text = value
    ET.SubElement(result, "Time", {"Name": "t0"}).text = str(_EPOCH)
    array = ET.SubElement(result, "Array", {"Type": "floatComplex"})
    ET.SubElement(array, "Dim").text = "2"
    ET.SubElement(array, "Dim").text = str(len(samples))
    stream = ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"})
    stream.text = base64.b64encode(raw).decode("ascii")
    path = tmp_path / filename
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return str(path), expected_frequencies, expected_samples


def _write_tf0(
    parent: ET.Element,
    *,
    name: str,
    samples: tuple[complex, ...],
    channel_a: str = _CHANNEL_A,
    channel_b: str = _CHANNEL_B,
) -> None:
    result = ET.SubElement(
        parent, "LIGO_LW", {"Name": name, "Type": "TransferFunction"}
    )
    for param_name, value in {
        "Subtype": "0",
        "M": "1",
        "N": str(len(samples)),
        "f0": str(_FREQUENCIES[0]),
        "df": "2.5",
        "ChannelA": channel_a,
        "ChannelB[0]": channel_b,
    }.items():
        ET.SubElement(result, "Param", {"Name": param_name}).text = value
    ET.SubElement(result, "Time", {"Name": "t0"}).text = str(_EPOCH)
    array = ET.SubElement(result, "Array", {"Type": "floatComplex"})
    ET.SubElement(array, "Dim").text = "1"
    ET.SubElement(array, "Dim").text = str(len(samples))
    stream = ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"})
    interleaved = tuple(component for z in samples for component in (z.real, z.imag))
    stream.text = base64.b64encode(
        struct.pack(f"<{len(interleaved)}f", *interleaved)
    ).decode("ascii")


def _write_collision(tmp_path, tf6_first: bool) -> tuple[str, np.ndarray]:
    root = ET.Element("LIGO_LW")
    tf6 = ET.Element("LIGO_LW", {"Name": "Result[6]", "Type": "TransferFunction"})
    for param_name, value in {
        "Subtype": "6",
        "M": "1",
        "N": str(len(_SAMPLES)),
        "ChannelA": _CHANNEL_A,
        "ChannelB[0]": _CHANNEL_B,
    }.items():
        ET.SubElement(tf6, "Param", {"Name": param_name}).text = value
    ET.SubElement(tf6, "Time", {"Name": "t0"}).text = str(_EPOCH)
    array = ET.SubElement(tf6, "Array", {"Type": "floatComplex"})
    ET.SubElement(array, "Dim").text = "2"
    ET.SubElement(array, "Dim").text = str(len(_SAMPLES))
    stream = ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"})
    stream.text = base64.b64encode(_oracle_bytes()).decode("ascii")

    tf0 = ET.Element("LIGO_LW", {"Name": "Result[0]", "Type": "TransferFunction"})
    for param_name, value in {
        "Subtype": "0",
        "M": "1",
        "N": str(len(_SAMPLES)),
        "f0": str(_FREQUENCIES[0]),
        "df": "2.5",
        "ChannelA": _CHANNEL_A,
        "ChannelB[0]": _CHANNEL_B,
    }.items():
        ET.SubElement(tf0, "Param", {"Name": param_name}).text = value
    ET.SubElement(tf0, "Time", {"Name": "t0"}).text = str(_EPOCH)
    tf0_array = ET.SubElement(tf0, "Array", {"Type": "floatComplex"})
    ET.SubElement(tf0_array, "Dim").text = "1"
    ET.SubElement(tf0_array, "Dim").text = str(len(_SAMPLES))
    tf0_stream = ET.SubElement(tf0_array, "Stream", {"Encoding": "LittleEndian,base64"})
    tf0_samples = tuple(z * 2 for z in _SAMPLES)
    components = tuple(c for z in tf0_samples for c in (z.real, z.imag))
    tf0_stream.text = base64.b64encode(
        struct.pack(f"<{len(components)}f", *components)
    ).decode("ascii")

    for block in (tf6, tf0) if tf6_first else (tf0, tf6):
        root.append(block)
    path = tmp_path / f"tf0_tf6_{'tf6first' if tf6_first else 'tf0first'}.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return str(path), np.asarray(tf0_samples, dtype=np.complex64)


def _assert_tf6_series(series, frequencies: np.ndarray, samples: np.ndarray) -> None:
    assert isinstance(series, FrequencySeries)
    assert series.dtype == np.dtype(np.complex64)
    assert series.shape == samples.shape
    np.testing.assert_array_equal(series.value, samples)
    np.testing.assert_array_equal(series.frequencies.value, frequencies)
    assert float(series.epoch.value) == pytest.approx(_EPOCH)
    assert np.any(np.angle(series.value) != 0)


@pytest.mark.parametrize("tf6_first", [False, True], ids=["tf0-first", "tf6-first"])
def test_fixture_metadata_and_dttxml_result_identity_and_source_order(
    tmp_path, tf6_first
):
    """Pin the installed parser's ChannelA key and XML source-order overwrite behavior."""
    pytest.importorskip("dttxml")
    path, _ = _write_collision(tmp_path, tf6_first)
    root = ET.parse(path).getroot()
    blocks = root.findall("LIGO_LW")
    assert [block.get("Type") for block in blocks] == [
        "TransferFunction",
        "TransferFunction",
    ]
    assert [block.get("Name") for block in blocks] == (
        ["Result[6]", "Result[0]"] if tf6_first else ["Result[0]", "Result[6]"]
    )
    assert [block.findtext("Param[@Name='Subtype']") for block in blocks] == (
        ["6", "0"] if tf6_first else ["0", "6"]
    )
    assert all(
        block.findtext("Param[@Name='ChannelA']") == _CHANNEL_A for block in blocks
    )
    assert all(
        block.findtext("Param[@Name='ChannelB[0]']") == _CHANNEL_B for block in blocks
    )

    import dttxml

    assert dttxml.__version__ == "1.1.8"
    parsed = dttxml.DiagAccess(path).results.TF
    assert list(parsed) == [_CHANNEL_A]
    last_subtype = 0 if tf6_first else 6
    assert parsed[_CHANNEL_A].subtype_raw == last_subtype
    np.testing.assert_array_equal(parsed[_CHANNEL_A].channelB, [_CHANNEL_B])


def test_independent_oracle_decodes_only_little_endian_mixed_fixture(tmp_path):
    path, frequencies, samples = _write_tf6(tmp_path)
    stream = ET.parse(path).findtext(".//Stream")
    assert stream is not None
    raw = base64.b64decode(stream, validate=True)
    oracle_frequencies, oracle_samples = _oracle_decode(raw, len(_SAMPLES))
    assert raw == _oracle_bytes()
    np.testing.assert_array_equal(
        oracle_frequencies, np.asarray([17.5, 20.0, 22.5], dtype=np.float64)
    )
    np.testing.assert_array_equal(
        oracle_samples,
        np.asarray([1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 4.0j], dtype=np.complex64),
    )
    np.testing.assert_array_equal(oracle_frequencies, frequencies)
    np.testing.assert_array_equal(oracle_samples, samples)
    assert oracle_samples.dtype == np.dtype(np.complex64)
    assert np.any(np.angle(oracle_samples) != 0)


@pytest.mark.skipif(not HAS_DTTXML, reason="external route requires dttxml==1.1.8")
def test_external_tf6_preserves_or_fails_before_returning_real_only_data(tmp_path):
    path, frequencies, samples = _write_tf6(tmp_path)
    try:
        products = load_dttxml_products(path)
    except ValueError as exc:
        assert re.search(_TF6_REFUSAL, str(exc)), str(exc)
        return
    assert "TF" in products and _PAIR in products["TF"]
    _assert_tf6_series(products["TF"][_PAIR], frequencies, samples)


@pytest.mark.skipif(not HAS_DTTXML, reason="external route requires dttxml==1.1.8")
def test_external_tf0_survives_distinct_tf6_repair_or_specific_refusal(
    tmp_path, monkeypatch
):
    tf6_path, tf6_frequencies, tf6_samples = _write_tf6(
        tmp_path,
        name="Result[6]",
        channel_a="K1:TEST-TF6-INPUT",
        channel_b="K1:TEST-TF6-OUTPUT",
        filename="distinct_tf6.xml",
    )
    root = ET.Element("LIGO_LW")
    tf0_values = (2.0 + 3.0j, -1.0 + 0.5j, 4.0 - 2.0j)
    _write_tf0(root, name="Result[0]", samples=tf0_values)
    tf6_block = ET.parse(tf6_path).getroot().find("LIGO_LW")
    assert tf6_block is not None
    root.append(tf6_block)
    path = tmp_path / "distinct_tf0_tf6.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    import dttxml

    original_diag_access = dttxml.DiagAccess
    external_tf0_sentinel = np.asarray(
        [101.0 + 7.0j, -103.0 + 11.0j, 107.0 - 13.0j], dtype=np.complex64
    )

    class SentinelDiagAccess:
        def __init__(self, source):
            parsed = original_diag_access(source)
            parsed.results.TF[_CHANNEL_A].xfer = external_tf0_sentinel.reshape(1, -1)
            self.results = parsed.results

    # Keep the source XML untouched. This sentinel exists only in the installed
    # parser's in-memory TF/0 result and detects wholesale native TF replacement.
    monkeypatch.setattr(dttxml, "DiagAccess", SentinelDiagAccess)

    tf6_pair = ("K1:TEST-TF6-OUTPUT", "K1:TEST-TF6-INPUT")
    try:
        products = load_dttxml_products(path)
    except ValueError as exc:
        assert re.search(_TF6_REFUSAL, str(exc)), str(exc)
        return

    assert "TF" in products
    tf0 = products["TF"][_PAIR]
    assert isinstance(tf0, FrequencySeries)
    assert tf0.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(tf0.value, external_tf0_sentinel)
    np.testing.assert_array_equal(tf0.frequencies.value, _FREQUENCIES)
    assert float(tf0.epoch.value) == pytest.approx(_EPOCH)
    _assert_tf6_series(products["TF"][tf6_pair], tf6_frequencies, tf6_samples)


def test_native_tf6_preserves_complex_values_axis_epoch_and_pair(tmp_path):
    path, frequencies, samples = _write_tf6(tmp_path)
    products = load_dttxml_products(path, native=True)
    assert set(products["TF"]) == {_PAIR}
    native = products["TF"][_PAIR]
    assert isinstance(native, dict)
    assert native["data"].dtype == np.dtype("<c8")
    assert native["data"].shape == samples.shape
    np.testing.assert_array_equal(native["data"], samples)
    np.testing.assert_array_equal(native["frequencies"], frequencies)
    assert native["epoch"] == pytest.approx(_EPOCH)


@pytest.mark.parametrize("tf6_first", [False, True], ids=["tf0-first", "tf6-first"])
@pytest.mark.parametrize("native", [False, True], ids=["external", "native"])
def test_same_pair_tf0_and_tf6_fail_closed_in_either_xml_order(
    tmp_path, tf6_first, native
):
    if not native and not HAS_DTTXML:
        pytest.skip("external route requires dttxml==1.1.8")
    path, _ = _write_collision(tmp_path, tf6_first)
    with pytest.raises(ValueError, match="ambiguous|collision|multiple|subtype"):
        load_dttxml_products(path, native=native)


def test_subtype_raw_none_cannot_authorize_tf6_recovery(tmp_path, monkeypatch):
    path, frequencies, _ = _write_tf6(tmp_path)
    info = SimpleNamespace(
        FHz=frequencies,
        gps_second=_EPOCH,
        channelB=np.asarray([_CHANNEL_B]),
        xfer=np.asarray([np.real(_SAMPLES)], dtype=np.float32),
    )

    class FakeDiagAccess:
        def __init__(self, source):
            assert source == path
            self.results = SimpleNamespace(TF={_CHANNEL_A: info})

    monkeypatch.setattr(
        dttxml_common,
        "dttxml",
        SimpleNamespace(DiagAccess=FakeDiagAccess),
    )
    with pytest.raises(ValueError, match="subtype|identity|phase|ambiguous|raw"):
        load_dttxml_products(path, native=False)


def test_raw_tf6_identity_mismatch_fails_closed(tmp_path, monkeypatch):
    path, frequencies, _ = _write_tf6(tmp_path)
    wrong_channel_a = "K1:OTHER-INPUT"
    info = SimpleNamespace(
        FHz=frequencies,
        gps_second=_EPOCH,
        subtype_raw=6,
        channelB=np.asarray([_CHANNEL_B]),
        xfer=np.asarray([np.real(_SAMPLES)], dtype=np.float32),
    )

    class FakeDiagAccess:
        def __init__(self, source):
            assert source == path
            self.results = SimpleNamespace(TF={wrong_channel_a: info})

    monkeypatch.setattr(
        dttxml_common,
        "dttxml",
        SimpleNamespace(DiagAccess=FakeDiagAccess),
    )
    with pytest.raises(ValueError, match="identity|match|subtype|ambiguous|raw"):
        load_dttxml_products(path, native=False)


@pytest.mark.parametrize("native", [False, True], ids=["external", "native"])
def test_noncolliding_tf0_is_unchanged(tmp_path, native):
    if not native and not HAS_DTTXML:
        pytest.skip("external route requires dttxml==1.1.8")
    root = ET.Element("LIGO_LW")
    values = (1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 4.0j)
    _write_tf0(root, name="Result[0]", samples=values)
    path = tmp_path / "noncolliding_tf0.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    products = load_dttxml_products(path, native=native)
    assert set(products["TF"]) == {_PAIR}
    if native:
        np.testing.assert_array_equal(products["TF"][_PAIR]["data"], values)
    else:
        series = products["TF"][_PAIR]
        assert isinstance(series, FrequencySeries)
        assert series.dtype == np.dtype(np.complex64)
        np.testing.assert_array_equal(series.value, values)
        np.testing.assert_array_equal(series.frequencies.value, _FREQUENCIES)
        assert float(series.epoch.value) == pytest.approx(_EPOCH)
