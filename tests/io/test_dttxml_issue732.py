"""Regression coverage for source-grounded DiagGUI STF layouts (#732).

The values below are synthetic and characterize the serialized product only.
They do not define an STF physical convention, unit, or pair direction.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.io.dttxml_common import HAS_DTTXML

_EPOCH = 1_234_567_890.25
_CHANNEL_A = "K1:ISSUE732-CHANNEL-A"
_CHANNEL_B = "K1:ISSUE732-CHANNEL-B"
_CHANNEL_B_SECOND = "K1:ISSUE732-CHANNEL-Z"
_F0 = 17.5
_DF = 2.5
_FREQUENCIES = np.array([10.0, 11.0, 13.5, 17.0], dtype=np.float32)
_VALUES = np.array(
    [1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 4.0j, -2.0 - 1.5j], dtype=np.complex64
)


def _stf_xml(
    tmp_path: Path,
    *,
    subtype: int,
    values: np.ndarray = _VALUES,
    frequencies: np.ndarray = _FREQUENCIES,
    channel_b: tuple[str, ...] = (_CHANNEL_B,),
    m: int = 1,
    array_type: str = "floatComplex",
) -> Path:
    """Write one labeled STF row with the surveyed /1 or /4 XML layout."""
    values = np.asarray(values, dtype=np.complex64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    frequencies = np.asarray(frequencies, dtype=np.complex64).reshape(-1)
    n_points = values.shape[1]
    embedded = subtype == 4
    words = (
        np.concatenate((frequencies.reshape(1, -1), values), axis=0)
        if embedded
        else values
    )

    root = ET.Element("LIGO_LW")
    result = ET.SubElement(
        root, "LIGO_LW", {"Name": "Result[0]", "Type": "TransferFunction"}
    )
    ET.SubElement(result, "Time", {"Name": "t0", "Type": "GPS"}).text = str(_EPOCH)
    params = {
        "Subtype": str(subtype),
        "M": str(m),
        "N": str(n_points),
        "f0": str(_F0 if not embedded else 0.0),
        "df": str(_DF if not embedded else 0.0),
        "ChannelA": _CHANNEL_A,
    }
    params.update(
        {f"ChannelB[{index}]": channel for index, channel in enumerate(channel_b)}
    )
    for name, value in params.items():
        ET.SubElement(result, "Param", {"Name": name, "Type": "lstring"}).text = value

    array = ET.SubElement(result, "Array", {"Type": array_type})
    ET.SubElement(array, "Dim").text = str(m + int(embedded))
    ET.SubElement(array, "Dim").text = str(n_points)
    ET.SubElement(
        array, "Stream", {"Encoding": "LittleEndian,base64"}
    ).text = base64.b64encode(
        words.astype("<c16" if array_type == "doubleComplex" else "<c8").tobytes()
    ).decode("ascii")

    path = tmp_path / f"stf_{subtype}.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


@pytest.fixture(params=[1, 4], ids=["linear-fhz", "embedded-fhz"])
def stf_xml(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    return _stf_xml(tmp_path, subtype=request.param)


@pytest.mark.skipif(not HAS_DTTXML, reason="requires dttxml==1.1.8")
def test_dttxml_118_stf_object_matches_xml_layout(stf_xml: Path) -> None:
    """Pin parser object fields against Params, Array/Dim, and stream bytes."""
    import dttxml

    assert dttxml.__version__ == "1.1.8"
    result = dttxml.DiagAccess(str(stf_xml)).results["STF"][_CHANNEL_A]
    assert result.type_name == "STF"
    assert result.subtype_raw in (1, 4)
    assert result.gps_second == pytest.approx(_EPOCH)
    assert result.channelA == _CHANNEL_A
    np.testing.assert_array_equal(result.channelB, [_CHANNEL_B])
    assert result.channelB_inv[_CHANNEL_B] == 0

    response = result.response
    assert response.dtype == np.dtype("complex64")
    assert response.shape == (1, len(_VALUES))
    np.testing.assert_array_equal(response[0], _VALUES)
    assert np.any(response[0].imag != 0), "fixture must retain complex sample phase"

    expected_frequencies = (
        _F0 + _DF * np.arange(len(_VALUES)) if result.subtype_raw == 1 else _FREQUENCIES
    )
    np.testing.assert_array_equal(result.FHz, expected_frequencies)
    if result.subtype_raw == 4:
        assert result.FHz.dtype == np.dtype("float32")

    tree = ET.parse(stf_xml)
    xml_result = tree.find(".//LIGO_LW[@Type='TransferFunction']")
    assert xml_result is not None
    xml_params = {
        item.get("Name"): (item.text or "").strip()
        for item in xml_result.findall("Param")
    }
    assert xml_params["Subtype"] == str(result.subtype_raw)
    assert xml_params["M"] == "1"
    assert xml_params["N"] == str(len(_VALUES))
    assert xml_params["ChannelA"] == _CHANNEL_A
    assert xml_params["ChannelB[0]"] == _CHANNEL_B
    array = xml_result.find("Array")
    assert array is not None and array.get("Type") == "floatComplex"
    assert [int(dim.text) for dim in array.findall("Dim")] == [
        2 if result.subtype_raw == 4 else 1,
        len(_VALUES),
    ]
    stream = array.find("Stream")
    assert stream is not None and stream.text is not None
    decoded_words = np.frombuffer(base64.b64decode(stream.text), dtype="<c8")
    if result.subtype_raw == 4:
        assert decoded_words.size == 2 * len(_VALUES)
        np.testing.assert_array_equal(decoded_words[: len(_VALUES)].real, _FREQUENCIES)
        np.testing.assert_array_equal(decoded_words[: len(_VALUES)].imag, 0.0)
        np.testing.assert_array_equal(decoded_words[len(_VALUES) :], _VALUES)
    else:
        np.testing.assert_array_equal(decoded_words, _VALUES)


@pytest.mark.parametrize("native", [False, True], ids=["external", "native"])
def test_stf_matrix_reader_preserves_labeled_complex_row_and_axis(
    stf_xml: Path, native: bool
) -> None:
    if not native and not HAS_DTTXML:
        pytest.skip("external route requires dttxml==1.1.8")
    result = FrequencySeriesMatrix.read(
        stf_xml,
        format="xml.diaggui",
        products="STF",
        native=native,
    )

    assert isinstance(result, FrequencySeriesMatrix)
    assert result.shape == (1, 1, len(_VALUES))
    assert list(result.rows) == [_CHANNEL_B]
    assert list(result.cols) == [_CHANNEL_A]
    series = result[_CHANNEL_B, _CHANNEL_A]
    assert series.dtype == np.dtype("complex64")
    np.testing.assert_array_equal(series.value, _VALUES)
    assert np.any(series.value.imag != 0), "complex phase must survive normalization"
    xml_result = ET.parse(stf_xml).find(".//LIGO_LW[@Type='TransferFunction']")
    assert xml_result is not None
    subtype = int(xml_result.find("Param[@Name='Subtype']").text)
    expected_frequencies = (
        _F0 + _DF * np.arange(len(_VALUES)) if subtype == 1 else _FREQUENCIES
    )
    np.testing.assert_array_equal(series.frequencies.value, expected_frequencies)
    assert float(series.epoch.value) == pytest.approx(_EPOCH)


@pytest.mark.parametrize("subtype", [1, 4], ids=["linear", "embedded"])
@pytest.mark.parametrize("native", [False, True], ids=["external", "native"])
def test_stf_reader_preserves_one_bin_axis(
    tmp_path: Path, subtype: int, native: bool
) -> None:
    frequencies = np.array([_F0 if subtype == 1 else _FREQUENCIES[0]], dtype=np.float32)
    values = _VALUES[:1]
    path = _stf_xml(
        tmp_path,
        subtype=subtype,
        values=values,
        frequencies=frequencies,
    )
    result = FrequencySeriesMatrix.read(
        path,
        format="xml.diaggui",
        products="STF",
        native=native,
    )

    series = result[_CHANNEL_B, _CHANNEL_A]
    np.testing.assert_array_equal(series.value, values)
    np.testing.assert_array_equal(series.frequencies.value, frequencies)


@pytest.mark.parametrize("native", [False, True], ids=["external", "native"])
def test_stf_reader_preserves_every_indexed_response_row(
    tmp_path: Path, native: bool
) -> None:
    values = np.array(
        [
            [1.0 + 2.0j, 3.0 - 4.0j, -2.0 + 0.5j, 0.25 - 1.0j],
            [-1.0 + 0.5j, 2.0 + 1.0j, 0.75 - 3.0j, 4.0 + 0.25j],
        ],
        dtype=np.complex64,
    )
    path = _stf_xml(
        tmp_path,
        subtype=4,
        values=values,
        channel_b=(_CHANNEL_B_SECOND, _CHANNEL_B),
        m=2,
    )
    result = FrequencySeriesMatrix.read(
        path,
        format="xml.diaggui",
        products="STF",
        native=native,
    )

    assert list(result.rows) == sorted((_CHANNEL_B, _CHANNEL_B_SECOND))
    assert list(result.cols) == [_CHANNEL_A]
    np.testing.assert_array_equal(
        result[_CHANNEL_B_SECOND, _CHANNEL_A].value, values[0]
    )
    np.testing.assert_array_equal(result[_CHANNEL_B, _CHANNEL_A].value, values[1])


def test_stf_matrix_reader_uses_real_no_dttxml_interpreter(stf_xml: Path) -> None:
    """Fallback coverage must run in a separate interpreter without dttxml."""
    if importlib.util.find_spec("dttxml") is None:
        # Base-only CI already runs this test under the required interpreter.
        python = sys.executable
    else:
        # Installed-parser jobs need a separately verified base-only Python.
        python = os.environ.get("GWEXPY_NO_DTTXML_PYTHON")
        if python:
            try:
                probe = subprocess.run(
                    [
                        python,
                        "-c",
                        "import importlib.util; "
                        "raise SystemExit(importlib.util.find_spec('dttxml') is not None)",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                python = None
            else:
                if probe.returncode != 0:
                    python = None
        if not python:
            pytest.skip("no available interpreter without dttxml was configured")

    project_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.util, json, sys
assert importlib.util.find_spec('dttxml') is None
import numpy as np
from gwexpy.frequencyseries import FrequencySeriesMatrix
path, channel_a, channel_b = sys.argv[1:]
result = FrequencySeriesMatrix.read(path, format='xml.diaggui', products='STF')
series = result[channel_b, channel_a]
print(json.dumps({
    'type': type(result).__name__, 'shape': result.shape,
    'rows': list(result.rows), 'cols': list(result.cols),
    'dtype': str(series.dtype),
    'values': [[float(value.real), float(value.imag)] for value in series.value],
    'frequencies': series.frequencies.value.tolist(),
    'epoch': float(series.epoch.value),
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [str(python), "-c", code, str(stf_xml), _CHANNEL_A, _CHANNEL_B],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = json.loads(completed.stdout)
    xml_result = ET.parse(stf_xml).find(".//LIGO_LW[@Type='TransferFunction']")
    assert xml_result is not None
    subtype = int(xml_result.find("Param[@Name='Subtype']").text)
    expected_frequencies = (
        _F0 + _DF * np.arange(len(_VALUES)) if subtype == 1 else _FREQUENCIES
    )
    assert result["type"] == "FrequencySeriesMatrix"
    assert result["shape"] == [1, 1, len(_VALUES)]
    assert result["rows"] == [_CHANNEL_B]
    assert result["cols"] == [_CHANNEL_A]
    assert result["dtype"] == "complex64"
    np.testing.assert_array_equal(
        result["values"], np.column_stack((_VALUES.real, _VALUES.imag))
    )
    np.testing.assert_array_equal(result["frequencies"], expected_frequencies)
    assert float(result["epoch"]) == pytest.approx(_EPOCH)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing-channel-b", "ChannelB"),
        ("ambiguous-channel-b", "ChannelB"),
        ("duplicate-channel-label", "unique"),
        ("row-count-mismatch", "M.*row|row.*M"),
        ("imaginary-frequency", "frequency.*imaginary|imaginary.*frequency"),
        ("double-complex", "floatComplex"),
    ],
)
@pytest.mark.parametrize("native", [False, True], ids=["external", "native"])
def test_stf_reader_reports_ambiguous_or_inconsistent_layout(
    tmp_path: Path, case: str, match: str, native: bool
) -> None:
    """Reject layouts whose labels or embedded frequency words are ambiguous."""
    if not native and not HAS_DTTXML:
        pytest.skip("external route requires dttxml==1.1.8")
    kwargs: dict[str, object] = {"subtype": 4}
    if case == "missing-channel-b":
        kwargs["channel_b"] = ()
    elif case == "ambiguous-channel-b":
        kwargs["channel_b"] = (_CHANNEL_B, "K1:ISSUE732-SECOND-OUTPUT")
    elif case == "duplicate-channel-label":
        kwargs["channel_b"] = (_CHANNEL_B, _CHANNEL_B)
        kwargs["m"] = 2
        kwargs["values"] = np.vstack((_VALUES, _VALUES))
    elif case == "row-count-mismatch":
        kwargs["m"] = 2
    elif case == "imaginary-frequency":
        frequencies = _FREQUENCIES.astype(np.complex64)
        frequencies[1] += np.complex64(0.5j)
        kwargs["frequencies"] = frequencies
    elif case == "double-complex":
        kwargs["array_type"] = "doubleComplex"
    path = _stf_xml(tmp_path, **kwargs)

    with pytest.raises(ValueError, match=match):
        FrequencySeriesMatrix.read(
            path,
            format="xml.diaggui",
            products="STF",
            native=native,
        )
