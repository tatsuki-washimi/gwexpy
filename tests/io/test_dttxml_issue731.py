"""Regression coverage for source-backed DiagGUI Spectrum FFT products (#731)."""

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

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
from gwexpy.io.dttxml_common import HAS_DTTXML, load_dttxml_products

EPOCH = 1_234_567_890.25
LINEAR_CHANNEL = "K1:AUDIT-FFT-LINEAR"
EMBEDDED_CHANNEL = "K1:AUDIT-FFT-EMBEDDED"
LINEAR_VALUES = np.array([1 + 2j, -0.5 + 0.25j, 3 - 4j, -2 - 1.5j], dtype=np.complex64)
EMBEDDED_VALUES = np.array(
    [0.25 - 0.75j, -2 + 3j, 1.5 + 0.5j, -4 - 2j], dtype=np.complex64
)
LINEAR_FREQUENCIES = np.array([17.5, 20.0, 22.5, 25.0])
EMBEDDED_FREQUENCIES = np.array([10.0, 11.0, 13.5, 17.0])


def _stream(values: np.ndarray, *, array_type: str) -> str:
    storage_type = "<c16" if array_type == "doubleComplex" else "<c8"
    return base64.b64encode(np.asarray(values, dtype=storage_type).tobytes()).decode(
        "ascii"
    )


def _add_spectrum(
    root: ET.Element,
    *,
    result_index: int,
    subtype: int,
    channel: str,
    values: np.ndarray,
    array_type: str,
    n_points: int,
    rows: int,
    f0: float,
    df: float,
    dims: tuple[int, ...],
) -> None:
    result = ET.SubElement(
        root, "LIGO_LW", {"Name": f"Result[{result_index}]", "Type": "Spectrum"}
    )
    ET.SubElement(result, "Time", {"Name": "t0", "Type": "GPS"}).text = str(EPOCH)
    for name, value in {
        "Subtype": str(subtype),
        "M": str(rows),
        "N": str(n_points),
        "f0": str(f0),
        "df": str(df),
        "ChannelA": channel,
    }.items():
        ET.SubElement(result, "Param", {"Name": name}).text = value
    array = ET.SubElement(result, "Array", {"Type": array_type})
    for dimension in dims:
        ET.SubElement(array, "Dim").text = str(dimension)
    ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = _stream(
        values, array_type=array_type
    )


def _write_fft_xml(
    path: Path,
    *,
    bad_embedded_frequency: bool = False,
    multirow: bool = False,
    count_mismatch: bool = False,
    unsupported_precision: bool = False,
) -> Path:
    root = ET.Element("LIGO_LW")
    linear_values = LINEAR_VALUES
    linear_rows = 1
    linear_dims = (1, LINEAR_VALUES.size)
    if multirow:
        linear_values = np.concatenate((LINEAR_VALUES, EMBEDDED_VALUES))
        linear_rows = 2
        linear_dims = (2, LINEAR_VALUES.size)
    if count_mismatch:
        linear_values = np.concatenate((LINEAR_VALUES, EMBEDDED_VALUES))
        linear_dims = (1, 2 * LINEAR_VALUES.size)
    _add_spectrum(
        root,
        result_index=0,
        subtype=0,
        channel=LINEAR_CHANNEL,
        values=linear_values,
        array_type="doubleComplex" if unsupported_precision else "floatComplex",
        n_points=LINEAR_VALUES.size,
        rows=linear_rows,
        f0=17.5,
        df=2.5,
        dims=linear_dims,
    )

    embedded_frequencies = EMBEDDED_FREQUENCIES.astype(np.complex64)
    if bad_embedded_frequency:
        embedded_frequencies[1] += np.complex64(0.125j)
    embedded_values = np.concatenate((embedded_frequencies, EMBEDDED_VALUES))
    _add_spectrum(
        root,
        result_index=1,
        subtype=4,
        channel=EMBEDDED_CHANNEL,
        values=embedded_values,
        array_type="floatComplex",
        n_points=EMBEDDED_VALUES.size,
        rows=1,
        f0=0,
        df=0,
        dims=(2, EMBEDDED_VALUES.size),
    )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _write_one_bin_fft_xml(path: Path) -> Path:
    root = ET.Element("LIGO_LW")
    _add_spectrum(
        root,
        result_index=0,
        subtype=0,
        channel="K1:AUDIT-FFT-ONE-LINEAR",
        values=np.array([2.5 - 1.25j], dtype=np.complex64),
        array_type="floatComplex",
        n_points=1,
        rows=1,
        f0=17.5,
        df=2.5,
        dims=(1, 1),
    )
    _add_spectrum(
        root,
        result_index=1,
        subtype=4,
        channel="K1:AUDIT-FFT-ONE-EMBEDDED",
        values=np.array([31.25 + 0j, -0.75 + 3.5j], dtype=np.complex64),
        array_type="floatComplex",
        n_points=1,
        rows=1,
        f0=0,
        df=0,
        dims=(2, 1),
    )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


@pytest.fixture
def fft_xml(tmp_path: Path) -> Path:
    """Build the characterized Spectrum/0 and Spectrum/4 raw XML layouts."""
    return _write_fft_xml(tmp_path / "spectrum_fft.xml")


def _assert_series(
    series: FrequencySeries, values: np.ndarray, frequencies: np.ndarray
):
    assert isinstance(series, FrequencySeries)
    assert series.dtype == np.dtype(np.complex64)
    assert series.shape == values.shape
    assert np.isfinite(series.value).all()
    np.testing.assert_array_equal(series.value, values)
    np.testing.assert_array_equal(np.angle(series.value), np.angle(values))
    np.testing.assert_array_equal(series.frequencies.value, frequencies)
    assert float(series.epoch.value) == pytest.approx(EPOCH)
    assert str(series.unit) == ""


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_fft_frequency_readers_preserve_both_channel_layouts(
    fft_xml: Path, native: bool
) -> None:
    normalized = load_dttxml_products(fft_xml, native=native)
    fft = normalized["FFT"]
    assert list(fft) == [LINEAR_CHANNEL, EMBEDDED_CHANNEL]
    if native or not HAS_DTTXML:
        for channel, values, frequencies in (
            (LINEAR_CHANNEL, LINEAR_VALUES, LINEAR_FREQUENCIES),
            (EMBEDDED_CHANNEL, EMBEDDED_VALUES, EMBEDDED_FREQUENCIES),
        ):
            assert isinstance(fft[channel], dict)
            assert fft[channel]["unit"] is None
            assert fft[channel]["epoch"] == EPOCH
            np.testing.assert_array_equal(fft[channel]["data"], values)
            np.testing.assert_array_equal(fft[channel]["frequencies"], frequencies)
    else:
        assert all(isinstance(series, FrequencySeries) for series in fft.values())

    result = FrequencySeriesDict.read(
        fft_xml, format="xml.diaggui", products="FFT", native=native
    )
    assert isinstance(result, FrequencySeriesDict)
    assert list(result) == [LINEAR_CHANNEL, EMBEDDED_CHANNEL]
    _assert_series(result[LINEAR_CHANNEL], LINEAR_VALUES, LINEAR_FREQUENCIES)
    _assert_series(result[EMBEDDED_CHANNEL], EMBEDDED_VALUES, EMBEDDED_FREQUENCIES)

    for channel, values, frequencies in (
        (LINEAR_CHANNEL, LINEAR_VALUES, LINEAR_FREQUENCIES),
        (EMBEDDED_CHANNEL, EMBEDDED_VALUES, EMBEDDED_FREQUENCIES),
    ):
        direct = FrequencySeries.read(
            fft_xml,
            format="xml.diaggui",
            products="FFT",
            channels=[channel],
            native=native,
        )
        assert isinstance(direct, FrequencySeries)
        assert direct.name == channel
        assert str(direct.channel) == channel
        _assert_series(direct, values, frequencies)


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("bad_embedded_frequency", "nonzero imaginary"),
        ("multirow", "one-row FFT layout"),
        ("count_mismatch", "Frequency axis.*samples|row count|Dimensions"),
        ("unsupported_precision", "only floatComplex|Frequency axis.*values"),
    ],
)
def test_fft_reader_rejects_ambiguous_or_invalid_layouts(
    tmp_path: Path, native: bool, malformation: str, message: str
) -> None:
    path = _write_fft_xml(tmp_path / f"{malformation}.xml", **{malformation: True})
    with pytest.raises(ValueError, match=message):
        load_dttxml_products(path, native=native)


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_fft_one_bin_frequency_axis_is_preserved(tmp_path: Path, native: bool) -> None:
    path = _write_one_bin_fft_xml(tmp_path / "spectrum_fft_one_bin.xml")
    expected = {
        "K1:AUDIT-FFT-ONE-LINEAR": (
            np.array([2.5 - 1.25j], dtype=np.complex64),
            np.array([17.5]),
        ),
        "K1:AUDIT-FFT-ONE-EMBEDDED": (
            np.array([-0.75 + 3.5j], dtype=np.complex64),
            np.array([31.25]),
        ),
    }

    normalized = load_dttxml_products(path, native=native)["FFT"]
    result = FrequencySeriesDict.read(
        path, format="xml.diaggui", products="FFT", native=native
    )
    assert list(result) == list(expected)
    for channel, (values, frequencies) in expected.items():
        if native or not HAS_DTTXML:
            assert isinstance(normalized[channel], dict)
            np.testing.assert_array_equal(normalized[channel]["data"], values)
            np.testing.assert_array_equal(
                normalized[channel]["frequencies"], frequencies
            )
        else:
            _assert_series(normalized[channel], values, frequencies)
        _assert_series(result[channel], values, frequencies)
        direct = FrequencySeries.read(
            path,
            format="xml.diaggui",
            products="FFT",
            channels=[channel],
            native=native,
        )
        _assert_series(direct, values, frequencies)


@pytest.mark.parametrize("native", [False, True], ids=["dttxml", "native"])
def test_fft_reader_rejects_double_complex_n8_even_when_external_parser_accepts(
    tmp_path: Path, native: bool
) -> None:
    path = tmp_path / "spectrum_fft_double_complex_n8.xml"
    _add_root = ET.Element("LIGO_LW")
    _add_spectrum(
        _add_root,
        result_index=0,
        subtype=0,
        channel="K1:AUDIT-FFT-DOUBLE-COMPLEX",
        values=np.arange(8, dtype=np.float32).astype(np.complex64)
        + 1j * np.arange(8, dtype=np.float32),
        array_type="doubleComplex",
        n_points=8,
        rows=1,
        f0=0,
        df=1,
        dims=(1, 8),
    )
    ET.ElementTree(_add_root).write(path, encoding="utf-8", xml_declaration=True)

    with pytest.raises(ValueError, match="only floatComplex is characterized"):
        load_dttxml_products(path, native=native)


def test_fft_fallback_in_separate_no_dttxml_interpreter(fft_xml: Path) -> None:
    """Exercise the actual fallback environment without package monkeypatching."""
    if importlib.util.find_spec("dttxml") is None:
        python = sys.executable
    else:
        python = os.environ.get("GWEXPY_NO_DTTXML_PYTHON")
    if not python:
        pytest.skip("fallback route needs a separate interpreter without dttxml")

    project_root = Path(__file__).resolve().parents[2]
    one_bin_xml = _write_one_bin_fft_xml(fft_xml.parent / "spectrum_fft_one_bin.xml")
    code = """
import importlib.util, json, sys
import numpy as np
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesDict
assert importlib.util.find_spec('dttxml') is None
path = sys.argv[1]
channels = ('K1:AUDIT-FFT-LINEAR', 'K1:AUDIT-FFT-EMBEDDED')
result = FrequencySeriesDict.read(path, format='xml.diaggui', products='FFT')
assert isinstance(result, FrequencySeriesDict)
payload = {'dict_type': type(result).__name__, 'channels': list(result), 'series': {}}
for channel in channels:
    series = result[channel]
    assert isinstance(series, FrequencySeries)
    direct = FrequencySeries.read(path, format='xml.diaggui', products='FFT', channels=[channel])
    assert isinstance(direct, FrequencySeries)
    payload['series'][channel] = {
        'real': series.value.real.tolist(), 'imag': series.value.imag.tolist(),
        'dtype': str(series.dtype),
        'phase': np.angle(series.value).tolist(),
        'frequencies': series.frequencies.value.tolist(),
        'epoch': float(series.epoch.value), 'name': direct.name,
        'direct_type': type(direct).__name__,
    }
one_bin = FrequencySeriesDict.read(sys.argv[2], format='xml.diaggui', products='FFT')
for channel, frequency in (
    ('K1:AUDIT-FFT-ONE-LINEAR', 17.5),
    ('K1:AUDIT-FFT-ONE-EMBEDDED', 31.25),
):
    series = one_bin[channel]
    direct = FrequencySeries.read(
        sys.argv[2], format='xml.diaggui', products='FFT', channels=[channel]
    )
    assert series.frequencies.value.tolist() == [frequency]
    assert direct.frequencies.value.tolist() == [frequency]
print(json.dumps(payload))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [python, "-c", code, str(fft_xml), str(one_bin_xml)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = json.loads(completed.stdout)
    assert result["dict_type"] == "FrequencySeriesDict"
    assert result["channels"] == [LINEAR_CHANNEL, EMBEDDED_CHANNEL]
    for channel, values, frequencies in (
        (LINEAR_CHANNEL, LINEAR_VALUES, LINEAR_FREQUENCIES),
        (EMBEDDED_CHANNEL, EMBEDDED_VALUES, EMBEDDED_FREQUENCIES),
    ):
        actual = result["series"][channel]
        assert actual["dtype"] == "complex64"
        assert actual["direct_type"] == "FrequencySeries"
        assert actual["name"] == channel
        actual_values = np.asarray(actual["real"]) + 1j * np.asarray(actual["imag"])
        np.testing.assert_array_equal(actual_values, values)
        np.testing.assert_allclose(actual["phase"], np.angle(values), atol=2e-7, rtol=0)
        np.testing.assert_array_equal(actual["frequencies"], frequencies)
        assert actual["epoch"] == pytest.approx(EPOCH)
