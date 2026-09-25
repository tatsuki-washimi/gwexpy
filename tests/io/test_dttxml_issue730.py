"""Regression coverage for base-install DiagGUI TimeSeries reads (#730)."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gwexpy.io.dttxml_common import load_dttxml_native, load_dttxml_products
from gwexpy.timeseries import TimeSeriesDict

CHANNEL = "K1:AUDIT-TS"
VALUES = np.array([1.5, -2.0, 0.25, 8.0], dtype="<f4")
DT = 0.125
EPOCH = 1234567890.25


def _stream(data: np.ndarray) -> str:
    return base64.b64encode(data.tobytes()).decode("ascii")


@pytest.fixture
def ts_xml(tmp_path: Path) -> Path:
    """Build the audited Type=TimeSeries/Subtype=0 float32 serialization."""
    import xml.etree.ElementTree as ET

    root = ET.Element("LIGO_LW")
    result = ET.SubElement(root, "LIGO_LW", {"Name": "Result[0]", "Type": "TimeSeries"})
    ET.SubElement(result, "Time", {"Name": "t0", "Type": "GPS"}).text = str(EPOCH)
    for name, value in {
        "Subtype": "0",
        "N": str(VALUES.size),
        "dt": str(DT),
        "Channel": CHANNEL,
    }.items():
        ET.SubElement(result, "Param", {"Name": name, "Type": "string"}).text = value
    array = ET.SubElement(result, "Array", {"Type": "float"})
    ET.SubElement(array, "Dim").text = str(VALUES.size)
    ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = _stream(
        VALUES
    )

    path = tmp_path / "auditts.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _append_spectrum(path: Path) -> None:
    """Add an established Spectrum/1 PSD block beside the TS block."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    result = ET.SubElement(
        tree.getroot(), "LIGO_LW", {"Name": "Result[1]", "Type": "Spectrum"}
    )
    ET.SubElement(result, "Time", {"Name": "t0", "Type": "GPS"}).text = str(EPOCH)
    for name, value in {
        "Subtype": "1",
        "M": "1",
        "N": "3",
        "f0": "17.5",
        "df": "2.5",
        "ChannelA": "K1:AUDIT-PSD",
        "BUnit": "strain^2/Hz",
    }.items():
        ET.SubElement(result, "Param", {"Name": name, "Type": "string"}).text = value
    array = ET.SubElement(result, "Array", {"Type": "float"})
    ET.SubElement(array, "Dim").text = "3"
    psd = np.array([0.5, 1.25, 4.0], dtype="<f4")
    ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = _stream(
        psd
    )
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _append_transfer_function(path: Path) -> None:
    """Add a supported TransferFunction/0 block beside the TS block."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    result = ET.SubElement(
        tree.getroot(), "LIGO_LW", {"Name": "Result[2]", "Type": "TransferFunction"}
    )
    ET.SubElement(result, "Time", {"Name": "t0", "Type": "GPS"}).text = str(EPOCH)
    for name, value in {
        "Subtype": "0",
        "M": "1",
        "N": "3",
        "f0": "17.5",
        "df": "2.5",
        "ChannelA": "K1:AUDIT-INPUT",
        "ChannelB[0]": "K1:AUDIT-OUTPUT",
    }.items():
        ET.SubElement(result, "Param", {"Name": name, "Type": "string"}).text = value
    array = ET.SubElement(result, "Array", {"Type": "floatComplex"})
    ET.SubElement(array, "Dim").text = "1"
    ET.SubElement(array, "Dim").text = "3"
    transfer = np.array([1 + 2j, -0.5 + 0.25j, 3 - 4j], dtype="<c8")
    ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = _stream(
        transfer
    )
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _assert_public_read(path: Path, fmt: str | None) -> None:
    series_dict = TimeSeriesDict.read(path, format=fmt, products="TS")
    assert isinstance(series_dict, TimeSeriesDict)
    assert list(series_dict) == [CHANNEL]
    series = series_dict[CHANNEL]
    np.testing.assert_array_equal(series.value, VALUES)
    assert series.dtype == np.dtype("float32")
    assert np.isclose(float(series.dt.value), DT)
    assert float(series.t0.value) == pytest.approx(EPOCH, rel=0, abs=1e-6)
    assert str(series.channel) == CHANNEL


def test_native_parser_reads_source_grounded_timeseries(ts_xml: Path) -> None:
    products = load_dttxml_native(str(ts_xml))
    assert list(products["TS"]) == [CHANNEL]
    info = products["TS"][CHANNEL]
    np.testing.assert_array_equal(info["data"], VALUES)
    assert info["data"].dtype == np.dtype("float32")
    assert info["dt"] == DT
    assert info["epoch"] == EPOCH


def test_native_product_loader_reads_timeseries(ts_xml: Path) -> None:
    products = load_dttxml_products(str(ts_xml), native=True)
    assert list(products["TS"]) == [CHANNEL]
    info = products["TS"][CHANNEL]
    np.testing.assert_array_equal(info["data"], VALUES)
    assert info["data"].dtype == np.dtype("float32")
    assert info["dt"] == DT
    assert info["epoch"] == EPOCH


def test_native_timeseries_rejects_invalid_base64(ts_xml: Path) -> None:
    import xml.etree.ElementTree as ET

    tree = ET.parse(ts_xml)
    tree.find(".//Array/Stream").text = "AADAPwAA!MAAAIA+AAAAQQA=="
    tree.write(ts_xml, encoding="utf-8", xml_declaration=True)

    with pytest.warns(UserWarning, match="Failed to decode time series"):
        products = load_dttxml_native(str(ts_xml))
    assert "TS" not in products


def test_native_timeseries_accepts_base64_line_breaks(ts_xml: Path) -> None:
    import xml.etree.ElementTree as ET

    tree = ET.parse(ts_xml)
    stream = tree.find(".//Array/Stream")
    encoded = stream.text
    stream.text = f"{encoded[:8]}\n{encoded[8:]}"
    tree.write(ts_xml, encoding="utf-8", xml_declaration=True)

    info = load_dttxml_native(str(ts_xml))["TS"][CHANNEL]
    np.testing.assert_array_equal(info["data"], VALUES)


def test_native_timeseries_duplicate_channel_fails_closed(ts_xml: Path) -> None:
    import copy
    import xml.etree.ElementTree as ET

    tree = ET.parse(ts_xml)
    duplicate = copy.deepcopy(tree.find(".//LIGO_LW[@Type='TimeSeries']"))
    duplicate.set("Name", "Result[1]")
    tree.getroot().append(duplicate)
    tree.write(ts_xml, encoding="utf-8", xml_declaration=True)

    with pytest.raises(ValueError, match="Duplicate TimeSeries channel.*K1:AUDIT-TS"):
        load_dttxml_native(str(ts_xml))


def test_native_timeseries_keeps_existing_frequency_products(ts_xml: Path) -> None:
    _append_spectrum(ts_xml)
    _append_transfer_function(ts_xml)
    products = load_dttxml_native(str(ts_xml))
    assert list(products["TS"]) == [CHANNEL]
    np.testing.assert_array_equal(products["TS"][CHANNEL]["data"], VALUES)
    for product in ("PSD", "ASD"):
        assert list(products[product]) == ["K1:AUDIT-PSD"]
        np.testing.assert_array_equal(
            products[product]["K1:AUDIT-PSD"]["data"], [0.5, 1.25, 4.0]
        )
        np.testing.assert_array_equal(
            products[product]["K1:AUDIT-PSD"]["frequencies"], [17.5, 20.0, 22.5]
        )
    tf_key = ("K1:AUDIT-OUTPUT", "K1:AUDIT-INPUT")
    np.testing.assert_array_equal(
        products["TF"][tf_key]["data"], [1 + 2j, -0.5 + 0.25j, 3 - 4j]
    )
    assert products["TF"][tf_key]["data"].dtype == np.dtype("complex64")
    np.testing.assert_array_equal(
        products["TF"][tf_key]["frequencies"], [17.5, 20.0, 22.5]
    )


def test_installed_dttxml_route_reads_timeseries(ts_xml: Path) -> None:
    if importlib.util.find_spec("dttxml") is None:
        pytest.skip("installed dttxml route is covered in the dttxml-enabled env")
    import dttxml

    assert getattr(dttxml, "__version__", None) == "1.1.8"
    _assert_public_read(ts_xml, "xml.diaggui")


def test_installed_dttxml_route_preserves_mixed_frequency_products(
    ts_xml: Path,
) -> None:
    if importlib.util.find_spec("dttxml") is None:
        pytest.skip("installed dttxml route is covered in the dttxml-enabled env")
    _append_spectrum(ts_xml)
    _append_transfer_function(ts_xml)
    products = load_dttxml_products(str(ts_xml), native=False)
    assert list(products["TS"]) == [CHANNEL]
    np.testing.assert_array_equal(products["TS"][CHANNEL]["data"], VALUES)
    for product in ("PSD", "ASD"):
        series = products[product]["K1:AUDIT-PSD"]
        np.testing.assert_array_equal(series.value, [0.5, 1.25, 4.0])
        assert series.dtype == np.dtype("float32")
        np.testing.assert_array_equal(series.frequencies.value, [17.5, 20.0, 22.5])
        assert float(series.f0.value) == pytest.approx(17.5)
        assert float(series.df.value) == pytest.approx(2.5)
        assert float(series.epoch.value) == pytest.approx(EPOCH, rel=0, abs=1e-6)
    tf_key = ("K1:AUDIT-OUTPUT", "K1:AUDIT-INPUT")
    tf_series = products["TF"][tf_key]
    np.testing.assert_array_equal(tf_series.value, [1 + 2j, -0.5 + 0.25j, 3 - 4j])
    assert tf_series.dtype == np.dtype("complex64")
    np.testing.assert_array_equal(tf_series.frequencies.value, [17.5, 20.0, 22.5])
    assert float(tf_series.f0.value) == pytest.approx(17.5)
    assert float(tf_series.df.value) == pytest.approx(2.5)
    assert float(tf_series.epoch.value) == pytest.approx(EPOCH, rel=0, abs=1e-6)


@pytest.mark.parametrize("fmt", ["xml.diaggui", None], ids=["explicit", "auto"])
def test_public_read_in_real_no_dttxml_process(ts_xml: Path, fmt: str | None) -> None:
    """Exercise the fallback in a separate interpreter without dttxml installed."""
    if importlib.util.find_spec("dttxml") is None:
        # In the base-install CI job, use a child with the same real interpreter.
        python = sys.executable
    else:
        # Developers with dttxml installed can provide a real base-only Python.
        python = os.environ.get("GWEXPY_NO_DTTXML_PYTHON")
    if not python:
        pytest.skip("fallback route needs a separate interpreter without dttxml")

    project_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.util, json, sys
from gwexpy.timeseries import TimeSeriesDict
assert importlib.util.find_spec('dttxml') is None
path, fmt = sys.argv[1], sys.argv[2]
fmt = None if fmt == 'None' else fmt
result = TimeSeriesDict.read(path, format=fmt, products='TS')
channel = 'K1:AUDIT-TS'
series = result[channel]
print(json.dumps({
    'type': type(result).__name__, 'channels': list(result),
    'data': series.value.tolist(), 'dtype': str(series.dtype),
    'dt': float(series.dt.value), 'epoch': float(series.t0.value),
    'channel': str(series.channel),
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [python, "-c", code, str(ts_xml), str(fmt)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = json.loads(completed.stdout)
    assert result["type"] == "TimeSeriesDict"
    assert result["channels"] == [CHANNEL]
    np.testing.assert_array_equal(result["data"], VALUES)
    assert result["dtype"] == "float32"
    assert result["dt"] == DT
    assert result["epoch"] == pytest.approx(EPOCH, rel=0, abs=1e-6)
    assert result["channel"] == CHANNEL
