"""Regression coverage for DiagGUI TimeSeriesMatrix identification (#734)."""

from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

CHANNELS = ("K1:AUDIT-MATRIX-A", "K1:AUDIT-MATRIX-B")
VALUES = np.array([[1.5, -2.0, 0.25, 8.0], [3.0, 4.5, -1.0, 0.0]], dtype="<f4")
DT = 0.125
EPOCH = 1234567890.25


def _xml_bytes() -> bytes:
    """Build the characterized Type=TimeSeries/Subtype=0 float32 layout."""
    import xml.etree.ElementTree as ET

    root = ET.Element("LIGO_LW")
    for index, (channel, data) in enumerate(zip(CHANNELS, VALUES, strict=True)):
        result = ET.SubElement(
            root,
            "LIGO_LW",
            {"Name": f"Result[{index}]", "Type": "TimeSeries"},
        )
        ET.SubElement(result, "Time", {"Name": "t0", "Type": "GPS"}).text = str(EPOCH)
        for name, value in {
            "Subtype": "0",
            "N": str(data.size),
            "dt": str(DT),
            "Channel": channel,
        }.items():
            ET.SubElement(
                result, "Param", {"Name": name, "Type": "string"}
            ).text = value
        array = ET.SubElement(result, "Array", {"Type": "float"})
        ET.SubElement(array, "Dim").text = str(data.size)
        encoded = base64.b64encode(data.tobytes()).decode("ascii")
        ET.SubElement(
            array, "Stream", {"Encoding": "LittleEndian,base64"}
        ).text = encoded

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


@pytest.fixture(params=["xml", "xml.gz"])
def ts_matrix_xml(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    suffix = request.param
    path = tmp_path / f"audit-matrix.{suffix}"
    content = _xml_bytes()
    if suffix.endswith(".gz"):
        with gzip.open(path, "wb") as stream:
            stream.write(content)
    else:
        path.write_bytes(content)
    return path


@pytest.mark.parametrize("fmt", ["xml.diaggui", None], ids=["explicit", "auto"])
def test_matrix_read_in_real_base_only_process(
    ts_matrix_xml: Path, fmt: str | None
) -> None:
    """Identify and read the matrix in a real interpreter without dttxml."""
    python = os.environ.get("GWEXPY_NO_DTTXML_PYTHON")
    if not python and importlib.util.find_spec("dttxml") is None:
        import sys

        python = sys.executable
    if not python:
        pytest.skip("base-only route needs a separate interpreter without dttxml")

    project_root = Path(__file__).resolve().parents[2]
    code = """
import importlib.util, json, sys
from gwexpy.timeseries import TimeSeriesMatrix
assert importlib.util.find_spec('dttxml') is None
path, fmt = sys.argv[1], sys.argv[2]
fmt = None if fmt == 'None' else fmt
matrix = TimeSeriesMatrix.read(path, format=fmt, products='TS')
print(json.dumps({
    'type': type(matrix).__name__, 'data': matrix.value.tolist(),
    'times': matrix.times.value.tolist(),
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [python, "-c", code, str(ts_matrix_xml), str(fmt)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = json.loads(completed.stdout)
    assert result["type"] == "TimeSeriesMatrix"
    matrix_values = np.asarray(result["data"])
    np.testing.assert_allclose(np.squeeze(matrix_values, axis=1), VALUES)
    expected_times = EPOCH + np.arange(VALUES.shape[1]) * DT
    np.testing.assert_allclose(result["times"], expected_times, rtol=0, atol=1e-6)
