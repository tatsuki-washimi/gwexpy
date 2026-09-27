"""Public-reader regressions for malformed GL500 GBD headers."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _write_gl500_gbd(
    tmp_path,
    *,
    omit=(),
    model="GL500",
    firmware="Ver1.00",
    replacements=None,
    amp_lines=None,
    amplifier_lines=(),
    common_lines=(),
):
    """Write a small GL500-style file with two interleaved channels."""
    replacements = replacements or {}
    fields = [
        ("Model", f'Model = "{model}"'),
        ("Firmware", f'Firmware = "{firmware}"'),
        *[(f"Common{index}", line) for index, line in enumerate(common_lines)],
        ("Data", "$$Data"),
        ("Type", "Type = BigEndian, Short, Setup, Current"),
        ("Order", "Order = CH1, Alarm"),
        ("Sample", "Sample = 100ms"),
        ("Counts", "Counts = 3"),
        ("Time", "$$Time"),
        ("Start", "Start = 2024-01-02,03:04:05"),
        ("Stop", "Stop = 2024-01-02,03:04:06"),
    ]
    if "Amp" not in omit:
        fields.append(("Amp", "$Amp"))
        fields.extend(
            amp_lines
            if amp_lines is not None
            else [
                ("CH1", "CH1 = M, DC, 10V, Off, TC_K, +0"),
                ("CH2", "CH2 = M, DC, 10V, Off, TC_K, +0"),
            ]
        )
    if amplifier_lines:
        fields.append(("Amplifier", "$Amplifier"))
        fields.extend(amplifier_lines)
    fields.append(("EndHeader", "$EndHeader"))
    lines = [
        line if key not in replacements else replacements[key]
        for key, line in fields
        if key not in omit
    ]
    # Keep the normal GL500 data section marker when testing the $Amp section.
    lines.insert(0, "$Common")
    lines.insert(0, "HeaderSiz = 2048")
    header = ("\r\n".join(lines) + "\r\n").encode("ascii")
    assert len(header) <= 2048
    rows = ((2000, 0), (-4000, 2), (10000, 8))
    payload = b"".join(struct.pack(">hh", *row) for row in rows)
    path = tmp_path / "gl500.gbd"
    path.write_bytes(header.ljust(2048, b" ") + payload)
    return path


@pytest.mark.parametrize(
    ("missing", "replacements", "expected_message"),
    [
        ({"Sample"}, None, "Sample"),
        (set(), {"Sample": "Sample = 0"}, "Sample"),
        ({"Order"}, None, "Order"),
        (set(), {"Order": "Order = CH1,,Alarm"}, "Order"),
        ({"Counts"}, None, "Counts"),
        (set(), {"Counts": "Counts = not-a-count"}, "Counts"),
        ({"Amp"}, None, "Amp"),
        (set(), {"CH1": "CH1 = M, DC, unknown"}, "Amp"),
        (set(), {"CH1": "CH1 = M, DC, unknown10V"}, "Amp"),
        (set(), {"CH1": "CH1 = M, DC, 1e309V"}, "Amp"),
        (set(), {"CH1": "CH1 = M, DC, 1 0V"}, "Amp"),
        (set(), {"CH1": "CH1 = M, DC, , 10V"}, "Amp"),
    ],
)
@pytest.mark.parametrize(
    ("reader", "channels"),
    [
        (TimeSeries.read, ["CH1"]),
        (TimeSeriesDict.read, None),
        (TimeSeriesMatrix.read, None),
    ],
)
def test_gl500_public_readers_reject_malformed_required_header_metadata(
    tmp_path, missing, replacements, expected_message, reader, channels
):
    path = _write_gl500_gbd(
        tmp_path,
        omit=missing,
        replacements=replacements,
    )

    kwargs = {"format": "gbd", "timezone": "UTC"}
    if channels is not None:
        kwargs["channels"] = channels
    with pytest.raises(ValueError, match=expected_message):
        reader(path, **kwargs)


def test_valid_gl500_public_readers_keep_timing_scaling_and_digital_values(tmp_path):
    path = _write_gl500_gbd(tmp_path)

    tsd = TimeSeriesDict.read(path, format="gbd", timezone="UTC")
    np.testing.assert_allclose(tsd["CH1"].value, [1.0, -2.0, 5.0])
    np.testing.assert_array_equal(tsd["Alarm"].value, [0.0, 1.0, 1.0])
    assert np.isclose(tsd["CH1"].dt.value, 0.1)

    ts = TimeSeries.read(path, format="gbd", timezone="UTC", channels=["CH1"])
    np.testing.assert_allclose(ts.value, [1.0, -2.0, 5.0])

    matrix = TimeSeriesMatrix.read(path, format="gbd", timezone="UTC")
    assert matrix.shape == (2, 1, 3)
    assert np.isclose(matrix.dt.value, 0.1)


def test_gl500_public_reader_uses_sample_from_data_section(tmp_path):
    path = _write_gl500_gbd(tmp_path, common_lines=["Sample = 1s"])

    ts = TimeSeries.read(path, format="gbd", timezone="UTC", channels=["CH1"])

    assert np.isclose(ts.dt.value, 0.1)


def test_gl500_public_reader_uses_order_from_data_section(tmp_path):
    path = _write_gl500_gbd(tmp_path, common_lines=["Order = CH2, Alarm"])

    tsd = TimeSeriesDict.read(path, format="gbd", timezone="UTC")

    assert set(tsd) == {"Alarm", "CH1"}
    np.testing.assert_allclose(tsd["CH1"].value, [1.0, -2.0, 5.0])


def test_gl500_public_reader_uses_counts_from_data_section(tmp_path):
    path = _write_gl500_gbd(tmp_path, common_lines=["Counts = 1"])

    tsd = TimeSeriesDict.read(path, format="gbd", timezone="UTC")

    assert len(tsd["CH1"]) == 3
    np.testing.assert_allclose(tsd["CH1"].value, [1.0, -2.0, 5.0])


def test_gl500_empty_amp_section_does_not_use_later_amplifier_range(tmp_path):
    path = _write_gl500_gbd(
        tmp_path,
        amp_lines=[],
        amplifier_lines=[("AmplifierCH1", "CH1 = M, DC, 10V, Off, TC_K, +0")],
    )

    with pytest.raises(ValueError, match="Amp"):
        TimeSeries.read(path, format="gbd", timezone="UTC", channels=["CH1"])


@pytest.mark.parametrize(
    ("model", "firmware"),
    [("GL400", "Ver1.00"), ("GL500", "Ver1.22")],
)
def test_gl500_header_validation_does_not_extend_outside_model_firmware_scope(
    tmp_path, model, firmware
):
    path = _write_gl500_gbd(tmp_path, omit={"Sample"}, model=model, firmware=firmware)

    ts = TimeSeries.read(path, format="gbd", timezone="UTC", channels=["CH1"])
    assert np.isclose(ts.dt.value, 1.0)
