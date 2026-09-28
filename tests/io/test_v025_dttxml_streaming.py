"""Native PSD selection and DTTXML compatibility regressions for #589."""

from __future__ import annotations

import hashlib
import warnings
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

import gwexpy.io.dttxml_common as dttxml_common
from benchmarks.io.f3_dttxml_fixtures import (
    SELECTED_CHANNEL,
    DecodedPayloadSpy,
    write_fixtures,
)
from gwexpy.frequencyseries import FrequencySeriesDict


def test_native_selected_psd_does_not_decode_unselected_streams(tmp_path):
    """Public selection must reach the parser before base64 decoding."""
    fixture = write_fixtures(tmp_path, unselected_results=3, points=8)
    case = fixture["cases"]["many_valid"]

    with DecodedPayloadSpy(case["stream_roles"]) as decoded:
        result = FrequencySeriesDict.read(
            case["path"], format="dttxml", **case["call_kwargs"]
        )

    assert list(result) == [SELECTED_CHANNEL]
    assert decoded.report()["selected_decoded_bytes"] == 8 * 4
    assert decoded.report()["unselected_decoded_bytes"] == 0
    assert decoded.report()["decode_calls"].get("unselected", 0) == 0


def test_native_selected_psd_does_not_enter_unselected_decoder(tmp_path, monkeypatch):
    """The base64 byte counter cannot hide a replacement decoder path."""
    fixture = write_fixtures(tmp_path, unselected_results=3, points=8)
    case = fixture["cases"]["many_valid"]
    original = dttxml_common._decode_dtt_stream
    calls: list[str] = []

    def counted(stream_text, *args, **kwargs):
        digest = hashlib.sha256(
            "".join(stream_text.split()).encode("ascii")
        ).hexdigest()
        calls.append(case["stream_roles"].get(digest, "unknown"))
        return original(stream_text, *args, **kwargs)

    monkeypatch.setattr(dttxml_common, "_decode_dtt_stream", counted)
    result = FrequencySeriesDict.read(
        case["path"], format="dttxml", **case["call_kwargs"]
    )

    assert list(result) == [SELECTED_CHANNEL]
    assert calls == ["selected"]


@pytest.mark.parametrize("case_name", ["many_valid", "many_valid_gzip"])
def test_native_selected_psd_preserves_values_axis_and_epoch(tmp_path, case_name):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    case = fixture["cases"][case_name]

    result = FrequencySeriesDict.read(
        case["path"], format="dttxml", **case["call_kwargs"]
    )

    assert list(result) == [SELECTED_CHANNEL]
    selected = result[SELECTED_CHANNEL]
    np.testing.assert_array_equal(selected.value, fixture["selected_values"])
    np.testing.assert_array_equal(
        selected.frequencies.value,
        fixture["frequency_axis"]["f0"]
        + np.arange(8) * fixture["frequency_axis"]["df"],
    )
    assert selected.epoch.value == fixture["epoch"]
    assert str(selected.unit) == ""


@pytest.mark.parametrize(
    "case_name",
    [
        "late_xml_syntax_fault",
        "late_xml_selected_payload_fault",
        "late_xml_unselected_payload_fault",
    ],
)
def test_xml_syntax_error_precedes_any_payload_decode(tmp_path, case_name):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    case = fixture["cases"][case_name]

    with DecodedPayloadSpy(case["stream_roles"]) as decoded:
        with pytest.warns(UserWarning, match="Failed to parse DTT XML:") as caught:
            result = FrequencySeriesDict.read(
                case["path"], format="dttxml", **case["call_kwargs"]
            )

    assert list(result) == []
    assert len(caught) == 1
    assert decoded.report()["total_decoded_bytes"] == 0
    assert decoded.report()["decode_calls"] == {}


def test_unselected_metadata_warning_is_preserved(tmp_path):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    case = fixture["cases"]["unselected_metadata_fault"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = FrequencySeriesDict.read(
            case["path"], format="dttxml", **case["call_kwargs"]
        )

    assert list(result) == [SELECTED_CHANNEL]
    assert [(warning.category, str(warning.message)) for warning in caught] == [
        (
            UserWarning,
            "Invalid frequency metadata for Result[1]: "
            "invalid literal for int() with base 10: 'bad-N'",
        )
    ]


def test_selected_payload_warning_is_preserved(tmp_path):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    case = fixture["cases"]["selected_payload_fault"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = FrequencySeriesDict.read(
            case["path"], format="dttxml", **case["call_kwargs"]
        )

    assert list(result) == []
    assert [(warning.category, str(warning.message)) for warning in caught] == [
        (
            UserWarning,
            "Failed to decode stream for Result[0]: Invalid base64-encoded "
            "string: number of data characters (1) cannot be 1 more than "
            "a multiple of 4",
        )
    ]


def test_empty_selection_and_namespace_keep_public_empty_result(tmp_path):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    for case_name in ("many_valid_empty_channels", "namespace_qualified_control"):
        case = fixture["cases"][case_name]
        result = FrequencySeriesDict.read(
            case["path"], format="dttxml", **case["call_kwargs"]
        )
        assert list(result) == []


def test_non_path_source_uses_existing_native_decode_route(tmp_path):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    case = fixture["cases"]["many_valid"]
    source = BytesIO(Path(case["path"]).read_bytes())

    with DecodedPayloadSpy(case["stream_roles"]) as decoded:
        result = FrequencySeriesDict.read(
            source, format="dttxml", **case["call_kwargs"]
        )

    assert list(result) == [SELECTED_CHANNEL]
    assert decoded.report()["unselected_decoded_bytes"] == 2 * 8 * 4


def test_no_selection_keeps_source_order_and_full_decode(tmp_path):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    case = fixture["cases"]["many_valid_all_channels"]

    with DecodedPayloadSpy(case["stream_roles"]) as decoded:
        result = FrequencySeriesDict.read(
            case["path"], format="dttxml", **case["call_kwargs"]
        )

    assert list(result) == [SELECTED_CHANNEL, *fixture["unselected_channels"]]
    assert decoded.report()["unselected_decoded_bytes"] == 2 * 8 * 4


def test_generator_selector_keeps_existing_consumption_behavior(tmp_path):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    case = fixture["cases"]["many_valid"]
    channels = iter([SELECTED_CHANNEL])

    with DecodedPayloadSpy(case["stream_roles"]) as decoded:
        result = FrequencySeriesDict.read(
            case["path"],
            format="dttxml",
            products="PSD",
            native=True,
            channels=channels,
        )

    assert list(result) == []
    assert decoded.report()["unselected_decoded_bytes"] == 2 * 8 * 4


def test_reference_channel_identity_is_used_for_selection(tmp_path):
    fixture = write_fixtures(tmp_path, unselected_results=2, points=8)
    case = fixture["cases"]["many_valid"]
    tree = ET.parse(case["path"])
    selected_result = tree.getroot().find("LIGO_LW")
    assert selected_result is not None
    selected_result.set("Name", "Reference[4]")
    path = tmp_path / "reference.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)

    selected_name = f"{SELECTED_CHANNEL}(REF4)"
    result = FrequencySeriesDict.read(
        path, format="dttxml", products="PSD", channels=[selected_name], native=True
    )

    assert list(result) == [selected_name]
    np.testing.assert_array_equal(
        result[selected_name].value, fixture["selected_values"]
    )


def test_unselected_nonfinite_df_keeps_old_runtime_warning(tmp_path):
    fixture = write_fixtures(tmp_path, unselected_results=1, points=8)
    case = fixture["cases"]["many_valid"]
    tree = ET.parse(case["path"])
    unselected = tree.getroot().findall("LIGO_LW")[1]
    df = unselected.find("Param[@Name='df']")
    assert df is not None
    df.text = "inf"
    path = tmp_path / "unselected-nonfinite-df.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)

    with pytest.warns(RuntimeWarning, match="invalid value encountered in multiply"):
        result = FrequencySeriesDict.read(
            path,
            format="dttxml",
            products="PSD",
            native=True,
            channels=[SELECTED_CHANNEL],
        )

    assert list(result) == [SELECTED_CHANNEL]
