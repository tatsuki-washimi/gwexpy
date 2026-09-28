"""Deterministic, package-independent DTTXML fixtures for the B-F3 lane.

This module creates files and independent byte-level oracles. It imports no
gwexpy code until a caller explicitly enters one of the optional spies.
The external ``dttxml`` parser has a separate contract and is not measured by
the native selected-decode gate.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import struct
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

SELECTED_CHANNEL = "K1:F3-SELECTED"
EPOCH = 1_234_567_890.25
F0 = 17.5
DF = 2.5


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stream_hash(stream: str | bytes) -> str:
    """Hash the base64 input after XML whitespace normalization."""
    if isinstance(stream, bytes):
        stream = stream.decode("ascii")
    return _sha256("".join(stream.split()).encode("ascii"))


def _spectrum(
    parent: ET.Element,
    index: int,
    channel: str,
    raw: bytes,
    points: int,
    *,
    malformed_stream: bool = False,
    malformed_metadata: bool = False,
    subtype: int = 1,
    array_type: str = "float",
    row_dimension: int = 1,
) -> str:
    result = ET.SubElement(
        parent, "LIGO_LW", {"Name": f"Result[{index}]", "Type": "Spectrum"}
    )
    fields = {
        "Subtype": str(subtype),
        "M": "1",
        "N": "bad-N" if malformed_metadata else str(points),
        "f0": str(F0),
        "df": str(DF),
        "ChannelA": channel,
    }
    for name, value in fields.items():
        ET.SubElement(result, "Param", {"Name": name, "Type": "string"}).text = value
    ET.SubElement(result, "Time", {"Name": "t0"}).text = str(EPOCH)
    array = ET.SubElement(result, "Array", {"Type": array_type})
    ET.SubElement(array, "Dim").text = str(row_dimension)
    ET.SubElement(array, "Dim").text = str(points)
    encoded = "A===" if malformed_stream else base64.b64encode(raw).decode("ascii")
    ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = encoded
    return _stream_hash(encoded)


def _psd_raw(index: int, points: int) -> bytes:
    # Exact binary32 values, so the oracle does not depend on NumPy rounding.
    return struct.pack(
        f"<{points}f", *(float(index + 1) + (j % 8) / 8 for j in range(points))
    )


def _timeseries(
    parent: ET.Element,
    index: int,
    channel: str,
    samples: tuple[float, ...],
    *,
    declared_points: int | None = None,
) -> str:
    result = ET.SubElement(
        parent, "LIGO_LW", {"Name": f"Result[{index}]", "Type": "TimeSeries"}
    )
    for name, value in {
        "Subtype": "0",
        "N": str(declared_points if declared_points is not None else len(samples)),
        "dt": "0.125",
        "Channel": channel,
    }.items():
        ET.SubElement(result, "Param", {"Name": name}).text = value
    ET.SubElement(result, "Time", {"Name": "t0"}).text = str(EPOCH)
    array = ET.SubElement(result, "Array", {"Type": "float"})
    ET.SubElement(array, "Dim").text = str(
        declared_points if declared_points is not None else len(samples)
    )
    encoded = base64.b64encode(struct.pack(f"<{len(samples)}f", *samples)).decode(
        "ascii"
    )
    ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = encoded
    return _stream_hash(encoded)


def _stf(
    parent: ET.Element,
    index: int,
    channel_a: str,
    channel_b: str,
    samples: tuple[complex, ...],
    *,
    declared_points: int = 3,
) -> str:
    result = ET.SubElement(
        parent, "LIGO_LW", {"Name": f"Result[{index}]", "Type": "TransferFunction"}
    )
    for name, value in {
        "Subtype": "1",
        "M": "1",
        "N": str(declared_points),
        "f0": str(F0),
        "df": str(DF),
        "ChannelA": channel_a,
        "ChannelB[0]": channel_b,
    }.items():
        ET.SubElement(result, "Param", {"Name": name}).text = value
    ET.SubElement(result, "Time", {"Name": "t0"}).text = str(EPOCH)
    array = ET.SubElement(result, "Array", {"Type": "floatComplex"})
    ET.SubElement(array, "Dim").text = "1"
    ET.SubElement(array, "Dim").text = str(declared_points)
    raw = struct.pack(
        f"<{2 * len(samples)}f",
        *(part for sample in samples for part in (sample.real, sample.imag)),
    )
    encoded = base64.b64encode(raw).decode("ascii")
    ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = encoded
    return _stream_hash(encoded)


def _tf6(
    parent: ET.Element,
    *,
    result_index: int = 6,
    channel_a: str = "K1:F3-TF-INPUT",
    channel_b: str = "K1:F3-TF-OUTPUT",
    malformed_stream: bool = False,
    nonfinite_frequency: bool = False,
    sample_offset: float = 0.0,
) -> tuple[str, bytes]:
    frequencies = (17.5, 20.0, float("nan") if nonfinite_frequency else 22.5)
    samples = tuple(
        sample + sample_offset for sample in (1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 4.0j)
    )
    raw = struct.pack("<3d", *frequencies) + struct.pack(
        "<6f", *(part for sample in samples for part in (sample.real, sample.imag))
    )
    result = ET.SubElement(
        parent,
        "LIGO_LW",
        {"Name": f"Result[{result_index}]", "Type": "TransferFunction"},
    )
    fields = {
        "Subtype": "6",
        "M": "1",
        "N": "3",
        "f0": "0",
        "df": "0",
        "ChannelA": channel_a,
        "ChannelB[0]": channel_b,
    }
    for name, value in fields.items():
        ET.SubElement(result, "Param", {"Name": name}).text = value
    ET.SubElement(result, "Time", {"Name": "t0"}).text = str(EPOCH)
    array = ET.SubElement(result, "Array", {"Type": "floatComplex"})
    ET.SubElement(array, "Dim").text = "2"
    ET.SubElement(array, "Dim").text = "3"
    encoded = "A===" if malformed_stream else base64.b64encode(raw).decode("ascii")
    ET.SubElement(array, "Stream", {"Encoding": "LittleEndian,base64"}).text = encoded
    return _stream_hash(encoded), raw


def _xml(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _write(path: Path, content: bytes) -> dict[str, Any]:
    if path.suffix == ".gz":
        with (
            path.open("wb") as handle,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=handle, mtime=0
            ) as compressed,
        ):
            compressed.write(content)
    else:
        path.write_bytes(content)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path.read_bytes()),
        "bytes": path.stat().st_size,
    }


def write_fixtures(
    output_dir: str | Path, *, unselected_results: int = 64, points: int = 1024
) -> dict[str, Any]:
    """Write deterministic native fixtures and return a JSON-ready manifest.

    The many-result case contains one selected PSD and ``unselected_results``
    other PSD results. ``stream_roles`` maps normalized base64 input SHA256 to
    its role for :class:`DecodedPayloadSpy`. Defaults are for a smoke check;
    the harness must freeze a larger shape before its peak-RSS baseline.
    """
    if unselected_results < 1 or points < 3:
        raise ValueError("unselected_results must be >= 1 and points must be >= 3")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    root = ET.Element("LIGO_LW")
    selected_raw = _psd_raw(0, points)
    selected_hash = _spectrum(root, 0, SELECTED_CHANNEL, selected_raw, points)
    roles = {selected_hash: "selected"}
    unselected_channels = []
    for index in range(1, unselected_results + 1):
        channel = f"K1:F3-OTHER-{index:05d}"
        unselected_channels.append(channel)
        roles[_spectrum(root, index, channel, _psd_raw(index, points), points)] = (
            "unselected"
        )
    valid_xml = _xml(root)

    def scenario(
        filename: str,
        content: bytes,
        category: str,
        *,
        product: str = "PSD",
        selected: str | None = SELECTED_CHANNEL,
        role_map: Mapping[str, str] | None = None,
        native_parser_exception_applies: bool = False,
    ) -> dict[str, Any]:
        entry = _write(directory / filename, content)
        selector = {"channels": [selected] if selected is not None else None}
        entry.update(
            {
                "oracle_category": category,
                "call_kwargs": {"products": product, "native": True, **selector},
                "external_call_kwargs": {
                    "products": product,
                    "native": False,
                    **selector,
                },
                "stream_roles": dict(role_map or {}),
                "native_parser_exception_applies": native_parser_exception_applies,
            }
        )
        return entry

    cases: dict[str, dict[str, Any]] = {
        "many_valid": scenario(
            "many-valid.xml", valid_xml, "selected_valid", role_map=roles
        ),
        "many_valid_gzip": scenario(
            "many-valid.xml.gz", valid_xml, "selected_valid", role_map=roles
        ),
    }
    all_channels = dict(cases["many_valid"])
    all_channels["oracle_category"] = "all_channels_source_order"
    all_channels["call_kwargs"] = {
        "products": "PSD",
        "native": True,
        "channels": None,
    }
    all_channels["external_call_kwargs"] = {
        "products": "PSD",
        "native": False,
        "channels": None,
    }
    cases["many_valid_all_channels"] = all_channels
    empty_channels = dict(all_channels)
    empty_channels["oracle_category"] = "empty_channels_returns_empty_dict"
    empty_channels["call_kwargs"] = {
        "products": "PSD",
        "native": True,
        "channels": [],
    }
    empty_channels["external_call_kwargs"] = {
        "products": "PSD",
        "native": False,
        "channels": [],
    }
    cases["many_valid_empty_channels"] = empty_channels

    # Start from a fresh tree for each fault. The malformed unselected stream
    # remains fully outside the selected channel but inside the selected product.
    for case_name, fault_index, fault_stream, fault_metadata in (
        ("selected_payload_fault", 0, True, False),
        ("unselected_payload_fault", 1, True, False),
        ("selected_metadata_fault", 0, False, True),
        ("unselected_metadata_fault", 1, False, True),
    ):
        fault_root = ET.Element("LIGO_LW")
        fault_roles: dict[str, str] = {}
        for index in range(2):
            channel = SELECTED_CHANNEL if index == 0 else unselected_channels[0]
            stream_hash = _spectrum(
                fault_root,
                index,
                channel,
                _psd_raw(index, points),
                points,
                malformed_stream=index == fault_index and fault_stream,
                malformed_metadata=index == fault_index and fault_metadata,
            )
            fault_roles[stream_hash] = "selected" if index == 0 else "unselected"
        if fault_stream:
            category = (
                "selected_payload_warning"
                if fault_index == 0
                else "unselected_payload_warning_may_disappear"
            )
        else:
            category = (
                "selected_structural_metadata_warning"
                if fault_index == 0
                else "unselected_structural_metadata_warning"
            )
        cases[case_name] = scenario(
            f"{case_name.replace('_', '-')}.xml",
            _xml(fault_root),
            category,
            role_map=fault_roles,
            native_parser_exception_applies=case_name == "unselected_payload_fault",
        )

    cases["late_xml_syntax_fault"] = scenario(
        "late-xml-syntax-fault.xml",
        valid_xml + b"<broken>",
        "xml_parse_warning_empty",
        role_map=roles,
    )
    for name in ("selected_payload_fault", "unselected_payload_fault"):
        original = cases[name]
        cases[f"late_xml_{name}"] = scenario(
            f"late-xml-{name.replace('_', '-')}.xml",
            Path(original["path"]).read_bytes() + b"<broken>",
            "xml_parse_warning_supersedes_payload_fault",
            role_map=original["stream_roles"],
        )

    tf_root = ET.Element("LIGO_LW")
    tf_hash, tf_raw = _tf6(tf_root)
    cases["tf6_raw_layout"] = scenario(
        "tf6-raw-layout.xml",
        _xml(tf_root),
        "tf6_complex_phase_valid",
        product="TF",
        selected=None,
        role_map={tf_hash: "selected"},
    )
    matrix_selector = {
        "rows": ["K1:F3-TF-OUTPUT"],
        "cols": ["K1:F3-TF-INPUT"],
    }
    cases["tf6_raw_layout"]["call_kwargs"] = {
        "products": "TF",
        "native": True,
        **matrix_selector,
    }
    cases["tf6_raw_layout"]["external_call_kwargs"] = {
        "products": "TF",
        "native": False,
        **matrix_selector,
    }

    for case_name, malformed_index in (
        ("tf6_selected_payload_fault", 0),
        ("tf6_unselected_payload_fault", 1),
    ):
        fault_root = ET.Element("LIGO_LW")
        fault_roles = {}
        for index in range(2):
            stream_hash, _ = _tf6(
                fault_root,
                result_index=6 + index,
                channel_a=("K1:F3-TF-INPUT" if index == 0 else "K1:F3-TF-OTHER-INPUT"),
                channel_b=(
                    "K1:F3-TF-OUTPUT" if index == 0 else "K1:F3-TF-OTHER-OUTPUT"
                ),
                malformed_stream=index == malformed_index,
                sample_offset=float(index),
            )
            fault_roles[stream_hash] = "selected" if index == 0 else "unselected"
        entry = scenario(
            f"{case_name.replace('_', '-')}.xml",
            _xml(fault_root),
            (
                "selected_payload_decode_error"
                if malformed_index == 0
                else "unselected_payload_decode_error_may_disappear"
            ),
            product="TF",
            selected=None,
            role_map=fault_roles,
            native_parser_exception_applies=malformed_index == 1,
        )
        entry["call_kwargs"] = {"products": "TF", "native": True, **matrix_selector}
        entry["external_call_kwargs"] = {
            "products": "TF",
            "native": False,
            **matrix_selector,
        }
        cases[case_name] = entry

    identity_root = ET.Element("LIGO_LW")
    identity_roles = {}
    for index in range(2):
        stream_hash, _ = _tf6(
            identity_root,
            result_index=6,
            channel_a=("K1:F3-TF-INPUT" if index == 0 else "K1:F3-TF-OTHER-INPUT"),
            channel_b=("K1:F3-TF-OUTPUT" if index == 0 else "K1:F3-TF-OTHER-OUTPUT"),
            sample_offset=float(index),
        )
        identity_roles[stream_hash] = "selected" if index == 0 else "unselected"
    identity_case = scenario(
        "tf6-unselected-identity-fault.xml",
        _xml(identity_root),
        "unselected_structural_identity_error",
        product="TF",
        selected=None,
        role_map=identity_roles,
    )
    identity_case["call_kwargs"] = {"products": "TF", "native": True, **matrix_selector}
    identity_case["external_call_kwargs"] = {
        "products": "TF",
        "native": False,
        **matrix_selector,
    }
    cases["tf6_unselected_identity_fault"] = identity_case

    nonfinite_root = ET.Element("LIGO_LW")
    nonfinite_roles = {}
    for index in range(2):
        stream_hash, _ = _tf6(
            nonfinite_root,
            result_index=6 + index,
            channel_a="K1:F3-TF-INPUT" if index == 0 else "K1:F3-TF-OTHER-INPUT",
            channel_b="K1:F3-TF-OUTPUT" if index == 0 else "K1:F3-TF-OTHER-OUTPUT",
            nonfinite_frequency=index == 1,
            sample_offset=float(index),
        )
        nonfinite_roles[stream_hash] = "selected" if index == 0 else "unselected"
    nonfinite_case = scenario(
        "tf6-unselected-nonfinite-frequency-fault.xml",
        _xml(nonfinite_root),
        "unselected_payload_semantic_error_hold",
        product="TF",
        selected=None,
        role_map=nonfinite_roles,
    )
    nonfinite_case["call_kwargs"] = {
        "products": "TF",
        "native": True,
        **matrix_selector,
    }
    nonfinite_case["external_call_kwargs"] = {
        "products": "TF",
        "native": False,
        **matrix_selector,
    }
    cases["tf6_unselected_nonfinite_frequency_fault"] = nonfinite_case

    psd_short_root = ET.Element("LIGO_LW")
    psd_short_roles = {
        _spectrum(psd_short_root, 0, SELECTED_CHANNEL, _psd_raw(0, 3), 3): "selected",
        _spectrum(
            psd_short_root, 1, "K1:F3-OTHER-00001", _psd_raw(1, 2), 3
        ): "unselected",
    }
    cases["psd_unselected_short_payload_warning"] = scenario(
        "psd-unselected-short-payload-warning.xml",
        _xml(psd_short_root),
        "unselected_payload_warning_may_disappear",
        role_map=psd_short_roles,
        native_parser_exception_applies=True,
    )

    fft_short_root = ET.Element("LIGO_LW")
    fft_short_roles = {}
    for index, count in ((0, 3), (1, 2)):
        raw = struct.pack(
            f"<{2 * count}f",
            *(part for sample in range(count) for part in (float(sample), 0.5)),
        )
        stream_hash = _spectrum(
            fft_short_root,
            index,
            SELECTED_CHANNEL if index == 0 else "K1:F3-OTHER-00001",
            raw,
            3,
            subtype=0,
            array_type="floatComplex",
        )
        fft_short_roles[stream_hash] = "selected" if index == 0 else "unselected"
    cases["fft_unselected_short_payload_error"] = scenario(
        "fft-unselected-short-payload-error.xml",
        _xml(fft_short_root),
        "unselected_payload_semantic_error_hold",
        product="FFT",
        role_map=fft_short_roles,
    )
    fft_selected_short_root = ET.Element("LIGO_LW")
    fft_selected_short_roles = {}
    for index, count in ((0, 2), (1, 3)):
        raw = struct.pack(
            f"<{2 * count}f",
            *(part for sample in range(count) for part in (float(sample), 0.5)),
        )
        stream_hash = _spectrum(
            fft_selected_short_root,
            index,
            SELECTED_CHANNEL if index == 0 else "K1:F3-OTHER-00001",
            raw,
            3,
            subtype=0,
            array_type="floatComplex",
        )
        fft_selected_short_roles[stream_hash] = (
            "selected" if index == 0 else "unselected"
        )
    cases["fft_selected_short_payload_error"] = scenario(
        "fft-selected-short-payload-error.xml",
        _xml(fft_selected_short_root),
        "selected_payload_semantic_error_preserve",
        product="FFT",
        role_map=fft_selected_short_roles,
    )

    fft_axis_root = ET.Element("LIGO_LW")
    fft_axis_roles = {}
    for index in range(2):
        frequency_imaginary = 1.0 if index == 1 else 0.0
        words = (
            17.5,
            frequency_imaginary,
            20.0,
            0.0,
            22.5,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        )
        stream_hash = _spectrum(
            fft_axis_root,
            index,
            SELECTED_CHANNEL if index == 0 else "K1:F3-OTHER-00001",
            struct.pack("<12f", *words),
            3,
            subtype=4,
            array_type="floatComplex",
            row_dimension=2,
        )
        fft_axis_roles[stream_hash] = "selected" if index == 0 else "unselected"
    cases["fft_unselected_nonreal_axis_error"] = scenario(
        "fft-unselected-nonreal-axis-error.xml",
        _xml(fft_axis_root),
        "unselected_payload_semantic_error_hold",
        product="FFT",
        role_map=fft_axis_roles,
    )

    stf_short_root = ET.Element("LIGO_LW")
    stf_short_roles = {}
    for index, values in ((0, (1 + 2j, 3 + 4j, 5 + 6j)), (1, (7 + 8j, 9 + 10j))):
        stream_hash = _stf(
            stf_short_root,
            index,
            "K1:F3-TF-INPUT" if index == 0 else "K1:F3-TF-OTHER-INPUT",
            "K1:F3-TF-OUTPUT" if index == 0 else "K1:F3-TF-OTHER-OUTPUT",
            values,
        )
        stf_short_roles[stream_hash] = "selected" if index == 0 else "unselected"
    stf_case = scenario(
        "stf-unselected-short-payload-error.xml",
        _xml(stf_short_root),
        "unselected_payload_semantic_error_hold",
        product="STF",
        selected=None,
        role_map=stf_short_roles,
    )
    stf_case["call_kwargs"] = {
        "products": "STF",
        "native": True,
        **matrix_selector,
    }
    stf_case["external_call_kwargs"] = {
        "products": "STF",
        "native": False,
        **matrix_selector,
    }
    cases["stf_unselected_short_payload_error"] = stf_case
    stf_selected_short_root = ET.Element("LIGO_LW")
    stf_selected_short_roles = {}
    for index, values in (
        (0, (1 + 2j, 3 + 4j)),
        (1, (5 + 6j, 7 + 8j, 9 + 10j)),
    ):
        stream_hash = _stf(
            stf_selected_short_root,
            index,
            "K1:F3-TF-INPUT" if index == 0 else "K1:F3-TF-OTHER-INPUT",
            "K1:F3-TF-OUTPUT" if index == 0 else "K1:F3-TF-OTHER-OUTPUT",
            values,
        )
        stf_selected_short_roles[stream_hash] = (
            "selected" if index == 0 else "unselected"
        )
    stf_selected_case = scenario(
        "stf-selected-short-payload-error.xml",
        _xml(stf_selected_short_root),
        "selected_payload_semantic_error_preserve",
        product="STF",
        selected=None,
        role_map=stf_selected_short_roles,
    )
    stf_selected_case["call_kwargs"] = {
        "products": "STF",
        "native": True,
        **matrix_selector,
    }
    stf_selected_case["external_call_kwargs"] = {
        "products": "STF",
        "native": False,
        **matrix_selector,
    }
    cases["stf_selected_short_payload_error"] = stf_selected_case

    ts_valid_root = ET.Element("LIGO_LW")
    ts_selected_samples = (1.0, 1.125, 1.25)
    ts_valid_roles = {
        _timeseries(
            ts_valid_root, 0, SELECTED_CHANNEL, ts_selected_samples
        ): "selected",
        _timeseries(
            ts_valid_root, 1, "K1:F3-OTHER-00001", (2.0, 2.125, 2.25)
        ): "unselected",
    }
    cases["ts_two_valid"] = scenario(
        "ts-two-valid.xml",
        _xml(ts_valid_root),
        "ts_public_native_arg_ignored",
        product="TS",
        role_map=ts_valid_roles,
    )
    ts_short_root = ET.Element("LIGO_LW")
    ts_short_roles = {
        _timeseries(
            ts_short_root, 0, SELECTED_CHANNEL, ts_selected_samples
        ): "selected",
        _timeseries(
            ts_short_root,
            1,
            "K1:F3-OTHER-00001",
            (2.0, 2.125),
            declared_points=3,
        ): "unselected",
    }
    cases["ts_unselected_short_payload_warning"] = scenario(
        "ts-unselected-short-payload-warning.xml",
        _xml(ts_short_root),
        "unselected_payload_warning_may_disappear_in_fallback_only",
        product="TS",
        role_map=ts_short_roles,
        native_parser_exception_applies=True,
    )
    ts_selected_short_root = ET.Element("LIGO_LW")
    ts_selected_short_roles = {
        _timeseries(
            ts_selected_short_root,
            0,
            SELECTED_CHANNEL,
            (1.0, 1.125),
            declared_points=3,
        ): "selected",
        _timeseries(
            ts_selected_short_root,
            1,
            "K1:F3-OTHER-00001",
            (2.0, 2.125, 2.25),
        ): "unselected",
    }
    cases["ts_selected_short_payload_warning"] = scenario(
        "ts-selected-short-payload-warning.xml",
        _xml(ts_selected_short_root),
        "selected_payload_warning_preserve",
        product="TS",
        role_map=ts_selected_short_roles,
    )
    ts_duplicate_root = ET.Element("LIGO_LW")
    ts_duplicate_roles = {
        _timeseries(
            ts_duplicate_root, 0, SELECTED_CHANNEL, ts_selected_samples
        ): "selected",
        _timeseries(
            ts_duplicate_root, 1, SELECTED_CHANNEL, (2.0, 2.125, 2.25)
        ): "unselected",
    }
    cases["ts_duplicate_channel_error"] = scenario(
        "ts-duplicate-channel-error.xml",
        _xml(ts_duplicate_root),
        "structural_identity_error_hold",
        product="TS",
        role_map=ts_duplicate_roles,
    )

    # Namespace-qualified elements are a control for literal tag matching in
    # old R. Its native parser currently returns no product for this XML.
    ns_root = ET.Element("{urn:gwexpy:f3-control}LIGO_LW")
    ns_result = ET.SubElement(
        ns_root,
        "{urn:gwexpy:f3-control}LIGO_LW",
        {"Name": "Result[0]", "Type": "Spectrum"},
    )
    for name, value in {
        "Subtype": "1",
        "M": "1",
        "N": str(points),
        "f0": str(F0),
        "df": str(DF),
        "ChannelA": SELECTED_CHANNEL,
    }.items():
        ET.SubElement(
            ns_result, "{urn:gwexpy:f3-control}Param", {"Name": name}
        ).text = value
    ET.SubElement(ns_result, "{urn:gwexpy:f3-control}Time", {"Name": "t0"}).text = str(
        EPOCH
    )
    ns_array = ET.SubElement(
        ns_result, "{urn:gwexpy:f3-control}Array", {"Type": "float"}
    )
    for dimension in (1, points):
        ET.SubElement(ns_array, "{urn:gwexpy:f3-control}Dim").text = str(dimension)
    ET.SubElement(
        ns_array, "{urn:gwexpy:f3-control}Stream", {"Encoding": "LittleEndian,base64"}
    ).text = base64.b64encode(selected_raw).decode("ascii")
    cases["namespace_qualified_control"] = scenario(
        "namespace-qualified-control.xml",
        _xml(ns_root),
        "old_r_namespace_empty",
        role_map={selected_hash: "selected"},
    )

    return {
        "schema": "gwexpy-b-f3-fixtures-v1",
        "selected_channel": SELECTED_CHANNEL,
        "unselected_channels": unselected_channels,
        "unselected_results": unselected_results,
        "points": points,
        "selected_raw_sha256": _sha256(selected_raw),
        "selected_raw_bytes": len(selected_raw),
        "old_r_many_valid_unselected_decoded_bytes": unselected_results
        * len(selected_raw),
        "selected_values": [
            float(value) for value in struct.unpack(f"<{points}f", selected_raw)
        ],
        "frequency_axis": {"f0": F0, "df": DF, "count": points},
        "epoch": EPOCH,
        "tf6_raw_sha256": _sha256(tf_raw),
        "tf6_oracle": {
            "pair": ["K1:F3-TF-OUTPUT", "K1:F3-TF-INPUT"],
            "frequencies": [17.5, 20.0, 22.5],
            "real": [1.0, -0.5, 3.0],
            "imag": [2.0, 0.25, -4.0],
            "dtype": "complex64",
        },
        "ts_oracle": {
            "selected_values": list(ts_selected_samples),
            "dt": 0.125,
            "t0": EPOCH,
        },
        "cases": cases,
        "native_selection_scope": {
            "PSD": {
                "status": "eligible",
                "allowed_unselected_payload_faults": [
                    "base64_decode_warning",
                    "decoded_length_warning",
                ],
                "preserve": [
                    "XML_parse_warning",
                    "metadata_warning",
                    "selected_payload_warning",
                ],
            },
            "TS": {
                "status": "fallback_only",
                "allowed_unselected_payload_faults": ["decoded_length_warning"],
                "preserve": ["duplicate_channel_error", "metadata_warning"],
                "note": "Public TS native=True is ignored with dttxml installed.",
            },
            "FFT": {
                "status": "HOLD",
                "preserve": [
                    "unselected_invalid_length_ValueError",
                    "unselected_nonreal_axis_ValueError",
                ],
            },
            "STF": {
                "status": "HOLD",
                "preserve": ["unselected_invalid_length_ValueError"],
            },
            "TF6": {
                "status": "HOLD",
                "preserve": [
                    "duplicate_identity_ValueError",
                    "unselected_nonfinite_frequency_ValueError",
                ],
            },
        },
        "external_route_note": (
            "native=False may use optional dttxml; characterize its warnings and "
            "errors separately and exclude it from the #589 decode/memory gate."
        ),
    }


class DecodedPayloadSpy(AbstractContextManager["DecodedPayloadSpy"]):
    """Count exact bytes returned by each native ``base64.b64decode`` call.

    ``stream_roles`` comes from a scenario manifest. The wrapper is local to
    a context manager and preserves decoder errors. Inspect ``report()`` only
    after the parser call has finished or raised.
    """

    def __init__(self, stream_roles: Mapping[str, str]):
        self.roles = dict(stream_roles)
        self.bytes_by_role: Counter[str] = Counter()
        self.calls_by_role: Counter[str] = Counter()
        self._patch: Any = None

    def __enter__(self) -> DecodedPayloadSpy:
        import gwexpy.io.dttxml_common as common

        original = common.base64.b64decode

        def counted(stream: str | bytes, *args: Any, **kwargs: Any) -> bytes:
            role = self.roles.get(_stream_hash(stream), "unknown")
            self.calls_by_role[role] += 1
            raw = original(stream, *args, **kwargs)
            self.bytes_by_role[role] += len(raw)
            return raw

        self._patch = patch.object(common.base64, "b64decode", counted)
        self._patch.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Literal[False]:
        self._patch.stop()
        return False

    def report(self) -> dict[str, Any]:
        """Return exact successful decoder byte counts and attempted calls."""
        return {
            "decoded_bytes": dict(self.bytes_by_role),
            "decode_calls": dict(self.calls_by_role),
            "total_decoded_bytes": sum(self.bytes_by_role.values()),
            "selected_decoded_bytes": self.bytes_by_role["selected"],
            "unselected_decoded_bytes": self.bytes_by_role["unselected"],
            "unknown_decode_calls": self.calls_by_role["unknown"],
        }


class FrequencyMaterializationSpy(
    AbstractContextManager["FrequencyMaterializationSpy"]
):
    """Count public reader FrequencySeries construction by channel name."""

    def __init__(self, selected_channels: set[str] | None):
        self.selected_channels = selected_channels
        self.channels: list[str] = []
        self._patch: Any = None

    def __enter__(self) -> FrequencyMaterializationSpy:
        import gwexpy.frequencyseries.io.dttxml as reader

        original: Any = reader.FrequencySeries
        self_spy = self

        class CountedFrequencySeries(original):
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                instance = super().__new__(cls, *args, **kwargs)
                self_spy.channels.append(str(kwargs.get("channel", "")))
                return instance

        self._patch = patch.object(reader, "FrequencySeries", CountedFrequencySeries)
        self._patch.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Literal[False]:
        self._patch.stop()
        return False

    def report(self) -> dict[str, Any]:
        """Return all constructed channels and the unselected count."""
        return {
            "series_constructed": len(self.channels),
            "unselected_series_constructed": sum(
                self.selected_channels is not None
                and channel not in self.selected_channels
                for channel in self.channels
            ),
            "channels": self.channels,
        }
