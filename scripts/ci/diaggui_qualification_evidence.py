#!/usr/bin/env python3
"""Record and aggregate the v0.2.4 installed DiagGUI qualification lane."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

VERSION = "0.2.4"
PAYLOAD_SCHEMA = "gwexpy-v024-release-payload-v1"
CELL_SCHEMA = "gwexpy-v024-diaggui-qualification-cell-v1"
AGGREGATE_SCHEMA = "gwexpy-v024-diaggui-qualification-evidence-v1"
EXPECTED_DTTXML_VERSION = "1.1.8"
CELLS = ("base-wheel", "base-sdist", "dttxml-wheel", "dttxml-sdist")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 2 * 1024 * 1024

# Keep the tested surface explicit by environment. The base list selects only
# native and true no-dttxml subprocess cases. The dttxml list selects the same
# relevant format families with both native and external routes, omitting only
# tests whose purpose is to provision a separate parser-free interpreter.
BASE_TEST_NODES = (
    "io/test_dttxml_issue730.py::test_native_parser_reads_source_grounded_timeseries",
    "io/test_dttxml_issue730.py::test_native_product_loader_reads_timeseries",
    "io/test_dttxml_issue730.py::test_native_timeseries_rejects_invalid_base64",
    "io/test_dttxml_issue730.py::test_native_timeseries_accepts_base64_line_breaks",
    "io/test_dttxml_issue730.py::test_native_timeseries_duplicate_channel_fails_closed",
    "io/test_dttxml_issue730.py::test_native_timeseries_keeps_existing_frequency_products",
    "io/test_dttxml_issue730.py::test_public_read_in_real_no_dttxml_process",
    "io/test_dttxml_issue731.py::test_fft_frequency_readers_preserve_both_channel_layouts[native]",
    "io/test_dttxml_issue731.py::test_fft_reader_rejects_ambiguous_or_invalid_layouts[bad_embedded_frequency-nonzero imaginary-native]",
    "io/test_dttxml_issue731.py::test_fft_reader_rejects_ambiguous_or_invalid_layouts[multirow-one-row FFT layout-native]",
    "io/test_dttxml_issue731.py::test_fft_reader_rejects_ambiguous_or_invalid_layouts[count_mismatch-Frequency axis.*samples|row count|Dimensions-native]",
    "io/test_dttxml_issue731.py::test_fft_reader_rejects_ambiguous_or_invalid_layouts[unsupported_precision-only floatComplex|Frequency axis.*values-native]",
    "io/test_dttxml_issue731.py::test_fft_one_bin_frequency_axis_is_preserved[native]",
    "io/test_dttxml_issue731.py::test_fft_reader_rejects_double_complex_n8_even_when_external_parser_accepts[native]",
    "io/test_dttxml_issue731.py::test_fft_fallback_in_separate_no_dttxml_interpreter",
    "io/test_dttxml_issue732.py::test_stf_matrix_reader_preserves_labeled_complex_row_and_axis[linear-fhz-native]",
    "io/test_dttxml_issue732.py::test_stf_matrix_reader_preserves_labeled_complex_row_and_axis[embedded-fhz-native]",
    "io/test_dttxml_issue732.py::test_stf_subtype4_axis_does_not_require_f0_df[native]",
    "io/test_dttxml_issue732.py::test_matrix_reader_scopes_stf_validation_to_requested_product[native]",
    "io/test_dttxml_issue732.py::test_stf_matrix_reader_rejects_different_result_epochs[native]",
    "io/test_dttxml_issue732.py::test_stf_reader_preserves_one_bin_axis[native-linear]",
    "io/test_dttxml_issue732.py::test_stf_reader_preserves_one_bin_axis[native-embedded]",
    "io/test_dttxml_issue732.py::test_stf_reader_preserves_every_indexed_response_row[native]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-nonpositive-n-N must be positive]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-missing-stream-missing Array/Stream]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-missing-channel-b-ChannelB]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-ambiguous-channel-b-ChannelB]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-duplicate-channel-label-unique]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-row-count-mismatch-M.*row|row.*M]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-imaginary-frequency-frequency.*imaginary|imaginary.*frequency]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-double-complex-floatComplex]",
    "io/test_dttxml_issue732.py::test_stf_matrix_reader_uses_real_no_dttxml_interpreter[linear-fhz-with-f0-df]",
    "io/test_dttxml_issue732.py::test_stf_matrix_reader_uses_real_no_dttxml_interpreter[embedded-fhz-with-f0-df]",
    "io/test_dttxml_issue732.py::test_stf_matrix_reader_uses_real_no_dttxml_interpreter[embedded-fhz-no-f0-df]",
    "io/test_dttxml_issue733.py::test_tf6_subtype_integer_text_with_leading_zero_is_supported[native]",
    "io/test_dttxml_issue733.py::test_native_tf6_preserves_complex_values_axis_epoch_and_pair",
    "io/test_dttxml_issue733.py::test_tf6_preserves_near_uniform_serialized_axis_through_matrix[native]",
    "io/test_dttxml_issue733.py::test_same_pair_tf0_and_tf6_fail_closed_in_either_xml_order[native-tf0-first]",
    "io/test_dttxml_issue733.py::test_same_pair_tf0_and_tf6_fail_closed_in_either_xml_order[native-tf6-first]",
    "io/test_dttxml_issue733.py::test_same_channel_a_tf_results_fail_closed_when_parser_key_is_ambiguous[native]",
    "io/test_dttxml_issue733.py::test_uncharacterized_tf6_stream_layout_fails_closed[big-endian-native]",
    "io/test_dttxml_issue733.py::test_uncharacterized_tf6_stream_layout_fails_closed[truncated-native]",
    "io/test_dttxml_issue733.py::test_duplicate_raw_tf6_candidates_fail_closed[native]",
    "io/test_dttxml_issue733.py::test_noncolliding_tf0_is_unchanged[native]",
    "io/test_dttxml_issue733.py::test_native_noncolliding_tf0_accepts_unindexed_channel_b",
    "io/test_dttxml_issue734.py::test_matrix_read_in_real_base_only_process",
    "io/test_dttxml_common.py::TestLoadDttxmlProducts::test_ts_entries_stay_dict_shaped_and_reader_consumes_them",
)

DTTXML_TEST_NODES = (
    "io/test_dttxml_issue730.py::test_native_parser_reads_source_grounded_timeseries",
    "io/test_dttxml_issue730.py::test_native_product_loader_reads_timeseries",
    "io/test_dttxml_issue730.py::test_native_timeseries_rejects_invalid_base64",
    "io/test_dttxml_issue730.py::test_native_timeseries_accepts_base64_line_breaks",
    "io/test_dttxml_issue730.py::test_native_timeseries_duplicate_channel_fails_closed",
    "io/test_dttxml_issue730.py::test_native_timeseries_keeps_existing_frequency_products",
    "io/test_dttxml_issue730.py::test_installed_dttxml_route_reads_timeseries",
    "io/test_dttxml_issue730.py::test_installed_dttxml_route_preserves_mixed_frequency_products",
    "io/test_dttxml_issue731.py::test_fft_frequency_readers_preserve_both_channel_layouts",
    "io/test_dttxml_issue731.py::test_fft_reader_rejects_ambiguous_or_invalid_layouts",
    "io/test_dttxml_issue731.py::test_fft_one_bin_frequency_axis_is_preserved",
    "io/test_dttxml_issue731.py::test_fft_reader_rejects_double_complex_n8_even_when_external_parser_accepts",
    "io/test_dttxml_issue732.py::test_dttxml_118_stf_object_matches_xml_layout",
    "io/test_dttxml_issue732.py::test_stf_matrix_reader_preserves_labeled_complex_row_and_axis",
    "io/test_dttxml_issue732.py::test_external_stf_preflight_ignores_reference_nodes",
    "io/test_dttxml_issue732.py::test_stf_subtype4_axis_does_not_require_f0_df",
    "io/test_dttxml_issue732.py::test_matrix_reader_scopes_stf_validation_to_requested_product",
    "io/test_dttxml_issue732.py::test_stf_matrix_reader_rejects_different_result_epochs",
    "io/test_dttxml_issue732.py::test_stf_reader_preserves_one_bin_axis",
    "io/test_dttxml_issue732.py::test_stf_reader_preserves_every_indexed_response_row",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[external-nonpositive-n-N must be positive]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[external-missing-channel-b-ChannelB]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[external-ambiguous-channel-b-ChannelB]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[external-duplicate-channel-label-unique]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[external-row-count-mismatch-M.*row|row.*M]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[external-imaginary-frequency-frequency.*imaginary|imaginary.*frequency]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[external-double-complex-floatComplex]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-nonpositive-n-N must be positive]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-missing-stream-missing Array/Stream]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-missing-channel-b-ChannelB]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-ambiguous-channel-b-ChannelB]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-duplicate-channel-label-unique]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-row-count-mismatch-M.*row|row.*M]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-imaginary-frequency-frequency.*imaginary|imaginary.*frequency]",
    "io/test_dttxml_issue732.py::test_stf_reader_reports_ambiguous_or_inconsistent_layout[native-double-complex-floatComplex]",
    "io/test_dttxml_issue733.py",
    "io/test_dttxml_common.py::TestLoadDttxmlProducts::test_ts_entries_stay_dict_shaped_and_reader_consumes_them",
)


class DiagGUIQualificationError(ValueError):
    """Raised when DiagGUI qualification facts are incomplete or inconsistent."""


def _read_json(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise DiagGUIQualificationError(f"{description} must be a regular file")
        if metadata.st_size > MAX_JSON_BYTES:
            raise DiagGUIQualificationError(f"{description} exceeds size limit")
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
    except DiagGUIQualificationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey) as exc:
        raise DiagGUIQualificationError(f"invalid {description} JSON") from exc
    if not isinstance(data, dict):
        raise DiagGUIQualificationError(f"{description} must be a JSON object")
    return data, raw


class _DuplicateKey(ValueError):
    """Raised for an ambiguous JSON object."""


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _canonical_json(data: object) -> bytes:
    try:
        text = json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise DiagGUIQualificationError(
            "evidence contains a noncanonical value"
        ) from exc
    return (text + "\n").encode("utf-8")


def _require_keys(value: object, keys: set[str], *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DiagGUIQualificationError(f"{description} has unknown or missing keys")
    return value


def _payload_files(path: Path, *, source_sha: str) -> dict[str, dict[str, str]]:
    data, _ = _read_json(path, description="payload manifest")
    _require_keys(
        data,
        {"files", "schema", "source_sha", "version"},
        description="payload manifest",
    )
    if (
        data["schema"] != PAYLOAD_SCHEMA
        or data["version"] != VERSION
        or data["source_sha"] != source_sha
    ):
        raise DiagGUIQualificationError(
            "payload manifest is not bound to the v0.2.4 candidate"
        )
    files = _require_keys(
        data["files"], {"wheel", "sdist"}, description="payload files"
    )
    result: dict[str, dict[str, str]] = {}
    for kind, suffix in (("wheel", ".whl"), ("sdist", ".tar.gz")):
        entry = _require_keys(
            files[kind], {"name", "sha256"}, description=f"payload {kind}"
        )
        name = entry["name"]
        digest = entry["sha256"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or not name.endswith(suffix)
            or not isinstance(digest, str)
            or SHA256.fullmatch(digest) is None
        ):
            raise DiagGUIQualificationError(f"invalid payload {kind} entry")
        if (kind == "sdist" and name != f"gwexpy-{VERSION}.tar.gz") or (
            kind == "wheel"
            and re.fullmatch(
                rf"gwexpy-{re.escape(VERSION)}-[^-]+-[^-]+-[^-]+\.whl", name
            )
            is None
        ):
            raise DiagGUIQualificationError(
                f"payload {kind} entry does not match candidate version"
            )
        result[kind] = {"name": name, "sha256": digest}
    return result


def _sha256_file(path: Path) -> str:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise DiagGUIQualificationError("candidate artifact must be a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except DiagGUIQualificationError:
        raise
    except OSError as exc:
        raise DiagGUIQualificationError("cannot read candidate artifact") from exc


def _candidate_facts(
    *, cell: str, payload_files: dict[str, dict[str, str]], artifact_path: Path
) -> dict[str, str]:
    kind = "wheel" if cell.endswith("wheel") else "sdist"
    expected = payload_files[kind]
    if artifact_path.name != expected["name"]:
        raise DiagGUIQualificationError(
            "candidate artifact filename does not match payload"
        )
    digest = _sha256_file(artifact_path)
    if digest != expected["sha256"]:
        raise DiagGUIQualificationError(
            "candidate artifact digest does not match payload"
        )
    return {"filename": expected["name"], "kind": kind, "sha256": digest}


def _installed_candidate_facts(
    *, artifact_path: Path, artifact_sha256: str
) -> dict[str, Any]:
    try:
        import gwexpy

        distribution = importlib.metadata.distribution("gwexpy")
        version = importlib.metadata.version("gwexpy")
        module_path = Path(gwexpy.__file__).resolve()
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else None
    except (
        ImportError,
        importlib.metadata.PackageNotFoundError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise DiagGUIQualificationError(
            "installed gwexpy candidate is unavailable"
        ) from exc
    if version != VERSION or getattr(gwexpy, "__version__", None) != VERSION:
        raise DiagGUIQualificationError(
            "installed gwexpy version does not match candidate"
        )
    if not any(
        part in {"site-packages", "dist-packages"} for part in module_path.parts
    ):
        raise DiagGUIQualificationError(
            "gwexpy did not import from installed site-packages"
        )
    if not isinstance(direct_url, dict) or set(direct_url) != {"url", "archive_info"}:
        raise DiagGUIQualificationError(
            "installed gwexpy lacks direct artifact provenance"
        )
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        raise DiagGUIQualificationError(
            "installed gwexpy lacks archive digest provenance"
        )
    hash_value = archive_info.get("hash")
    hash_values = {hash_value} if isinstance(hash_value, str) else set()
    hashes = archive_info.get("hashes")
    if isinstance(hashes, dict):
        sha256_value = hashes.get("sha256")
        if isinstance(sha256_value, str):
            hash_values.add(f"sha256={sha256_value}")
    if f"sha256={artifact_sha256}" not in hash_values:
        raise DiagGUIQualificationError(
            "installed gwexpy digest does not match candidate"
        )
    parsed_url = urlparse(direct_url.get("url", ""))
    if parsed_url.scheme != "file" or not parsed_url.path:
        raise DiagGUIQualificationError(
            "installed gwexpy URL is not a local candidate artifact"
        )
    installed_from = Path(unquote(parsed_url.path)).resolve()
    if installed_from != artifact_path.resolve():
        raise DiagGUIQualificationError("installed gwexpy came from another artifact")
    return {
        "candidate_installed_from_payload": True,
        "gwexpy_module_in_site_packages": True,
        "gwexpy_version": version,
    }


def _dttxml_facts(*, expected_present: bool) -> dict[str, Any]:
    spec = importlib.util.find_spec("dttxml")
    present = spec is not None
    if present != expected_present:
        expected = "present" if expected_present else "absent"
        raise DiagGUIQualificationError(f"dttxml must be {expected} in this cell")
    if not present:
        return {"dttxml_present": False, "dttxml_version": None}
    try:
        import dttxml

        version = getattr(dttxml, "__version__", None)
    except ImportError as exc:
        raise DiagGUIQualificationError("dttxml cannot be imported") from exc
    if version != EXPECTED_DTTXML_VERSION:
        raise DiagGUIQualificationError(
            f"dttxml version must be {EXPECTED_DTTXML_VERSION}"
        )
    return {"dttxml_present": True, "dttxml_version": version}


def _junit_facts(path: Path) -> dict[str, Any]:
    path_to_script = Path(__file__).with_name("qualification_evidence.py")
    try:
        spec = importlib.util.spec_from_file_location(
            "qualification_evidence_for_diaggui", path_to_script
        )
        if spec is None or spec.loader is None:
            raise DiagGUIQualificationError("qualification JUnit reader is unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except DiagGUIQualificationError:
        raise
    except (ImportError, OSError) as exc:
        raise DiagGUIQualificationError(
            "qualification JUnit reader is unavailable"
        ) from exc
    try:
        testcase_count, skips = module._parse_junit(path)
    except module.QualificationEvidenceError as exc:
        raise DiagGUIQualificationError(str(exc)) from exc
    if skips:
        raise DiagGUIQualificationError(
            "DiagGUI qualification contains unexpected skipped testcases"
        )
    return {
        "observed_skips": [],
        "test_status": "passed",
        "testcase_count": testcase_count,
    }


def record_cell(
    *,
    cell: str,
    source_sha: str,
    payload_manifest: Path | str,
    artifact_path: Path | str,
    junit_path: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    if cell not in CELLS:
        raise DiagGUIQualificationError(f"unknown DiagGUI qualification cell: {cell}")
    if SHA40.fullmatch(source_sha) is None:
        raise DiagGUIQualificationError("source_sha must be a full lowercase SHA")
    files = _payload_files(Path(payload_manifest), source_sha=source_sha)
    artifact = _candidate_facts(
        cell=cell, payload_files=files, artifact_path=Path(artifact_path)
    )
    is_dttxml = cell.startswith("dttxml-")
    installed = _installed_candidate_facts(
        artifact_path=Path(artifact_path), artifact_sha256=artifact["sha256"]
    )
    report = {
        "artifact": artifact,
        "cell": cell,
        "environment": {
            **_dttxml_facts(expected_present=is_dttxml),
            **installed,
        },
        "schema": CELL_SCHEMA,
        "source_sha": source_sha,
        **_junit_facts(Path(junit_path)),
        "version": VERSION,
    }
    _write_json(Path(report_path), report)
    return report


def _write_json(path: Path, data: object) -> None:
    if path.exists() or path.is_symlink():
        raise DiagGUIQualificationError("evidence output already exists")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_json(data))
    except DiagGUIQualificationError:
        raise
    except OSError as exc:
        raise DiagGUIQualificationError("cannot write DiagGUI evidence") from exc


def _load_reports(reports_dir: Path) -> list[dict[str, Any]]:
    if reports_dir.is_symlink() or not reports_dir.is_dir():
        raise DiagGUIQualificationError(
            "DiagGUI report directory must be a real directory"
        )
    paths = sorted(reports_dir.rglob("*.json"))
    if len(paths) != len(CELLS) or any(
        path.name != "diaggui-qualification.json" for path in paths
    ):
        raise DiagGUIQualificationError(
            "DiagGUI evidence must contain exactly four cell reports"
        )
    reports: list[dict[str, Any]] = []
    for path in paths:
        data, raw = _read_json(path, description="DiagGUI cell report")
        if raw != _canonical_json(data):
            raise DiagGUIQualificationError(
                "DiagGUI cell report must be canonical JSON"
            )
        reports.append(data)
    return reports


def aggregate_reports(
    *,
    source_sha: str,
    payload_manifest: Path | str,
    reports_dir: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    if SHA40.fullmatch(source_sha) is None:
        raise DiagGUIQualificationError("source_sha must be a full lowercase SHA")
    files = _payload_files(Path(payload_manifest), source_sha=source_sha)
    reports = _load_reports(Path(reports_dir))
    seen: set[str] = set()
    summaries: list[dict[str, Any]] = []
    expected_report_keys = {
        "artifact",
        "cell",
        "environment",
        "observed_skips",
        "schema",
        "source_sha",
        "test_status",
        "testcase_count",
        "version",
    }
    for report in reports:
        _require_keys(report, expected_report_keys, description="DiagGUI cell report")
        cell = report["cell"]
        if not isinstance(cell, str) or cell not in CELLS or cell in seen:
            raise DiagGUIQualificationError(
                "DiagGUI reports have unknown or duplicate cells"
            )
        artifact_kind = "wheel" if cell.endswith("wheel") else "sdist"
        artifact = _require_keys(
            report["artifact"],
            {"filename", "kind", "sha256"},
            description="artifact facts",
        )
        environment = _require_keys(
            report["environment"],
            {
                "candidate_installed_from_payload",
                "dttxml_present",
                "dttxml_version",
                "gwexpy_module_in_site_packages",
                "gwexpy_version",
            },
            description="environment facts",
        )
        expected_present = cell.startswith("dttxml-")
        testcase_count = report["testcase_count"]
        if (
            report["schema"] != CELL_SCHEMA
            or report["source_sha"] != source_sha
            or report["version"] != VERSION
            or artifact
            != {
                "filename": files[artifact_kind]["name"],
                "kind": artifact_kind,
                "sha256": files[artifact_kind]["sha256"],
            }
            or environment
            != {
                "candidate_installed_from_payload": True,
                "dttxml_present": expected_present,
                "dttxml_version": EXPECTED_DTTXML_VERSION if expected_present else None,
                "gwexpy_module_in_site_packages": True,
                "gwexpy_version": VERSION,
            }
            or report["test_status"] != "passed"
            or not isinstance(testcase_count, int)
            or isinstance(testcase_count, bool)
            or testcase_count <= 0
            or report["observed_skips"] != []
        ):
            raise DiagGUIQualificationError(
                "DiagGUI report has mismatched candidate, environment, or test facts"
            )
        summaries.append(
            {
                "artifact": artifact,
                "cell": cell,
                "environment": environment,
                "observed_skips": [],
                "test_status": "passed",
                "testcase_count": testcase_count,
            }
        )
        seen.add(cell)
    if seen != set(CELLS):
        raise DiagGUIQualificationError("DiagGUI reports have missing cells")
    aggregate = {
        "cells": sorted(summaries, key=lambda item: item["cell"]),
        "files": files,
        "schema": AGGREGATE_SCHEMA,
        "source_sha": source_sha,
        "version": VERSION,
    }
    _write_json(Path(output_path), aggregate)
    return aggregate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    nodes = commands.add_parser("test-nodes")
    nodes.add_argument("--mode", choices=("base", "dttxml"), required=True)
    record = commands.add_parser("record")
    record.add_argument("--cell", required=True)
    record.add_argument("--source-sha", required=True)
    record.add_argument("--payload-manifest", type=Path, required=True)
    record.add_argument("--artifact", type=Path, required=True)
    record.add_argument("--junit", type=Path, required=True)
    record.add_argument("--report", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--source-sha", required=True)
    aggregate.add_argument("--payload-manifest", type=Path, required=True)
    aggregate.add_argument("--reports-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "test-nodes":
            nodes = BASE_TEST_NODES if args.mode == "base" else DTTXML_TEST_NODES
            print(*nodes, sep="\n")
        elif args.command == "record":
            record_cell(
                cell=args.cell,
                source_sha=args.source_sha,
                payload_manifest=args.payload_manifest,
                artifact_path=args.artifact,
                junit_path=args.junit,
                report_path=args.report,
            )
        else:
            aggregate_reports(
                source_sha=args.source_sha,
                payload_manifest=args.payload_manifest,
                reports_dir=args.reports_dir,
                output_path=args.output,
            )
    except DiagGUIQualificationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
