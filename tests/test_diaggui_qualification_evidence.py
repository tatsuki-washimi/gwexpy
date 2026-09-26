"""Fail-closed evidence checks for v0.2.4 DiagGUI qualification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "diaggui_qualification_evidence.py"
SOURCE_SHA = "a" * 40
CELLS = ("base-wheel", "base-sdist", "dttxml-wheel", "dttxml-sdist")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "diaggui_qualification_evidence", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_json(data: object) -> bytes:
    return (
        json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def files_for() -> dict[str, dict[str, str]]:
    return {
        "sdist": {"name": "gwexpy-0.2.4.tar.gz", "sha256": "c" * 64},
        "wheel": {"name": "gwexpy-0.2.4-py3-none-any.whl", "sha256": "b" * 64},
    }


def write_payload(path: Path) -> Path:
    path.write_bytes(
        canonical_json(
            {
                "files": files_for(),
                "schema": "gwexpy-v024-release-payload-v1",
                "source_sha": SOURCE_SHA,
                "version": "0.2.4",
            }
        )
    )
    return path


def report_for(cell: str, *, skips: list[list[str]] | None = None) -> dict[str, Any]:
    kind = "wheel" if cell.endswith("wheel") else "sdist"
    dttxml = cell.startswith("dttxml-")
    return {
        "artifact": {
            "filename": files_for()[kind]["name"],
            "kind": kind,
            "sha256": files_for()[kind]["sha256"],
        },
        "cell": cell,
        "environment": {
            "candidate_installed_from_payload": True,
            "dttxml_present": dttxml,
            "dttxml_version": "1.1.8" if dttxml else None,
            "gwexpy_module_in_site_packages": True,
            "gwexpy_version": "0.2.4",
        },
        "observed_skips": skips or [],
        "schema": "gwexpy-v024-diaggui-qualification-cell-v1",
        "source_sha": SOURCE_SHA,
        "test_status": "passed",
        "testcase_count": 10,
        "version": "0.2.4",
    }


def write_cell_reports(
    directory: Path, *, altered: dict[str, Any] | None = None
) -> None:
    for cell in CELLS:
        report = (
            altered if altered and altered.get("cell") == cell else report_for(cell)
        )
        target = directory / cell / "diaggui-qualification.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(canonical_json(report))


def test_mode_node_lists_are_explicit_and_avoid_skip_prone_route_probes() -> None:
    evidence = load_module()

    assert evidence.BASE_TEST_NODES
    assert evidence.DTTXML_TEST_NODES
    assert any("native" in node for node in evidence.BASE_TEST_NODES)
    assert any(
        "test_installed_dttxml_route_reads_timeseries" in node
        for node in evidence.DTTXML_TEST_NODES
    )
    assert any(
        "test_native_tf6_preserves_complex_values" in node
        for node in evidence.BASE_TEST_NODES
    )
    assert any(
        "test_fft_fallback_in_separate_no_dttxml_interpreter" in node
        for node in evidence.BASE_TEST_NODES
    )
    assert any(
        "test_stf_matrix_reader_uses_real_no_dttxml_interpreter[linear-fhz-with-f0-df]"
        in node
        for node in evidence.BASE_TEST_NODES
    )
    assert any(
        "test_stf_matrix_reader_uses_real_no_dttxml_interpreter[embedded-fhz-with-f0-df]"
        in node
        for node in evidence.BASE_TEST_NODES
    )
    assert any(
        "test_stf_matrix_reader_uses_real_no_dttxml_interpreter[embedded-fhz-no-f0-df]"
        in node
        for node in evidence.BASE_TEST_NODES
    )
    assert not any(
        "test_stf_matrix_reader_uses_real_no_dttxml_interpreter[linear-fhz-no-f0-df]"
        in node
        for node in evidence.BASE_TEST_NODES
    )
    assert not any(
        "test_public_read_in_real_no_dttxml_process" in node
        for node in evidence.DTTXML_TEST_NODES
    )
    assert not any(
        "test_matrix_read_in_real_base_only_process" in node
        for node in evidence.DTTXML_TEST_NODES
    )
    assert not any(
        "test_fft_fallback_in_separate_no_dttxml_interpreter" in node
        for node in evidence.DTTXML_TEST_NODES
    )
    assert not any(
        "test_stf_matrix_reader_uses_real_no_dttxml_interpreter" in node
        for node in evidence.DTTXML_TEST_NODES
    )
    assert any(
        "test_external_fft_preserves_quantized_nonuniform_embedded_axis" in node
        for node in evidence.DTTXML_TEST_NODES
    )
    assert not any(
        "test_external_fft_preserves_quantized_nonuniform_embedded_axis" in node
        for node in evidence.BASE_TEST_NODES
    )
    layout_nodes = [
        node
        for node in evidence.DTTXML_TEST_NODES
        if "test_stf_reader_reports_ambiguous_or_inconsistent_layout" in node
    ]
    assert len(layout_nodes) == 15
    assert all("[" in node and node.endswith("]") for node in layout_nodes)
    assert not any("external-missing-stream" in node for node in layout_nodes)
    assert any("native-missing-stream" in node for node in layout_nodes)


def test_payload_candidate_hash_and_filename_are_checked(tmp_path: Path) -> None:
    evidence = load_module()
    payload_path = write_payload(tmp_path / "payload.json")
    payload = evidence._payload_files(payload_path, source_sha=SOURCE_SHA)
    artifact = tmp_path / files_for()["wheel"]["name"]
    content = b"candidate wheel bytes"
    artifact.write_bytes(content)
    expected_digest = hashlib.sha256(content).hexdigest()
    payload["wheel"]["sha256"] = expected_digest

    facts = evidence._candidate_facts(
        cell="base-wheel", payload_files=payload, artifact_path=artifact
    )
    assert facts == {
        "filename": artifact.name,
        "kind": "wheel",
        "sha256": expected_digest,
    }

    artifact.write_bytes(b"different bytes")
    with pytest.raises(evidence.DiagGUIQualificationError, match="digest"):
        evidence._candidate_facts(
            cell="base-wheel", payload_files=payload, artifact_path=artifact
        )


def test_aggregate_requires_exact_four_candidate_bound_cells(tmp_path: Path) -> None:
    evidence = load_module()
    payload = write_payload(tmp_path / "payload.json")
    reports_dir = tmp_path / "reports"
    write_cell_reports(reports_dir)

    aggregate = evidence.aggregate_reports(
        source_sha=SOURCE_SHA,
        payload_manifest=payload,
        reports_dir=reports_dir,
        output_path=tmp_path / "aggregate.json",
    )
    assert aggregate["schema"] == "gwexpy-v024-diaggui-qualification-evidence-v1"
    assert aggregate["source_sha"] == SOURCE_SHA
    assert aggregate["files"] == files_for()
    assert [cell["cell"] for cell in aggregate["cells"]] == sorted(CELLS)
    assert all(cell["observed_skips"] == [] for cell in aggregate["cells"])

    extra = reports_dir / "unexpected.json"
    extra.write_bytes(canonical_json({"cell": "extra"}))
    with pytest.raises(evidence.DiagGUIQualificationError, match="exactly four"):
        evidence.aggregate_reports(
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            reports_dir=reports_dir,
            output_path=tmp_path / "aggregate-extra.json",
        )


@pytest.mark.parametrize(
    ("altered", "message"),
    [
        (
            report_for("base-wheel", skips=[["tests.optional", "test_x", "skip"]]),
            "mismatched",
        ),
        ({**report_for("dttxml-wheel"), "source_sha": "b" * 40}, "mismatched"),
        (
            {
                **report_for("dttxml-wheel"),
                "environment": {
                    **report_for("dttxml-wheel")["environment"],
                    "dttxml_version": "1.1.7",
                },
            },
            "mismatched",
        ),
    ],
)
def test_aggregate_rejects_unexpected_skips_and_mismatched_facts(
    tmp_path: Path, altered: dict[str, Any], message: str
) -> None:
    evidence = load_module()
    payload = write_payload(tmp_path / "payload.json")
    reports_dir = tmp_path / "reports"
    write_cell_reports(reports_dir, altered=altered)

    with pytest.raises(evidence.DiagGUIQualificationError, match=message):
        evidence.aggregate_reports(
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            reports_dir=reports_dir,
            output_path=tmp_path / "aggregate.json",
        )


def test_junit_skip_set_must_be_empty(tmp_path: Path) -> None:
    evidence = load_module()
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        '<testsuites><testsuite tests="2" errors="0" failures="0" skipped="1">'
        '<testcase classname="tests.required" name="test_passes" />'
        '<testcase classname="tests.optional" name="test_skips">'
        '<skipped message="unexpected environment skip" /></testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )

    with pytest.raises(evidence.DiagGUIQualificationError, match="unexpected skipped"):
        evidence._junit_facts(junit)
