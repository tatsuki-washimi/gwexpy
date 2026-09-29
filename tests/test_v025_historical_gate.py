"""Focused release gate checks for the historical 74-scenario accounting."""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ci/v025_historical_gate.py"


def gate():
    spec = importlib.util.spec_from_file_location("v025_historical_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    module = gate()
    source = "a" * 40
    hashes = {"wheel": "b" * 64, "sdist": "c" * 64}
    manifest = tmp_path / "payload.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gwexpy-v025-release-payload-v1",
                "source_sha": source,
                "version": "0.2.5",
                "files": {
                    kind: {"name": name, "sha256": hashes[kind]}
                    for kind, name in (
                        ("wheel", "gwexpy-0.2.5-py3-none-any.whl"),
                        ("sdist", "gwexpy-0.2.5.tar.gz"),
                    )
                },
            }
        )
    )
    matrix = json.loads(
        (
            ROOT
            / "docs/developers/reports/2026-09-27-public-io-cross-format-post-fix-matrix.json"
        ).read_text()
    )
    fixed = {
        f["finding_id"]
        for f in matrix["findings"]
        if f["post_fix_status"] == "FIX_VERIFIED"
    }
    remaining = [
        f for f in matrix["findings"] if f["post_fix_status"] == "NOT_REQUALIFIED"
    ]
    behaviors = {
        f["finding_id"] for f in remaining if f["baseline_finding_status"] != "BLOCKED"
    }
    blocked = {
        f["finding_id"] for f in remaining if f["baseline_finding_status"] == "BLOCKED"
    }
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "qualification.json").write_text(
        json.dumps(
            {
                "schema": "gwexpy-v025-audit-qualification-v1",
                "source_sha": source,
                "artifact_sha256": hashes,
                "status_counts": {
                    "PASS": 26,
                    "FAIL": 0,
                    "NEEDS_HARNESS": 0,
                    "BLOCKED": 12,
                },
                "release_gate": "OPEN",
                "findings": [
                    {
                        "finding_id": finding["finding_id"],
                        "status": "BLOCKED"
                        if finding["finding_id"] in blocked
                        else "PASS",
                        "artifact_checks": {
                            kind: {
                                "status": "CHARACTERIZED"
                                if finding["finding_id"] in blocked
                                else "PASS"
                            }
                            for kind in ("wheel", "sdist")
                        },
                    }
                    for finding in remaining
                ],
            }
        )
    )
    nodes = {
        node
        for finding in matrix["findings"]
        if finding["finding_id"] in fixed
        for node in finding["regression_tests"]
    }
    for kind in ("wheel", "sdist"):
        suite = ET.Element(
            "testsuite", tests=str(len(nodes)), errors="0", failures="0", skipped="0"
        )
        for node in sorted(nodes):
            path, *names = node.split("::")
            classname = path.removesuffix(".py").replace("/", ".")
            if len(names) > 1:
                classname += "." + ".".join(names[:-1])
            ET.SubElement(suite, "testcase", classname=classname, name=names[-1])
        ET.ElementTree(suite).write(reports / f"fixed-{kind}.xml")
        (reports / f"{kind}.json").write_text(
            json.dumps(
                {
                    "schema": module.CELL_SCHEMA,
                    "source_sha": source,
                    "distribution": {"kind": kind, "sha256": hashes[kind]},
                    "fixed_finding_ids": sorted(fixed),
                    "behavior_finding_ids": sorted(behaviors),
                    "blocked_finding_ids": sorted(blocked),
                    "fixed_junit_sha256": module._sha(reports / f"fixed-{kind}.xml"),
                    "qualification_sha256": module._sha(reports / "qualification.json"),
                }
            )
        )
    approval = tmp_path / "approval.yaml"
    write_approval(approval, "f" * 64)
    return module, manifest, reports, approval, source


def write_approval(path: Path, digest: str) -> None:
    payload = json.dumps({"human_approval": {"disposition_digest": digest}})
    path.write_text("review_evidence_json: |\n  " + payload + "\n")


def test_aggregate_accounts_for_62_runtime_and_12_dispositions(tmp_path):
    module, manifest, reports, approval, source = fixture(tmp_path)
    result = module.aggregate(manifest, reports, approval, source, "f" * 64)
    assert result["runtime_pass_count"] == 62
    assert result["disposition_count"] == 12
    assert result["disposition_finding_ids"]
    assert not set(result["disposition_finding_ids"]) & set(
        result["runtime_finding_ids"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "wrong_source",
        "wrong_hash",
        "missing_behavior",
        "blocked_as_pass",
        "missing_approval",
        "wrong_digest",
        "tampered_junit",
        "qualification_blocked_pass",
        "missing_qualification",
    ],
)
def test_aggregate_rejects_incomplete_or_unbound_evidence(tmp_path, change):
    module, manifest, reports, approval, source = fixture(tmp_path)
    path = reports / "wheel.json"
    data = json.loads(path.read_text())
    if change == "wrong_source":
        data["source_sha"] = "0" * 40
    elif change == "wrong_hash":
        data["distribution"]["sha256"] = "0" * 64
    elif change == "missing_behavior":
        data["behavior_finding_ids"].pop()
    elif change == "blocked_as_pass":
        data["behavior_finding_ids"].append(data["blocked_finding_ids"][0])
    elif change == "missing_approval":
        approval.write_text("review_evidence_json: |\n  {}\n")
    elif change == "wrong_digest":
        write_approval(approval, "0" * 64)
    elif change == "tampered_junit":
        (reports / "fixed-wheel.xml").write_text(
            "<testsuite tests='0' errors='0' failures='0' skipped='0' />"
        )
    elif change == "qualification_blocked_pass":
        qualified = json.loads((reports / "qualification.json").read_text())
        qualified["findings"][0]["status"] = "PASS"
        (reports / "qualification.json").write_text(json.dumps(qualified))
    elif change == "missing_qualification":
        (reports / "qualification.json").unlink()
    path.write_text(json.dumps(data))
    with pytest.raises(module.HistoricalGateError):
        module.aggregate(manifest, reports, approval, source, "f" * 64)


def test_junit_skip_error_identifies_testcase_and_reason(tmp_path):
    module = gate()
    junit = tmp_path / "fixed.xml"
    suite = ET.Element("testsuite", tests="1", errors="0", failures="0", skipped="1")
    case = ET.SubElement(
        suite,
        "testcase",
        classname="tests.io.test_audio_metadata",
        name="test_public_audio_registry_preserves_available_metadata[flac]",
    )
    skipped = ET.SubElement(case, "skipped", message="ffmpeg unavailable")
    skipped.text = "codec backend could not be loaded"
    ET.ElementTree(suite).write(junit)

    with pytest.raises(module.HistoricalGateError) as exc_info:
        module._check_junit(tmp_path, junit, set())

    message = str(exc_info.value)
    assert (
        "tests.io.test_audio_metadata::test_public_audio_registry_preserves_available_metadata[flac]"
        in message
    )
    assert "ffmpeg unavailable" in message


def test_failed_pytest_reports_skipped_junit_details(tmp_path, monkeypatch):
    module = gate()
    junit = tmp_path / "fixed.xml"
    testcase_id = "tests.io.test_audio_metadata::test_audio[flac]"

    def failed_pytest(command, *, cwd, env):
        suite = ET.Element(
            "testsuite", tests="1", errors="0", failures="0", skipped="1"
        )
        case = ET.SubElement(
            suite,
            "testcase",
            classname="tests.io.test_audio_metadata",
            name="test_audio[flac]",
        )
        ET.SubElement(case, "skipped", message="ffmpeg unavailable")
        ET.ElementTree(suite).write(junit)
        raise module.HistoricalGateError("pytest exited 1")

    monkeypatch.setattr(module, "_run", failed_pytest)
    with pytest.raises(module.HistoricalGateError) as exc_info:
        module._run_fixed_regressions(
            tmp_path,
            tmp_path / "python",
            tmp_path,
            {},
            junit,
            set(),
            ["io/test_audio_metadata.py::test_audio[flac]"],
        )

    assert testcase_id in str(exc_info.value)
    assert "ffmpeg unavailable" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, module.HistoricalGateError)
    assert str(exc_info.value.__cause__) == "pytest exited 1"


def test_failed_pytest_preserves_original_error_when_junit_passes(
    tmp_path, monkeypatch
):
    module = gate()
    junit = tmp_path / "fixed.xml"
    classname = "tests.io.test_audio_metadata"
    test_name = "test_audio"
    nodes = {"tests/io/test_audio_metadata.py::test_audio"}
    original_error = module.HistoricalGateError("pytest exited 1")

    def failed_pytest(command, *, cwd, env):
        suite = ET.Element(
            "testsuite", tests="1", errors="0", failures="0", skipped="0"
        )
        ET.SubElement(suite, "testcase", classname=classname, name=test_name)
        ET.ElementTree(suite).write(junit)
        raise original_error

    monkeypatch.setattr(module, "_run", failed_pytest)
    with pytest.raises(module.HistoricalGateError) as exc_info:
        module._run_fixed_regressions(
            tmp_path,
            tmp_path / "python",
            tmp_path,
            {},
            junit,
            nodes,
            ["io/test_audio_metadata.py::test_audio"],
        )

    assert exc_info.value is original_error


def test_publish_workflow_requires_historical_gate_before_upload():
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/publish-release.yml").read_text()
    )
    jobs = workflow["jobs"]
    gate_job = jobs["historical_74_gate"]
    assert (
        jobs["verify"]["outputs"]["review_evidence"]
        == "${{ steps.validate.outputs.review_evidence }}"
    )
    assert set(gate_job["needs"]) == {"verify", "build"}
    assert "historical_74_gate" in jobs["publish"]["needs"]
    run = next(
        step["run"]
        for step in gate_job["steps"]
        if step["name"].startswith("Run historical probes")
    )
    assert "for kind in wheel sdist" in run
    assert "v025_historical_gate.py run" in run
    assert "v025_historical_gate.py finalize" in run
    assert '--approval "$GITHUB_WORKSPACE/source/$REVIEW_EVIDENCE"' in run
    assert "--payload-manifest" in run and "--source-sha" in run
