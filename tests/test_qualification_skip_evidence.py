"""Fail-closed contracts for v0.2.3 qualification skip evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "qualification_evidence.py"
BASELINE = ROOT / "scripts" / "ci" / "v023_qualification_expected_skips.json"

EXPECTED_CELLS = (
    "install-ubuntu-3.11-wheel",
    "install-ubuntu-3.11-sdist",
    "install-ubuntu-3.12-wheel",
    "install-ubuntu-3.12-sdist",
    "install-ubuntu-3.13-wheel",
    "install-ubuntu-3.13-sdist",
    "install-ubuntu-3.14-wheel",
    "install-ubuntu-3.14-sdist",
    "install-macos-3.11-wheel",
    "install-macos-3.14-wheel",
    "install-windows-3.11-wheel",
    "install-windows-3.14-wheel",
    "gwpy-4.0.1-wheel",
    "gwpy-4.0.2-wheel",
    "sdist-3.12-claims",
    "conda-3.11",
    "conda-3.14",
    "scientific-3.11-wheel",
    "docs-en-ja-3.11-wheel",
)
SOURCE_SHA = "a" * 40
BASELINE_SHA256 = "6a19168dab95b582e65b612ac7bc5ad2853b7df723b8d4cae73e6edf32802430"


def files_for(version: str) -> dict[str, dict[str, str]]:
    return {
        "sdist": {"name": f"gwexpy-{version}.tar.gz", "sha256": "c" * 64},
        "wheel": {
            "name": f"gwexpy-{version}-py3-none-any.whl",
            "sha256": "b" * 64,
        },
    }


FILES = files_for("0.2.3")


def load_module():
    spec = importlib.util.spec_from_file_location("qualification_evidence", SCRIPT)
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


def baseline_data(
    optional: dict[str, list[list[str]]] | None = None,
) -> dict[str, object]:
    optional = optional or {}
    return {
        "cells": [
            {"cell": cell, "optional_skips": optional.get(cell, [])}
            for cell in EXPECTED_CELLS
        ],
        "schema": "gwexpy-v023-qualification-expected-skips-v1",
        "version": "0.2.3",
    }


def write_baseline(
    path: Path,
    optional: dict[str, list[list[str]]] | None = None,
) -> Path:
    path.write_bytes(canonical_json(baseline_data(optional)))
    return path


def write_payload(path: Path, *, version: str = "0.2.3") -> Path:
    path.write_bytes(
        canonical_json(
            {
                "files": files_for(version),
                "schema": f"gwexpy-v{version.replace('.', '')}-release-payload-v1",
                "source_sha": SOURCE_SHA,
                "version": version,
            }
        )
    )
    return path


def write_junit(path: Path, skips: list[tuple[str, str, str]]) -> Path:
    cases = '<testcase classname="tests.required" name="test_passes"/>' + "".join(
        f'<testcase classname="{classname}" name="{name}">'
        f'<skipped message="{message}"/></testcase>'
        for classname, name, message in skips
    )
    path.write_text(
        '<testsuites name="pytest tests"><testsuite name="pytest" '
        f'errors="0" failures="0" skipped="{len(skips)}" '
        f'tests="{len(skips) + 1}">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )
    return path


def test_repository_baseline_is_canonical_complete_and_initially_empty() -> None:
    evidence = load_module()
    loaded = evidence.load_expected_skips(BASELINE)
    raw = BASELINE.read_bytes()

    assert tuple(loaded.cells) == EXPECTED_CELLS
    assert all(not skips for skips in loaded.cells.values())
    assert raw == canonical_json(baseline_data())
    assert loaded.sha256 == hashlib.sha256(raw).hexdigest() == BASELINE_SHA256


@pytest.mark.parametrize(
    ("autocrlf", "eol", "attributes_newline"),
    [("false", "lf", b"\n"), ("true", "crlf", b"\r\n")],
    ids=["lf", "crlf"],
)
def test_repository_baseline_survives_crlf_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    autocrlf: str,
    eol: str,
    attributes_newline: bytes,
) -> None:
    source_attributes = ROOT / ".gitattributes"
    assert source_attributes.is_file()

    repository = tmp_path / "repository"
    repository.mkdir()
    attributes = repository / ".gitattributes"
    attributes.write_bytes(
        source_attributes.read_text(encoding="utf-8").encode("utf-8")
    )

    baseline = repository / "scripts" / "ci" / "v023_qualification_expected_skips.json"
    baseline.parent.mkdir(parents=True)
    shutil.copyfile(BASELINE, baseline)

    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Qualification test"],
        ["config", "core.autocrlf", autocrlf],
        ["config", "core.eol", eol],
    ):
        subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["git", "add", "."],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "baseline"],
        cwd=repository,
        check=True,
        capture_output=True,
    )

    attributes.unlink()
    baseline.unlink()
    subprocess.run(
        ["git", "checkout", "--", "."],
        cwd=repository,
        check=True,
        capture_output=True,
    )

    assert attributes.read_bytes() == (
        b"scripts/ci/v023_qualification_expected_skips.json text eol=lf"
        + attributes_newline
    )
    # Exercise the actual workflow contract against both checkout conditions.
    spec = importlib.util.spec_from_file_location(
        "release_workflow_contract", ROOT / "tests" / "test_publish_release_workflow.py"
    )
    assert spec and spec.loader
    workflow_contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow_contract)
    monkeypatch.setattr(
        workflow_contract,
        "WORKFLOW",
        repository / ".github" / "workflows" / "publish-release.yml",
    )
    workflow_contract.test_v023_expected_skip_baseline_declares_lf_checkout_contract()

    raw = baseline.read_bytes()
    assert raw == BASELINE.read_bytes()
    assert b"\r\n" not in raw
    assert hashlib.sha256(raw).hexdigest() == BASELINE_SHA256
    evidence = load_module()
    assert evidence.load_expected_skips(baseline).sha256 == BASELINE_SHA256

    baseline.write_bytes(raw.replace(b"\n", b"\r\n"))
    with pytest.raises(evidence.QualificationEvidenceError, match="canonical"):
        evidence.load_expected_skips(baseline)


def test_version_contract_preserves_v022_and_adds_v023() -> None:
    evidence = load_module()

    assert evidence.qualification_contract("0.2.2") == {
        "artifact_prefix": "v022-qualification-evidence",
        "evidence_schema": "gwexpy-v022-qualification-evidence-v1",
        "expected_skips_schema": None,
    }
    assert evidence.qualification_contract("0.2.3") == {
        "artifact_prefix": "v023-qualification-evidence",
        "evidence_schema": "gwexpy-v023-qualification-evidence-v1",
        "expected_skips_schema": "gwexpy-v023-qualification-expected-skips-v1",
    }
    with pytest.raises(evidence.QualificationEvidenceError, match="unsupported"):
        evidence.qualification_contract("0.2.4")


def test_record_v023_accepts_reviewed_optional_skip_and_records_baseline_sha(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    skip = ["tests.optional", "test_backend", "optional backend is unavailable"]
    baseline = write_baseline(tmp_path / "baseline.json", {EXPECTED_CELLS[0]: [skip]})
    junit = write_junit(tmp_path / "pytest.xml", [(skip[0], skip[1], skip[2])])
    payload = write_payload(tmp_path / "payload.json")
    report = tmp_path / "qualification.json"
    baseline_before = baseline.read_bytes()

    result = evidence.record_cell(
        version="0.2.3",
        cell=EXPECTED_CELLS[0],
        source_sha=SOURCE_SHA,
        payload_manifest=payload,
        report_path=report,
        junit_path=junit,
        expected_skips_path=baseline,
    )

    baseline_sha = hashlib.sha256(baseline_before).hexdigest()
    assert result == {
        "baseline_sha256": baseline_sha,
        "cell": EXPECTED_CELLS[0],
        "files": FILES,
        "observed_optional_skips": [skip],
        "observed_required_skips": [],
        "observed_skips": [skip],
        "source_sha": SOURCE_SHA,
        "status": "passed",
        "testcase_count": 2,
        "version": "0.2.3",
    }
    assert json.loads(report.read_text(encoding="utf-8")) == result
    assert baseline.read_bytes() == baseline_before


def test_record_v023_rejects_every_unreviewed_observed_skip(tmp_path: Path) -> None:
    evidence = load_module()
    baseline = write_baseline(tmp_path / "baseline.json")
    junit = write_junit(
        tmp_path / "pytest.xml",
        [("tests.required", "test_contract", "new unexpected skip")],
    )
    payload = write_payload(tmp_path / "payload.json")
    report = tmp_path / "qualification.json"

    with pytest.raises(
        evidence.QualificationEvidenceError, match="required observed skips"
    ):
        evidence.record_cell(
            version="0.2.3",
            cell=EXPECTED_CELLS[0],
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=report,
            junit_path=junit,
            expected_skips_path=baseline,
        )
    assert not report.exists()


def test_reviewed_optional_skip_may_be_absent_from_an_observed_cell(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    approved = ["tests.optional", "test_backend", "optional backend unavailable"]
    baseline = write_baseline(
        tmp_path / "baseline.json", {EXPECTED_CELLS[0]: [approved]}
    )
    junit = write_junit(tmp_path / "pytest.xml", [])
    payload = write_payload(tmp_path / "payload.json")

    result = evidence.record_cell(
        version="0.2.3",
        cell=EXPECTED_CELLS[0],
        source_sha=SOURCE_SHA,
        payload_manifest=payload,
        report_path=tmp_path / "qualification.json",
        junit_path=junit,
        expected_skips_path=baseline,
    )

    assert result["observed_skips"] == []
    assert result["observed_optional_skips"] == []
    assert result["observed_required_skips"] == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(schema="unknown"), "schema"),
        (lambda data: data.update(version="0.2.4"), "version"),
        (lambda data: data["cells"].pop(), "exactly 19"),
        (
            lambda data: data["cells"].append(data["cells"][0]),
            "exactly 19|duplicate",
        ),
        (
            lambda data: data["cells"][0].update(cell="unknown-cell"),
            "unknown|missing",
        ),
        (
            lambda data: data["cells"][0].update(
                optional_skips=[["bad\nclass", "test_name", "reason"]]
            ),
            "unsafe",
        ),
        (
            lambda data: data["cells"][0].update(
                optional_skips=[
                    ["tests.optional", "test_backend", "reason"],
                    ["tests.optional", "test_backend", "reason"],
                ]
            ),
            "sorted and unique",
        ),
        (lambda data: data.update(extra=True), "keys"),
    ],
)
def test_baseline_validation_fails_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    evidence = load_module()
    data = baseline_data()
    mutation(data)
    path = tmp_path / "baseline.json"
    path.write_bytes(canonical_json(data))

    with pytest.raises(evidence.QualificationEvidenceError, match=message):
        evidence.load_expected_skips(path)


def test_baseline_rejects_noncanonical_json_and_symlink(tmp_path: Path) -> None:
    evidence = load_module()
    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(json.dumps(baseline_data(), indent=2), encoding="utf-8")
    with pytest.raises(evidence.QualificationEvidenceError, match="canonical"):
        evidence.load_expected_skips(noncanonical)

    canonical = write_baseline(tmp_path / "canonical.json")
    symlink = tmp_path / "baseline-link.json"
    symlink.symlink_to(canonical)
    with pytest.raises(evidence.QualificationEvidenceError, match="regular file"):
        evidence.load_expected_skips(symlink)


@pytest.mark.parametrize("protected_input", ["baseline", "junit", "payload"])
def test_record_rejects_symlink_ancestor_for_protected_inputs(
    tmp_path: Path,
    protected_input: str,
) -> None:
    evidence = load_module()
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    real_paths = {
        "baseline": write_baseline(real / "baseline.json"),
        "junit": write_junit(real / "pytest.xml", []),
        "payload": write_payload(real / "payload.json"),
    }
    selected_paths = {
        name: alias / path.name if name == protected_input else path
        for name, path in real_paths.items()
    }

    with pytest.raises(evidence.QualificationEvidenceError, match="symlink ancestor"):
        evidence.record_cell(
            version="0.2.3",
            cell=EXPECTED_CELLS[0],
            source_sha=SOURCE_SHA,
            payload_manifest=selected_paths["payload"],
            report_path=tmp_path / "qualification.json",
            junit_path=selected_paths["junit"],
            expected_skips_path=selected_paths["baseline"],
        )


def test_aggregate_rejects_reports_directory_symlink_ancestor(tmp_path: Path) -> None:
    evidence = load_module()
    real = tmp_path / "real"
    reports = real / "reports"
    reports.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(evidence.QualificationEvidenceError, match="symlink ancestor"):
        evidence.aggregate_reports(
            version="0.2.3",
            source_sha=SOURCE_SHA,
            payload_manifest=write_payload(tmp_path / "payload.json"),
            reports_dir=alias / "reports",
            output_path=tmp_path / "aggregate.json",
            expected_skips_path=write_baseline(tmp_path / "baseline.json"),
        )


def test_baseline_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    evidence = load_module()
    path = tmp_path / "duplicate-key.json"
    path.write_text(
        '{"cells":[],"schema":"one","schema":"two","version":"0.2.3"}\n',
        encoding="utf-8",
    )

    with pytest.raises(evidence.QualificationEvidenceError, match="invalid"):
        evidence.load_expected_skips(path)


def test_junit_skip_parser_rejects_noncanonical_or_duplicate_cases(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    missing = tmp_path / "missing.xml"
    missing.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="1">'
        '<testcase name="test_name"><skipped message="reason"/></testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    with pytest.raises(evidence.QualificationEvidenceError, match="classname"):
        evidence.parse_junit_skips(missing)

    duplicate = write_junit(
        tmp_path / "duplicate.xml",
        [
            ("tests.optional", "test_backend", "reason"),
            ("tests.optional", "test_backend", "reason"),
        ],
    )
    with pytest.raises(evidence.QualificationEvidenceError, match="duplicate"):
        evidence.parse_junit_skips(duplicate)

    failure = tmp_path / "failure.xml"
    failure.write_text(
        '<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0">'
        '<testcase classname="tests.required" name="test_failure">'
        "<failure/></testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    with pytest.raises(evidence.QualificationEvidenceError, match="failed|errored"):
        evidence.parse_junit_skips(failure)


@pytest.mark.parametrize(
    "document",
    [
        (
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="tests.required" name="test_passes"/>'
            "</testsuite>"
        ),
        (
            "<testsuites>"
            '<testcase classname="tests.required" name="test_passes"/>'
            "</testsuites>"
        ),
        (
            "<testsuites>"
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="tests.required" name="test_one"/>'
            "</testsuite>"
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="tests.required" name="test_two"/>'
            "</testsuite>"
            "</testsuites>"
        ),
        (
            '<testsuites><testsuite failures="0" errors="0" skipped="0">'
            '<testcase classname="tests.required" name="test_passes"/>'
            "</testsuite></testsuites>"
        ),
        (
            '<testsuites><testsuite tests="-1" failures="0" errors="0" '
            'skipped="0"><testcase classname="tests.required" '
            'name="test_passes"/></testsuite></testsuites>'
        ),
        (
            '<testsuites><testsuite tests="one" failures="0" errors="0" '
            'skipped="0"><testcase classname="tests.required" '
            'name="test_passes"/></testsuite></testsuites>'
        ),
        (
            '<testsuites><testsuite tests="2" failures="0" errors="0" '
            'skipped="0"><testcase classname="tests.required" '
            'name="test_passes"/></testsuite></testsuites>'
        ),
        (
            '<testsuites><testsuite tests="1" failures="1" errors="0" '
            'skipped="0"><testcase classname="tests.required" '
            'name="test_passes"/></testsuite></testsuites>'
        ),
        (
            '<testsuites><testsuite tests="1" failures="0" errors="1" '
            'skipped="0"><testcase classname="tests.required" '
            'name="test_passes"/></testsuite></testsuites>'
        ),
        (
            '<testsuites><testsuite tests="1" failures="0" errors="0" '
            'skipped="1"><testcase classname="tests.required" '
            'name="test_passes"/></testsuite></testsuites>'
        ),
        (
            '<testsuites><testsuite tests="1" failures="0" errors="0" '
            'skipped="0"><testcase classname="tests.required" '
            'name="test_passes"/><failure/></testsuite></testsuites>'
        ),
    ],
)
def test_junit_parser_requires_pytest_single_suite_and_truthful_counters(
    tmp_path: Path,
    document: str,
) -> None:
    evidence = load_module()
    junit = tmp_path / "forged.xml"
    junit.write_text(document, encoding="utf-8")

    with pytest.raises(
        evidence.QualificationEvidenceError,
        match="hierarchy|counter|declared|declaration",
    ):
        evidence.parse_junit_skips(junit)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le"])
def test_junit_parser_rejects_utf16_dtd_entity_bypass(
    tmp_path: Path,
    encoding: str,
) -> None:
    evidence = load_module()
    junit = tmp_path / "utf16-entity.xml"
    document = (
        '<?xml version="1.0" encoding="utf-16"?>'
        '<!DOCTYPE testsuites [<!ENTITY reason "forged optional skip">]>'
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="1">'
        '<testcase classname="tests.optional" name="test_backend">'
        '<skipped message="&reason;"/></testcase></testsuite></testsuites>'
    )
    junit.write_bytes(document.encode(encoding))

    with pytest.raises(evidence.QualificationEvidenceError, match="UTF-8"):
        evidence.parse_junit_skips(junit)


def test_parser_consumes_pytests_builtin_junit_skip_tuple(tmp_path: Path) -> None:
    evidence = load_module()
    test_file = tmp_path / "test_builtin_junit.py"
    test_file.write_text(
        "import pytest\n\n"
        "@pytest.mark.skip(reason='optional dependency missing')\n"
        "def test_optional():\n"
        "    raise AssertionError('must not run')\n",
        encoding="utf-8",
    )
    junit = tmp_path / "pytest.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
            str(test_file),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert evidence.parse_junit_skips(junit) == [
        ["test_builtin_junit", "test_optional", "optional dependency missing"]
    ]


@pytest.mark.parametrize(
    "document",
    [
        (
            '<testsuites xmlns:z="urn:forged"><testsuite tests="1" failures="0" '
            'errors="0" skipped="0"><testcase classname="tests.required" '
            'name="test_hidden"><z:skipped message="hidden skip"/></testcase>'
            "</testsuite></testsuites>"
        ),
        (
            '<testsuites><testsuite tests="1" failures="0" errors="0" '
            'skipped="0"><testcase classname="tests.required" '
            'name="test_hidden"><evil/></testcase></testsuite></testsuites>'
        ),
        (
            '<testsuites xmlns:z="urn:forged"><testsuite tests="1" failures="0" '
            'errors="0" skipped="0"><z:testcase classname="tests.required" '
            'name="test_hidden"/></testsuite></testsuites>'
        ),
        (
            '<testsuites xmlns:z="urn:forged"><testsuite tests="1" failures="0" '
            'errors="0" skipped="0"><testcase classname="tests.required" '
            'name="test_hidden"><properties><z:property name="role" '
            'value="forged"/></properties></testcase></testsuite></testsuites>'
        ),
        (
            '<testsuites><testsuite tests="1" failures="0" errors="0" '
            'skipped="0"><system-out>forged suite output</system-out>'
            '<testcase classname="tests.required" name="test_hidden"/>'
            "</testsuite></testsuites>"
        ),
        (
            '<testsuites xmlns:z="urn:forged"><z:testsuite tests="1" '
            'failures="0" errors="0" skipped="0"><testcase '
            'classname="tests.required" name="test_hidden"/>'
            "</z:testsuite></testsuites>"
        ),
        (
            '<testsuites><evil/><testsuite tests="1" failures="0" errors="0" '
            'skipped="0"><testcase classname="tests.required" '
            'name="test_hidden"/></testsuite></testsuites>'
        ),
        (
            '<z:testsuites xmlns:z="urn:forged"><z:testsuite tests="1" '
            'failures="0" errors="0" skipped="0"><z:testcase '
            'classname="tests.required" name="test_hidden"/>'
            "</z:testsuite></z:testsuites>"
        ),
    ],
)
def test_junit_parser_rejects_unknown_or_namespaced_tags(
    tmp_path: Path,
    document: str,
) -> None:
    evidence = load_module()
    junit = tmp_path / "forged-tag.xml"
    junit.write_text(document, encoding="utf-8")

    with pytest.raises(evidence.QualificationEvidenceError, match="tag|hierarchy"):
        evidence.parse_junit_skips(junit)


def test_parser_accepts_pytest_properties_and_captured_output(tmp_path: Path) -> None:
    evidence = load_module()
    test_file = tmp_path / "test_builtin_metadata.py"
    test_file.write_text(
        "import sys\n\n"
        "def test_metadata(record_property, record_testsuite_property):\n"
        "    record_property('detector', 'H1')\n"
        "    record_testsuite_property('oracle', 'GWpy')\n"
        "    print('captured stdout')\n"
        "    print('captured stderr', file=sys.stderr)\n",
        encoding="utf-8",
    )
    junit = tmp_path / "pytest-metadata.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "junit_logging=all",
            f"--junitxml={junit}",
            str(test_file),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    xml = junit.read_text(encoding="utf-8")
    for tag in ("<properties>", "<property ", "<system-out>", "<system-err>"):
        assert tag in xml
    assert evidence.parse_junit_skips(junit) == []


def test_aggregate_v023_requires_one_valid_report_for_every_cell(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    baseline = write_baseline(tmp_path / "baseline.json")
    payload = write_payload(tmp_path / "payload.json")
    reports = tmp_path / "reports"
    reports.mkdir()
    junit = write_junit(tmp_path / "pytest.xml", [])
    baseline_sha = hashlib.sha256(baseline.read_bytes()).hexdigest()

    for index, cell in enumerate(EXPECTED_CELLS):
        evidence.record_cell(
            version="0.2.3",
            cell=cell,
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=reports / str(index) / "qualification.json",
            junit_path=junit,
            expected_skips_path=baseline,
        )

    output = tmp_path / "aggregate.json"
    result = evidence.aggregate_reports(
        version="0.2.3",
        source_sha=SOURCE_SHA,
        payload_manifest=payload,
        reports_dir=reports,
        output_path=output,
        expected_skips_path=baseline,
    )

    assert result["schema"] == "gwexpy-v023-qualification-evidence-v1"
    assert result["baseline_sha256"] == baseline_sha
    assert [item["cell"] for item in result["cells"]] == sorted(EXPECTED_CELLS)
    assert all(item["observed_required_skips"] == [] for item in result["cells"])
    assert all(item["observed_optional_skips"] == [] for item in result["cells"])
    assert json.loads(output.read_text(encoding="utf-8")) == result

    (reports / "0" / "qualification.json").unlink()
    with pytest.raises(evidence.QualificationEvidenceError, match="exactly 19"):
        evidence.aggregate_reports(
            version="0.2.3",
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            reports_dir=reports,
            output_path=output,
            expected_skips_path=baseline,
        )


def test_aggregate_rejects_duplicate_cell_and_baseline_identity_mismatch(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    baseline = write_baseline(tmp_path / "baseline.json")
    payload = write_payload(tmp_path / "payload.json")
    reports = tmp_path / "reports"
    reports.mkdir()
    junit = write_junit(tmp_path / "pytest.xml", [])
    for index, cell in enumerate(EXPECTED_CELLS):
        evidence.record_cell(
            version="0.2.3",
            cell=cell,
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=reports / str(index) / "qualification.json",
            junit_path=junit,
            expected_skips_path=baseline,
        )

    duplicate = reports / "duplicate" / "qualification.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes((reports / "0" / "qualification.json").read_bytes())
    with pytest.raises(evidence.QualificationEvidenceError, match="exactly 19"):
        evidence.aggregate_reports(
            version="0.2.3",
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            reports_dir=reports,
            output_path=tmp_path / "aggregate.json",
            expected_skips_path=baseline,
        )
    duplicate.unlink()

    report = reports / "0" / "qualification.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    data["baseline_sha256"] = "0" * 64
    report.write_bytes(canonical_json(data))
    with pytest.raises(evidence.QualificationEvidenceError, match="baseline"):
        evidence.aggregate_reports(
            version="0.2.3",
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            reports_dir=reports,
            output_path=tmp_path / "aggregate.json",
            expected_skips_path=baseline,
        )


def test_aggregate_rejects_noncanonical_v023_cell_report(tmp_path: Path) -> None:
    evidence = load_module()
    baseline = write_baseline(tmp_path / "baseline.json")
    payload = write_payload(tmp_path / "payload.json")
    reports = tmp_path / "reports"
    reports.mkdir()
    junit = write_junit(tmp_path / "pytest.xml", [])
    for index, cell in enumerate(EXPECTED_CELLS):
        evidence.record_cell(
            version="0.2.3",
            cell=cell,
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=reports / str(index) / "qualification.json",
            junit_path=junit,
            expected_skips_path=baseline,
        )
    report = reports / "0" / "qualification.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    report.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(evidence.QualificationEvidenceError, match="canonical"):
        evidence.aggregate_reports(
            version="0.2.3",
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            reports_dir=reports,
            output_path=tmp_path / "aggregate.json",
            expected_skips_path=baseline,
        )


def test_record_preserves_noncanonical_payload_manifest_serialization(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    payload = write_payload(tmp_path / "payload.json")
    data = json.loads(payload.read_text(encoding="utf-8"))
    payload.write_text(json.dumps(data, indent=2), encoding="utf-8")

    result = evidence.record_cell(
        version="0.2.3",
        cell=EXPECTED_CELLS[0],
        source_sha=SOURCE_SHA,
        payload_manifest=payload,
        report_path=tmp_path / "qualification.json",
        junit_path=write_junit(tmp_path / "pytest.xml", []),
        expected_skips_path=write_baseline(tmp_path / "baseline.json"),
    )

    assert result["files"] == FILES


def test_aggregate_rejects_duplicate_cell_with_exactly_nineteen_records(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    baseline = write_baseline(tmp_path / "baseline.json")
    payload = write_payload(tmp_path / "payload.json")
    reports = tmp_path / "reports"
    reports.mkdir()
    junit = write_junit(tmp_path / "pytest.xml", [])
    for index, cell in enumerate(EXPECTED_CELLS):
        evidence.record_cell(
            version="0.2.3",
            cell=cell,
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=reports / str(index) / "qualification.json",
            junit_path=junit,
            expected_skips_path=baseline,
        )
    last = reports / str(len(EXPECTED_CELLS) - 1) / "qualification.json"
    data = json.loads(last.read_text(encoding="utf-8"))
    data["cell"] = EXPECTED_CELLS[0]
    last.write_bytes(canonical_json(data))

    with pytest.raises(evidence.QualificationEvidenceError, match="duplicate"):
        evidence.aggregate_reports(
            version="0.2.3",
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            reports_dir=reports,
            output_path=tmp_path / "aggregate.json",
            expected_skips_path=baseline,
        )


def test_v022_cell_and_aggregate_reports_remain_byte_contract_compatible(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    payload = write_payload(tmp_path / "payload.json", version="0.2.2")
    reports = tmp_path / "reports"
    reports.mkdir()
    v022_files = files_for("0.2.2")

    for index, cell in enumerate(EXPECTED_CELLS):
        report_path = reports / str(index) / "qualification.json"
        report = evidence.record_cell(
            version="0.2.2",
            cell=cell,
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=report_path,
        )
        assert report == {
            "cell": cell,
            "source_sha": SOURCE_SHA,
            "version": "0.2.2",
            "files": v022_files,
            "status": "passed",
        }
        assert report_path.read_bytes() == (
            json.dumps(report, sort_keys=True) + "\n"
        ).encode("utf-8")

    aggregate_path = tmp_path / "aggregate.json"
    aggregate = evidence.aggregate_reports(
        version="0.2.2",
        source_sha=SOURCE_SHA,
        payload_manifest=payload,
        reports_dir=reports,
        output_path=aggregate_path,
    )
    assert aggregate == {
        "schema": "gwexpy-v022-qualification-evidence-v1",
        "source_sha": SOURCE_SHA,
        "version": "0.2.2",
        "files": v022_files,
        "cells": sorted(EXPECTED_CELLS),
    }
    assert aggregate_path.read_bytes() == (
        json.dumps(aggregate, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_record_rejects_unknown_version_cell_and_missing_v023_inputs(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    payload = write_payload(tmp_path / "payload.json")

    with pytest.raises(evidence.QualificationEvidenceError, match="unsupported"):
        evidence.record_cell(
            version="0.2.4",
            cell=EXPECTED_CELLS[0],
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=tmp_path / "report.json",
        )
    with pytest.raises(evidence.QualificationEvidenceError, match="unknown cell"):
        evidence.record_cell(
            version="0.2.2",
            cell="unknown-cell",
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=tmp_path / "report.json",
        )
    with pytest.raises(evidence.QualificationEvidenceError, match="requires"):
        evidence.record_cell(
            version="0.2.3",
            cell=EXPECTED_CELLS[0],
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=tmp_path / "report.json",
        )


def test_record_rejects_payload_filename_for_a_different_version(
    tmp_path: Path,
) -> None:
    evidence = load_module()
    payload = write_payload(tmp_path / "payload.json")
    data = json.loads(payload.read_text(encoding="utf-8"))
    data["files"]["wheel"]["name"] = "gwexpy-0.2.2-py3-none-any.whl"
    payload.write_bytes(canonical_json(data))

    with pytest.raises(evidence.QualificationEvidenceError, match="candidate version"):
        evidence.record_cell(
            version="0.2.3",
            cell=EXPECTED_CELLS[0],
            source_sha=SOURCE_SHA,
            payload_manifest=payload,
            report_path=tmp_path / "report.json",
            junit_path=write_junit(tmp_path / "pytest.xml", []),
            expected_skips_path=write_baseline(tmp_path / "baseline.json"),
        )
