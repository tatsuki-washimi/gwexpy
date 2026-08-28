"""Contract tests for the published-release qualification harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "qualify_published_release.py"
CLAIMS = ROOT / "tests" / "qualification" / "v0.2.0-claims.json"


@pytest.fixture(scope="module")
def qualifier():
    spec = importlib.util.spec_from_file_location("qualifier", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _claims(tmp_path: Path) -> Path:
    data = json.loads(CLAIMS.read_text(encoding="utf-8"))
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_claims_are_strict_and_have_the_complete_first_run_ledger(qualifier) -> None:
    claims = qualifier.load_claims(CLAIMS)
    assert claims.version == "0.2.0"
    assert set(claims.required_cells) == {
        *(f"install-ubuntu-{python}-{kind}" for python in ("3.11", "3.12", "3.13", "3.14") for kind in ("wheel", "sdist")),
        "install-macos-3.11-wheel", "install-macos-3.14-wheel",
        "install-windows-3.11-wheel", "install-windows-3.14-wheel",
        "gwpy-4.0.1-wheel", "gwpy-4.0.2-wheel", "sdist-3.12-claims",
        "conda-3.11", "conda-3.14", "scientific-3.11-wheel", "docs-en-ja-3.11-wheel",
    }


@pytest.mark.parametrize("case", ("unknown", "version", "unsafe-cell"))
def test_claims_reject_unknown_invalid_and_unsafe_values(
    qualifier, tmp_path: Path, case: str
) -> None:
    path = _claims(tmp_path)
    data = json.loads(path.read_text())
    if case == "unknown":
        data["unexpected"] = True
    elif case == "version":
        data["version"] = "0.2"
    else:
        data["required_cells"]["bad/../cell"] = next(iter(data["required_cells"].values())).copy()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier.load_claims(path)


def test_claims_reject_duplicate_json_keys(qualifier, tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier.load_claims(path)


def test_artifact_directory_rejects_symlinks_extras_and_wrong_hash(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    for artifact in claims.artifacts.values():
        (directory / artifact.name).write_bytes(b"wrong")
    with pytest.raises(qualifier.QualificationError, match="hash"):
        qualifier.verify_artifact_directory(claims, directory)
    (directory / "extra").write_text("x", encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier.verify_artifact_directory(claims, directory)


def test_artifact_directory_rejects_a_symlink(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    for artifact in claims.artifacts.values():
        (directory / artifact.name).symlink_to(target)
    with pytest.raises(qualifier.QualificationError, match="non-regular"):
        qualifier.verify_artifact_directory(claims, directory)


def test_pypi_and_sidecar_corroboration_fail_closed(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    pypi = {"info": {"name": "gwexpy", "version": "0.2.0"}, "urls": []}
    with pytest.raises(qualifier.QualificationError):
        qualifier.validate_pypi_json(claims, pypi)
    sidecar = {"schema": claims.payload_sidecar_schema, "source_sha": "0" * 40,
               "version": claims.version, "files": {}}
    with pytest.raises(qualifier.QualificationError):
        qualifier.validate_payload_sidecar(claims, sidecar)


def test_pypi_rejects_yanked_or_wrong_distribution_facts(qualifier) -> None:
    claims = qualifier.load_claims(CLAIMS)
    urls = []
    for artifact in claims.artifacts.values():
        urls.append({"filename": artifact.name, "packagetype": artifact.packagetype,
                     "digests": {"sha256": artifact.sha256}, "yanked": True,
                     "url": f"https://files.pythonhosted.org/packages/{artifact.name}"})
    data = {"info": {"name": "gwexpy", "version": claims.version}, "urls": urls}
    with pytest.raises(qualifier.QualificationError, match="non-yanked"):
        qualifier.validate_pypi_json(claims, data)
    for entry in urls:
        entry["yanked"] = False
    urls[0]["digests"] = {"sha256": "0" * 64}
    with pytest.raises(qualifier.QualificationError, match="do not match"):
        qualifier.validate_pypi_json(claims, data)


def test_aggregate_writes_evidence_on_missing_extra_and_counter_mismatch(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    json_out = tmp_path / "aggregate.json"
    junit_out = tmp_path / "aggregate.xml"
    result = qualifier.aggregate(claims, artifacts, reports, None, None, json_out, junit_out)
    assert result is False
    assert json_out.is_file() and junit_out.is_file()
    report = json.loads(json_out.read_text())
    assert report["passed"] is False
    assert "missing" in report["error"].lower()


def test_aggregate_rejects_counter_mismatch(qualifier, monkeypatch, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    monkeypatch.setattr(qualifier, "verify_artifact_directory", lambda *_: {})
    for cell, specification in claims.required_cells.items():
        counters = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
        report = {
            "schema": "gwexpy-published-release-cell-report-v1",
            "cell": cell,
            "claims_sha256": claims.digest,
            "version": claims.version,
            "python": f"{specification.python}.0",
            "artifact": claims.artifacts[specification.artifact_kind].name
            if specification.channel == "pypi"
            else None,
            "passed": True,
            "counters": counters,
            "error": None,
            "duration_seconds": 0,
        }
        (reports / f"{cell}.json").write_text(json.dumps(report), encoding="utf-8")
        qualifier._write_junit(reports / f"{cell}.xml", [(cell, None)])
    changed = reports / "conda-3.11.json"
    report = json.loads(changed.read_text())
    report["counters"]["tests"] = 2
    changed.write_text(json.dumps(report), encoding="utf-8")
    assert not qualifier.aggregate(claims, artifacts, reports, None, None, tmp_path / "out.json", tmp_path / "out.xml")


def test_parse_junit_rejects_dtd_zero_skips_and_failures(qualifier, tmp_path: Path) -> None:
    for name, payload in {
        "dtd.xml": "<!DOCTYPE x [<!ENTITY y 'z'>]><testsuite tests='1'/>",
        "zero.xml": "<testsuite tests='0' failures='0' errors='0' skipped='0'/>",
        "skip.xml": "<testsuite tests='1' failures='0' errors='0' skipped='1'/>",
        "fail.xml": "<testsuite tests='1' failures='1' errors='0' skipped='0'/>",
        "malformed.xml": "<testsuite tests='1'>",
    }.items():
        path = tmp_path / name
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(qualifier.QualificationError):
            qualifier.parse_junit(path)


def test_run_cell_records_preflight_failure_and_writes_both_outputs(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    artifact = tmp_path / claims.artifacts["wheel"].name
    artifact.write_bytes(b"not the published wheel")
    json_out = tmp_path / "cell.json"
    junit_out = tmp_path / "cell.xml"
    assert not qualifier.run_cell(claims, "install-ubuntu-3.11-wheel", ROOT, artifact, json_out, junit_out)
    assert json.loads(json_out.read_text())["passed"] is False
    assert "artifact" in junit_out.read_text(encoding="utf-8")


def test_run_cell_rejects_source_shadowed_origin_and_records_it(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    json_out = tmp_path / "cell.json"
    junit_out = tmp_path / "cell.xml"
    assert not qualifier.run_cell(claims, "conda-3.11", ROOT, None, json_out, junit_out)
    assert "source-shadowed" in json.loads(json_out.read_text())["error"]
