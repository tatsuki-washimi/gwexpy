"""Contract tests for the published-release qualification harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
        "conda-3.11", "conda-3.14", "scientific-3.11-wheel", "docs-en-ja-3.11",
    }
    assert claims.required_cells["gwpy-4.0.1-wheel"].gwpy == "4.0.1"
    assert claims.required_cells["install-ubuntu-3.11-wheel"].platform == "linux"
    assert claims.required_cells["install-ubuntu-3.11-wheel"].artifact == "wheel"
    assert claims.required_cells["docs-en-ja-3.11"].channel == "docs"
    assert claims.required_cells["docs-en-ja-3.11"].artifact == "none"


def test_inside_resolves_both_sides_without_string_prefixes(qualifier, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    child = root / "child"
    child.mkdir()
    sibling = tmp_path / "root-escape"
    sibling.mkdir()
    assert qualifier._inside(child, root)
    assert not qualifier._inside(sibling, root)


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


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_json_rejects_nonfinite_constants(qualifier, tmp_path: Path, constant: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(f'{{"value": {constant}}}', encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier._json_file(path, "test")


def test_file_uri_rejects_authorities_queries_and_fragments(qualifier, tmp_path: Path) -> None:
    artifact = tmp_path / "wheel.whl"
    artifact.write_bytes(b"x")
    assert qualifier._local_file_uri(artifact.as_uri()) == artifact
    assert qualifier._local_file_uri("file:///C:/wheel.whl", lambda value: f"WIN:{value}") == Path("WIN:/C:/wheel.whl")
    for value in ("file://server/share/wheel.whl", "file:///tmp/wheel.whl?x=1", "file:///tmp/wheel.whl#part"):
        with pytest.raises(qualifier.QualificationError):
            qualifier._local_file_uri(value)


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
    with pytest.raises(qualifier.QualificationError, match="yanked"):
        qualifier.validate_pypi_json(claims, data)


def test_pypi_rejects_missing_yanked_flag_and_extra_non_yanked_file(qualifier) -> None:
    claims = qualifier.load_claims(CLAIMS)
    urls = [{"filename": item.name, "packagetype": item.packagetype, "digests": {"sha256": item.sha256}, "yanked": False, "url": f"https://files.pythonhosted.org/a/{item.name}"} for item in claims.artifacts.values()]
    data = {"info": {"name": "gwexpy", "version": claims.version}, "urls": urls}
    del urls[0]["yanked"]
    with pytest.raises(qualifier.QualificationError):
        qualifier.validate_pypi_json(claims, data)
    urls[0]["yanked"] = False
    urls.append({"filename": "other.whl", "packagetype": "bdist_wheel", "digests": {"sha256": "0" * 64}, "yanked": False, "url": "https://files.pythonhosted.org/a/other.whl"})
    with pytest.raises(qualifier.QualificationError):
        qualifier.validate_pypi_json(claims, data)
    urls.pop()
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
    assert "missing" in " ".join(report["errors"].values()).lower()


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
            "platform": specification.platform,
            "gwpy": getattr(specification, "gwpy", None),
            "channel": specification.channel,
            "artifact": claims.artifacts[specification.artifact].name
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


def test_run_cell_stages_and_executes_a_real_selector(qualifier, monkeypatch, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repository = tmp_path / "repo"
    test_file = repository / "tests" / "test_trivial.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_staged():\n    assert True\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/test_trivial.py::test_staged",), support_paths=(), timeout=30
    )
    claims.required_cells["conda-3.11"].suite = "core"
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: None)
    json_out, junit_out = tmp_path / "result.json", tmp_path / "result.xml"
    assert qualifier.run_cell(claims, "conda-3.11", repository, None, json_out, junit_out)
    assert json.loads(json_out.read_text())["counters"]["tests"] == 1
    assert qualifier.parse_junit(junit_out)["tests"] == 1


def test_run_cell_rejects_equivalent_output_paths(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    output = tmp_path / "same.json"
    assert not qualifier.run_cell(claims, "conda-3.11", ROOT, None, output, output)
    assert not output.exists()


def test_non_pypi_cell_rejects_artifact_and_traversal_selector(qualifier, monkeypatch, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"x")
    assert not qualifier.run_cell(claims, "conda-3.11", ROOT, artifact, tmp_path / "a.json", tmp_path / "a.xml")
    claims.suites["release-claims"] = SimpleNamespace(selectors=("../escape.py",), support_paths=(), timeout=30)
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: None)
    assert not qualifier.run_cell(claims, "conda-3.11", ROOT, None, tmp_path / "b.json", tmp_path / "b.xml")
