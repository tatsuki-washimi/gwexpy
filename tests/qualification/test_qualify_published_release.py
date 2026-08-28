"""Contract tests for the published-release qualification harness."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def _preflight_fixture(qualifier, monkeypatch, tmp_path: Path, *, conda: bool = False):
    import gwexpy
    prefix=tmp_path/"prefix"; site=prefix/"site-packages"; package=site/"gwexpy"; package.mkdir(parents=True); init=package/"__init__.py"; init.write_text("")
    info=site/"gwexpy-0.2.0.dist-info"; info.mkdir(); direct=info/"direct_url.json"; artifact: Path | None=tmp_path/"gwexpy-0.2.0-py3-none-any.whl"; assert artifact is not None; artifact.write_bytes(b"wheel")
    direct.write_text(json.dumps({"url":artifact.as_uri(),"archive_info":{"hash":f"sha256={qualifier._digest(artifact)}"}}))
    class Distribution:
        files: Any=(Path("gwexpy-0.2.0.dist-info/direct_url.json"),)
        def locate_file(self,item): return site if str(item)=="." else site/item
    monkeypatch.setattr(gwexpy,"__file__",str(init)); monkeypatch.setattr(gwexpy,"__version__","0.2.0")
    monkeypatch.setattr(qualifier.importlib.metadata,"distribution",lambda _:Distribution()); monkeypatch.setattr(qualifier.importlib.metadata,"version",lambda name:"4.0.1" if name=="gwpy" else "0.2.0")
    monkeypatch.setattr(qualifier.importlib.util,"find_spec",lambda _:SimpleNamespace(origin=str(init))); monkeypatch.setattr(qualifier.sys,"prefix",str(prefix)); monkeypatch.setattr(qualifier.sys,"platform","linux"); monkeypatch.setattr(qualifier.sys,"version_info",SimpleNamespace(major=3,minor=11))
    claims=qualifier.load_claims(CLAIMS); cell=claims.required_cells["gwpy-4.0.1-wheel"]; repo=tmp_path/"repo"; repo.mkdir()
    if conda:
        cell=claims.required_cells["conda-3.11"]; direct.unlink(); Distribution.files=(); (prefix/"conda-meta").mkdir(); (prefix/"conda-meta"/"gwexpy-0.2.0-0.json").write_text(json.dumps({"name":"gwexpy","version":"0.2.0","channel":"conda-forge","subdir":"linux-64"})); monkeypatch.setenv("CONDA_PREFIX",str(prefix)); artifact=None
    return claims,cell,repo,artifact,direct,init,site,prefix


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


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"channel": "conda-forge", "subdir": "linux-64"}, "conda-forge"),
        ({"channel": "https://conda.anaconda.org/conda-forge/linux-64", "subdir": "linux-64"}, "conda-forge"),
        ({"channel": "https://evil/conda-forge/linux-64", "subdir": "linux-64"}, None),
        ({"channel": "https://conda.anaconda.org/conda-forge/linux-64?x=1", "subdir": "linux-64"}, None),
    ],
)
def test_conda_channel_is_canonical_and_rejects_malicious_urls(qualifier, record, expected) -> None:
    assert qualifier._conda_channel(record) == expected


@pytest.mark.parametrize("channel, extra", [("pypi", "conda_channel"), ("docs", "gwpy"), ("conda", "gwpy")])
def test_cell_schema_rejects_cross_channel_fields(qualifier, tmp_path: Path, channel: str, extra: str) -> None:
    path = _claims(tmp_path)
    data = json.loads(path.read_text())
    cell = data["required_cells"]["install-ubuntu-3.11-wheel"]
    cell["channel"] = channel
    cell["artifact"] = "none" if channel != "pypi" else "wheel"
    cell[extra] = "conda-forge" if extra == "conda_channel" else "4.0.1"
    path.write_text(json.dumps(data), encoding="utf-8")
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


def test_artifact_oversize(qualifier, monkeypatch, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS); directory = tmp_path / "artifacts"; directory.mkdir()
    monkeypatch.setattr(qualifier, "MAX_ARTIFACT", 1)
    for item in claims.artifacts.values(): (directory / item.name).write_bytes(b"xx")
    with pytest.raises(qualifier.QualificationError, match="size"):
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


def test_aggregate_extra_report(qualifier, monkeypatch, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS); reports = tmp_path / "reports"; reports.mkdir(); (reports / "extra.txt").write_text("x")
    monkeypatch.setattr(qualifier, "verify_artifact_directory", lambda *_: {})
    output = tmp_path / "out.json"
    assert not qualifier.aggregate(claims, tmp_path, reports, None, None, output, tmp_path / "out.xml")
    assert "extra" in " ".join(json.loads(output.read_text())["errors"].values())


def test_aggregate_boolean_counters(qualifier, monkeypatch, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS); reports = tmp_path / "reports"; reports.mkdir(); monkeypatch.setattr(qualifier, "verify_artifact_directory", lambda *_: {})
    for cell, spec in claims.required_cells.items():
        report = {"schema":"gwexpy-published-release-cell-report-v1","cell":cell,"claims_sha256":claims.digest,"version":claims.version,"python":f"{spec.python}.0","platform":spec.platform,"gwpy":getattr(spec,"gwpy",None),"channel":spec.channel,"artifact":claims.artifacts[spec.artifact].name if spec.channel=="pypi" else None,"passed":True,"counters":{"tests":1,"failures":0,"errors":0,"skipped":0},"error":None,"duration_seconds":0}
        (reports/f"{cell}.json").write_text(json.dumps(report)); qualifier._write_junit(reports/f"{cell}.xml",[(cell,None)])
    changed=reports/"conda-3.11.json"; data=json.loads(changed.read_text()); data["counters"]={"tests":True,"failures":False,"errors":False,"skipped":False}; changed.write_text(json.dumps(data))
    assert not qualifier.aggregate(claims,tmp_path,reports,None,None,tmp_path/"o.json",tmp_path/"o.xml")


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


def test_report_oversize(qualifier, monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "report.xml"; path.write_text("x" * 20)
    monkeypatch.setattr(qualifier, "MAX_OUTPUT", 10)
    with pytest.raises(qualifier.QualificationError, match="too large"):
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


@pytest.mark.parametrize("kind", ("selected_symlink", "parent_symlink", "file_oversize", "total_oversize"))
def test_staged_path_bounds_and_symlinks(qualifier, monkeypatch, tmp_path: Path, kind: str) -> None:
    claims=qualifier.load_claims(CLAIMS); repo=tmp_path/"repo"; tests=repo/"tests"; tests.mkdir(parents=True); target=tmp_path/"target.py"; target.write_text("def test_x(): assert True\n")
    source=tests/"one.py"; source.write_text("def test_x(): assert True\n")
    if kind=="selected_symlink": source.unlink(); source.symlink_to(target)
    if kind=="parent_symlink": source.unlink(); tests.rmdir(); tests.symlink_to(tmp_path); (tmp_path/"one.py").write_text("def test_x(): assert True\n")
    if kind=="file_oversize": monkeypatch.setattr(qualifier,"MAX_STAGE_FILE",1)
    if kind=="total_oversize": monkeypatch.setattr(qualifier,"MAX_STAGE_TOTAL",1)
    claims.suites["core"]=SimpleNamespace(selectors=("tests/one.py",),support_paths=(),timeout=1); claims.required_cells["conda-3.11"].suite="core"; monkeypatch.setattr(qualifier,"_preflight",lambda *_:None)
    output=tmp_path/"x.json"; assert not qualifier.run_cell(claims,"conda-3.11",repo,None,output,tmp_path/"x.xml")
    assert ("unsafe" in json.loads(output.read_text())["error"] if "symlink" in kind else "size limit" in json.loads(output.read_text())["error"])


@pytest.mark.parametrize("outcome", ("timeout", "missing-junit", "bad-junit", "skipped-junit", "nonzero"))
def test_run_cell_failure_branches_write_evidence(qualifier, monkeypatch, tmp_path: Path, outcome: str) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repo = tmp_path / "repo"
    source = repo / "tests" / "test_one.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_one(): assert True\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(selectors=("tests/test_one.py",), support_paths=(), timeout=1)
    claims.required_cells["conda-3.11"].suite = "core"
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: None)
    def fake_run(command, **kwargs):
        junit = Path(next(item.split("=", 1)[1] for item in command if item.startswith("--junitxml=")))
        if outcome == "timeout": raise subprocess.TimeoutExpired(command, 1)
        if outcome == "bad-junit": junit.write_text("<bad>", encoding="utf-8")
        elif outcome == "skipped-junit": junit.write_text("<testsuite tests='1' skipped='1'><testcase><skipped/></testcase></testsuite>", encoding="utf-8")
        elif outcome != "missing-junit": return SimpleNamespace(returncode=1, stdout="fail", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(qualifier.subprocess, "run", fake_run)
    json_out, junit_out = tmp_path / f"{outcome}.json", tmp_path / f"{outcome}.xml"
    assert not qualifier.run_cell(claims, "conda-3.11", repo, None, json_out, junit_out)
    assert json.loads(json_out.read_text())["passed"] is False and junit_out.exists()


def test_preflight_accepts_exact_pep610_and_rejects_missing_or_editable(qualifier, tmp_path: Path) -> None:
    monkeypatch=pytest.MonkeyPatch(); claims,cell,repo,artifact,direct,*_= _preflight_fixture(qualifier,monkeypatch,tmp_path)
    qualifier._preflight(claims,cell,repo,artifact); direct.unlink()
    with pytest.raises(qualifier.QualificationError,match="direct_url"): qualifier._preflight(claims,cell,repo,artifact)
    direct.write_text(json.dumps({"url":artifact.as_uri(),"dir_info":{"editable":True}}))
    with pytest.raises(qualifier.QualificationError,match="editable"): qualifier._preflight(claims,cell,repo,artifact)
    monkeypatch.undo()


def test_preflight_rejects_origin_root_interpreter_and_gwpy_mismatches(qualifier, tmp_path: Path) -> None:
    monkeypatch=pytest.MonkeyPatch(); claims,cell,repo,artifact,_,init,site,prefix=_preflight_fixture(qualifier,monkeypatch,tmp_path); qualifier._preflight(claims,cell,repo,artifact)
    monkeypatch.setattr(qualifier.importlib.util,"find_spec",lambda _:SimpleNamespace(origin=str(repo/"x.py")))
    with pytest.raises(qualifier.QualificationError): qualifier._preflight(claims,cell,repo,artifact)
    monkeypatch.setattr(qualifier.importlib.util,"find_spec",lambda _:SimpleNamespace(origin=str(init))); monkeypatch.setattr(qualifier.sys,"platform","win32")
    with pytest.raises(qualifier.QualificationError): qualifier._preflight(claims,cell,repo,artifact)
    monkeypatch.setattr(qualifier.sys,"platform","linux"); monkeypatch.setattr(qualifier.sys,"version_info",SimpleNamespace(major=3,minor=12))
    with pytest.raises(qualifier.QualificationError): qualifier._preflight(claims,cell,repo,artifact)
    monkeypatch.setattr(qualifier.sys,"version_info",SimpleNamespace(major=3,minor=11)); monkeypatch.setattr(qualifier.importlib.metadata,"version",lambda name:"4.0.2" if name=="gwpy" else "0.2.0")
    with pytest.raises(qualifier.QualificationError): qualifier._preflight(claims,cell,repo,artifact)
    monkeypatch.setattr(qualifier.importlib.metadata,"version",lambda name:"4.0.1" if name=="gwpy" else "0.2.0")
    class OutsideDistribution:
        files=(Path("gwexpy-0.2.0.dist-info/direct_url.json"),)
        def locate_file(self,item): return site/Path(item) if str(item) else tmp_path/"outside-site"
    monkeypatch.setattr(qualifier.importlib.metadata,"distribution",lambda _:OutsideDistribution())
    with pytest.raises(qualifier.QualificationError,match="source-shadowed"): qualifier._preflight(claims,cell,repo,artifact)
    monkeypatch.undo()


def test_preflight_conda_binds_active_prefix_and_record(qualifier, tmp_path: Path, monkeypatch) -> None:
    claims,cell,repo,artifact,_,*_,prefix=_preflight_fixture(qualifier,monkeypatch,tmp_path,conda=True); qualifier._preflight(claims,cell,repo,artifact)
    monkeypatch.setenv("CONDA_PREFIX",str(tmp_path/"other"))
    with pytest.raises(qualifier.QualificationError,match="CONDA_PREFIX"): qualifier._preflight(claims,cell,repo,artifact)


def test_run_cell_records_second_preflight_failure(qualifier, monkeypatch, tmp_path: Path) -> None:
    claims=qualifier.load_claims(CLAIMS); repo=tmp_path/"repo"; source=repo/"tests"/"one.py"; source.parent.mkdir(parents=True); source.write_text("def test_one(): assert True\n")
    claims.suites["core"]=SimpleNamespace(selectors=("tests/one.py",),support_paths=(),timeout=1); claims.required_cells["conda-3.11"].suite="core"; calls=[]
    def preflight(*_):
        calls.append(1)
        if len(calls)==2: raise qualifier.QualificationError("post-run origin changed")
    def run(command, **_):
        Path(next(x.split("=",1)[1] for x in command if x.startswith("--junitxml="))).write_text("<testsuite tests='1' failures='0' errors='0' skipped='0'><testcase/></testsuite>")
        return SimpleNamespace(returncode=0,stdout="",stderr="")
    monkeypatch.setattr(qualifier,"_preflight",preflight); monkeypatch.setattr(qualifier.subprocess,"run",run)
    output,junit=tmp_path/"o.json",tmp_path/"o.xml"; assert not qualifier.run_cell(claims,"conda-3.11",repo,None,output,junit)
    assert "post-run origin changed" in json.loads(output.read_text())["error"] and "post-run origin changed" in junit.read_text()
