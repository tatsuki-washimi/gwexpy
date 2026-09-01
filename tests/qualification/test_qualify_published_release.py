"""Contract tests for the published-release qualification harness."""

from __future__ import annotations

import importlib.util
import json
import nturl2path
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
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


def _runtime_provenance(*, channel: str = "conda") -> SimpleNamespace:
    return SimpleNamespace(
        installer="conda" if channel == "conda" else "pip",
        channel="conda-forge" if channel == "conda" else "pypi",
        gwpy_version="4.0.2",
        pip_version="26.0",
    )


def _pip_facts(version: str = "0.2.0") -> tuple[list[str], dict[str, Any]]:
    freeze = [f"gwexpy=={version}", "GWpy==4.0.2", "pip==26.0"]
    inspect = {
        "version": "1",
        "pip_version": "26.0",
        "installed": [
            {
                "metadata": {"name": "gwexpy", "version": version},
                "installer": "conda",
            },
            {"metadata": {"name": "GWpy", "version": "4.0.2"}},
        ],
        "environment": {
            "implementation_name": "cpython",
            "implementation_version": platform.python_version(),
            "os_name": os.name,
            "platform_machine": platform.machine(),
            "platform_python_implementation": platform.python_implementation(),
            "platform_release": platform.release(),
            "platform_system": platform.system(),
            "platform_version": platform.version(),
            "python_full_version": platform.python_version(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "sys_platform": sys.platform,
        },
    }
    return freeze, inspect


def _valid_cell_report(qualifier, claims, cell_id: str) -> dict[str, Any]:
    specification = claims.required_cells[cell_id]
    gwpy_version = getattr(specification, "gwpy", "4.0.2")
    sys_platform = {
        "linux": "linux",
        "macos": "darwin",
        "windows": "win32",
    }[specification.platform]
    platform_system = {
        "linux": "Linux",
        "macos": "Darwin",
        "windows": "Windows",
    }[specification.platform]
    artifact = (
        {
            "filename": claims.artifacts[specification.artifact].name,
            "sha256": claims.artifacts[specification.artifact].sha256,
        }
        if specification.channel == "pypi"
        else None
    )
    freeze, inspect = _pip_facts(claims.version)
    inspect["installed"][1]["metadata"]["version"] = gwpy_version
    freeze[1] = f"GWpy=={gwpy_version}"
    inspect["environment"].update(
        {
            "implementation_name": "cpython",
            "implementation_version": f"{specification.python}.9",
            "os_name": "nt" if specification.platform == "windows" else "posix",
            "platform_machine": "AMD64"
            if specification.platform == "windows"
            else "x86_64",
            "platform_python_implementation": "CPython",
            "platform_system": platform_system,
            "python_full_version": f"{specification.python}.9",
            "python_version": specification.python,
            "sys_platform": sys_platform,
        }
    )
    if specification.channel == "pypi":
        expected_artifact = claims.artifacts[specification.artifact]
        freeze[0] = (
            f"gwexpy @ file:///tmp/{expected_artifact.name}"
            f"#sha256={expected_artifact.sha256}"
        )
        inspect["installed"][0].update(
            {
                "installer": "pip",
                "direct_url": {
                    "url": f"file:///tmp/{expected_artifact.name}",
                    "archive_info": {"hash": f"sha256={expected_artifact.sha256}"},
                },
            }
        )
    return {
        "schema": "gwexpy-published-release-cell-report-v2",
        "cell": cell_id,
        "claims_sha256": claims.digest,
        "version": claims.version,
        "python": f"{specification.python}.9 (qualification build)",
        "sys_platform": sys_platform,
        "platform_system": platform_system,
        "platform_machine": "AMD64"
        if specification.platform == "windows"
        else "x86_64",
        "installer": "pip" if specification.channel == "pypi" else "conda",
        "channel": "pypi"
        if specification.channel == "pypi"
        else specification.conda_channel,
        "gwpy_version": gwpy_version,
        "pip_version": "26.0",
        "artifact": artifact,
        "pip_freeze": freeze,
        "pip_inspect": inspect,
        "passed": True,
        "counters": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
        "error": None,
        "duration_seconds": 0,
    }


def _valid_aggregate_fixture(qualifier, tmp_path: Path) -> SimpleNamespace:
    wheel = b"qualification wheel"
    sdist = b"qualification sdist"
    data = json.loads(CLAIMS.read_text(encoding="utf-8"))
    payloads = {"wheel": wheel, "sdist": sdist}
    for kind, payload in payloads.items():
        data["artifacts"][kind]["sha256"] = qualifier.hashlib.sha256(
            payload
        ).hexdigest()
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(data), encoding="utf-8")
    claims = qualifier.load_claims(claims_path)

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for kind, payload in payloads.items():
        (artifacts / claims.artifacts[kind].name).write_bytes(payload)

    pypi_data = {
        "info": {"name": claims.project, "version": claims.version},
        "urls": [
            {
                "filename": artifact.name,
                "packagetype": artifact.packagetype,
                "digests": {"sha256": artifact.sha256},
                "yanked": False,
                "url": f"https://files.pythonhosted.org/packages/{artifact.name}",
            }
            for artifact in claims.artifacts.values()
        ],
    }
    pypi_json = tmp_path / "pypi.json"
    pypi_json.write_text(json.dumps(pypi_data), encoding="utf-8")

    sidecar_data = {
        "schema": claims.payload_sidecar_schema,
        "source_sha": claims.source_sha,
        "version": claims.version,
        "files": {
            kind: {"name": artifact.name, "sha256": artifact.sha256}
            for kind, artifact in claims.artifacts.items()
        },
    }
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps(sidecar_data), encoding="utf-8")

    reports = tmp_path / "reports"
    reports.mkdir()
    for cell_id in claims.required_cells:
        report = _valid_cell_report(qualifier, claims, cell_id)
        (reports / f"{cell_id}.json").write_text(json.dumps(report), encoding="utf-8")
        qualifier._write_junit(reports / f"{cell_id}.xml", [(cell_id, None)])
    return SimpleNamespace(
        claims=claims,
        artifacts=artifacts,
        reports=reports,
        pypi_json=pypi_json,
        sidecar=sidecar,
        json_out=tmp_path / "aggregate.json",
        junit_out=tmp_path / "aggregate.xml",
    )


def _preflight_fixture(qualifier, monkeypatch, tmp_path: Path, *, conda: bool = False):
    import gwexpy

    prefix = tmp_path / "prefix"
    site = prefix / "site-packages"
    package = site / "gwexpy"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("")
    info = site / "gwexpy-0.2.0.dist-info"
    info.mkdir()
    direct = info / "direct_url.json"
    installer = info / "INSTALLER"
    installer.write_text("pip\n")
    artifact: Path | None = tmp_path / "gwexpy-0.2.0-py3-none-any.whl"
    assert artifact is not None
    artifact.write_bytes(b"wheel")
    direct.write_text(
        json.dumps(
            {
                "url": artifact.as_uri(),
                "archive_info": {"hash": f"sha256={qualifier._digest(artifact)}"},
            }
        )
    )

    class Distribution:
        files: Any = (
            Path("gwexpy-0.2.0.dist-info/direct_url.json"),
            Path("gwexpy-0.2.0.dist-info/INSTALLER"),
        )

        def locate_file(self, item):
            return site if str(item) == "." else site / item

    monkeypatch.setattr(gwexpy, "__file__", str(init))
    monkeypatch.setattr(gwexpy, "__version__", "0.2.0")
    monkeypatch.setattr(
        qualifier.importlib.metadata, "distribution", lambda _: Distribution()
    )
    monkeypatch.setattr(
        qualifier.importlib.metadata,
        "version",
        lambda name: "4.0.1" if name == "gwpy" else "0.2.0",
    )
    monkeypatch.setattr(
        qualifier.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(init)),
    )
    monkeypatch.setattr(qualifier.sys, "prefix", str(prefix))
    monkeypatch.setattr(qualifier.sys, "platform", "linux")
    monkeypatch.setattr(
        qualifier.sys, "version_info", SimpleNamespace(major=3, minor=11)
    )
    claims = qualifier.load_claims(CLAIMS)
    cell = claims.required_cells["gwpy-4.0.1-wheel"]
    repo = tmp_path / "repo"
    repo.mkdir()
    if conda:
        cell = claims.required_cells["conda-3.11"]
        direct.unlink()
        Distribution.files = ()
        (prefix / "conda-meta").mkdir()
        (prefix / "conda-meta" / "gwexpy-0.2.0-0.json").write_text(
            json.dumps(
                {
                    "name": "gwexpy",
                    "version": "0.2.0",
                    "channel": "conda-forge",
                    "subdir": "linux-64",
                }
            )
        )
        monkeypatch.setenv("CONDA_PREFIX", str(prefix))
        artifact = None
    return claims, cell, repo, artifact, direct, init, site, prefix


def test_claims_are_strict_and_have_the_complete_first_run_ledger(qualifier) -> None:
    claims = qualifier.load_claims(CLAIMS)
    assert claims.version == "0.2.0"
    assert set(claims.required_cells) == {
        *(
            f"install-ubuntu-{python}-{kind}"
            for python in ("3.11", "3.12", "3.13", "3.14")
            for kind in ("wheel", "sdist")
        ),
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
    }
    assert claims.required_cells["gwpy-4.0.1-wheel"].gwpy == "4.0.1"
    assert claims.required_cells["install-ubuntu-3.11-wheel"].platform == "linux"
    assert claims.required_cells["install-ubuntu-3.11-wheel"].artifact == "wheel"
    assert claims.required_cells["docs-en-ja-3.11-wheel"].channel == "pypi"
    assert claims.required_cells["docs-en-ja-3.11-wheel"].artifact == "wheel"


def test_b0_contract_suite_runs_in_exactly_one_required_cell(qualifier) -> None:
    claims = qualifier.load_claims(CLAIMS)
    b0_selector = (
        "tests/types/test_series_matrix_contract_manifest.py::"
        "test_every_b0_cell_executes_once_through_the_typed_adapter"
    )
    selected = [
        cell_id
        for cell_id, cell in claims.required_cells.items()
        if b0_selector in claims.suites[cell.suite].selectors
    ]
    assert selected == ["gwpy-4.0.2-wheel"]
    assert claims.required_cells["gwpy-4.0.1-wheel"].suite == (
        "release-contracts-compat"
    )
    assert claims.required_cells["gwpy-4.0.2-wheel"].suite == ("release-contracts-full")
    assert claims.required_cells["sdist-3.12-claims"].suite == (
        "release-contracts-compat"
    )
    assert claims.required_cells["conda-3.11"].suite == ("release-contracts-compat")
    assert claims.required_cells["conda-3.14"].suite == "core"

    compat = set(claims.suites["release-contracts-compat"].selectors)
    full = set(claims.suites["release-contracts-full"].selectors)
    assert full - compat == {
        "tests/types/test_series_matrix_contract_manifest.py::"
        "test_b0_manifest_has_a_literal_cell_count_and_unique_ids",
        b0_selector,
    }
    assert compat < full
    assert claims.suites["release-contracts-full"].support_paths == (
        "tests/types/series_matrix_contract_manifest.py",
    )


def test_inside_resolves_both_sides_without_string_prefixes(
    qualifier, tmp_path: Path
) -> None:
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
        data["required_cells"]["bad/../cell"] = next(
            iter(data["required_cells"].values())
        ).copy()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier.load_claims(path)


def test_claims_reject_duplicate_json_keys(qualifier, tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier.load_claims(path)


def test_claims_reject_a_symlink_to_a_valid_manifest(qualifier, tmp_path: Path) -> None:
    target = _claims(tmp_path)
    symlink = tmp_path / "claims-symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(qualifier.QualificationError, match="non-symlink"):
        qualifier.load_claims(symlink)


def test_claims_parse_and_digest_the_same_byte_snapshot(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    path = _claims(tmp_path)
    snapshot = path.read_bytes()

    def forbid_digest_reread(_path):
        raise AssertionError("claims digest must not reread the path")

    monkeypatch.setattr(qualifier, "_digest", forbid_digest_reread)
    claims = qualifier.load_claims(path)
    assert claims.digest == qualifier.hashlib.sha256(snapshot).hexdigest()


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_json_rejects_nonfinite_constants(
    qualifier, tmp_path: Path, constant: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(f'{{"value": {constant}}}', encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier._json_file(path, "test")


def test_file_uri_rejects_authorities_queries_and_fragments(
    qualifier, tmp_path: Path
) -> None:
    artifact = tmp_path / "wheel.whl"
    artifact.write_bytes(b"x")
    assert qualifier._local_file_uri(artifact.as_uri()) == artifact
    assert qualifier._local_file_uri(
        "file:///C:/wheel.whl", lambda value: f"WIN:{value}"
    ) == Path("WIN:/C:/wheel.whl")
    for value in (
        "file://server/share/wheel.whl",
        "file:///tmp/wheel.whl?x=1",
        "file:///tmp/wheel.whl#part",
    ):
        with pytest.raises(qualifier.QualificationError):
            qualifier._local_file_uri(value)


def test_file_uri_decodes_percent_escapes_exactly_once(
    qualifier, tmp_path: Path
) -> None:
    artifact = tmp_path / "literal%2Fwheel.whl"
    artifact.write_bytes(b"x")
    assert "%252F" in artifact.as_uri()
    assert qualifier._local_file_uri(artifact.as_uri()) == artifact


def test_file_uri_rejects_relative_and_nul_paths(qualifier) -> None:
    for value in ("file:relative.whl", "file:///tmp/bad%00name.whl"):
        with pytest.raises(qualifier.QualificationError):
            qualifier._local_file_uri(value)


def test_file_uri_rejects_windows_unc_paths(qualifier) -> None:
    for value in (
        "file:////server/share/wheel.whl",
        "file:///%5Cserver/share/wheel.whl",
        "file:///%5C%5Cserver/share/wheel.whl",
        "file:///%2F%2Fserver/share/wheel.whl",
        "file:////%5Cserver/share/wheel.whl",
    ):
        with pytest.raises(qualifier.QualificationError, match="local file URI"):
            qualifier._local_file_uri(value, nturl2path.url2pathname)


@pytest.mark.parametrize("failure", ("write", "replace"))
def test_atomic_removes_temporary_file_after_failure(
    qualifier, monkeypatch, tmp_path: Path, failure: str
) -> None:
    output = tmp_path / "result.json"
    if failure == "replace":

        def fail_replace(self, target):
            raise OSError("replace failed")

        monkeypatch.setattr(qualifier.Path, "replace", fail_replace)
    else:
        named_temporary_file = qualifier.tempfile.NamedTemporaryFile

        @contextmanager
        def fail_write(*args, **kwargs):
            with named_temporary_file(*args, **kwargs) as stream:

                class BrokenWriter:
                    name = stream.name

                    def write(self, content: str) -> None:
                        stream.write(content[:1])
                        stream.flush()
                        raise OSError("write failed")

                yield BrokenWriter()

        monkeypatch.setattr(qualifier.tempfile, "NamedTemporaryFile", fail_write)

    with pytest.raises(OSError, match=failure):
        qualifier._atomic(output, "payload")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"channel": "conda-forge", "subdir": "linux-64"}, "conda-forge"),
        (
            {
                "channel": "https://conda.anaconda.org/conda-forge/linux-64",
                "subdir": "linux-64",
            },
            "conda-forge",
        ),
        ({"channel": "https://evil/conda-forge/linux-64", "subdir": "linux-64"}, None),
        ({"channel": "conda-forge", "subdir": "evil"}, None),
        (
            {
                "channel": "https://conda.anaconda.org/conda-forge/linux-64?x=1",
                "subdir": "linux-64",
            },
            None,
        ),
    ],
)
def test_conda_channel_is_canonical_and_rejects_malicious_urls(
    qualifier, record, expected
) -> None:
    assert qualifier._conda_channel(record) == expected


@pytest.mark.parametrize(
    "channel, extra", [("pypi", "conda_channel"), ("docs", "gwpy"), ("conda", "gwpy")]
)
def test_cell_schema_rejects_cross_channel_fields(
    qualifier, tmp_path: Path, channel: str, extra: str
) -> None:
    path = _claims(tmp_path)
    data = json.loads(path.read_text())
    cell = data["required_cells"]["install-ubuntu-3.11-wheel"]
    cell["channel"] = channel
    cell["artifact"] = "none" if channel != "pypi" else "wheel"
    cell[extra] = "conda-forge" if extra == "conda_channel" else "4.0.1"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier.load_claims(path)


def test_artifact_directory_rejects_symlinks_extras_and_wrong_hash(
    qualifier, tmp_path: Path
) -> None:
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
    claims = qualifier.load_claims(CLAIMS)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    monkeypatch.setattr(qualifier, "MAX_ARTIFACT", 1)
    for item in claims.artifacts.values():
        (directory / item.name).write_bytes(b"xx")
    with pytest.raises(qualifier.QualificationError, match="size"):
        qualifier.verify_artifact_directory(claims, directory)


def test_pypi_and_sidecar_corroboration_fail_closed(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    pypi = {"info": {"name": "gwexpy", "version": "0.2.0"}, "urls": []}
    with pytest.raises(qualifier.QualificationError):
        qualifier.validate_pypi_json(claims, pypi)
    sidecar = {
        "schema": claims.payload_sidecar_schema,
        "source_sha": "0" * 40,
        "version": claims.version,
        "files": {},
    }
    with pytest.raises(qualifier.QualificationError):
        qualifier.validate_payload_sidecar(claims, sidecar)


def test_pypi_rejects_yanked_or_wrong_distribution_facts(qualifier) -> None:
    claims = qualifier.load_claims(CLAIMS)
    urls = []
    for artifact in claims.artifacts.values():
        urls.append(
            {
                "filename": artifact.name,
                "packagetype": artifact.packagetype,
                "digests": {"sha256": artifact.sha256},
                "yanked": True,
                "url": f"https://files.pythonhosted.org/packages/{artifact.name}",
            }
        )
    data = {"info": {"name": "gwexpy", "version": claims.version}, "urls": urls}
    with pytest.raises(qualifier.QualificationError, match="yanked"):
        qualifier.validate_pypi_json(claims, data)


def test_pypi_rejects_missing_yanked_flag_and_extra_non_yanked_file(qualifier) -> None:
    claims = qualifier.load_claims(CLAIMS)
    urls = [
        {
            "filename": item.name,
            "packagetype": item.packagetype,
            "digests": {"sha256": item.sha256},
            "yanked": False,
            "url": f"https://files.pythonhosted.org/a/{item.name}",
        }
        for item in claims.artifacts.values()
    ]
    data = {"info": {"name": "gwexpy", "version": claims.version}, "urls": urls}
    del urls[0]["yanked"]
    with pytest.raises(qualifier.QualificationError):
        qualifier.validate_pypi_json(claims, data)
    urls[0]["yanked"] = False
    urls.append(
        {
            "filename": "other.whl",
            "packagetype": "bdist_wheel",
            "digests": {"sha256": "0" * 64},
            "yanked": False,
            "url": "https://files.pythonhosted.org/a/other.whl",
        }
    )
    with pytest.raises(qualifier.QualificationError):
        qualifier.validate_pypi_json(claims, data)
    urls.pop()
    for entry in urls:
        entry["yanked"] = False
    urls[0]["digests"] = {"sha256": "0" * 64}
    with pytest.raises(qualifier.QualificationError, match="do not match"):
        qualifier.validate_pypi_json(claims, data)


def test_aggregate_accepts_a_full_valid_19_cell_ledger(
    qualifier, tmp_path: Path
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    assert len(fixture.claims.required_cells) == 19
    assert qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        fixture.json_out,
        fixture.junit_out,
    )
    result = json.loads(fixture.json_out.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["errors"] == {}
    assert qualifier.parse_junit(fixture.junit_out)["tests"] == 23


def test_aggregate_accepts_conda_pip_as_a_local_direct_reference(
    qualifier, tmp_path: Path
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    path = fixture.reports / "conda-3.11.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["pip_freeze"][2] = "pip @ file:///home/conda/feedstock_root/pip/work"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        fixture.json_out,
        fixture.junit_out,
    )


def test_cell_report_writer_and_aggregate_share_the_same_size_bound(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    original_json_file = qualifier._json_file
    cell_limits: list[int] = []

    def observed_json_file(path, label, limit=qualifier.MAX_INPUT):
        if label == "cell JSON":
            cell_limits.append(limit)
        return original_json_file(path, label, limit)

    monkeypatch.setattr(qualifier, "_json_file", observed_json_file)
    assert qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        fixture.json_out,
        fixture.junit_out,
    )
    assert cell_limits == [qualifier.MAX_CELL_REPORT] * 19
    oversized = tmp_path / "oversized.json"
    with pytest.raises(qualifier.QualificationError, match="output limit"):
        qualifier._result(oversized, {"fact": "x" * 100}, limit=32)
    assert not oversized.exists()


def test_aggregate_checks_artifacts_pypi_sidecar_and_reports_independently(
    qualifier, tmp_path: Path
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    wheel = fixture.claims.artifacts["wheel"]
    (fixture.artifacts / wheel.name).write_bytes(b"corrupt artifact")

    pypi = json.loads(fixture.pypi_json.read_text(encoding="utf-8"))
    pypi["urls"][0]["digests"]["sha256"] = "0" * 64
    fixture.pypi_json.write_text(json.dumps(pypi), encoding="utf-8")

    sidecar = json.loads(fixture.sidecar.read_text(encoding="utf-8"))
    sidecar["source_sha"] = "0" * 40
    fixture.sidecar.write_text(json.dumps(sidecar), encoding="utf-8")

    forged_cell = "gwpy-4.0.1-wheel"
    report_path = fixture.reports / f"{forged_cell}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["sys_platform"] = "forged"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert not qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        fixture.json_out,
        fixture.junit_out,
    )
    errors = json.loads(fixture.json_out.read_text(encoding="utf-8"))["errors"]
    assert forged_cell in errors
    assert {"identity-artifacts", "identity-pypi", "identity-sidecar"} <= set(errors)
    junit, _ = qualifier._junit_counts(fixture.junit_out, clean=False)
    assert junit["failures"] >= 4


def test_aggregate_requires_pypi_and_payload_identity_evidence(
    qualifier, tmp_path: Path
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    assert not qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        None,
        None,
        fixture.json_out,
        fixture.junit_out,
    )
    errors = json.loads(fixture.json_out.read_text(encoding="utf-8"))["errors"]
    assert {"identity-pypi", "identity-sidecar"} <= set(errors)
    junit, _ = qualifier._junit_counts(fixture.junit_out, clean=False)
    assert junit["failures"] == 2


def test_aggregate_refuses_to_overwrite_an_input_report(
    qualifier, tmp_path: Path
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    report = fixture.reports / "conda-3.11.json"
    original = report.read_bytes()
    assert not qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        report,
        fixture.junit_out,
    )
    assert report.read_bytes() == original
    assert not fixture.junit_out.exists()


def test_aggregate_refuses_to_write_inside_an_input_directory(
    qualifier, tmp_path: Path
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    output = fixture.reports / "summary.json"
    original_entries = {path.name for path in fixture.reports.iterdir()}
    assert not qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        output,
        fixture.junit_out,
    )
    assert {path.name for path in fixture.reports.iterdir()} == original_entries
    assert not fixture.junit_out.exists()


def test_aggregate_json_is_the_final_commit_marker(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)

    def fail_junit(*_args, **_kwargs):
        raise OSError("JUnit write failed")

    monkeypatch.setattr(qualifier, "_write_junit", fail_junit)
    with pytest.raises(OSError, match="JUnit write failed"):
        qualifier.aggregate(
            fixture.claims,
            fixture.artifacts,
            fixture.reports,
            fixture.pypi_json,
            fixture.sidecar,
            fixture.json_out,
            fixture.junit_out,
        )
    assert not fixture.json_out.exists()


@pytest.mark.parametrize(
    "forgery",
    (
        "python",
        "sys-platform",
        "platform-system",
        "platform-machine",
        "installer",
        "channel",
        "gwpy",
        "pip-version",
        "artifact-name",
        "artifact-hash",
        "artifact-extra",
        "empty-freeze",
        "freeze-version",
        "freeze-control",
        "freeze-oversize",
        "inspect-version",
        "inspect-gwexpy",
        "inspect-gwpy",
        "inspect-extra",
        "inspect-type",
        "inspect-duplicate",
        "inspect-environment",
        "inspect-os-name",
        "inspect-implementation-version",
        "inspect-implementation-name",
        "inspect-installer",
        "inspect-direct-url-hash",
        "freeze-channel",
        "freeze-hash",
        "freeze-gwpy",
        "freeze-pip",
        "freeze-duplicate-gwpy",
        "conda-channel-facts",
        "unknown-field",
    ),
)
def test_aggregate_rejects_forged_v2_runtime_facts(
    qualifier, tmp_path: Path, forgery: str
) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    cell_id = "conda-3.11" if forgery == "conda-channel-facts" else "gwpy-4.0.1-wheel"
    path = fixture.reports / f"{cell_id}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if forgery == "python":
        report["python"] = "3.12.9 (forged)"
    elif forgery == "sys-platform":
        report["sys_platform"] = "darwin"
    elif forgery == "platform-system":
        report["platform_system"] = "Darwin"
    elif forgery == "platform-machine":
        report["platform_machine"] = ""
    elif forgery == "installer":
        report["installer"] = "editable"
    elif forgery == "channel":
        report["channel"] = "conda-forge"
    elif forgery == "gwpy":
        report["gwpy_version"] = "4.0.2"
    elif forgery == "pip-version":
        report["pip_version"] = "99.0"
    elif forgery == "artifact-name":
        report["artifact"]["filename"] = "other.whl"
    elif forgery == "artifact-hash":
        report["artifact"]["sha256"] = "0" * 64
    elif forgery == "artifact-extra":
        report["artifact"]["url"] = "file:///forged"
    elif forgery == "empty-freeze":
        report["pip_freeze"] = []
    elif forgery == "freeze-version":
        report["pip_freeze"][0] = "gwexpy==9.9.9"
    elif forgery == "freeze-control":
        report["pip_freeze"][0] = "gwexpy==0.2.0\nforged==1"
    elif forgery == "freeze-oversize":
        report["pip_freeze"] = ["gwexpy==0.2.0"] + [
            f"package-{index}==1" for index in range(qualifier.MAX_FREEZE_LINES)
        ]
    elif forgery == "inspect-version":
        report["pip_inspect"]["version"] = "2"
    elif forgery == "inspect-gwexpy":
        report["pip_inspect"]["installed"][0]["metadata"]["version"] = "9.9.9"
    elif forgery == "inspect-gwpy":
        report["pip_inspect"]["installed"][1]["metadata"]["version"] = "9.9.9"
    elif forgery == "inspect-extra":
        report["pip_inspect"]["unknown"] = True
    elif forgery == "inspect-type":
        report["pip_inspect"]["installed"] = {}
    elif forgery == "inspect-duplicate":
        report["pip_inspect"]["installed"].append(
            {"metadata": {"name": "GWEXPY", "version": "0.2.0"}}
        )
    elif forgery == "inspect-environment":
        report["pip_inspect"]["environment"]["sys_platform"] = "darwin"
    elif forgery == "inspect-os-name":
        report["pip_inspect"]["environment"]["os_name"] = "nt"
    elif forgery == "inspect-implementation-version":
        report["pip_inspect"]["environment"]["implementation_version"] = "9.9.9"
    elif forgery == "inspect-implementation-name":
        report["pip_inspect"]["environment"]["implementation_name"] = "pypy"
    elif forgery == "inspect-installer":
        report["pip_inspect"]["installed"][0]["installer"] = "conda"
    elif forgery == "inspect-direct-url-hash":
        report["pip_inspect"]["installed"][0]["direct_url"]["archive_info"] = {
            "hash": f"sha256={'0' * 64}"
        }
    elif forgery == "freeze-channel":
        report["pip_freeze"][0] = "gwexpy==0.2.0"
    elif forgery == "freeze-hash":
        report["pip_freeze"][0] = (
            report["pip_freeze"][0].rsplit("=", 1)[0] + "=" + "0" * 64
        )
    elif forgery == "freeze-gwpy":
        report["pip_freeze"][1] = "GWpy==999.0.0"
    elif forgery == "freeze-pip":
        report["pip_freeze"][2] = "pip==999.0"
    elif forgery == "freeze-duplicate-gwpy":
        report["pip_freeze"].append(report["pip_freeze"][1])
    elif forgery == "conda-channel-facts":
        report["pip_freeze"][0] = "gwexpy @ file:///tmp/forged.whl"
        report["pip_inspect"]["installed"][0].update(
            {
                "installer": "pip",
                "direct_url": {
                    "url": "file:///tmp/forged.whl",
                    "archive_info": {"hash": f"sha256={'0' * 64}"},
                },
            }
        )
    else:
        report["unknown"] = True
    path.write_text(json.dumps(report), encoding="utf-8")

    assert not qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        fixture.json_out,
        fixture.junit_out,
    )
    errors = json.loads(fixture.json_out.read_text(encoding="utf-8"))["errors"]
    assert cell_id in errors


def test_aggregate_rejects_counter_mismatch(qualifier, tmp_path: Path) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    changed = fixture.reports / "conda-3.11.json"
    report = json.loads(changed.read_text(encoding="utf-8"))
    report["counters"]["tests"] = 2
    changed.write_text(json.dumps(report), encoding="utf-8")
    assert not qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        fixture.json_out,
        fixture.junit_out,
    )


def test_aggregate_extra_report(qualifier, tmp_path: Path) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    (fixture.reports / "extra.txt").write_text("x", encoding="utf-8")
    assert not qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        fixture.json_out,
        fixture.junit_out,
    )
    errors = json.loads(fixture.json_out.read_text(encoding="utf-8"))["errors"]
    assert "extra" in " ".join(errors.values())


def test_aggregate_boolean_counters(qualifier, tmp_path: Path) -> None:
    fixture = _valid_aggregate_fixture(qualifier, tmp_path)
    changed = fixture.reports / "conda-3.11.json"
    data = json.loads(changed.read_text(encoding="utf-8"))
    data["counters"] = {
        "tests": True,
        "failures": False,
        "errors": False,
        "skipped": False,
    }
    changed.write_text(json.dumps(data), encoding="utf-8")
    assert not qualifier.aggregate(
        fixture.claims,
        fixture.artifacts,
        fixture.reports,
        fixture.pypi_json,
        fixture.sidecar,
        fixture.json_out,
        fixture.junit_out,
    )


def test_parse_junit_rejects_dtd_zero_skips_and_failures(
    qualifier, tmp_path: Path
) -> None:
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

    utf16 = tmp_path / "utf16-dtd.xml"
    utf16.write_bytes(
        (
            "<?xml version='1.0' encoding='UTF-16'?>"
            "<!DOCTYPE x [<!ENTITY y 'z'>]>"
            "<testsuite tests='1' failures='0' errors='0' skipped='0'>"
            "<testcase/></testsuite>"
        ).encode("utf-16")
    )
    with pytest.raises(qualifier.QualificationError, match="DTD"):
        qualifier.parse_junit(utf16)


def test_junit_accepts_only_pytest_compatible_root_shapes(
    qualifier, tmp_path: Path
) -> None:
    suite_root = tmp_path / "suite.xml"
    suite_root.write_text(
        "<testsuite tests='1' failures='0' errors='0' skipped='0'>"
        "<testcase name='one'/></testsuite>",
        encoding="utf-8",
    )
    assert qualifier.parse_junit(suite_root)["tests"] == 1

    suites_root = tmp_path / "suites.xml"
    suites_root.write_text(
        "<testsuites tests='2' failures='0' errors='0' skipped='0'>"
        "<testsuite tests='1' failures='0' errors='0' skipped='0'>"
        "<testcase name='one'/></testsuite>"
        "<testsuite tests='1' failures='0' errors='0' skipped='0'>"
        "<testcase name='two'/></testsuite></testsuites>",
        encoding="utf-8",
    )
    assert qualifier.parse_junit(suites_root)["tests"] == 2

    suites_without_root_counters = tmp_path / "suites-no-counters.xml"
    suites_without_root_counters.write_text(
        "<testsuites><testsuite tests='1' failures='0' errors='0' skipped='0'>"
        "<testcase name='one'/></testsuite></testsuites>",
        encoding="utf-8",
    )
    assert qualifier.parse_junit(suites_without_root_counters)["tests"] == 1


@pytest.mark.parametrize(
    "payload",
    (
        "<wrapper><testsuite tests='1'><testcase/></testsuite></wrapper>",
        "<testsuites><testsuite tests='1'><testsuite tests='1'>"
        "<testcase/></testsuite></testsuite></testsuites>",
        "<testsuites><testcase/><testsuite tests='1'><testcase/></testsuite>"
        "</testsuites>",
        "<testsuites tests='1' failures='1' errors='0' skipped='0'>"
        "<testsuite tests='1' failures='0' errors='0' skipped='0'>"
        "<testcase/></testsuite><failure/></testsuites>",
        "<testsuite tests='1' failures='1'><testcase><wrapper>"
        "<failure/></wrapper></testcase></testsuite>",
        "<testsuites tests='2'><testsuite tests='1'><testcase/>"
        "</testsuite></testsuites>",
        "<testsuite tests='1' failures='0' errors='0' skipped='0'>"
        "<properties><evil/></properties><testcase/></testsuite>",
    ),
)
def test_junit_rejects_nested_misplaced_or_forged_results(
    qualifier, tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "forged.xml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier._junit_counts(path, clean=False)


@pytest.mark.parametrize(
    "payload",
    (
        "<testsuite tests='1' failures='0' errors='0'><testcase/></testsuite>",
        "<testsuite tests='١' failures='0' errors='0' skipped='0'>"
        "<testcase/></testsuite>",
        "<testsuite tests='1' failures='1' errors='0' skipped='1'>"
        "<testcase><failure/><skipped/></testcase></testsuite>",
        "<testsuites tests='1'><testsuite tests='1' failures='0' errors='0' "
        "skipped='0'><testcase/></testsuite></testsuites>",
        "<testsuites tests='2' failures='0' errors='0' skipped='0'>"
        "<testsuite tests='1' failures='0' errors='0' skipped='0'>"
        "<testcase/></testsuite></testsuites>",
    ),
)
def test_junit_rejects_incomplete_non_ascii_multiple_or_wrong_counters(
    qualifier, tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "counters.xml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(qualifier.QualificationError):
        qualifier._junit_counts(path, clean=False)


def test_junit_reconciles_every_result_element(qualifier, tmp_path: Path) -> None:
    path = tmp_path / "results.xml"
    path.write_text(
        "<testsuites tests='4' failures='1' errors='1' skipped='1'>"
        "<testsuite tests='4' failures='1' errors='1' skipped='1'>"
        "<testcase/><testcase><failure/></testcase>"
        "<testcase><error/></testcase><testcase><skipped/></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    counters, _ = qualifier._junit_counts(path, clean=False)
    assert counters == {"tests": 4, "failures": 1, "errors": 1, "skipped": 1}


def test_junit_writer_replaces_every_xml_1_0_forbidden_character(
    qualifier, tmp_path: Path
) -> None:
    path = tmp_path / "sanitized.xml"
    qualifier._write_junit(path, [("case", "bad-\ud800-\ufffe-\uffff")])
    counters, raw = qualifier._junit_counts(path, clean=False)
    assert counters["failures"] == 1
    assert b"bad-?-?-?" in raw


def test_write_junit_treats_empty_error_as_a_failure(qualifier, tmp_path: Path) -> None:
    path = tmp_path / "empty-error.xml"
    qualifier._write_junit(path, [("empty", "")])
    counters, raw = qualifier._junit_counts(path, clean=False)
    assert counters["failures"] == 1
    assert b"<failure" in raw


def test_report_oversize(qualifier, monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "report.xml"
    path.write_text("x" * 20)
    monkeypatch.setattr(qualifier, "MAX_OUTPUT", 10)
    with pytest.raises(qualifier.QualificationError, match="too large"):
        qualifier.parse_junit(path)


def test_bounded_runner_caps_merged_output_and_preserves_nonzero_exit(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(qualifier, "MAX_OUTPUT", 128)
    completed = qualifier._bounded_run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; print('stdout', flush=True); "
            "sys.stderr.buffer.write(b'\\xff' * 1000); sys.exit(7)",
        ],
        cwd=tmp_path,
        env={},
        timeout=5,
    )
    assert completed.returncode == 7
    assert completed.output.startswith("stdout\n")
    assert len(completed.output.encode("utf-8")) <= 128
    assert completed.truncated is True


def test_bounded_runner_kills_and_reaps_on_timeout(qualifier, tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        qualifier._bounded_run(
            [
                sys.executable,
                "-I",
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            cwd=tmp_path,
            env={},
            timeout=0.1,
        )
    assert time.monotonic() - started < 5


def test_bounded_runner_kills_descendants_on_timeout(qualifier, tmp_path: Path) -> None:
    marker = tmp_path / "grandchild-survived"
    grandchild = (
        "import time; from pathlib import Path; time.sleep(0.5); "
        f"Path({str(marker)!r}).write_text('alive')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-I', '-c', {grandchild!r}]); "
        "time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        qualifier._bounded_run(
            [sys.executable, "-I", "-c", parent],
            cwd=tmp_path,
            env={},
            timeout=0.1,
        )
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert not marker.exists()


def test_bounded_runner_kills_descendants_after_leader_exits(
    qualifier, tmp_path: Path
) -> None:
    marker = tmp_path / "orphan-survived"
    grandchild = (
        "import time; from pathlib import Path; time.sleep(0.5); "
        f"Path({str(marker)!r}).write_text('alive')"
    )
    parent = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-I', '-c', {grandchild!r}])"
    )
    started = time.monotonic()
    completed = qualifier._bounded_run(
        [sys.executable, "-I", "-c", parent],
        cwd=tmp_path,
        env={},
        timeout=5,
    )
    assert completed.returncode == 0
    assert completed.truncated is True
    assert time.monotonic() - started < 1
    time.sleep(0.6)
    assert not marker.exists()


def test_bounded_runner_kills_silent_descendants_after_leader_exits(
    qualifier, tmp_path: Path
) -> None:
    marker = tmp_path / "silent-orphan-survived"
    grandchild = (
        "import time; from pathlib import Path; time.sleep(0.5); "
        f"Path({str(marker)!r}).write_text('alive')"
    )
    parent = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-I', '-c', {grandchild!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
    )
    completed = qualifier._bounded_run(
        [sys.executable, "-I", "-c", parent],
        cwd=tmp_path,
        env={},
        timeout=5,
    )
    assert completed.returncode == 0
    assert completed.truncated is True
    time.sleep(0.6)
    assert not marker.exists()


@pytest.mark.parametrize("failure", ("thread-start", "wait"))
def test_bounded_runner_cleans_up_after_unexpected_internal_failure(
    qualifier, monkeypatch, tmp_path: Path, failure: str
) -> None:
    class FakeStdout:
        closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        pid = 424242
        stdout = FakeStdout()
        killed = False
        waited = False

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            if timeout is not None and failure == "wait":
                raise RuntimeError("wait failed")
            self.waited = True
            return -9

    class FakeThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            if failure == "thread-start":
                raise RuntimeError("thread start failed")

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    process = FakeProcess()
    groups: list[int] = []
    monkeypatch.setattr(qualifier.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(qualifier.threading, "Thread", FakeThread)
    monkeypatch.setattr(qualifier.os, "killpg", lambda pid, _signal: groups.append(pid))
    with pytest.raises(RuntimeError, match="failed"):
        qualifier._bounded_run(
            [sys.executable, "-c", "pass"], cwd=tmp_path, env={}, timeout=1
        )
    assert groups == [process.pid]
    assert process.killed and process.waited and process.stdout.closed


def test_pip_evidence_uses_isolated_bounded_commands(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    freeze, inspect = _pip_facts()

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = (
            "\n".join(freeze) + "\n" if "freeze" in command else json.dumps(inspect)
        )
        return SimpleNamespace(returncode=0, output=output, truncated=False)

    monkeypatch.setattr(qualifier, "_bounded_run", fake_run)
    actual_freeze, actual_inspect = qualifier._pip_evidence(
        {"HOME": str(tmp_path)}, timeout=30
    )
    assert actual_freeze == freeze
    assert actual_inspect == inspect
    assert [call[0][1:4] for call in calls] == [
        ["-I", "-m", "pip"],
        ["-I", "-m", "pip"],
    ]
    assert "--all" in calls[0][0]
    assert "--local" in calls[1][0]


def test_run_cell_records_preflight_failure_and_writes_both_outputs(
    qualifier, tmp_path: Path
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    artifact = tmp_path / claims.artifacts["wheel"].name
    artifact.write_bytes(b"not the published wheel")
    json_out = tmp_path / "cell.json"
    junit_out = tmp_path / "cell.xml"
    assert not qualifier.run_cell(
        claims, "install-ubuntu-3.11-wheel", ROOT, artifact, json_out, junit_out
    )
    report = json.loads(json_out.read_text())
    assert report["passed"] is False
    assert report["artifact"] == {
        "filename": artifact.name,
        "sha256": qualifier._digest(artifact),
    }
    assert "artifact" in junit_out.read_text(encoding="utf-8")


def test_run_cell_rejects_source_shadowed_origin_and_records_it(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    import gwexpy

    claims = qualifier.load_claims(CLAIMS)
    installed_version = qualifier.importlib.metadata.version

    def version_for_claims(name: str) -> str:
        if name.lower() == "gwexpy":
            return claims.version
        return installed_version(name)

    monkeypatch.setattr(qualifier.importlib.metadata, "version", version_for_claims)
    monkeypatch.setattr(gwexpy, "__version__", claims.version)
    json_out = tmp_path / "cell.json"
    junit_out = tmp_path / "cell.xml"
    assert not qualifier.run_cell(claims, "conda-3.11", ROOT, None, json_out, junit_out)
    assert "source-shadowed" in json.loads(json_out.read_text())["error"]


def test_run_cell_stages_and_executes_a_real_selector(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repository = tmp_path / "repo"
    test_file = repository / "tests" / "test_trivial.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_staged():\n    assert True\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/test_trivial.py::test_staged",), support_paths=(), timeout=30
    )
    claims.required_cells["conda-3.11"].suite = "core"
    provenance = _runtime_provenance()
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: provenance)
    monkeypatch.setattr(
        qualifier, "_pip_evidence", lambda *_args, **_kwargs: _pip_facts()
    )
    json_out, junit_out = tmp_path / "result.json", tmp_path / "result.xml"
    assert qualifier.run_cell(
        claims, "conda-3.11", repository, None, json_out, junit_out
    )
    assert json.loads(json_out.read_text())["counters"]["tests"] == 1
    assert qualifier.parse_junit(junit_out)["tests"] == 1


def test_run_cell_stages_support_for_a_relative_test_import(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repository = tmp_path / "repo"
    tests = repository / "tests" / "types"
    tests.mkdir(parents=True)
    (tests / "test_relative.py").write_text(
        "from .support import VALUE\n\ndef test_relative(): assert VALUE == 480\n",
        encoding="utf-8",
    )
    (tests / "support.py").write_text("VALUE = 480\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/types/test_relative.py::test_relative",),
        support_paths=("tests/types/support.py",),
        timeout=30,
    )
    claims.required_cells["conda-3.11"].suite = "core"
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: _runtime_provenance())
    monkeypatch.setattr(
        qualifier, "_pip_evidence", lambda *_args, **_kwargs: _pip_facts()
    )
    json_out, junit_out = tmp_path / "result.json", tmp_path / "result.xml"

    assert qualifier.run_cell(
        claims, "conda-3.11", repository, None, json_out, junit_out
    )
    assert qualifier.parse_junit(junit_out)["tests"] == 1


def test_run_cell_v2_records_actual_runtime_facts_and_isolates_pytest(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repository = tmp_path / "repo"
    source = repository / "tests" / "test_one.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_one(): assert True\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/test_one.py",), support_paths=(), timeout=30
    )
    claims.required_cells["conda-3.11"].suite = "core"
    provenance = _runtime_provenance()
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: provenance)
    monkeypatch.setattr(
        qualifier, "_pip_evidence", lambda *_args, **_kwargs: _pip_facts()
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "malicious"))
    observed: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        junit = Path(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("--junitxml=")
            )
        )
        junit.write_text(
            "<testsuite tests='1' failures='0' errors='0' skipped='0'>"
            "<testcase/></testsuite>",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, output="one passed", truncated=False)

    monkeypatch.setattr(qualifier, "_bounded_run", fake_run)
    json_out, junit_out = tmp_path / "result.json", tmp_path / "result.xml"
    assert qualifier.run_cell(
        claims, "conda-3.11", repository, None, json_out, junit_out
    )
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["schema"] == "gwexpy-published-release-cell-report-v2"
    assert report["python"] == sys.version
    assert report["sys_platform"] == sys.platform
    assert report["platform_system"] == platform.system()
    assert report["platform_machine"] == platform.machine()
    assert report["installer"] == "conda"
    assert report["channel"] == "conda-forge"
    assert report["gwpy_version"] == "4.0.2"
    assert report["pip_version"] == "26.0"
    assert report["artifact"] is None
    assert report["pip_freeze"]
    assert report["pip_inspect"]["installed"]
    assert observed["command"][1] == "-I"
    assert "-P" not in observed["command"]
    assert observed["environment"]["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in observed["environment"]


def test_run_cell_v2_records_the_exact_selected_artifact(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    artifact_payload = b"selected wheel"
    claims_data = json.loads(CLAIMS.read_text(encoding="utf-8"))
    claims_data["artifacts"]["wheel"]["sha256"] = qualifier.hashlib.sha256(
        artifact_payload
    ).hexdigest()
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(claims_data), encoding="utf-8")
    claims = qualifier.load_claims(claims_path)
    artifact = tmp_path / claims.artifacts["wheel"].name
    artifact.write_bytes(artifact_payload)

    repository = tmp_path / "repo"
    source = repository / "tests" / "test_one.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_one(): assert True\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/test_one.py",), support_paths=(), timeout=30
    )
    monkeypatch.setattr(
        qualifier, "_preflight", lambda *_: _runtime_provenance(channel="pypi")
    )
    monkeypatch.setattr(
        qualifier, "_pip_evidence", lambda *_args, **_kwargs: _pip_facts()
    )

    def fake_run(command, **_kwargs):
        junit = Path(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("--junitxml=")
            )
        )
        qualifier._write_junit(junit, [("one", None)])
        return SimpleNamespace(returncode=0, output="one passed", truncated=False)

    monkeypatch.setattr(qualifier, "_bounded_run", fake_run)
    json_out, junit_out = tmp_path / "result.json", tmp_path / "result.xml"
    assert qualifier.run_cell(
        claims,
        "install-ubuntu-3.11-wheel",
        repository,
        artifact,
        json_out,
        junit_out,
    )
    assert json.loads(json_out.read_text(encoding="utf-8"))["artifact"] == {
        "filename": artifact.name,
        "sha256": qualifier._digest(artifact),
    }


def test_run_cell_normalizes_an_empty_exception(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    claims = qualifier.load_claims(CLAIMS)

    class EmptyFailure(Exception):
        pass

    def fail(*_args):
        raise EmptyFailure()

    monkeypatch.setattr(qualifier, "_preflight", fail)
    json_out, junit_out = tmp_path / "failure.json", tmp_path / "failure.xml"
    assert not qualifier.run_cell(claims, "conda-3.11", ROOT, None, json_out, junit_out)
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["error"] == "EmptyFailure"
    assert "EmptyFailure" in junit_out.read_text(encoding="utf-8")


def test_run_cell_rejects_equivalent_output_paths(qualifier, tmp_path: Path) -> None:
    claims = qualifier.load_claims(CLAIMS)
    output = tmp_path / "same.json"
    assert not qualifier.run_cell(claims, "conda-3.11", ROOT, None, output, output)
    assert not output.exists()


def test_outputs_reject_case_variants_on_case_insensitive_platforms(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(qualifier.sys, "platform", "darwin")
    with pytest.raises(qualifier.QualificationError, match="distinct"):
        qualifier._distinct_outputs(tmp_path / "Result.JSON", tmp_path / "result.json")
    with pytest.raises(qualifier.QualificationError, match="overwrite"):
        qualifier._disjoint_outputs(
            tmp_path / "REPORT.JSON",
            tmp_path / "result.xml",
            [tmp_path / "report.json"],
        )


@pytest.mark.parametrize("collision", ("claims", "artifact", "selector"))
def test_run_cell_refuses_to_overwrite_inputs(
    qualifier, monkeypatch, tmp_path: Path, collision: str
) -> None:
    claims_path = _claims(tmp_path)
    claims = qualifier.load_claims(claims_path)
    repository = tmp_path / "repo"
    selector = repository / "tests" / "test_one.py"
    selector.parent.mkdir(parents=True)
    selector.write_text("def test_one(): assert True\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/test_one.py",), support_paths=(), timeout=30
    )
    claims.required_cells["conda-3.11"].suite = "core"
    artifact = tmp_path / claims.artifacts["wheel"].name
    artifact.write_bytes(b"not-the-published-artifact")
    if collision == "claims":
        cell_id, selected, output = "conda-3.11", None, claims_path
    elif collision == "artifact":
        cell_id, selected, output = (
            "install-ubuntu-3.11-wheel",
            artifact,
            artifact,
        )
    else:
        cell_id, selected, output = "conda-3.11", None, selector
    original = output.read_bytes()
    monkeypatch.setattr(
        qualifier,
        "_preflight",
        lambda *_: (_ for _ in ()).throw(qualifier.QualificationError("stop")),
    )
    assert not qualifier.run_cell(
        claims,
        cell_id,
        repository,
        selected,
        output,
        tmp_path / f"{collision}.xml",
    )
    assert output.read_bytes() == original


def test_main_error_fallback_does_not_overwrite_invalid_claims(
    qualifier, tmp_path: Path
) -> None:
    claims = tmp_path / "invalid-claims.json"
    claims.write_text("{invalid", encoding="utf-8")
    original = claims.read_bytes()
    assert (
        qualifier.main(
            [
                "run-cell",
                "--claims",
                str(claims),
                "--cell",
                "conda-3.11",
                "--repo-root",
                str(tmp_path),
                "--json-out",
                str(claims),
                "--junit-out",
                str(tmp_path / "failure.xml"),
            ]
        )
        == 1
    )
    assert claims.read_bytes() == original
    assert not (tmp_path / "failure.xml").exists()


def test_non_pypi_cell_rejects_artifact_and_traversal_selector(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"x")
    assert not qualifier.run_cell(
        claims, "conda-3.11", ROOT, artifact, tmp_path / "a.json", tmp_path / "a.xml"
    )
    claims.suites["release-claims"] = SimpleNamespace(
        selectors=("../escape.py",), support_paths=(), timeout=30
    )
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: None)
    assert not qualifier.run_cell(
        claims, "conda-3.11", ROOT, None, tmp_path / "b.json", tmp_path / "b.xml"
    )


@pytest.mark.parametrize(
    "kind", ("selected_symlink", "parent_symlink", "file_oversize", "total_oversize")
)
def test_staged_path_bounds_and_symlinks(
    qualifier, monkeypatch, tmp_path: Path, kind: str
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repo = tmp_path / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    target = tmp_path / "target.py"
    target.write_text("def test_x(): assert True\n")
    source = tests / "one.py"
    source.write_text("def test_x(): assert True\n")
    if kind == "selected_symlink":
        source.unlink()
        source.symlink_to(target)
    if kind == "parent_symlink":
        source.unlink()
        tests.rmdir()
        tests.symlink_to(tmp_path)
        (tmp_path / "one.py").write_text("def test_x(): assert True\n")
    if kind == "file_oversize":
        monkeypatch.setattr(qualifier, "MAX_STAGE_FILE", 1)
    if kind == "total_oversize":
        monkeypatch.setattr(qualifier, "MAX_STAGE_TOTAL", 1)
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/one.py",), support_paths=(), timeout=1
    )
    claims.required_cells["conda-3.11"].suite = "core"
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: None)
    output = tmp_path / "x.json"
    assert not qualifier.run_cell(
        claims, "conda-3.11", repo, None, output, tmp_path / "x.xml"
    )
    assert (
        "unsafe" in json.loads(output.read_text())["error"]
        if "symlink" in kind
        else "size limit" in json.loads(output.read_text())["error"]
    )


@pytest.mark.parametrize(
    "outcome", ("timeout", "missing-junit", "bad-junit", "skipped-junit", "nonzero")
)
def test_run_cell_failure_branches_write_evidence(
    qualifier, monkeypatch, tmp_path: Path, outcome: str
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repo = tmp_path / "repo"
    source = repo / "tests" / "test_one.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_one(): assert True\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/test_one.py",), support_paths=(), timeout=1
    )
    claims.required_cells["conda-3.11"].suite = "core"
    provenance = _runtime_provenance()
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: provenance)
    monkeypatch.setattr(
        qualifier, "_pip_evidence", lambda *_args, **_kwargs: _pip_facts()
    )

    def fake_run(command, **kwargs):
        junit = Path(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("--junitxml=")
            )
        )
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if outcome == "bad-junit":
            junit.write_text("<bad>", encoding="utf-8")
        elif outcome == "skipped-junit":
            junit.write_text(
                "<testsuite tests='1' skipped='1'><testcase><skipped/></testcase></testsuite>",
                encoding="utf-8",
            )
        elif outcome != "missing-junit":
            return SimpleNamespace(returncode=1, output="fail", truncated=False)
        return SimpleNamespace(returncode=0, output="", truncated=False)

    monkeypatch.setattr(qualifier, "_bounded_run", fake_run)
    json_out, junit_out = tmp_path / f"{outcome}.json", tmp_path / f"{outcome}.xml"
    assert not qualifier.run_cell(claims, "conda-3.11", repo, None, json_out, junit_out)
    assert json.loads(json_out.read_text())["passed"] is False and junit_out.exists()


def test_run_cell_preserves_valid_pytest_junit_on_nonzero_exit(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repo = tmp_path / "repo"
    source = repo / "tests" / "test_two.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_one(): assert True\n", encoding="utf-8")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/test_two.py",), support_paths=(), timeout=30
    )
    claims.required_cells["conda-3.11"].suite = "core"
    provenance = _runtime_provenance()
    monkeypatch.setattr(qualifier, "_preflight", lambda *_: provenance)
    monkeypatch.setattr(
        qualifier, "_pip_evidence", lambda *_args, **_kwargs: _pip_facts()
    )

    def fail_with_junit(command, **_kwargs):
        junit = Path(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("--junitxml=")
            )
        )
        junit.write_text(
            "<testsuite tests='2' failures='1' errors='0' skipped='0'>"
            "<testcase name='passing'/><testcase name='failing'>"
            "<failure message='failed'>failed</failure>"
            "</testcase></testsuite>",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, output="one failed", truncated=False)

    monkeypatch.setattr(qualifier, "_bounded_run", fail_with_junit)
    json_out, junit_out = tmp_path / "result.json", tmp_path / "result.xml"
    assert not qualifier.run_cell(claims, "conda-3.11", repo, None, json_out, junit_out)
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["error"].startswith("QualificationError: pytest exit 1")
    assert report["counters"] == {
        "tests": 2,
        "failures": 1,
        "errors": 0,
        "skipped": 0,
    }
    counters, raw = qualifier._junit_counts(junit_out, clean=False)
    assert counters == report["counters"]
    assert raw.count(b"<testcase") == 2


def test_preflight_accepts_exact_pep610_and_rejects_missing_or_editable(
    qualifier, tmp_path: Path
) -> None:
    monkeypatch = pytest.MonkeyPatch()
    claims, cell, repo, artifact, direct, *_ = _preflight_fixture(
        qualifier, monkeypatch, tmp_path
    )
    provenance = qualifier._preflight(claims, cell, repo, artifact)
    assert provenance.installer == "pip" and provenance.channel == "pypi"
    assert provenance.gwpy_version == "4.0.1" and provenance.pip_version == "0.2.0"
    direct.write_text(
        json.dumps(
            {
                "url": artifact.as_uri(),
                "archive_info": {
                    "hash": f"sha256={qualifier._digest(artifact)}",
                    "hashes": {"sha256": "0" * 64},
                },
            }
        )
    )
    with pytest.raises(qualifier.QualificationError, match="does not attest"):
        qualifier._preflight(claims, cell, repo, artifact)
    direct.unlink()
    with pytest.raises(qualifier.QualificationError, match="direct_url"):
        qualifier._preflight(claims, cell, repo, artifact)
    direct.write_text(
        json.dumps({"url": artifact.as_uri(), "dir_info": {"editable": True}})
    )
    with pytest.raises(qualifier.QualificationError, match="editable"):
        qualifier._preflight(claims, cell, repo, artifact)
    monkeypatch.undo()


def test_preflight_rejects_origin_root_interpreter_and_gwpy_mismatches(
    qualifier, tmp_path: Path
) -> None:
    monkeypatch = pytest.MonkeyPatch()
    claims, cell, repo, artifact, _, init, site, prefix = _preflight_fixture(
        qualifier, monkeypatch, tmp_path
    )
    qualifier._preflight(claims, cell, repo, artifact)
    monkeypatch.setattr(
        qualifier.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(repo / "x.py")),
    )
    with pytest.raises(qualifier.QualificationError):
        qualifier._preflight(claims, cell, repo, artifact)
    monkeypatch.setattr(
        qualifier.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(init)),
    )
    monkeypatch.setattr(qualifier.sys, "platform", "win32")
    with pytest.raises(qualifier.QualificationError):
        qualifier._preflight(claims, cell, repo, artifact)
    monkeypatch.setattr(qualifier.sys, "platform", "linux")
    monkeypatch.setattr(
        qualifier.sys, "version_info", SimpleNamespace(major=3, minor=12)
    )
    with pytest.raises(qualifier.QualificationError):
        qualifier._preflight(claims, cell, repo, artifact)
    monkeypatch.setattr(
        qualifier.sys, "version_info", SimpleNamespace(major=3, minor=11)
    )
    monkeypatch.setattr(
        qualifier.importlib.metadata,
        "version",
        lambda name: "4.0.2" if name == "gwpy" else "0.2.0",
    )
    with pytest.raises(qualifier.QualificationError):
        qualifier._preflight(claims, cell, repo, artifact)
    monkeypatch.setattr(
        qualifier.importlib.metadata,
        "version",
        lambda name: "4.0.1" if name == "gwpy" else "0.2.0",
    )

    class OutsideDistribution:
        files = (Path("gwexpy-0.2.0.dist-info/direct_url.json"),)

        def locate_file(self, item):
            return site / Path(item) if str(item) else tmp_path / "outside-site"

    monkeypatch.setattr(
        qualifier.importlib.metadata, "distribution", lambda _: OutsideDistribution()
    )
    with pytest.raises(qualifier.QualificationError, match="source-shadowed"):
        qualifier._preflight(claims, cell, repo, artifact)
    monkeypatch.undo()


def test_preflight_conda_binds_active_prefix_and_record(
    qualifier, tmp_path: Path, monkeypatch
) -> None:
    claims, cell, repo, artifact, _, *_, prefix = _preflight_fixture(
        qualifier, monkeypatch, tmp_path, conda=True
    )
    provenance = qualifier._preflight(claims, cell, repo, artifact)
    assert provenance.installer == "conda" and provenance.channel == "conda-forge"
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "other"))
    with pytest.raises(qualifier.QualificationError, match="CONDA_PREFIX"):
        qualifier._preflight(claims, cell, repo, artifact)


def test_run_cell_records_second_preflight_failure(
    qualifier, monkeypatch, tmp_path: Path
) -> None:
    claims = qualifier.load_claims(CLAIMS)
    repo = tmp_path / "repo"
    source = repo / "tests" / "one.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_one(): assert True\n")
    claims.suites["core"] = SimpleNamespace(
        selectors=("tests/one.py",), support_paths=(), timeout=1
    )
    claims.required_cells["conda-3.11"].suite = "core"
    calls = []

    def preflight(*_):
        calls.append(1)
        if len(calls) == 2:
            raise qualifier.QualificationError("post-run origin changed")

    def run(command, **_):
        Path(
            next(x.split("=", 1)[1] for x in command if x.startswith("--junitxml="))
        ).write_text(
            "<testsuite tests='1' failures='0' errors='0' skipped='0'><testcase/></testsuite>"
        )
        return SimpleNamespace(returncode=0, output="", truncated=False)

    monkeypatch.setattr(qualifier, "_preflight", preflight)
    monkeypatch.setattr(
        qualifier, "_pip_evidence", lambda *_args, **_kwargs: _pip_facts()
    )
    monkeypatch.setattr(qualifier, "_bounded_run", run)
    output, junit = tmp_path / "o.json", tmp_path / "o.xml"
    assert not qualifier.run_cell(claims, "conda-3.11", repo, None, output, junit)
    assert (
        "post-run origin changed" in json.loads(output.read_text())["error"]
        and "post-run origin changed" in junit.read_text()
    )


def test_clean_python_helper_is_isolated_sanitized_and_bounded(monkeypatch) -> None:
    path = ROOT / "tests" / "qualification" / "test_v020_release_claims.py"
    spec = importlib.util.spec_from_file_location("release_claim_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("PYTHONPATH", "/forged/source")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._clean_python("pass")
    assert observed["command"][1] == "-I"
    assert observed["env"]["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in observed["env"]
    assert 0 < observed["timeout"] <= 60
    assert "stdout" not in observed
    assert "stderr" not in observed
    runtime = Path(observed["cwd"])
    for key in ("HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR"):
        assert observed["env"][key] == str(runtime)
    assert observed["env"]["XDG_CACHE_HOME"] == str(runtime / "cache")
    assert observed["env"]["XDG_CONFIG_HOME"] == str(runtime / "config")
