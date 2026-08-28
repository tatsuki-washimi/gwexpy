#!/usr/bin/env python3
"""Strict, offline evidence harness for a published GWexpy release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as etree
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MAX_INPUT = 4 * 1024 * 1024
MAX_OUTPUT = 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
CELL = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")


class QualificationError(ValueError):
    """Raised when qualification input or evidence is not trustworthy."""


class _DuplicateKey(ValueError):
    pass


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for key, value in pairs:
        if key in answer:
            raise _DuplicateKey(key)
        answer[key] = value
    return answer


def _regular(path: Path, label: str, limit: int = MAX_INPUT) -> Path:
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise QualificationError(f"{label} must be a regular non-symlink file")
    if path.stat().st_size > limit:
        raise QualificationError(f"{label} is too large")
    return path.resolve()


def _json_file(path: Path, label: str) -> dict[str, Any]:
    path = _regular(path, label)
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey) as exc:
        raise QualificationError(f"invalid {label}") from exc
    if not isinstance(data, dict):
        raise QualificationError(f"{label} must be an object")
    return data


def _keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise QualificationError(f"{label} has missing or unknown keys")
    return value


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value or value.startswith("-"):
        raise QualificationError(f"unsafe {label}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise QualificationError(f"unsafe {label}")
    return value


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().is_relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _artifact(value: object, kind: str, version: str) -> SimpleNamespace:
    entry = _keys(value, {"filename", "packagetype", "sha256"}, "artifact")
    name = _safe_relative(entry["filename"], "artifact filename")
    expected_name = f"gwexpy-{version}-py3-none-any.whl" if kind == "wheel" else f"gwexpy-{version}.tar.gz"
    expected_type = "bdist_wheel" if kind == "wheel" else "sdist"
    if name != expected_name or entry["packagetype"] != expected_type or not isinstance(entry["sha256"], str) or not SHA256.fullmatch(entry["sha256"]):
        raise QualificationError("invalid artifact claim")
    return SimpleNamespace(name=name, packagetype=expected_type, sha256=entry["sha256"])


def load_claims(path: Path | str) -> SimpleNamespace:
    """Load the versioned claims manifest, rejecting any ambiguity."""
    data = _json_file(Path(path), "claims")
    fields = {"schema", "project", "version", "tag", "source_sha", "release_run_id", "repository", "payload_sidecar_schema", "artifacts", "suites", "required_cells"}
    _keys(data, fields, "claims")
    if data["schema"] != "gwexpy-published-release-claims-v1" or data["project"] != "gwexpy" or not isinstance(data["version"], str) or not SEMVER.fullmatch(data["version"]):
        raise QualificationError("invalid release identity")
    if data["tag"] != f"v{data['version']}" or not isinstance(data["source_sha"], str) or not SHA1.fullmatch(data["source_sha"]):
        raise QualificationError("invalid release tag or source SHA")
    if not isinstance(data["release_run_id"], str) or not data["release_run_id"].isdecimal() or not data["release_run_id"]:
        raise QualificationError("invalid release run ID")
    if data["repository"] != "tatsuki-washimi/gwexpy" or not isinstance(data["payload_sidecar_schema"], str) or not data["payload_sidecar_schema"]:
        raise QualificationError("invalid repository or payload schema")
    artifacts = _keys(data["artifacts"], {"wheel", "sdist"}, "artifacts")
    parsed_artifacts = {kind: _artifact(artifacts[kind], kind, data["version"]) for kind in artifacts}
    suites = data["suites"]
    if not isinstance(suites, dict) or not suites:
        raise QualificationError("invalid suites")
    parsed_suites: dict[str, SimpleNamespace] = {}
    for name, suite in suites.items():
        if not CELL.fullmatch(name):
            raise QualificationError("invalid suite name")
        suite = _keys(suite, {"selectors", "support_paths", "timeout"}, "suite")
        selectors, support = suite["selectors"], suite["support_paths"]
        if not isinstance(selectors, list) or not selectors or not isinstance(support, list) or not isinstance(suite["timeout"], int) or not 0 < suite["timeout"] <= 3600:
            raise QualificationError("invalid suite")
        if any(_safe_relative(item, "selector") != item for item in selectors) or any(_safe_relative(item, "support path") != item for item in support) or len(set(selectors + support)) != len(selectors + support):
            raise QualificationError("invalid suite paths")
        parsed_suites[name] = SimpleNamespace(selectors=tuple(selectors), support_paths=tuple(support), timeout=suite["timeout"])
    cells = data["required_cells"]
    if not isinstance(cells, dict) or not cells:
        raise QualificationError("invalid required cells")
    parsed_cells: dict[str, SimpleNamespace] = {}
    for cell_id, cell in cells.items():
        if not CELL.fullmatch(cell_id):
            raise QualificationError("invalid cell ID")
        cell = _keys(cell, {"python", "channel", "artifact_kind", "suite", "required"}, "cell")
        if not isinstance(cell["python"], str) or not re.fullmatch(r"3\.(1[1-4])", cell["python"]) or cell["channel"] not in {"pypi", "conda"} or cell["artifact_kind"] not in {"wheel", "sdist", "none"} or cell["suite"] not in parsed_suites or cell["required"] is not True:
            raise QualificationError("invalid cell")
        if (cell["channel"] == "pypi") != (cell["artifact_kind"] in parsed_artifacts):
            raise QualificationError("invalid cell artifact selection")
        parsed_cells[cell_id] = SimpleNamespace(**cell)
    result = dict(data)
    result.update(artifacts=parsed_artifacts, suites=parsed_suites, required_cells=parsed_cells, digest=_digest(Path(path)))
    return SimpleNamespace(**result)


def verify_artifact_directory(claims: SimpleNamespace, directory: Path | str) -> dict[str, Path]:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise QualificationError("artifact directory must be a real directory")
    entries: dict[str, Path] = {}
    for item in root.iterdir():
        if item.is_symlink() or not stat.S_ISREG(item.stat(follow_symlinks=False).st_mode):
            raise QualificationError("artifact directory contains non-regular entry")
        entries[item.name] = item.resolve()
    expected = {artifact.name for artifact in claims.artifacts.values()}
    if set(entries) != expected:
        raise QualificationError("artifact directory has missing or extra artifacts")
    for artifact in claims.artifacts.values():
        if _digest(entries[artifact.name]) != artifact.sha256:
            raise QualificationError(f"artifact hash mismatch: {artifact.name}")
    return entries


def validate_pypi_json(claims: SimpleNamespace, data: object) -> None:
    if not isinstance(data, dict) or set(data) != {"info", "urls"} or not isinstance(data["info"], dict) or data["info"].get("name") != claims.project or data["info"].get("version") != claims.version or not isinstance(data["urls"], list):
        raise QualificationError("invalid PyPI JSON")
    files = [item for item in data["urls"] if isinstance(item, dict) and not item.get("yanked", False)]
    if len(files) != 2:
        raise QualificationError("PyPI JSON must contain exactly two non-yanked files")
    expected = {(artifact.name, artifact.packagetype, artifact.sha256) for artifact in claims.artifacts.values()}
    actual = set()
    for item in files:
        digest = item.get("digests", {}).get("sha256") if isinstance(item.get("digests"), dict) else None
        url = item.get("url")
        if not isinstance(url, str) or not re.fullmatch(r"https://files\.pythonhosted\.org/.+", url):
            raise QualificationError("PyPI URL is not files.pythonhosted.org HTTPS")
        actual.add((item.get("filename"), item.get("packagetype"), digest))
    if actual != expected:
        raise QualificationError("PyPI artifacts do not match claims")


def validate_payload_sidecar(claims: SimpleNamespace, data: object) -> None:
    sidecar = _keys(data, {"schema", "source_sha", "version", "files"}, "payload sidecar")
    if sidecar["schema"] != claims.payload_sidecar_schema or sidecar["source_sha"] != claims.source_sha or sidecar["version"] != claims.version:
        raise QualificationError("payload sidecar identity mismatch")
    files = _keys(sidecar["files"], {"wheel", "sdist"}, "payload sidecar files")
    for kind, artifact in claims.artifacts.items():
        item = _keys(files[kind], {"name", "sha256"}, "payload sidecar file")
        if item["name"] != artifact.name or item["sha256"] != artifact.sha256:
            raise QualificationError("payload sidecar artifact mismatch")


def _atomic(path: Path, content: str) -> None:
    if path.exists() and path.is_symlink():
        raise QualificationError("output path cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def _write_junit(
    path: Path, cases: list[tuple[str, str | None]], tests: int | None = None
) -> None:
    suite = etree.Element("testsuite", name="gwexpy-published-release", tests=str(tests if tests is not None else len(cases)), failures=str(sum(error is not None for _, error in cases)), errors="0", skipped="0")
    for name, error in cases:
        case = etree.SubElement(suite, "testcase", classname="qualification", name=name)
        if error:
            failure = etree.SubElement(case, "failure", message=error[:4096])
            failure.text = error[:4096]
    _atomic(path, etree.tostring(suite, encoding="unicode") + "\n")


def parse_junit(path: Path | str) -> dict[str, int]:
    source = _regular(Path(path), "JUnit", MAX_OUTPUT)
    raw = source.read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise QualificationError("JUnit must not contain DTD or entities")
    try:
        root = etree.fromstring(raw)
    except etree.ParseError as exc:
        raise QualificationError("malformed JUnit") from exc
    nodes = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not nodes:
        raise QualificationError("JUnit has no testsuite")
    totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for node in nodes:
        for key in totals:
            value = node.attrib.get(key, "0")
            if not value.isdecimal():
                raise QualificationError("invalid JUnit counter")
            totals[key] += int(value)
    if totals["tests"] <= 0 or any(totals[key] for key in ("failures", "errors", "skipped")):
        raise QualificationError("JUnit is not a clean non-empty pass")
    return totals


def _preflight(claims: SimpleNamespace, repo: Path, artifact: Path | None) -> None:
    import gwexpy
    try:
        distribution = importlib.metadata.distribution("gwexpy")
        if importlib.metadata.version("gwexpy") != claims.version or gwexpy.__version__ != claims.version:
            raise QualificationError("installed version does not match claims")
        specification = importlib.util.find_spec("gwexpy")
        if specification is None or specification.origin is None:
            raise QualificationError("gwexpy module has no import origin")
        origin = Path(specification.origin).resolve()
        package_file = Path(gwexpy.__file__ or "").resolve()
        root = Path(str(distribution.locate_file(""))).resolve()
    except (importlib.metadata.PackageNotFoundError, AttributeError) as exc:
        raise QualificationError("gwexpy distribution preflight failed") from exc
    prefix = Path(sys.prefix).resolve()
    if origin != package_file or not _inside(origin, prefix) or not _inside(root, prefix) or _inside(origin, repo):
        raise QualificationError("source-shadowed or editable installation")
    if artifact is not None:
        direct = root / "gwexpy-0.2.0.dist-info" / "direct_url.json"
        data = _json_file(direct, "direct_url metadata")
        url = data.get("url")
        archive = data.get("archive_info")
        if not isinstance(url, str) or Path(url.removeprefix("file://")).resolve() != artifact.resolve() or not isinstance(archive, dict) or archive.get("hash") != f"sha256={_digest(artifact)}":
            raise QualificationError("direct_url metadata does not attest selected artifact")


def _result(path: Path, payload: dict[str, Any]) -> None:
    _atomic(path, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def run_cell(claims: SimpleNamespace, cell_id: str, repo_root: Path, artifact: Path | None, json_out: Path, junit_out: Path) -> bool:
    """Preflight an installed distribution, stage selected tests, and run pytest."""
    started = time.monotonic()
    error: str | None = None
    cell: SimpleNamespace | None = None
    counters: dict[str, int] = {"tests": 0, "failures": 1, "errors": 0, "skipped": 0}
    try:
        cell = claims.required_cells[cell_id]
        repo = Path(repo_root).resolve()
        if not repo.is_dir() or repo.is_symlink():
            raise QualificationError("repo root must be a real directory")
        selected: Path | None = None
        if cell.channel == "pypi":
            if artifact is None:
                raise QualificationError("PyPI cell requires an artifact")
            selected = _regular(Path(artifact), "artifact")
            expected = claims.artifacts[cell.artifact_kind]
            if selected.name != expected.name or _digest(selected) != expected.sha256:
                raise QualificationError("selected artifact does not match claims")
        elif artifact is not None:
            selected = _regular(Path(artifact), "artifact")
        _preflight(claims, repo, selected if cell.channel == "pypi" else None)
        suite = claims.suites[cell.suite]
        with tempfile.TemporaryDirectory(prefix="gwexpy-qualification-") as temporary:
            stage = Path(temporary) / "stage"
            stage.mkdir()
            for relative in (*suite.selectors, *suite.support_paths):
                source = (repo / relative).resolve()
                if not _inside(source, repo) or source.is_symlink() or not source.is_file():
                    raise QualificationError("selected test/support path is unsafe")
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            if (stage / "gwexpy").exists():
                raise QualificationError("staged tree contains source package")
            cwd = Path(temporary) / "cwd"
            cwd.mkdir()
            pytest_junit = Path(temporary) / "pytest.xml"
            config = Path(temporary) / "pytest.ini"
            config.write_text("[pytest]\n", encoding="utf-8")
            environment = os.environ.copy()
            for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
                environment.pop(key, None)
            environment.update({"PYTHONSAFEPATH": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "GWEXPY_POST_RELEASE_QUALIFICATION": "1"})
            command = [sys.executable, "-P", "-m", "pytest", "-c", str(config), "--import-mode=importlib", f"--junitxml={pytest_junit}", *(str(stage / selector) for selector in suite.selectors)]
            process = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=suite.timeout, check=False)
            if process.returncode != 0:
                raise QualificationError(f"pytest exit {process.returncode}: {(process.stderr or process.stdout)[:4096]}")
            counters = parse_junit(pytest_junit)
        _preflight(claims, repo, selected if cell.channel == "pypi" else None)
    except (KeyError, QualificationError, subprocess.TimeoutExpired, OSError) as exc:
        error = str(exc)[:4096]
    passed = error is None
    payload = {"schema": "gwexpy-published-release-cell-report-v1", "cell": cell_id, "claims_sha256": claims.digest, "version": claims.version, "python": sys.version.split()[0], "artifact": artifact.name if cell is not None and cell.channel == "pypi" and artifact else None, "passed": passed, "counters": counters, "error": error, "duration_seconds": round(time.monotonic() - started, 3)}
    _result(json_out, payload)
    _write_junit(junit_out, [(cell_id, error)], counters["tests"] if passed else None)
    return passed


def aggregate(claims: SimpleNamespace, artifact_dir: Path, reports_dir: Path, pypi_json: Path | None, payload_sidecar: Path | None, json_out: Path, junit_out: Path) -> bool:
    """Cross-check a complete cell ledger and write aggregate evidence on failure."""
    errors: list[str] = []
    try:
        verify_artifact_directory(claims, artifact_dir)
        if pypi_json is not None:
            validate_pypi_json(claims, _json_file(pypi_json, "PyPI JSON"))
        if payload_sidecar is not None:
            validate_payload_sidecar(claims, _json_file(payload_sidecar, "payload sidecar"))
        if reports_dir.is_symlink() or not reports_dir.is_dir():
            raise QualificationError("reports directory must be real")
        entries = {item.name: item for item in reports_dir.iterdir()}
        expected = {f"{cell}.json" for cell in claims.required_cells} | {f"{cell}.xml" for cell in claims.required_cells}
        if set(entries) != expected:
            raise QualificationError("missing or extra qualification reports")
        for cell in claims.required_cells:
            report = _json_file(entries[f"{cell}.json"], "cell JSON")
            counters = parse_junit(entries[f"{cell}.xml"])
            selected = claims.required_cells[cell]
            expected_artifact = claims.artifacts[selected.artifact_kind].name if selected.channel == "pypi" else None
            if set(report) != {"schema", "cell", "claims_sha256", "version", "python", "artifact", "passed", "counters", "error", "duration_seconds"} or report["cell"] != cell or report["claims_sha256"] != claims.digest or report["version"] != claims.version or not isinstance(report["python"], str) or not report["python"].startswith(f"{selected.python}.") or report["artifact"] != expected_artifact or report["passed"] is not True or report["counters"] != counters:
                raise QualificationError(f"mismatched evidence for {cell}")
    except QualificationError as exc:
        errors.append(str(exc)[:4096])
    passed = not errors
    _result(json_out, {"schema": "gwexpy-published-release-aggregate-v1", "claims_sha256": claims.digest, "passed": passed, "required_cells": sorted(claims.required_cells), "error": errors[0] if errors else None})
    _write_junit(junit_out, [(cell, errors[0] if errors else None) for cell in sorted(claims.required_cells)])
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-cell")
    run.add_argument("--claims", type=Path, required=True); run.add_argument("--cell", required=True); run.add_argument("--repo-root", type=Path, required=True); run.add_argument("--artifact", type=Path); run.add_argument("--json-out", type=Path, required=True); run.add_argument("--junit-out", type=Path, required=True)
    summary = commands.add_parser("aggregate")
    summary.add_argument("--claims", type=Path, required=True); summary.add_argument("--artifact-dir", type=Path, required=True); summary.add_argument("--reports-dir", type=Path, required=True); summary.add_argument("--pypi-json", type=Path); summary.add_argument("--payload-sidecar", type=Path); summary.add_argument("--json-out", type=Path, required=True); summary.add_argument("--junit-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        claims = load_claims(args.claims)
        passed = run_cell(claims, args.cell, args.repo_root, args.artifact, args.json_out, args.junit_out) if args.command == "run-cell" else aggregate(claims, args.artifact_dir, args.reports_dir, args.pypi_json, args.payload_sidecar, args.json_out, args.junit_out)
    except (QualificationError, OSError) as exc:
        message = str(exc)[:4096]
        try:
            _result(args.json_out, {"schema": "gwexpy-published-release-error-v1", "passed": False, "error": message})
            _write_junit(args.junit_out, [(args.command, message)])
        except (QualificationError, OSError):
            pass
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
