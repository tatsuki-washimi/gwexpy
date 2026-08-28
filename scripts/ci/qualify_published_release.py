#!/usr/bin/env python3
"""Strict, offline evidence harness for a published GWexpy release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as etree
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MAX_INPUT = 4 * 1024 * 1024
MAX_OUTPUT = 1024 * 1024
MAX_ARTIFACT = 128 * 1024 * 1024
MAX_STAGE_FILE = 2 * 1024 * 1024
MAX_STAGE_TOTAL = 8 * 1024 * 1024
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


def _no_json_constant(value: str) -> None:
    raise QualificationError(f"non-finite JSON constant: {value}")


def _regular(path: Path, label: str, limit: int = MAX_INPUT) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise QualificationError(f"{label} must be a regular non-symlink file") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise QualificationError(f"{label} must be a regular non-symlink file")
    if path.stat().st_size > limit:
        raise QualificationError(f"{label} is too large")
    return path.resolve()


def _json_file(path: Path, label: str) -> dict[str, Any]:
    path = _regular(path, label)
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates, parse_constant=_no_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, QualificationError) as exc:
        raise QualificationError(f"invalid {label}") from exc
    if not isinstance(data, dict):
        raise QualificationError(f"{label} must be an object")
    return data


def _keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise QualificationError(f"{label} has missing or unknown keys")
    return value


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 240 or "\x00" in value or "\\" in value or any(ord(char) < 32 for char in value):
        raise QualificationError(f"unsafe {label}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or any(part in {"", "."} or part.startswith("-") for part in path.parts):
        raise QualificationError(f"unsafe {label}")
    return value


def _inside(child: Path, parent: Path) -> bool:
    return child.resolve().is_relative_to(parent.resolve())


def _has_symlink_component(path: Path, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _local_file_uri(
    url: object, converter: Any = urllib.request.url2pathname
) -> Path:
    if not isinstance(url, str):
        raise QualificationError("direct_url is not a file URI")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
        raise QualificationError("direct_url must be a local file URI")
    return Path(converter(urllib.parse.unquote(parsed.path)))


def _conda_channel(record: dict[str, Any]) -> str | None:
    source = record.get("schannel", record.get("channel"))
    if not isinstance(source, str):
        return None
    if source == "conda-forge":
        return source
    parsed = urllib.parse.urlparse(source)
    subdir = record.get("subdir")
    known = {"linux-64", "linux-aarch64", "osx-64", "osx-arm64", "win-64", "noarch"}
    if not isinstance(subdir, str) or subdir not in known:
        return None
    expected = f"/conda-forge/{subdir}"
    return "conda-forge" if parsed.scheme == "https" and parsed.hostname == "conda.anaconda.org" and parsed.port in {None, 443} and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment and parsed.path == expected else None


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
    if not isinstance(data["release_run_id"], str) or re.fullmatch(r"[1-9][0-9]{0,19}", data["release_run_id"]) is None:
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
        if not CELL.fullmatch(name) or len(name) > 80:
            raise QualificationError("invalid suite name")
        suite = _keys(suite, {"selectors", "support_paths", "timeout"}, "suite")
        selectors, support = suite["selectors"], suite["support_paths"]
        if not isinstance(selectors, list) or not selectors or not isinstance(support, list) or isinstance(suite["timeout"], bool) or not isinstance(suite["timeout"], int) or not 0 < suite["timeout"] <= 3600:
            raise QualificationError("invalid suite")
        if any(_safe_relative(item, "selector") != item for item in selectors) or any(_safe_relative(item, "support path") != item for item in support) or len(set(selectors + support)) != len(selectors + support):
            raise QualificationError("invalid suite paths")
        parsed_suites[name] = SimpleNamespace(selectors=tuple(selectors), support_paths=tuple(support), timeout=suite["timeout"])
    cells = data["required_cells"]
    if not isinstance(cells, dict) or not cells:
        raise QualificationError("invalid required cells")
    parsed_cells: dict[str, SimpleNamespace] = {}
    for cell_id, cell in cells.items():
        if not CELL.fullmatch(cell_id) or len(cell_id) > 80:
            raise QualificationError("invalid cell ID")
        if not isinstance(cell, dict):
            raise QualificationError("cell has missing or unknown keys")
        base = {"python", "platform", "channel", "artifact", "suite", "required"}
        channel = cell.get("channel")
        allowed = base | ({"gwpy"} if channel == "pypi" and "gwpy" in cell else set()) | ({"conda_channel"} if channel == "conda" else set())
        if set(cell) != allowed:
            raise QualificationError("cell has missing or unknown keys")
        if not isinstance(cell["python"], str) or not re.fullmatch(r"3\.(1[1-4])", cell["python"]) or cell["platform"] not in {"linux", "macos", "windows"} or cell["channel"] not in {"pypi", "conda", "docs"} or cell["artifact"] not in {"wheel", "sdist", "none"} or cell["suite"] not in parsed_suites or cell["required"] is not True or ("gwpy" in cell and (not isinstance(cell["gwpy"], str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", cell["gwpy"]) is None)):
            raise QualificationError("invalid cell")
        if (cell["channel"] == "pypi" and cell["artifact"] not in parsed_artifacts) or (cell["channel"] != "pypi" and cell["artifact"] != "none"):
            raise QualificationError("invalid cell artifact selection")
        if cell["channel"] == "conda" and (not isinstance(cell.get("conda_channel"), str) or not cell["conda_channel"]):
            raise QualificationError("conda cell requires exact channel")
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
        if item.stat().st_size > MAX_ARTIFACT:
            raise QualificationError("artifact exceeds size limit")
        entries[item.name] = item.resolve()
    expected = {artifact.name for artifact in claims.artifacts.values()}
    if set(entries) != expected:
        raise QualificationError("artifact directory has missing or extra artifacts")
    for artifact in claims.artifacts.values():
        if _digest(entries[artifact.name]) != artifact.sha256:
            raise QualificationError(f"artifact hash mismatch: {artifact.name}")
    return entries


def validate_pypi_json(claims: SimpleNamespace, data: object) -> None:
    if not isinstance(data, dict) or not {"info", "urls"}.issubset(data) or not isinstance(data["info"], dict) or data["info"].get("name") != claims.project or data["info"].get("version") != claims.version or not isinstance(data["urls"], list):
        raise QualificationError("invalid PyPI JSON")
    if any(not isinstance(item, dict) or type(item.get("yanked")) is not bool for item in data["urls"]):
        raise QualificationError("invalid PyPI file entry")
    expected_names = {artifact.name for artifact in claims.artifacts.values()}
    if any(item.get("yanked", False) and item.get("filename") in expected_names for item in data["urls"]):
        raise QualificationError("PyPI expected artifact is yanked")
    files = [item for item in data["urls"] if not item["yanked"]]
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
    if actual != expected or len(actual) != len(files):
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
    if path.name.startswith("-") or path == path.parent:
        raise QualificationError("unsafe output path")
    parent = path.parent
    chain = [parent, *parent.parents]
    if any(item.exists() and item.is_symlink() for item in chain):
        raise QualificationError("output parent cannot contain symlink")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise QualificationError("output path cannot be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink():
        raise QualificationError("output parent must be a real directory")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=parent, delete=False) as stream:
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


def _distinct_outputs(json_out: Path, junit_out: Path) -> None:
    if json_out.resolve() == junit_out.resolve():
        raise QualificationError("JSON and JUnit outputs must be distinct")


def _junit_counts(path: Path | str, clean: bool = True) -> tuple[dict[str, int], bytes]:
    source = _regular(Path(path), "JUnit", MAX_OUTPUT)
    raw = source.read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise QualificationError("JUnit must not contain DTD or entities")
    try:
        root = etree.fromstring(raw)
    except etree.ParseError as exc:
        raise QualificationError("malformed JUnit") from exc
    nodes = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    nodes = [node for node in nodes if not list(node.findall("testsuite"))]
    if not nodes:
        raise QualificationError("JUnit has no testsuite")
    totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for node in nodes:
        cases = list(node.findall("testcase"))
        actual = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
        for case in cases:
            actual["failures"] += len(case.findall("failure"))
            actual["errors"] += len(case.findall("error"))
            actual["skipped"] += len(case.findall("skipped"))
        for key in totals:
            value = node.attrib.get(key, "0")
            if not value.isdecimal():
                raise QualificationError("invalid JUnit counter")
            if int(value) != actual[key]:
                raise QualificationError("JUnit declared counters do not match testcases")
            totals[key] += actual[key]
    if clean and (totals["tests"] <= 0 or any(totals[key] for key in ("failures", "errors", "skipped"))):
        raise QualificationError("JUnit is not a clean non-empty pass")
    return totals, raw


def parse_junit(path: Path | str) -> dict[str, int]:
    return _junit_counts(path)[0]


def _valid_counters(value: object) -> bool:
    return isinstance(value, dict) and set(value) == {"tests", "failures", "errors", "skipped"} and all(type(item) is int and 0 <= item <= 1_000_000 for item in value.values())


def _dist_info_file(distribution: importlib.metadata.Distribution, name: str) -> Path | None:
    for item in distribution.files or ():
        if item.name == name and ".dist-info" in str(item):
            return Path(str(distribution.locate_file(item)))
    return None


def _preflight(
    claims: SimpleNamespace, cell: SimpleNamespace, repo: Path, artifact: Path | None
) -> None:
    try:
        import gwexpy

        distribution = importlib.metadata.distribution("gwexpy")
        if importlib.metadata.version("gwexpy") != claims.version or gwexpy.__version__ != claims.version:
            raise QualificationError("installed version does not match claims")
        specification = importlib.util.find_spec("gwexpy")
        if specification is None or specification.origin is None:
            raise QualificationError("gwexpy module has no import origin")
        origin = Path(specification.origin).resolve()
        package_file = Path(gwexpy.__file__ or "").resolve()
        root = Path(str(distribution.locate_file(""))).resolve()
    except (ImportError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError, AttributeError, OSError) as exc:
        raise QualificationError("gwexpy distribution preflight failed") from exc
    prefix = Path(sys.prefix).resolve()
    if origin != package_file or not _inside(origin, prefix) or not _inside(root, prefix) or _inside(origin, repo) or _inside(root, repo):
        raise QualificationError("source-shadowed or editable installation")
    expected_platform = {"linux": "linux", "macos": "darwin", "windows": "win32"}[cell.platform]
    if not sys.platform.startswith(expected_platform) or f"{sys.version_info.major}.{sys.version_info.minor}" != cell.python:
        raise QualificationError("interpreter platform or Python version does not match cell")
    if getattr(cell, "gwpy", None) is not None and importlib.metadata.version("gwpy") != cell.gwpy:
        raise QualificationError("installed GWpy version does not match cell")
    direct = _dist_info_file(distribution, "direct_url.json")
    if direct is not None:
        data = _json_file(direct, "direct_url metadata")
        if isinstance(data.get("dir_info"), dict) and data["dir_info"].get("editable") is True:
            raise QualificationError("editable installation")
    if artifact is not None:
        if direct is None:
            raise QualificationError("direct_url metadata is required for artifact cell")
        data = _json_file(direct, "direct_url metadata")
        url = data.get("url")
        archive = data.get("archive_info")
        digest = _digest(artifact)
        local = _local_file_uri(url)
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        declared = archive.get("hash") if isinstance(archive, dict) else None
        if local.resolve() != artifact.resolve() or not isinstance(archive, dict) or (declared != f"sha256={digest}" and (not isinstance(hashes, dict) or hashes.get("sha256") != digest)):
            raise QualificationError("direct_url metadata does not attest selected artifact")
    if cell.channel == "conda":
        prefix_value = os.environ.get("CONDA_PREFIX")
        metadata = Path(prefix_value) / "conda-meta" if prefix_value else None
        if prefix_value is None or Path(prefix_value).resolve() != Path(sys.prefix).resolve():
            raise QualificationError("CONDA_PREFIX does not match interpreter prefix")
        matches = list(metadata.glob("gwexpy-*.json")) if metadata and metadata.is_dir() else []
        if len(matches) != 1:
            raise QualificationError("conda cell has no exact gwexpy conda record")
        record = _json_file(matches[0], "conda package record")
        if record.get("name") != "gwexpy" or record.get("version") != claims.version or _conda_channel(record) != cell.conda_channel:
            raise QualificationError("conda package record does not match claims")


def _result(path: Path, payload: dict[str, Any]) -> None:
    _atomic(path, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def run_cell(claims: SimpleNamespace, cell_id: str, repo_root: Path, artifact: Path | None, json_out: Path, junit_out: Path) -> bool:
    """Preflight an installed distribution, stage selected tests, and run pytest."""
    started = time.monotonic()
    error: str | None = None
    cell: SimpleNamespace | None = None
    counters: dict[str, int] = {"tests": 0, "failures": 1, "errors": 0, "skipped": 0}
    try:
        _distinct_outputs(json_out, junit_out)
    except QualificationError:
        return False
    try:
        cell = claims.required_cells[cell_id]
        repo_input = Path(repo_root)
        if repo_input.is_symlink() or not repo_input.is_dir():
            raise QualificationError("repo root must be a real directory")
        repo = repo_input.resolve()
        selected: Path | None = None
        if cell.channel == "pypi":
            if artifact is None:
                raise QualificationError("PyPI cell requires an artifact")
            selected = _regular(Path(artifact), "artifact")
            expected = claims.artifacts[cell.artifact]
            if selected.name != expected.name or _digest(selected) != expected.sha256:
                raise QualificationError("selected artifact does not match claims")
        elif artifact is not None:
            raise QualificationError("non-PyPI cells must not accept an artifact")
        _preflight(claims, cell, repo, selected if cell.channel == "pypi" else None)
        suite = claims.suites[cell.suite]
        with tempfile.TemporaryDirectory(prefix="gwexpy-qualification-") as temporary:
            stage = Path(temporary) / "stage"
            stage.mkdir()
            staged_bytes = 0
            mapped_selectors: list[str] = []
            for relative in (*suite.selectors, *suite.support_paths):
                file_part, marker, node = relative.partition("::")
                _safe_relative(file_part, "selector path")
                if marker and (not node or ".." in node or "\x00" in node or len(node) > 200):
                    raise QualificationError("unsafe selector node")
                unresolved = repo / file_part
                try:
                    source_mode = unresolved.lstat().st_mode
                except OSError as exc:
                    raise QualificationError("selected test/support path is missing") from exc
                if stat.S_ISLNK(source_mode) or not stat.S_ISREG(source_mode) or _has_symlink_component(unresolved, repo):
                    raise QualificationError("selected test/support path is unsafe")
                source = unresolved.resolve()
                if not _inside(source, repo):
                    raise QualificationError("selected test/support path escapes repo")
                size = source.stat().st_size
                staged_bytes += size
                if size > MAX_STAGE_FILE or staged_bytes > MAX_STAGE_TOTAL:
                    raise QualificationError("staged qualification files exceed size limit")
                destination = stage / file_part
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if relative in suite.selectors:
                    mapped_selectors.append(str(destination) + (f"::{node}" if marker else ""))
            if (stage / "gwexpy").exists():
                raise QualificationError("staged tree contains source package")
            cwd = Path(temporary) / "cwd"
            cwd.mkdir()
            pytest_junit = Path(temporary) / "pytest.xml"
            config = Path(temporary) / "pytest.ini"
            config.write_text("[pytest]\n", encoding="utf-8")
            environment = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL") if key in os.environ}
            environment.update({"PYTHONSAFEPATH": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "GWEXPY_POST_RELEASE_QUALIFICATION": "1", "MPLBACKEND": "Agg"})
            command = [sys.executable, "-P", "-m", "pytest", "-c", str(config), "--import-mode=importlib", f"--junitxml={pytest_junit}", *mapped_selectors]
            process = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=suite.timeout, check=False)
            if process.returncode != 0:
                raise QualificationError(f"pytest exit {process.returncode}: {(process.stderr or process.stdout)[:4096]}")
            counters, raw_junit = _junit_counts(pytest_junit)
            _atomic(junit_out, raw_junit.decode("utf-8"))
        _preflight(claims, cell, repo, selected if cell.channel == "pypi" else None)
    except Exception as exc:
        error = str(exc)[:4096]
    passed = error is None
    payload = {"schema": "gwexpy-published-release-cell-report-v1", "cell": cell_id, "claims_sha256": claims.digest, "version": claims.version, "python": sys.version.split()[0], "platform": cell.platform if cell else None, "gwpy": getattr(cell, "gwpy", None) if cell else None, "channel": cell.channel if cell else None, "artifact": artifact.name if cell is not None and cell.channel == "pypi" and artifact else None, "passed": passed, "counters": counters, "error": error, "duration_seconds": round(time.monotonic() - started, 3)}
    try:
        _result(json_out, payload)
        if not passed:
            _write_junit(junit_out, [(cell_id, error)])
    except Exception:
        return False
    return passed


def aggregate(claims: SimpleNamespace, artifact_dir: Path, reports_dir: Path, pypi_json: Path | None, payload_sidecar: Path | None, json_out: Path, junit_out: Path) -> bool:
    """Cross-check a complete cell ledger and write aggregate evidence on failure."""
    errors: dict[str, str] = {}
    try:
        _distinct_outputs(json_out, junit_out)
    except Exception:
        return False
    try:
        verify_artifact_directory(claims, artifact_dir)
    except Exception as exc:
        errors["aggregate"] = str(exc)[:4096]
    try:
        if pypi_json is not None:
            validate_pypi_json(claims, _json_file(pypi_json, "PyPI JSON"))
        if payload_sidecar is not None:
            validate_payload_sidecar(claims, _json_file(payload_sidecar, "payload sidecar"))
        if reports_dir.is_symlink() or not reports_dir.is_dir():
            raise QualificationError("reports directory must be real")
        entries = {item.name: item for item in reports_dir.iterdir()}
        expected = {f"{cell}.json" for cell in claims.required_cells} | {f"{cell}.xml" for cell in claims.required_cells}
        if set(entries) != expected:
            missing = sorted(expected - set(entries))
            extra = sorted(set(entries) - expected)
            for cell in claims.required_cells:
                if f"{cell}.json" in missing or f"{cell}.xml" in missing:
                    errors[cell] = "missing qualification report"
            if extra:
                errors["aggregate"] = "extra or unsafe qualification reports"
        for cell in claims.required_cells:
            if cell in errors:
                continue
            try:
                report = _json_file(entries[f"{cell}.json"], "cell JSON")
                counters = parse_junit(entries[f"{cell}.xml"])
            except QualificationError as exc:
                errors[cell] = str(exc)
                continue
            selected = claims.required_cells[cell]
            expected_artifact = claims.artifacts[selected.artifact].name if selected.channel == "pypi" else None
            fields = {"schema", "cell", "claims_sha256", "version", "python", "platform", "gwpy", "channel", "artifact", "passed", "counters", "error", "duration_seconds"}
            if set(report) != fields or report["schema"] != "gwexpy-published-release-cell-report-v1" or report["cell"] != cell or report["claims_sha256"] != claims.digest or report["version"] != claims.version or not isinstance(report["python"], str) or re.fullmatch(r"3\.[0-9]+\.[0-9]+", report["python"]) is None or not report["python"].startswith(f"{selected.python}.") or report["platform"] != selected.platform or report["gwpy"] != getattr(selected, "gwpy", None) or report["channel"] != selected.channel or report["artifact"] != expected_artifact or report["passed"] is not True or report["error"] is not None or not _valid_counters(report["counters"]) or report["counters"] != counters or isinstance(report["duration_seconds"], bool) or not isinstance(report["duration_seconds"], (int, float)) or not math.isfinite(report["duration_seconds"]) or not 0 <= report["duration_seconds"] <= 3600:
                errors[cell] = "mismatched evidence"
    except Exception as exc:
        errors["aggregate"] = str(exc)[:4096]
    passed = not errors
    _result(json_out, {"schema": "gwexpy-published-release-aggregate-v1", "claims_sha256": claims.digest, "passed": passed, "required_cells": sorted(claims.required_cells), "errors": errors})
    cases = [(cell, errors.get(cell)) for cell in sorted(claims.required_cells)]
    cases.append(("aggregate", errors.get("aggregate")))
    _write_junit(junit_out, cases)
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
    except Exception as exc:
        message = str(exc)[:4096]
        try:
            _distinct_outputs(args.json_out, args.junit_out)
            _result(args.json_out, {"schema": "gwexpy-published-release-error-v1", "passed": False, "error": message})
            _write_junit(args.junit_out, [(args.command, message)])
        except (QualificationError, OSError):
            pass
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
