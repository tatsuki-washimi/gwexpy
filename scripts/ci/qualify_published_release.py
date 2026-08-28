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
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as etree
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

MAX_INPUT = 4 * 1024 * 1024
MAX_OUTPUT = 4 * 1024 * 1024
MAX_ARTIFACT = 128 * 1024 * 1024
MAX_STAGE_FILE = 2 * 1024 * 1024
MAX_STAGE_TOTAL = 8 * 1024 * 1024
MAX_FREEZE_LINES = 4096
MAX_FREEZE_LINE = 4096
MAX_INSPECT = 3 * 1024 * 1024
MAX_CELL_REPORT = 8 * 1024 * 1024
MAX_FACT = 4096
MAX_JUNIT_COUNT = 1_000_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
CELL = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
ASCII_COUNTER = re.compile(r"^(0|[1-9][0-9]*)$")
INSPECT_ENVIRONMENT_FIELDS = {
    "implementation_name",
    "implementation_version",
    "os_name",
    "platform_machine",
    "platform_python_implementation",
    "platform_release",
    "platform_system",
    "platform_version",
    "python_full_version",
    "python_version",
    "sys_platform",
}


class QualificationError(ValueError):
    """Raised when qualification input or evidence is not trustworthy."""


class _DuplicateKey(ValueError):
    pass


class _BoundedResult(NamedTuple):
    returncode: int
    output: str
    truncated: bool


class _Provenance(NamedTuple):
    installer: str
    channel: str
    gwpy_version: str
    pip_version: str


def _exception_text(exc: BaseException) -> str:
    """Return a bounded, nonempty and type-qualified exception description."""
    name = type(exc).__name__
    message = str(exc).strip()
    return (f"{name}: {message}" if message else name)[:MAX_FACT]


def _xml_text(value: str) -> str:
    def legal(character: str) -> bool:
        codepoint = ord(character)
        return (
            character in "\t\n\r"
            or 0x20 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        )

    return "".join(character if legal(character) else "?" for character in value)[
        :MAX_FACT
    ]


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


def _json_bytes(content: bytes, label: str) -> dict[str, Any]:
    try:
        data = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_no_json_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateKey,
        QualificationError,
    ) as exc:
        raise QualificationError(f"invalid {label}") from exc
    if not isinstance(data, dict):
        raise QualificationError(f"{label} must be an object")
    return data


def _json_file(path: Path, label: str, limit: int = MAX_INPUT) -> dict[str, Any]:
    source = _regular(path, label, limit)
    try:
        content = source.read_bytes()
    except OSError as exc:
        raise QualificationError(f"invalid {label}") from exc
    return _json_bytes(content, label)


def _keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise QualificationError(f"{label} has missing or unknown keys")
    return value


def _safe_relative(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 240
        or "\x00" in value
        or "\\" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise QualificationError(f"unsafe {label}")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} or part.startswith("-") for part in path.parts)
    ):
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


def _local_file_uri(url: object, converter: Any = urllib.request.url2pathname) -> Path:
    if not isinstance(url, str):
        raise QualificationError("direct_url is not a file URI")
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise QualificationError("direct_url must be a local file URI")
    if not parsed.path.startswith("/") or "\x00" in parsed.path:
        raise QualificationError("direct_url has an unsafe path")
    converted = converter(parsed.path)
    if not isinstance(converted, str) or "\x00" in converted:
        raise QualificationError("direct_url has an unsafe path")
    if converted.replace("/", "\\").startswith("\\\\"):
        raise QualificationError("direct_url must be a local file URI")
    return Path(converted)


def _conda_channel(record: dict[str, Any]) -> str | None:
    source = record.get("schannel", record.get("channel"))
    if not isinstance(source, str):
        return None
    subdir = record.get("subdir")
    known = {"linux-64", "linux-aarch64", "osx-64", "osx-arm64", "win-64", "noarch"}
    if not isinstance(subdir, str) or subdir not in known:
        return None
    if source == "conda-forge":
        return source
    parsed = urllib.parse.urlparse(source)
    expected = (
        f"/conda-forge/{subdir}"
        if source.rstrip("/") != "https://conda.anaconda.org/conda-forge"
        else "/conda-forge"
    )
    return (
        "conda-forge"
        if parsed.scheme == "https"
        and parsed.hostname == "conda.anaconda.org"
        and parsed.port in {None, 443}
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and parsed.path == expected
        else None
    )


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _artifact(value: object, kind: str, version: str) -> SimpleNamespace:
    entry = _keys(value, {"filename", "packagetype", "sha256"}, "artifact")
    name = _safe_relative(entry["filename"], "artifact filename")
    expected_name = (
        f"gwexpy-{version}-py3-none-any.whl"
        if kind == "wheel"
        else f"gwexpy-{version}.tar.gz"
    )
    expected_type = "bdist_wheel" if kind == "wheel" else "sdist"
    if (
        name != expected_name
        or entry["packagetype"] != expected_type
        or not isinstance(entry["sha256"], str)
        or not SHA256.fullmatch(entry["sha256"])
    ):
        raise QualificationError("invalid artifact claim")
    return SimpleNamespace(name=name, packagetype=expected_type, sha256=entry["sha256"])


def load_claims(path: Path | str) -> SimpleNamespace:
    """Load the versioned claims manifest, rejecting any ambiguity."""
    source_path = _regular(Path(path), "claims")
    try:
        claims_bytes = source_path.read_bytes()
    except OSError as exc:
        raise QualificationError("invalid claims") from exc
    data = _json_bytes(claims_bytes, "claims")
    fields = {
        "schema",
        "project",
        "version",
        "tag",
        "source_sha",
        "release_run_id",
        "repository",
        "payload_sidecar_schema",
        "artifacts",
        "suites",
        "required_cells",
    }
    _keys(data, fields, "claims")
    if (
        data["schema"] != "gwexpy-published-release-claims-v1"
        or data["project"] != "gwexpy"
        or not isinstance(data["version"], str)
        or not SEMVER.fullmatch(data["version"])
    ):
        raise QualificationError("invalid release identity")
    if (
        data["tag"] != f"v{data['version']}"
        or not isinstance(data["source_sha"], str)
        or not SHA1.fullmatch(data["source_sha"])
    ):
        raise QualificationError("invalid release tag or source SHA")
    if (
        not isinstance(data["release_run_id"], str)
        or re.fullmatch(r"[1-9][0-9]{0,19}", data["release_run_id"]) is None
    ):
        raise QualificationError("invalid release run ID")
    if (
        data["repository"] != "tatsuki-washimi/gwexpy"
        or not isinstance(data["payload_sidecar_schema"], str)
        or not data["payload_sidecar_schema"]
    ):
        raise QualificationError("invalid repository or payload schema")
    artifacts = _keys(data["artifacts"], {"wheel", "sdist"}, "artifacts")
    parsed_artifacts = {
        kind: _artifact(artifacts[kind], kind, data["version"]) for kind in artifacts
    }
    suites = data["suites"]
    if not isinstance(suites, dict) or not suites:
        raise QualificationError("invalid suites")
    parsed_suites: dict[str, SimpleNamespace] = {}
    for name, suite in suites.items():
        if not CELL.fullmatch(name) or len(name) > 80:
            raise QualificationError("invalid suite name")
        suite = _keys(suite, {"selectors", "support_paths", "timeout"}, "suite")
        selectors, support = suite["selectors"], suite["support_paths"]
        if (
            not isinstance(selectors, list)
            or not selectors
            or not isinstance(support, list)
            or isinstance(suite["timeout"], bool)
            or not isinstance(suite["timeout"], int)
            or not 0 < suite["timeout"] <= 3600
        ):
            raise QualificationError("invalid suite")
        if (
            any(_safe_relative(item, "selector") != item for item in selectors)
            or any(_safe_relative(item, "support path") != item for item in support)
            or len(set(selectors + support)) != len(selectors + support)
        ):
            raise QualificationError("invalid suite paths")
        parsed_suites[name] = SimpleNamespace(
            selectors=tuple(selectors),
            support_paths=tuple(support),
            timeout=suite["timeout"],
        )
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
        allowed = (
            base
            | ({"gwpy"} if channel == "pypi" and "gwpy" in cell else set())
            | ({"conda_channel"} if channel == "conda" else set())
        )
        if set(cell) != allowed:
            raise QualificationError("cell has missing or unknown keys")
        if (
            not isinstance(cell["python"], str)
            or not re.fullmatch(r"3\.(1[1-4])", cell["python"])
            or cell["platform"] not in {"linux", "macos", "windows"}
            or cell["channel"] not in {"pypi", "conda"}
            or cell["artifact"] not in {"wheel", "sdist", "none"}
            or cell["suite"] not in parsed_suites
            or cell["required"] is not True
            or (
                "gwpy" in cell
                and (
                    not isinstance(cell["gwpy"], str)
                    or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", cell["gwpy"]) is None
                )
            )
        ):
            raise QualificationError("invalid cell")
        if (cell["channel"] == "pypi" and cell["artifact"] not in parsed_artifacts) or (
            cell["channel"] != "pypi" and cell["artifact"] != "none"
        ):
            raise QualificationError("invalid cell artifact selection")
        if cell["channel"] == "conda" and (
            not isinstance(cell.get("conda_channel"), str) or not cell["conda_channel"]
        ):
            raise QualificationError("conda cell requires exact channel")
        parsed_cells[cell_id] = SimpleNamespace(**cell)
    result = dict(data)
    result.update(
        artifacts=parsed_artifacts,
        suites=parsed_suites,
        required_cells=parsed_cells,
        digest=hashlib.sha256(claims_bytes).hexdigest(),
        source_path=source_path,
    )
    return SimpleNamespace(**result)


def verify_artifact_directory(
    claims: SimpleNamespace, directory: Path | str
) -> dict[str, Path]:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise QualificationError("artifact directory must be a real directory")
    entries: dict[str, Path] = {}
    for item in root.iterdir():
        if item.is_symlink() or not stat.S_ISREG(
            item.stat(follow_symlinks=False).st_mode
        ):
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
    if (
        not isinstance(data, dict)
        or not {"info", "urls"}.issubset(data)
        or not isinstance(data["info"], dict)
        or data["info"].get("name") != claims.project
        or data["info"].get("version") != claims.version
        or not isinstance(data["urls"], list)
    ):
        raise QualificationError("invalid PyPI JSON")
    if any(
        not isinstance(item, dict) or type(item.get("yanked")) is not bool
        for item in data["urls"]
    ):
        raise QualificationError("invalid PyPI file entry")
    expected_names = {artifact.name for artifact in claims.artifacts.values()}
    if any(
        item.get("yanked", False) and item.get("filename") in expected_names
        for item in data["urls"]
    ):
        raise QualificationError("PyPI expected artifact is yanked")
    files = [item for item in data["urls"] if not item["yanked"]]
    if len(files) != 2:
        raise QualificationError("PyPI JSON must contain exactly two non-yanked files")
    expected = {
        (artifact.name, artifact.packagetype, artifact.sha256)
        for artifact in claims.artifacts.values()
    }
    actual = set()
    for item in files:
        digest = (
            item.get("digests", {}).get("sha256")
            if isinstance(item.get("digests"), dict)
            else None
        )
        url = item.get("url")
        if not isinstance(url, str) or not re.fullmatch(
            r"https://files\.pythonhosted\.org/.+", url
        ):
            raise QualificationError("PyPI URL is not files.pythonhosted.org HTTPS")
        actual.add((item.get("filename"), item.get("packagetype"), digest))
    if actual != expected or len(actual) != len(files):
        raise QualificationError("PyPI artifacts do not match claims")


def validate_payload_sidecar(claims: SimpleNamespace, data: object) -> None:
    sidecar = _keys(
        data, {"schema", "source_sha", "version", "files"}, "payload sidecar"
    )
    if (
        sidecar["schema"] != claims.payload_sidecar_schema
        or sidecar["source_sha"] != claims.source_sha
        or sidecar["version"] != claims.version
    ):
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
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _write_junit(
    path: Path, cases: list[tuple[str, str | None]], tests: int | None = None
) -> None:
    suite = etree.Element(
        "testsuite",
        name="gwexpy-published-release",
        tests=str(tests if tests is not None else len(cases)),
        failures=str(sum(error is not None for _, error in cases)),
        errors="0",
        skipped="0",
    )
    for name, error in cases:
        case = etree.SubElement(suite, "testcase", classname="qualification", name=name)
        if error is not None:
            safe_error = _xml_text(error)
            failure = etree.SubElement(case, "failure", message=safe_error)
            failure.text = safe_error
    _atomic(path, etree.tostring(suite, encoding="unicode") + "\n")


def _path_key(path: Path) -> str:
    key = os.path.normcase(os.fspath(path.resolve()))
    return key.casefold() if sys.platform in {"darwin", "win32"} else key


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _distinct_outputs(json_out: Path, junit_out: Path) -> None:
    if _path_key(json_out) == _path_key(junit_out):
        raise QualificationError("JSON and JUnit outputs must be distinct")


def _disjoint_outputs(
    json_out: Path,
    junit_out: Path,
    inputs: list[Path | None],
    input_roots: list[Path] | None = None,
) -> None:
    _distinct_outputs(json_out, junit_out)
    outputs = {_path_key(json_out), _path_key(junit_out)}
    resolved_inputs = {_path_key(path) for path in inputs if path is not None}
    if outputs & resolved_inputs:
        raise QualificationError("qualification outputs must not overwrite inputs")
    roots = {_path_key(path) for path in input_roots or []}
    if any(_path_is_within(output, root) for output in outputs for root in roots):
        raise QualificationError("qualification outputs must be outside input roots")


def _junit_counts(path: Path | str, clean: bool = True) -> tuple[dict[str, int], bytes]:
    source = _regular(Path(path), "JUnit", MAX_OUTPUT)
    raw = source.read_bytes()
    declaration_bytes = raw.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in declaration_bytes or b"<!ENTITY" in declaration_bytes:
        raise QualificationError("JUnit must not contain DTD or entities")
    try:
        root = etree.fromstring(raw)
    except etree.ParseError as exc:
        raise QualificationError("malformed JUnit") from exc
    if root.tag == "testsuite":
        nodes = [root]
        root_summary: etree.Element | None = None
    elif root.tag == "testsuites":
        if not len(root) or any(child.tag != "testsuite" for child in root):
            raise QualificationError(
                "JUnit testsuites must contain only testsuite children"
            )
        nodes = list(root)
        root_summary = root
    else:
        raise QualificationError("JUnit root must be testsuite or testsuites")
    totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for node in nodes:
        if any(child.tag == "testsuite" for child in node):
            raise QualificationError("JUnit testsuites cannot be nested")
        allowed_suite_children = {"properties", "testcase", "system-out", "system-err"}
        if any(child.tag not in allowed_suite_children for child in node):
            raise QualificationError("JUnit testsuite contains an invalid result")
        for child in node:
            if child.tag == "properties" and any(
                item.tag != "property" or len(item) for item in child
            ):
                raise QualificationError("JUnit properties contain invalid children")
            if child.tag in {"system-out", "system-err"} and len(child):
                raise QualificationError("JUnit output contains invalid children")
        cases = [child for child in node if child.tag == "testcase"]
        actual = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
        for case in cases:
            allowed_case_children = {
                "properties",
                "failure",
                "error",
                "skipped",
                "system-out",
                "system-err",
            }
            if any(child.tag not in allowed_case_children for child in case):
                raise QualificationError("JUnit testcase contains an invalid result")
            for child in case:
                if child.tag == "properties" and any(
                    item.tag != "property" or len(item) for item in child
                ):
                    raise QualificationError(
                        "JUnit properties contain invalid children"
                    )
                if child.tag != "properties" and len(child):
                    raise QualificationError("JUnit result contains invalid children")
            outcomes = [
                child for child in case if child.tag in {"failure", "error", "skipped"}
            ]
            if len(outcomes) > 1:
                raise QualificationError("JUnit testcase has multiple outcomes")
            for outcome in outcomes:
                key = f"{outcome.tag}s" if outcome.tag != "skipped" else "skipped"
                actual[key] += 1
        if len(list(node.iter("testcase"))) != len(cases):
            raise QualificationError("JUnit testcase is outside a testsuite")
        for result in ("failure", "error", "skipped"):
            direct = sum(child.tag == result for case in cases for child in list(case))
            if len(list(node.iter(result))) != direct:
                raise QualificationError("JUnit result is outside a testcase")
        for key in totals:
            value = node.attrib.get(key)
            if value is None or ASCII_COUNTER.fullmatch(value) is None:
                raise QualificationError("invalid JUnit counter")
            declared = int(value)
            if declared > MAX_JUNIT_COUNT or declared != actual[key]:
                raise QualificationError(
                    "JUnit declared counters do not match testcases"
                )
            totals[key] += actual[key]
            if totals[key] > MAX_JUNIT_COUNT:
                raise QualificationError("JUnit counter exceeds limit")
    if root_summary is not None:
        present = {key for key in totals if key in root_summary.attrib}
        if present and present != set(totals):
            raise QualificationError("JUnit root counters must be complete")
        for key in present:
            value = root_summary.attrib[key]
            if ASCII_COUNTER.fullmatch(value) is None or int(value) != totals[key]:
                raise QualificationError("JUnit root counters do not match testsuites")
    if clean and (
        totals["tests"] <= 0
        or any(totals[key] for key in ("failures", "errors", "skipped"))
    ):
        raise QualificationError("JUnit is not a clean non-empty pass")
    return totals, raw


def parse_junit(path: Path | str) -> dict[str, int]:
    return _junit_counts(path)[0]


def _valid_counters(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"tests", "failures", "errors", "skipped"}
        and all(type(item) is int and 0 <= item <= 1_000_000 for item in value.values())
    )


def _fact_string(value: object, label: str, limit: int = MAX_FACT) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or any(ord(character) < 32 for character in value)
    ):
        raise QualificationError(f"invalid {label}")
    return value


def _isolated_environment(directory: Path) -> dict[str, str]:
    """Build a minimal environment for evidence and qualification subprocesses."""
    environment = {
        key: os.environ[key]
        for key in (
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "LANG",
            "LC_ALL",
        )
        if key in os.environ
    }
    runtime = directory / ".runtime"
    runtime.mkdir(exist_ok=True)
    environment.update(
        {
            "HOME": str(runtime),
            "USERPROFILE": str(runtime),
            "TEMP": str(runtime),
            "TMP": str(runtime),
            "TMPDIR": str(runtime),
            "XDG_CACHE_HOME": str(runtime / "cache"),
            "XDG_CONFIG_HOME": str(runtime / "config"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_NO_INPUT": "1",
            "PIP_NO_INDEX": "1",
            "GWEXPY_POST_RELEASE_QUALIFICATION": "1",
            "MPLBACKEND": "Agg",
        }
    )
    return environment


class _WindowsJob:
    """Kill-on-close Windows Job Object used to contain the entire child tree."""

    def __init__(self) -> None:
        import ctypes
        import ctypes.wintypes as wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _BasicAccountingInformation(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        library = getattr(ctypes, "WinDLL")
        self._kernel32 = library("kernel32", use_last_error=True)
        self._ntdll = library("ntdll")
        self._create = self._kernel32.CreateJobObjectW
        self._create.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._create.restype = wintypes.HANDLE
        self._set_information = self._kernel32.SetInformationJobObject
        self._set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._set_information.restype = wintypes.BOOL
        self._assign = self._kernel32.AssignProcessToJobObject
        self._assign.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._assign.restype = wintypes.BOOL
        self._query = self._kernel32.QueryInformationJobObject
        self._query.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self._query.restype = wintypes.BOOL
        self._close = self._kernel32.CloseHandle
        self._close.argtypes = [wintypes.HANDLE]
        self._close.restype = wintypes.BOOL
        self._resume = self._ntdll.NtResumeProcess
        self._resume.argtypes = [wintypes.HANDLE]
        self._resume.restype = wintypes.LONG
        self._accounting_type = _BasicAccountingInformation
        self._handle = self._create(None, None)
        if not self._handle:
            raise OSError(getattr(ctypes, "get_last_error")(), "CreateJobObjectW")
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x00002000
        if not self._set_information(
            self._handle,
            9,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = getattr(ctypes, "get_last_error")()
            self.close()
            raise OSError(error, "SetInformationJobObject")

    @property
    def creationflags(self) -> int:
        return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200) | getattr(
            subprocess, "CREATE_SUSPENDED", 0x00000004
        )

    def attach_and_resume(self, process: subprocess.Popen[bytes]) -> None:
        process_handle = getattr(process, "_handle")
        if not self._assign(self._handle, process_handle):
            raise OSError("AssignProcessToJobObject failed")
        if self._resume(process_handle) != 0:
            raise OSError("NtResumeProcess failed")

    def close(self) -> bool:
        import ctypes

        active = False
        if self._handle:
            accounting = self._accounting_type()
            active = (
                not self._query(
                    self._handle,
                    1,
                    ctypes.byref(accounting),
                    ctypes.sizeof(accounting),
                    None,
                )
                or accounting.ActiveProcesses > 0
            )
            try:
                self._close(self._handle)
            finally:
                self._handle = None
        return active


def _kill_process_tree(
    process: subprocess.Popen[bytes], windows_job: _WindowsJob | None = None
) -> bool:
    """Terminate the isolated process group and always reap its leader."""
    terminated_group = False
    if os.name == "nt":
        if windows_job is not None:
            terminated_group = windows_job.close()
        else:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            terminated_group = True
        except (ProcessLookupError, PermissionError):
            pass
    if process.poll() is None:
        process.kill()
    process.wait()
    return terminated_group


def _decode_bounded_output(raw: bytes) -> tuple[str, bool]:
    decoded = raw.decode("utf-8", errors="replace")
    encoded = decoded.encode("utf-8")
    if len(encoded) <= MAX_OUTPUT:
        return decoded, False
    return encoded[:MAX_OUTPUT].decode("utf-8", errors="ignore"), True


def _bounded_run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> _BoundedResult:
    """Run with a capped merged pipe while continuously draining child output."""
    windows_job = _WindowsJob() if os.name == "nt" else None
    group_options: dict[str, Any] = (
        {"creationflags": (windows_job.creationflags if windows_job is not None else 0)}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **group_options,
        )
        if windows_job is not None:
            windows_job.attach_and_resume(process)
    except Exception:
        if windows_job is not None:
            windows_job.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    if process is None:
        raise QualificationError("subprocess was not created")
    if process.stdout is None:
        _kill_process_tree(process, windows_job)
        raise QualificationError("subprocess output pipe was not created")
    stdout = process.stdout
    output = bytearray()
    truncated = False

    def drain() -> None:
        nonlocal truncated
        try:
            while True:
                block = stdout.read(64 * 1024)
                if not block:
                    break
                remaining = max(0, MAX_OUTPUT - len(output))
                if remaining:
                    output.extend(block[:remaining])
                if len(block) > remaining:
                    truncated = True
        except (OSError, ValueError):
            truncated = True

    reader = threading.Thread(target=drain, name="qualification-output", daemon=True)
    reader_started = False
    timeout_failure: subprocess.TimeoutExpired | None = None
    returncode: int | None = None
    try:
        reader.start()
        reader_started = True
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            timeout_failure = exc
    finally:
        if _kill_process_tree(process, windows_job):
            truncated = True
        if reader_started:
            reader.join(timeout=1)
        if reader_started and reader.is_alive():
            truncated = True
            try:
                os.close(stdout.fileno())
            except OSError:
                pass
            reader.join(timeout=1)
        if not reader_started or not reader.is_alive():
            try:
                stdout.close()
            except (OSError, ValueError):
                pass
    captured, decode_truncated = _decode_bounded_output(bytes(output))
    truncated = truncated or decode_truncated
    if timeout_failure is not None:
        raise subprocess.TimeoutExpired(
            command, timeout, output=captured
        ) from timeout_failure
    if returncode is None:
        raise QualificationError("subprocess did not report an exit status")
    return _BoundedResult(returncode=returncode, output=captured, truncated=truncated)


def _pip_evidence(
    environment: dict[str, str], *, timeout: float
) -> tuple[list[str], dict[str, Any]]:
    """Collect bounded, machine-readable package-manager evidence."""
    freeze_result = _bounded_run(
        [sys.executable, "-I", "-m", "pip", "freeze", "--all"],
        cwd=Path(environment["HOME"]),
        env=environment,
        timeout=timeout,
    )
    if freeze_result.returncode != 0:
        raise QualificationError(
            f"pip freeze exit {freeze_result.returncode}: {freeze_result.output[:MAX_FACT]}"
        )
    if freeze_result.truncated:
        raise QualificationError("pip freeze exceeds output limit")
    freeze = freeze_result.output.splitlines()
    if (
        not freeze
        or len(freeze) > MAX_FREEZE_LINES
        or any(
            not line
            or len(line) > MAX_FREEZE_LINE
            or any(ord(character) < 32 for character in line)
            for line in freeze
        )
    ):
        raise QualificationError("invalid pip freeze evidence")

    inspect_result = _bounded_run(
        [sys.executable, "-I", "-m", "pip", "inspect", "--local"],
        cwd=Path(environment["HOME"]),
        env=environment,
        timeout=timeout,
    )
    if inspect_result.returncode != 0:
        raise QualificationError(
            f"pip inspect exit {inspect_result.returncode}: {inspect_result.output[:MAX_FACT]}"
        )
    if inspect_result.truncated:
        raise QualificationError("pip inspect exceeds output limit")
    try:
        inspected = json.loads(
            inspect_result.output,
            object_pairs_hook=_duplicates,
            parse_constant=_no_json_constant,
        )
    except (
        json.JSONDecodeError,
        _DuplicateKey,
        QualificationError,
    ) as exc:
        raise QualificationError("invalid pip inspect evidence") from exc
    if not isinstance(inspected, dict):
        raise QualificationError("pip inspect evidence must be an object")
    try:
        compact_inspect = json.dumps(
            inspected, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QualificationError("invalid pip inspect evidence") from exc
    if len(compact_inspect) > MAX_INSPECT:
        raise QualificationError("pip inspect evidence exceeds report limit")
    return freeze, inspected


def _dist_info_file(
    distribution: importlib.metadata.Distribution, name: str
) -> Path | None:
    for item in distribution.files or ():
        if item.name == name and ".dist-info" in str(item):
            return Path(str(distribution.locate_file(item)))
    return None


def _distribution_text(
    distribution: importlib.metadata.Distribution, name: str
) -> str | None:
    try:
        value = distribution.read_text(name)
    except (AttributeError, OSError, UnicodeDecodeError):
        value = None
    if value is not None:
        return value
    path = _dist_info_file(distribution, name)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise QualificationError(f"invalid {name} metadata") from exc


def _preflight(
    claims: SimpleNamespace, cell: SimpleNamespace, repo: Path, artifact: Path | None
) -> _Provenance:
    try:
        import gwexpy

        distribution = importlib.metadata.distribution("gwexpy")
        installed_version = importlib.metadata.version("gwexpy")
        gwpy_version = importlib.metadata.version("gwpy")
        pip_version = importlib.metadata.version("pip")
        if installed_version != claims.version or gwexpy.__version__ != claims.version:
            raise QualificationError("installed version does not match claims")
        specification = importlib.util.find_spec("gwexpy")
        if specification is None or specification.origin is None:
            raise QualificationError("gwexpy module has no import origin")
        origin = Path(specification.origin).resolve()
        package_file = Path(gwexpy.__file__ or "").resolve()
        root = Path(str(distribution.locate_file(""))).resolve()
    except (
        ImportError,
        ModuleNotFoundError,
        importlib.metadata.PackageNotFoundError,
        AttributeError,
        OSError,
    ) as exc:
        raise QualificationError("gwexpy distribution preflight failed") from exc
    prefix = Path(sys.prefix).resolve()
    if (
        origin != package_file
        or not _inside(origin, prefix)
        or not _inside(root, prefix)
        or _inside(origin, repo)
        or _inside(root, repo)
    ):
        raise QualificationError("source-shadowed or editable installation")
    expected_platform = {
        "linux": ("linux", "Linux"),
        "macos": ("darwin", "Darwin"),
        "windows": ("win32", "Windows"),
    }[cell.platform]
    if (
        not sys.platform.startswith(expected_platform[0])
        or platform.system() != expected_platform[1]
        or f"{sys.version_info.major}.{sys.version_info.minor}" != cell.python
    ):
        raise QualificationError(
            "interpreter platform or Python version does not match cell"
        )
    _fact_string(sys.version, "Python runtime")
    _fact_string(platform.machine(), "platform machine")
    _fact_string(gwpy_version, "GWpy version")
    _fact_string(pip_version, "pip version")
    if getattr(cell, "gwpy", None) is not None and gwpy_version != cell.gwpy:
        raise QualificationError("installed GWpy version does not match cell")
    direct = _dist_info_file(distribution, "direct_url.json")
    if direct is not None:
        data = _json_file(direct, "direct_url metadata")
        if (
            isinstance(data.get("dir_info"), dict)
            and data["dir_info"].get("editable") is True
        ):
            raise QualificationError("editable installation")
    if artifact is not None:
        if direct is None:
            raise QualificationError(
                "direct_url metadata is required for artifact cell"
            )
        data = _json_file(direct, "direct_url metadata")
        url = data.get("url")
        archive = data.get("archive_info")
        digest = _digest(artifact)
        local = _local_file_uri(url)
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        declared = archive.get("hash") if isinstance(archive, dict) else None
        if (
            local.resolve() != artifact.resolve()
            or not isinstance(archive, dict)
            or not archive
            or set(archive) - {"hash", "hashes"}
            or (declared is None and hashes is None)
            or (declared is not None and declared != f"sha256={digest}")
            or (
                hashes is not None
                and (
                    not isinstance(hashes, dict)
                    or set(hashes) != {"sha256"}
                    or hashes.get("sha256") != digest
                )
            )
        ):
            raise QualificationError(
                "direct_url metadata does not attest selected artifact"
            )
        installer = _fact_string(
            (_distribution_text(distribution, "INSTALLER") or "").strip(),
            "installer",
        )
        if installer != "pip":
            raise QualificationError("artifact cell was not installed by pip")
        channel = "pypi"
    else:
        installer = "conda"
        channel = ""
    if cell.channel == "conda":
        prefix_value = os.environ.get("CONDA_PREFIX")
        metadata = Path(prefix_value) / "conda-meta" if prefix_value else None
        if (
            prefix_value is None
            or Path(prefix_value).resolve() != Path(sys.prefix).resolve()
        ):
            raise QualificationError("CONDA_PREFIX does not match interpreter prefix")
        matches = (
            list(metadata.glob("gwexpy-*.json"))
            if metadata and metadata.is_dir()
            else []
        )
        if len(matches) != 1:
            raise QualificationError("conda cell has no exact gwexpy conda record")
        record = _json_file(matches[0], "conda package record")
        observed_channel = _conda_channel(record)
        if (
            record.get("name") != "gwexpy"
            or record.get("version") != claims.version
            or observed_channel != cell.conda_channel
        ):
            raise QualificationError("conda package record does not match claims")
        channel = _fact_string(observed_channel, "conda channel")
    return _Provenance(
        installer=installer,
        channel=channel,
        gwpy_version=gwpy_version,
        pip_version=pip_version,
    )


def _result(path: Path, payload: dict[str, Any], limit: int = MAX_OUTPUT) -> None:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if len(content.encode("utf-8")) > limit:
        raise QualificationError("qualification JSON exceeds output limit")
    _atomic(path, content)


def run_cell(
    claims: SimpleNamespace,
    cell_id: str,
    repo_root: Path,
    artifact: Path | None,
    json_out: Path,
    junit_out: Path,
) -> bool:
    """Preflight an installed distribution, stage selected tests, and run pytest."""
    started = time.monotonic()
    error: str | None = None
    provenance: _Provenance | SimpleNamespace | None = None
    artifact_evidence: dict[str, str] | None = None
    pip_freeze: list[str] = []
    pip_inspect: dict[str, Any] = {}
    preserve_failure_junit = False
    counters: dict[str, int] = {
        "tests": 1,
        "failures": 1,
        "errors": 0,
        "skipped": 0,
    }
    python_runtime = sys.version
    sys_platform = sys.platform
    platform_system = platform.system()
    platform_machine = platform.machine()
    try:
        collision_inputs: list[Path | None] = [
            getattr(claims, "source_path", None),
            Path(artifact) if artifact is not None else None,
        ]
        collision_cell = claims.required_cells.get(cell_id)
        collision_suite = (
            claims.suites.get(collision_cell.suite)
            if collision_cell is not None
            else None
        )
        if collision_suite is not None:
            for relative in (
                *collision_suite.selectors,
                *collision_suite.support_paths,
            ):
                file_part = relative.partition("::")[0]
                try:
                    _safe_relative(file_part, "selector path")
                except QualificationError:
                    continue
                collision_inputs.append(Path(repo_root) / file_part)
        _disjoint_outputs(json_out, junit_out, collision_inputs, [Path(repo_root)])
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
            selected = _regular(Path(artifact), "artifact", MAX_ARTIFACT)
            expected = claims.artifacts[cell.artifact]
            artifact_evidence = {
                "filename": selected.name,
                "sha256": _digest(selected),
            }
            if (
                artifact_evidence["filename"] != expected.name
                or artifact_evidence["sha256"] != expected.sha256
            ):
                raise QualificationError("selected artifact does not match claims")
        elif artifact is not None:
            raise QualificationError("non-PyPI cells must not accept an artifact")
        provenance = _preflight(
            claims, cell, repo, selected if cell.channel == "pypi" else None
        )
        suite = claims.suites[cell.suite]
        with tempfile.TemporaryDirectory(prefix="gwexpy-qualification-") as temporary:
            stage = Path(temporary) / "stage"
            stage.mkdir()
            staged_bytes = 0
            mapped_selectors: list[str] = []
            for relative in (*suite.selectors, *suite.support_paths):
                file_part, marker, node = relative.partition("::")
                _safe_relative(file_part, "selector path")
                if marker and (
                    not node or ".." in node or "\x00" in node or len(node) > 200
                ):
                    raise QualificationError("unsafe selector node")
                unresolved = repo / file_part
                try:
                    source_mode = unresolved.lstat().st_mode
                except OSError as exc:
                    raise QualificationError(
                        "selected test/support path is missing"
                    ) from exc
                if (
                    stat.S_ISLNK(source_mode)
                    or not stat.S_ISREG(source_mode)
                    or _has_symlink_component(unresolved, repo)
                ):
                    raise QualificationError("selected test/support path is unsafe")
                source = unresolved.resolve()
                if not _inside(source, repo):
                    raise QualificationError("selected test/support path escapes repo")
                size = source.stat().st_size
                staged_bytes += size
                if size > MAX_STAGE_FILE or staged_bytes > MAX_STAGE_TOTAL:
                    raise QualificationError(
                        "staged qualification files exceed size limit"
                    )
                destination = stage / file_part
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if relative in suite.selectors:
                    mapped_selectors.append(
                        str(destination) + (f"::{node}" if marker else "")
                    )
            if (stage / "gwexpy").exists():
                raise QualificationError("staged tree contains source package")
            cwd = Path(temporary) / "cwd"
            cwd.mkdir()
            pytest_junit = Path(temporary) / "pytest.xml"
            config = Path(temporary) / "pytest.ini"
            config.write_text("[pytest]\n", encoding="utf-8")
            environment = _isolated_environment(cwd)
            evidence_timeout = min(float(suite.timeout), 60.0)
            pip_freeze, pip_inspect = _pip_evidence(
                environment, timeout=evidence_timeout
            )
            command = [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                "-c",
                str(config),
                "--import-mode=importlib",
                f"--junitxml={pytest_junit}",
                *mapped_selectors,
            ]
            process = _bounded_run(
                command,
                cwd=cwd,
                env=environment,
                timeout=float(suite.timeout),
            )
            counters, raw_junit = _junit_counts(
                pytest_junit, clean=process.returncode == 0
            )
            _atomic(junit_out, raw_junit.decode("utf-8"))
            if process.returncode != 0:
                preserve_failure_junit = True
                raise QualificationError(
                    f"pytest exit {process.returncode}: {process.output[:MAX_FACT]}"
                )
            if process.truncated:
                raise QualificationError("pytest output exceeds limit")
        after = _preflight(
            claims, cell, repo, selected if cell.channel == "pypi" else None
        )
        if after != provenance:
            raise QualificationError(
                "installed provenance changed during qualification"
            )
    except Exception as exc:
        error = _exception_text(exc)
    passed = error is None
    if not passed and not preserve_failure_junit:
        counters = {"tests": 1, "failures": 1, "errors": 0, "skipped": 0}
    payload = {
        "schema": "gwexpy-published-release-cell-report-v2",
        "cell": cell_id,
        "claims_sha256": claims.digest,
        "version": claims.version,
        "python": python_runtime,
        "sys_platform": sys_platform,
        "platform_system": platform_system,
        "platform_machine": platform_machine,
        "installer": getattr(provenance, "installer", None),
        "channel": getattr(provenance, "channel", None),
        "gwpy_version": getattr(provenance, "gwpy_version", None),
        "pip_version": getattr(provenance, "pip_version", None),
        "artifact": artifact_evidence,
        "pip_freeze": pip_freeze,
        "pip_inspect": pip_inspect,
        "passed": passed,
        "counters": counters,
        "error": error,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    try:
        _result(json_out, payload, MAX_CELL_REPORT)
        if not passed and not preserve_failure_junit:
            _write_junit(junit_out, [(cell_id, error)])
    except Exception:
        return False
    return passed


CELL_REPORT_FIELDS = {
    "schema",
    "cell",
    "claims_sha256",
    "version",
    "python",
    "sys_platform",
    "platform_system",
    "platform_machine",
    "installer",
    "channel",
    "gwpy_version",
    "pip_version",
    "artifact",
    "pip_freeze",
    "pip_inspect",
    "passed",
    "counters",
    "error",
    "duration_seconds",
}


def _validate_artifact_evidence(
    claims: SimpleNamespace, selected: SimpleNamespace, value: object
) -> None:
    if selected.channel == "conda":
        if value is not None:
            raise QualificationError("conda cell cannot claim an artifact")
        return
    artifact = _keys(value, {"filename", "sha256"}, "cell artifact")
    expected = claims.artifacts[selected.artifact]
    if artifact["filename"] != expected.name or artifact["sha256"] != expected.sha256:
        raise QualificationError("cell artifact does not match claims")


def _freeze_local_file(url: str, expected_sha256: str | None) -> Path:
    parsed = urllib.parse.urlparse(url)
    if expected_sha256 is not None:
        if parsed.fragment != f"sha256={expected_sha256}":
            raise QualificationError("pip freeze artifact hash does not match claims")
    elif parsed.fragment and (
        not parsed.fragment.startswith("sha256=")
        or SHA256.fullmatch(parsed.fragment.removeprefix("sha256=")) is None
    ):
        raise QualificationError("invalid pip freeze file hash")
    without_fragment = urllib.parse.urlunparse(parsed._replace(fragment=""))
    return _local_file_uri(without_fragment)


def _validate_freeze(
    value: object,
    expected_version: str,
    expected_gwpy: str,
    expected_pip: str,
    expected_installer: str,
    expected_artifact: SimpleNamespace | None,
) -> None:
    if not isinstance(value, list) or not value or len(value) > MAX_FREEZE_LINES:
        raise QualificationError("invalid pip freeze facts")
    total = 0
    fact_counts = {project: 0 for project in ("gwexpy", "gwpy", "pip")}
    for line in value:
        if (
            not isinstance(line, str)
            or not line
            or len(line) > MAX_FREEZE_LINE
            or any(ord(character) < 32 for character in line)
        ):
            raise QualificationError("invalid pip freeze line")
        total += len(line.encode("utf-8")) + 1
        if "==" in line:
            project_name, detail = line.split("==", 1)
            form = "pinned"
        elif " @ " in line:
            project_name, detail = line.split(" @ ", 1)
            form = "direct"
        else:
            continue
        project = _normalized_project(project_name)
        if project not in fact_counts:
            continue
        fact_counts[project] += 1
        if project == "gwexpy":
            direct_install = form == "direct" and detail.casefold().startswith("file:")
            if expected_installer == "conda":
                valid = (
                    form == "pinned" and detail == expected_version
                ) or direct_install
                if direct_install:
                    _freeze_local_file(detail, None)
            else:
                valid = direct_install
                if valid:
                    local = _freeze_local_file(
                        detail,
                        expected_artifact.sha256
                        if expected_artifact is not None
                        else None,
                    )
                    valid = (
                        expected_artifact is not None
                        and local.name == expected_artifact.name
                    )
            if not valid:
                raise QualificationError("pip freeze has a forged gwexpy fact")
        elif project == "gwpy":
            if form == "pinned":
                valid = detail == expected_gwpy
            else:
                valid = detail.casefold().startswith("file:")
                if valid:
                    _freeze_local_file(detail, None)
            if not valid:
                raise QualificationError("pip freeze contradicts observed GWpy")
        else:
            valid = (form == "pinned" and detail == expected_pip) or (
                form == "direct" and detail.casefold().startswith("file:")
            )
            if form == "direct" and valid:
                _freeze_local_file(detail, None)
            if not valid:
                raise QualificationError("pip freeze contradicts observed pip")
    if total > MAX_OUTPUT or any(count != 1 for count in fact_counts.values()):
        raise QualificationError("pip freeze omits or duplicates required facts")


def _normalized_project(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _inspect_versions(
    value: object,
    pip_version: str,
    *,
    python_version: str,
    python_series: str,
    sys_platform: str,
    platform_system: str,
    platform_machine: str,
    expected_installer: str,
    expected_artifact: SimpleNamespace | None,
) -> dict[str, str]:
    inspect = _keys(
        value,
        {"version", "pip_version", "installed", "environment"},
        "pip inspect",
    )
    if inspect["version"] != "1" or inspect["pip_version"] != pip_version:
        raise QualificationError("pip inspect identity mismatch")
    installed = inspect["installed"]
    environment = _keys(
        inspect["environment"], INSPECT_ENVIRONMENT_FIELDS, "pip inspect environment"
    )
    if not isinstance(installed, list) or not installed or len(installed) > 10_000:
        raise QualificationError("invalid pip inspect facts")
    for key, fact in environment.items():
        _fact_string(fact, f"pip inspect environment {key}")
    expected_os_name = "nt" if sys_platform == "win32" else "posix"
    if (
        environment["implementation_name"] != "cpython"
        or environment["platform_python_implementation"] != "CPython"
        or environment["implementation_version"] != python_version
        or environment["python_full_version"] != python_version
        or environment["python_version"] != python_series
        or environment["os_name"] != expected_os_name
        or environment["sys_platform"] != sys_platform
        or environment["platform_system"] != platform_system
        or environment["platform_machine"] != platform_machine
    ):
        raise QualificationError("pip inspect environment contradicts runtime facts")
    try:
        encoded = json.dumps(inspect, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise QualificationError("invalid pip inspect values") from exc
    if len(encoded) > MAX_INSPECT:
        raise QualificationError("pip inspect facts exceed output limit")
    versions: dict[str, str] = {}
    for item in installed:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            raise QualificationError("invalid pip inspect distribution")
        name = _fact_string(item["metadata"].get("name"), "distribution name", 256)
        version = _fact_string(
            item["metadata"].get("version"), "distribution version", 256
        )
        normalized = _normalized_project(name)
        if normalized in {"gwexpy", "gwpy"}:
            if normalized in versions:
                raise QualificationError("duplicate distribution in pip inspect")
            versions[normalized] = version
        if normalized == "gwexpy":
            installer = _fact_string(item.get("installer"), "gwexpy installer", 64)
            if installer != expected_installer:
                raise QualificationError("pip inspect installer contradicts channel")
            direct_url = item.get("direct_url")
            if expected_installer == "conda":
                if direct_url is not None:
                    if not isinstance(direct_url, dict) or "url" not in direct_url:
                        raise QualificationError("invalid conda direct URL")
                    _local_file_uri(direct_url["url"])
            else:
                direct = _keys(direct_url, {"url", "archive_info"}, "gwexpy direct URL")
                archive = direct["archive_info"]
                if (
                    expected_artifact is None
                    or not isinstance(archive, dict)
                    or not archive
                    or set(archive) - {"hash", "hashes"}
                ):
                    raise QualificationError("invalid gwexpy archive evidence")
                hashes = archive.get("hashes")
                declared = archive.get("hash")
                local = _local_file_uri(direct["url"])
                if (
                    local.name != expected_artifact.name
                    or (declared is None and hashes is None)
                    or (
                        declared is not None
                        and declared != f"sha256={expected_artifact.sha256}"
                    )
                    or (
                        hashes is not None
                        and (
                            not isinstance(hashes, dict)
                            or set(hashes) != {"sha256"}
                            or hashes.get("sha256") != expected_artifact.sha256
                        )
                    )
                ):
                    raise QualificationError(
                        "pip inspect does not attest selected artifact"
                    )
    if set(versions) != {"gwexpy", "gwpy"}:
        raise QualificationError("pip inspect omits gwexpy or GWpy")
    return versions


def _validate_cell_report(
    claims: SimpleNamespace,
    cell_id: str,
    report: dict[str, Any],
    junit_counters: dict[str, int],
) -> None:
    if set(report) != CELL_REPORT_FIELDS:
        raise QualificationError("cell report has missing or unknown keys")
    selected = claims.required_cells[cell_id]
    if (
        report["schema"] != "gwexpy-published-release-cell-report-v2"
        or report["cell"] != cell_id
        or report["claims_sha256"] != claims.digest
        or report["version"] != claims.version
    ):
        raise QualificationError("cell report identity mismatch")

    python_runtime = _fact_string(report["python"], "Python runtime")
    python_version = python_runtime.split(maxsplit=1)[0]
    if SEMVER.fullmatch(python_version) is None or not python_version.startswith(
        f"{selected.python}."
    ):
        raise QualificationError("cell Python does not match claims")
    expected_platform = {
        "linux": ("linux", "Linux"),
        "macos": ("darwin", "Darwin"),
        "windows": ("win32", "Windows"),
    }[selected.platform]
    if (
        report["sys_platform"] != expected_platform[0]
        or report["platform_system"] != expected_platform[1]
    ):
        raise QualificationError("cell platform does not match claims")
    _fact_string(report["platform_machine"], "platform machine", 256)

    expected_installer = "pip" if selected.channel == "pypi" else "conda"
    expected_channel = "pypi" if selected.channel == "pypi" else selected.conda_channel
    if (
        report["installer"] != expected_installer
        or report["channel"] != expected_channel
    ):
        raise QualificationError("installer or channel does not match claims")
    gwpy_version = _fact_string(report["gwpy_version"], "GWpy version", 256)
    pip_version = _fact_string(report["pip_version"], "pip version", 256)
    expected_gwpy = getattr(selected, "gwpy", None)
    if expected_gwpy is not None and gwpy_version != expected_gwpy:
        raise QualificationError("GWpy version does not match claims")
    _validate_artifact_evidence(claims, selected, report["artifact"])
    expected_artifact = (
        claims.artifacts[selected.artifact] if selected.channel == "pypi" else None
    )
    _validate_freeze(
        report["pip_freeze"],
        claims.version,
        gwpy_version,
        pip_version,
        expected_installer,
        expected_artifact,
    )
    inspected = _inspect_versions(
        report["pip_inspect"],
        pip_version,
        python_version=python_version,
        python_series=selected.python,
        sys_platform=report["sys_platform"],
        platform_system=report["platform_system"],
        platform_machine=report["platform_machine"],
        expected_installer=expected_installer,
        expected_artifact=expected_artifact,
    )
    if inspected["gwexpy"] != claims.version or inspected["gwpy"] != gwpy_version:
        raise QualificationError("pip inspect versions do not match observed facts")

    duration = report["duration_seconds"]
    if (
        report["passed"] is not True
        or report["error"] is not None
        or not _valid_counters(report["counters"])
        or report["counters"] != junit_counters
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 <= duration <= 3600
    ):
        raise QualificationError("cell result does not match JUnit")


def _record_error(errors: dict[str, str], key: str, error: BaseException | str) -> None:
    errors[key] = (
        _exception_text(error) if isinstance(error, BaseException) else error[:MAX_FACT]
    )


def aggregate(
    claims: SimpleNamespace,
    artifact_dir: Path,
    reports_dir: Path,
    pypi_json: Path | None,
    payload_sidecar: Path | None,
    json_out: Path,
    junit_out: Path,
) -> bool:
    """Cross-check a complete cell ledger and write aggregate evidence on failure."""
    errors: dict[str, str] = {}
    try:
        aggregate_inputs: list[Path | None] = [
            getattr(claims, "source_path", None),
            pypi_json,
            payload_sidecar,
        ]
        aggregate_inputs.extend(
            Path(artifact_dir) / artifact.name for artifact in claims.artifacts.values()
        )
        aggregate_inputs.extend(
            Path(reports_dir) / f"{cell}.{suffix}"
            for cell in claims.required_cells
            for suffix in ("json", "xml")
        )
        _disjoint_outputs(
            json_out,
            junit_out,
            aggregate_inputs,
            [Path(artifact_dir), Path(reports_dir)],
        )
    except Exception:
        return False
    try:
        verify_artifact_directory(claims, artifact_dir)
    except Exception as exc:
        _record_error(errors, "identity-artifacts", exc)
    try:
        if pypi_json is None:
            raise QualificationError("missing required PyPI identity evidence")
        validate_pypi_json(claims, _json_file(pypi_json, "PyPI JSON"))
    except Exception as exc:
        _record_error(errors, "identity-pypi", exc)
    try:
        if payload_sidecar is None:
            raise QualificationError("missing required payload identity evidence")
        validate_payload_sidecar(claims, _json_file(payload_sidecar, "payload sidecar"))
    except Exception as exc:
        _record_error(errors, "identity-sidecar", exc)

    entries: dict[str, Path] = {}
    try:
        if reports_dir.is_symlink() or not reports_dir.is_dir():
            raise QualificationError("reports directory must be real")
        entries = {item.name: item for item in reports_dir.iterdir()}
        expected = {
            f"{cell}.{suffix}"
            for cell in claims.required_cells
            for suffix in ("json", "xml")
        }
        if set(entries) - expected:
            raise QualificationError("extra or unsafe qualification reports")
    except Exception as exc:
        _record_error(errors, "identity-reports", exc)

    for cell_id in claims.required_cells:
        json_path = entries.get(f"{cell_id}.json")
        junit_path = entries.get(f"{cell_id}.xml")
        if json_path is None or junit_path is None:
            _record_error(errors, cell_id, "missing qualification report")
            continue
        try:
            report = _json_file(json_path, "cell JSON", MAX_CELL_REPORT)
            counters = parse_junit(junit_path)
            _validate_cell_report(claims, cell_id, report, counters)
        except Exception as exc:
            _record_error(errors, cell_id, exc)
    passed = not errors
    cases = [(cell, errors.get(cell)) for cell in sorted(claims.required_cells)]
    for identity in (
        "identity-artifacts",
        "identity-pypi",
        "identity-sidecar",
        "identity-reports",
    ):
        cases.append((identity, errors.get(identity)))
    _write_junit(junit_out, cases)
    _result(
        json_out,
        {
            "schema": "gwexpy-published-release-aggregate-v1",
            "claims_sha256": claims.digest,
            "passed": passed,
            "required_cells": sorted(claims.required_cells),
            "errors": errors,
        },
    )
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-cell")
    run.add_argument("--claims", type=Path, required=True)
    run.add_argument("--cell", required=True)
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--artifact", type=Path)
    run.add_argument("--json-out", type=Path, required=True)
    run.add_argument("--junit-out", type=Path, required=True)
    summary = commands.add_parser("aggregate")
    summary.add_argument("--claims", type=Path, required=True)
    summary.add_argument("--artifact-dir", type=Path, required=True)
    summary.add_argument("--reports-dir", type=Path, required=True)
    summary.add_argument("--pypi-json", type=Path, required=True)
    summary.add_argument("--payload-sidecar", type=Path, required=True)
    summary.add_argument("--json-out", type=Path, required=True)
    summary.add_argument("--junit-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        claims = load_claims(args.claims)
        passed = (
            run_cell(
                claims,
                args.cell,
                args.repo_root,
                args.artifact,
                args.json_out,
                args.junit_out,
            )
            if args.command == "run-cell"
            else aggregate(
                claims,
                args.artifact_dir,
                args.reports_dir,
                args.pypi_json,
                args.payload_sidecar,
                args.json_out,
                args.junit_out,
            )
        )
    except Exception as exc:
        message = _exception_text(exc)
        try:
            fallback_inputs: list[Path | None] = [args.claims]
            fallback_roots: list[Path] = []
            if args.command == "run-cell":
                fallback_inputs.append(args.artifact)
                fallback_roots.append(args.repo_root)
            else:
                fallback_inputs.extend([args.pypi_json, args.payload_sidecar])
                fallback_roots.extend([args.artifact_dir, args.reports_dir])
            _disjoint_outputs(
                args.json_out,
                args.junit_out,
                fallback_inputs,
                fallback_roots,
            )
            _result(
                args.json_out,
                {
                    "schema": "gwexpy-published-release-error-v1",
                    "passed": False,
                    "error": message,
                },
            )
            _write_junit(args.junit_out, [(args.command, message)])
        except (QualificationError, OSError):
            pass
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
