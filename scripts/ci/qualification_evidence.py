#!/usr/bin/env python3
"""Build fail-closed release qualification evidence for v0.2.2/v0.2.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import unicodedata
import xml.etree.ElementTree as etree
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUALIFICATION_CELLS = (
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

V023_BASELINE_SCHEMA = "gwexpy-v023-qualification-expected-skips-v1"
_CONTRACTS: dict[str, dict[str, str | None]] = {
    "0.2.2": {
        "artifact_prefix": "v022-qualification-evidence",
        "evidence_schema": "gwexpy-v022-qualification-evidence-v1",
        "expected_skips_schema": None,
    },
    "0.2.3": {
        "artifact_prefix": "v023-qualification-evidence",
        "evidence_schema": "gwexpy-v023-qualification-evidence-v1",
        "expected_skips_schema": V023_BASELINE_SCHEMA,
    },
}
_PAYLOAD_SCHEMAS = {
    "0.2.2": "gwexpy-v022-release-payload-v1",
    "0.2.3": "gwexpy-v023-release-payload-v1",
}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JUNIT_COUNTER = re.compile(r"^(0|[1-9][0-9]*)$")
_JUNIT_TAGS = {
    "error",
    "failure",
    "properties",
    "property",
    "skipped",
    "system-err",
    "system-out",
    "testcase",
    "testsuite",
    "testsuites",
}
_MAX_BASELINE_BYTES = 256 * 1024
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_JUNIT_BYTES = 32 * 1024 * 1024
_MAX_SKIP_FIELD_BYTES = 4096

SkipCase = tuple[str, str, str]


class QualificationEvidenceError(ValueError):
    """Raised when qualification inputs or evidence are incomplete or unsafe."""


class _DuplicateJSONKey(ValueError):
    """Raised when JSON contains an ambiguous duplicate object key."""


@dataclass(frozen=True)
class ExpectedSkips:
    """Reviewed optional skip sets and the byte identity of their baseline."""

    cells: dict[str, tuple[SkipCase, ...]]
    sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _canonical_json_bytes(data: object) -> bytes:
    try:
        text = json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise QualificationEvidenceError("JSON contains a noncanonical value") from exc
    return (text + "\n").encode("utf-8")


def _reject_symlink_ancestors(path: Path, *, description: str) -> None:
    try:
        for ancestor in path.absolute().parents:
            metadata = ancestor.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise QualificationEvidenceError(
                    f"{description} has a symlink ancestor"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise QualificationEvidenceError(
                    f"{description} ancestor must be a directory"
                )
    except QualificationEvidenceError:
        raise
    except OSError as exc:
        raise QualificationEvidenceError(
            f"cannot inspect {description} ancestors"
        ) from exc


def _read_regular_bytes(path: Path, *, maximum: int, description: str) -> bytes:
    try:
        _reject_symlink_ancestors(path, description=description)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise QualificationEvidenceError(f"{description} must be a regular file")
        if metadata.st_size > maximum:
            raise QualificationEvidenceError(f"{description} exceeds size limit")
        return path.read_bytes()
    except QualificationEvidenceError:
        raise
    except OSError as exc:
        raise QualificationEvidenceError(f"cannot read {description}") from exc


def _load_json(
    path: Path,
    *,
    maximum: int = _MAX_JSON_BYTES,
    description: str,
    require_canonical: bool = False,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, maximum=maximum, description=description)
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJSONKey) as exc:
        raise QualificationEvidenceError(f"invalid {description} JSON") from exc
    if not isinstance(data, dict):
        raise QualificationEvidenceError(f"{description} must be a JSON object")
    if require_canonical and raw != _canonical_json_bytes(data):
        raise QualificationEvidenceError(
            f"{description} must use canonical JSON serialization"
        )
    return data, raw


def _require_exact_keys(
    value: object, expected: set[str], *, description: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise QualificationEvidenceError(f"{description} has unknown or missing keys")
    return value


def _safe_skip_field(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise QualificationEvidenceError(f"skip {field} is unsafe or noncanonical")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise QualificationEvidenceError(
            f"skip {field} is unsafe or noncanonical"
        ) from exc
    if (
        not value
        or value != value.strip()
        or len(encoded) > _MAX_SKIP_FIELD_BYTES
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise QualificationEvidenceError(f"skip {field} is unsafe or noncanonical")
    return value


def _skip_case(value: object) -> SkipCase:
    if not isinstance(value, list) or len(value) != 3:
        raise QualificationEvidenceError(
            "skip record must be [classname, name, message]"
        )
    return (
        _safe_skip_field(value[0], field="classname"),
        _safe_skip_field(value[1], field="name"),
        _safe_skip_field(value[2], field="message"),
    )


def _skip_sort_key(value: SkipCase) -> tuple[bytes, bytes, bytes]:
    return (
        value[0].encode("utf-8"),
        value[1].encode("utf-8"),
        value[2].encode("utf-8"),
    )


def _sorted_skip_lists(values: set[SkipCase] | tuple[SkipCase, ...]) -> list[list[str]]:
    return [list(value) for value in sorted(values, key=_skip_sort_key)]


def qualification_contract(version: str) -> dict[str, str | None]:
    """Return the exact evidence contract for an allowed release version."""
    try:
        return dict(_CONTRACTS[version])
    except KeyError as exc:
        raise QualificationEvidenceError(
            f"unsupported qualification version: {version}"
        ) from exc


def load_expected_skips(path: Path | str) -> ExpectedSkips:
    """Load the canonical reviewed v0.2.3 optional-skip baseline."""
    data, raw = _load_json(
        Path(path),
        maximum=_MAX_BASELINE_BYTES,
        description="expected-skip baseline",
        require_canonical=True,
    )
    _require_exact_keys(data, {"cells", "schema", "version"}, description="baseline")
    if data["schema"] != V023_BASELINE_SCHEMA:
        raise QualificationEvidenceError("invalid expected-skip baseline schema")
    if data["version"] != "0.2.3":
        raise QualificationEvidenceError("invalid expected-skip baseline version")
    records = data["cells"]
    if not isinstance(records, list) or len(records) != len(QUALIFICATION_CELLS):
        raise QualificationEvidenceError(
            "expected-skip baseline must contain exactly 19 cells"
        )

    cells: dict[str, tuple[SkipCase, ...]] = {}
    observed_order: list[str] = []
    for raw_record in records:
        record = _require_exact_keys(
            raw_record, {"cell", "optional_skips"}, description="baseline cell"
        )
        cell = record["cell"]
        if not isinstance(cell, str) or cell not in QUALIFICATION_CELLS:
            raise QualificationEvidenceError("baseline contains an unknown cell")
        if cell in cells:
            raise QualificationEvidenceError("baseline contains a duplicate cell")
        raw_skips = record["optional_skips"]
        if not isinstance(raw_skips, list):
            raise QualificationEvidenceError("optional_skips must be a list")
        skips = tuple(_skip_case(value) for value in raw_skips)
        if list(raw_skips) != _sorted_skip_lists(set(skips)):
            raise QualificationEvidenceError(
                "optional skip records must be sorted and unique"
            )
        cells[cell] = skips
        observed_order.append(cell)

    if tuple(observed_order) != QUALIFICATION_CELLS:
        raise QualificationEvidenceError(
            "baseline cells are missing, duplicated, or noncanonical"
        )
    return ExpectedSkips(cells=cells, sha256=hashlib.sha256(raw).hexdigest())


def _parse_junit(path: Path) -> tuple[int, tuple[SkipCase, ...]]:
    raw = _read_regular_bytes(
        path, maximum=_MAX_JUNIT_BYTES, description="pytest JUnit report"
    )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QualificationEvidenceError(
            "pytest JUnit report must be strict UTF-8"
        ) from exc
    if "\x00" in text:
        raise QualificationEvidenceError("pytest JUnit report must be strict UTF-8 XML")
    upper = text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise QualificationEvidenceError("pytest JUnit report contains unsafe XML")
    try:
        root = etree.fromstring(text)
    except etree.ParseError as exc:
        raise QualificationEvidenceError("invalid pytest JUnit XML") from exc
    if any(
        not isinstance(element.tag, str) or element.tag not in _JUNIT_TAGS
        for element in root.iter()
    ):
        raise QualificationEvidenceError(
            "pytest JUnit report contains an unknown or namespaced tag"
        )
    root_children = list(root)
    if (
        root.tag != "testsuites"
        or len(root_children) != 1
        or root_children[0].tag != "testsuite"
    ):
        raise QualificationEvidenceError(
            "pytest JUnit report has an invalid single-suite hierarchy"
        )
    suite = root_children[0]
    testcases = [child for child in suite if child.tag == "testcase"]
    allowed_suite_children = {"properties", "testcase"}
    if (
        any(child.tag not in allowed_suite_children for child in suite)
        or sum(child.tag == "properties" for child in suite) > 1
        or list(suite.iter("testcase")) != testcases
        or any(
            descendant is not suite and descendant.tag in {"testsuite", "testsuites"}
            for descendant in suite.iter()
        )
    ):
        raise QualificationEvidenceError(
            "pytest JUnit report has an invalid testcase hierarchy"
        )
    if not testcases:
        raise QualificationEvidenceError("pytest JUnit report has no testcases")

    declared: dict[str, int] = {}
    for counter in ("tests", "failures", "errors", "skipped"):
        raw_counter = suite.get(counter)
        if raw_counter is None or _JUNIT_COUNTER.fullmatch(raw_counter) is None:
            raise QualificationEvidenceError(
                f"pytest JUnit report has an invalid {counter} counter"
            )
        declared[counter] = int(raw_counter)

    outcomes = {"failures": 0, "errors": 0, "skipped": 0}
    testcase_statuses: list[list[etree.Element[str]]] = []
    outcome_names = {"failure": "failures", "error": "errors", "skipped": "skipped"}
    allowed_testcase_children = {
        "error",
        "failure",
        "properties",
        "skipped",
        "system-err",
        "system-out",
    }
    for testcase in testcases:
        if any(child.tag not in allowed_testcase_children for child in testcase) or any(
            sum(child.tag == singleton for child in testcase) > 1
            for singleton in ("properties", "system-err", "system-out")
        ):
            raise QualificationEvidenceError(
                "pytest JUnit testcase has an invalid child hierarchy"
            )
        direct = [child for child in testcase if child.tag in outcome_names]
        nested = [
            descendant
            for descendant in testcase.iter()
            if descendant is not testcase and descendant.tag in outcome_names
        ]
        if nested != direct or len(direct) > 1:
            raise QualificationEvidenceError(
                "pytest JUnit testcase has an ambiguous outcome hierarchy"
            )
        testcase_statuses.append(direct)
        if direct:
            outcomes[outcome_names[direct[0].tag]] += 1

    for properties in root.iter("properties"):
        if any(child.tag != "property" for child in properties):
            raise QualificationEvidenceError(
                "pytest JUnit properties has an invalid child hierarchy"
            )
    for terminal_tag in (
        "error",
        "failure",
        "property",
        "skipped",
        "system-err",
        "system-out",
    ):
        if any(list(element) for element in root.iter(terminal_tag)):
            raise QualificationEvidenceError(
                "pytest JUnit terminal element has an invalid child hierarchy"
            )

    direct_status_ids = {
        id(status) for statuses in testcase_statuses for status in statuses
    }
    if any(
        descendant.tag in outcome_names and id(descendant) not in direct_status_ids
        for descendant in suite.iter()
    ):
        raise QualificationEvidenceError(
            "pytest JUnit report has an invalid outcome hierarchy"
        )

    if declared["tests"] != len(testcases) or any(
        declared[counter] != outcomes[counter]
        for counter in ("failures", "errors", "skipped")
    ):
        raise QualificationEvidenceError(
            "pytest JUnit declared counters do not match testcase outcomes"
        )
    if outcomes["failures"] or outcomes["errors"]:
        raise QualificationEvidenceError(
            "pytest JUnit report contains a failed or errored testcase"
        )

    skips: list[SkipCase] = []
    for testcase, statuses in zip(testcases, testcase_statuses, strict=True):
        if not statuses:
            continue
        skipped = statuses[0]
        classname = testcase.get("classname")
        name = testcase.get("name")
        message = skipped.get("message")
        if classname is None:
            raise QualificationEvidenceError("skipped testcase has no classname")
        if name is None:
            raise QualificationEvidenceError("skipped testcase has no name")
        if message is None:
            raise QualificationEvidenceError("skipped testcase has no message")
        skips.append(
            (
                _safe_skip_field(classname, field="classname"),
                _safe_skip_field(name, field="name"),
                _safe_skip_field(message, field="message"),
            )
        )
    if len(skips) != len(set(skips)):
        raise QualificationEvidenceError(
            "pytest JUnit report contains a duplicate skipped testcase"
        )
    return len(testcases), tuple(sorted(skips, key=_skip_sort_key))


def parse_junit_skips(path: Path | str) -> list[list[str]]:
    """Return sorted canonical ``[classname, name, message]`` skip records."""
    _, skips = _parse_junit(Path(path))
    return _sorted_skip_lists(skips)


def _payload_files(
    path: Path, *, version: str, source_sha: str
) -> dict[str, dict[str, str]]:
    data, _ = _load_json(path, description="payload manifest")
    _require_exact_keys(
        data,
        {"schema", "source_sha", "version", "files"},
        description="payload manifest",
    )
    if (
        data["schema"] != _PAYLOAD_SCHEMAS[version]
        or data["version"] != version
        or data["source_sha"] != source_sha
    ):
        raise QualificationEvidenceError(
            "payload manifest is not bound to the qualification candidate"
        )
    files = _require_exact_keys(
        data["files"], {"wheel", "sdist"}, description="payload files"
    )
    result: dict[str, dict[str, str]] = {}
    for kind, suffix in (("wheel", ".whl"), ("sdist", ".tar.gz")):
        entry = _require_exact_keys(
            files[kind], {"name", "sha256"}, description=f"payload {kind}"
        )
        name = entry["name"]
        digest = entry["sha256"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or not name.endswith(suffix)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise QualificationEvidenceError(f"invalid payload {kind} entry")
        if (kind == "sdist" and name != f"gwexpy-{version}.tar.gz") or (
            kind == "wheel"
            and re.fullmatch(
                rf"gwexpy-{re.escape(version)}-[^-]+-[^-]+-[^-]+\.whl", name
            )
            is None
        ):
            raise QualificationEvidenceError(
                f"payload {kind} entry does not match candidate version"
            )
        result[kind] = {"name": name, "sha256": digest}
    return result


def _validate_identity(version: str, cell: str, source_sha: str) -> None:
    qualification_contract(version)
    if cell not in QUALIFICATION_CELLS:
        raise QualificationEvidenceError(f"unknown cell: {cell}")
    if _SHA40.fullmatch(source_sha) is None:
        raise QualificationEvidenceError("source_sha must be a full lowercase SHA")


def _write_json(path: Path, data: object, *, canonical: bool) -> None:
    if path.exists() or path.is_symlink():
        raise QualificationEvidenceError("evidence output already exists")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            _canonical_json_bytes(data)
            if canonical
            else (json.dumps(data, sort_keys=True) + "\n").encode("utf-8")
        )
        path.write_bytes(content)
    except QualificationEvidenceError:
        raise
    except OSError as exc:
        raise QualificationEvidenceError("cannot write qualification evidence") from exc


def record_cell(
    *,
    version: str,
    cell: str,
    source_sha: str,
    payload_manifest: Path | str,
    report_path: Path | str,
    junit_path: Path | str | None = None,
    expected_skips_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one executed cell and write its immutable evidence record."""
    _validate_identity(version, cell, source_sha)
    files = _payload_files(
        Path(payload_manifest), version=version, source_sha=source_sha
    )
    if version == "0.2.2":
        if junit_path is not None or expected_skips_path is not None:
            raise QualificationEvidenceError(
                "v0.2.2 historical evidence does not accept skip inputs"
            )
        report: dict[str, Any] = {
            "cell": cell,
            "source_sha": source_sha,
            "version": version,
            "files": files,
            "status": "passed",
        }
        _write_json(Path(report_path), report, canonical=False)
        return report

    if junit_path is None or expected_skips_path is None:
        raise QualificationEvidenceError(
            "v0.2.3 evidence requires JUnit and expected-skip baseline"
        )
    baseline = load_expected_skips(expected_skips_path)
    testcase_count, observed = _parse_junit(Path(junit_path))
    approved = set(baseline.cells[cell])
    required = set(observed) - approved
    optional = set(observed) & approved
    if required:
        rendered = json.dumps(_sorted_skip_lists(required), ensure_ascii=True)
        raise QualificationEvidenceError(
            f"required observed skips are not empty: {rendered}"
        )

    report = {
        "baseline_sha256": baseline.sha256,
        "cell": cell,
        "files": files,
        "observed_optional_skips": _sorted_skip_lists(optional),
        "observed_required_skips": [],
        "observed_skips": _sorted_skip_lists(set(observed)),
        "source_sha": source_sha,
        "status": "passed",
        "testcase_count": testcase_count,
        "version": version,
    }
    _write_json(Path(report_path), report, canonical=True)
    return report


def _load_cell_reports(
    reports_dir: Path, *, require_canonical: bool
) -> list[dict[str, Any]]:
    _reject_symlink_ancestors(
        reports_dir, description="qualification reports directory"
    )
    if reports_dir.is_symlink() or not reports_dir.is_dir():
        raise QualificationEvidenceError(
            "qualification reports must be a real directory"
        )
    paths = sorted(reports_dir.rglob("qualification.json"))
    if len(paths) != len(QUALIFICATION_CELLS):
        raise QualificationEvidenceError(
            "qualification evidence must contain exactly 19 records"
        )
    return [
        _load_json(
            path,
            description="qualification cell report",
            require_canonical=require_canonical,
        )[0]
        for path in paths
    ]


def _validated_skip_list(value: object, *, description: str) -> tuple[SkipCase, ...]:
    if not isinstance(value, list):
        raise QualificationEvidenceError(f"{description} must be a list")
    skips = tuple(_skip_case(item) for item in value)
    if list(value) != _sorted_skip_lists(set(skips)):
        raise QualificationEvidenceError(f"{description} must be sorted and unique")
    return skips


def aggregate_reports(
    *,
    version: str,
    source_sha: str,
    payload_manifest: Path | str,
    reports_dir: Path | str,
    output_path: Path | str,
    expected_skips_path: Path | str | None = None,
) -> dict[str, Any]:
    """Require all 19 cells and emit the versioned aggregate ledger."""
    qualification_contract(version)
    if _SHA40.fullmatch(source_sha) is None:
        raise QualificationEvidenceError("source_sha must be a full lowercase SHA")
    files = _payload_files(
        Path(payload_manifest), version=version, source_sha=source_sha
    )
    reports = _load_cell_reports(
        Path(reports_dir), require_canonical=version == "0.2.3"
    )
    observed_cells: set[str] = set()

    if version == "0.2.2":
        if expected_skips_path is not None:
            raise QualificationEvidenceError(
                "v0.2.2 historical evidence does not accept a skip baseline"
            )
        for report in reports:
            cell = report.get("cell")
            if not isinstance(cell, str) or cell not in QUALIFICATION_CELLS:
                raise QualificationEvidenceError(
                    "qualification report has unknown cell"
                )
            if cell in observed_cells:
                raise QualificationEvidenceError(
                    "qualification reports duplicate a cell"
                )
            expected = {
                "cell": cell,
                "source_sha": source_sha,
                "version": version,
                "files": files,
                "status": "passed",
            }
            if report != expected:
                raise QualificationEvidenceError("invalid v0.2.2 qualification report")
            observed_cells.add(cell)
        if observed_cells != set(QUALIFICATION_CELLS):
            raise QualificationEvidenceError("qualification reports have missing cells")
        aggregate: dict[str, Any] = {
            "schema": str(_CONTRACTS[version]["evidence_schema"]),
            "source_sha": source_sha,
            "version": version,
            "files": files,
            "cells": sorted(observed_cells),
        }
        _write_json(Path(output_path), aggregate, canonical=False)
        return aggregate

    if expected_skips_path is None:
        raise QualificationEvidenceError(
            "v0.2.3 aggregate requires expected-skip baseline"
        )
    baseline = load_expected_skips(expected_skips_path)
    summaries: list[dict[str, Any]] = []
    expected_keys = {
        "baseline_sha256",
        "cell",
        "files",
        "observed_optional_skips",
        "observed_required_skips",
        "observed_skips",
        "source_sha",
        "status",
        "testcase_count",
        "version",
    }
    for report in reports:
        _require_exact_keys(report, expected_keys, description="v0.2.3 cell report")
        cell = report["cell"]
        if not isinstance(cell, str) or cell not in QUALIFICATION_CELLS:
            raise QualificationEvidenceError("qualification report has unknown cell")
        if cell in observed_cells:
            raise QualificationEvidenceError("qualification reports duplicate a cell")
        observed = _validated_skip_list(
            report["observed_skips"], description="observed skips"
        )
        optional = _validated_skip_list(
            report["observed_optional_skips"], description="observed optional skips"
        )
        required = _validated_skip_list(
            report["observed_required_skips"], description="observed required skips"
        )
        testcase_count = report["testcase_count"]
        if (
            report["baseline_sha256"] != baseline.sha256
            or report["source_sha"] != source_sha
            or report["version"] != version
            or report["files"] != files
            or report["status"] != "passed"
            or not isinstance(testcase_count, int)
            or isinstance(testcase_count, bool)
            or testcase_count <= 0
            or testcase_count < len(observed)
            or required
            or set(observed) != set(optional)
            or not set(optional) <= set(baseline.cells[cell])
        ):
            raise QualificationEvidenceError(
                "qualification report has invalid candidate or baseline evidence"
            )
        summaries.append(
            {
                "cell": cell,
                "observed_optional_skips": _sorted_skip_lists(set(optional)),
                "observed_required_skips": [],
                "observed_skips": _sorted_skip_lists(set(observed)),
                "testcase_count": testcase_count,
            }
        )
        observed_cells.add(cell)

    if observed_cells != set(QUALIFICATION_CELLS):
        raise QualificationEvidenceError("qualification reports have missing cells")
    aggregate = {
        "baseline_sha256": baseline.sha256,
        "cells": sorted(summaries, key=lambda item: item["cell"]),
        "files": files,
        "schema": str(_CONTRACTS[version]["evidence_schema"]),
        "source_sha": source_sha,
        "version": version,
    }
    _write_json(Path(output_path), aggregate, canonical=True)
    return aggregate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract = subparsers.add_parser("contract")
    contract.add_argument("--version", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--version", required=True)
    record.add_argument("--cell", required=True)
    record.add_argument("--source-sha", required=True)
    record.add_argument("--payload-manifest", type=Path, required=True)
    record.add_argument("--report", type=Path, required=True)
    record.add_argument("--junit", type=Path)
    record.add_argument("--expected-skips", type=Path)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--version", required=True)
    aggregate.add_argument("--source-sha", required=True)
    aggregate.add_argument("--payload-manifest", type=Path, required=True)
    aggregate.add_argument("--reports-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--expected-skips", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "contract":
            contract = qualification_contract(args.version)
            print(f"evidence_schema={contract['evidence_schema']}")
            print(f"artifact_prefix={contract['artifact_prefix']}")
        elif args.command == "record":
            record_cell(
                version=args.version,
                cell=args.cell,
                source_sha=args.source_sha,
                payload_manifest=args.payload_manifest,
                report_path=args.report,
                junit_path=args.junit,
                expected_skips_path=args.expected_skips,
            )
        else:
            aggregate_reports(
                version=args.version,
                source_sha=args.source_sha,
                payload_manifest=args.payload_manifest,
                reports_dir=args.reports_dir,
                output_path=args.output,
                expected_skips_path=args.expected_skips,
            )
    except QualificationEvidenceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
