#!/usr/bin/env python3
"""Run and aggregate the exact-artifact v0.2.5 historical release gate.

The twelve historical BLOCKED cases are characterized and dispositioned; they
are never counted among the 62 executable scenario passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

VERSION = "0.2.5"
CELL_SCHEMA = "gwexpy-v025-historical-cell-v1"
AGGREGATE_SCHEMA = "gwexpy-v025-historical-74-gate-v1"
MATRIX_REL = Path(
    "docs/developers/reports/2026-09-27-public-io-cross-format-post-fix-matrix.json"
)
AUDIT_REL = Path("docs/developers/reports/2026-09-27-public-io-cross-format-audit")
PREQUAL_REL = Path("docs/developers/reports/v0.2.5-s3-prequalification")
DISPOSITION_REL = PREQUAL_REL / "disposition-proposal.md"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HistoricalGateError(ValueError):
    """Release evidence is incomplete or not bound to this candidate."""


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HistoricalGateError("duplicate JSON key")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 5_000_000:
        raise HistoricalGateError(f"missing, symlinked, or oversized evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoricalGateError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise HistoricalGateError(f"expected JSON object: {path}")
    return value


def _approval(path: Path) -> dict[str, Any]:
    """Read the canonical YAML block using the release evidence parser."""
    if path.is_symlink() or not path.is_file():
        raise HistoricalGateError("approval must be a regular file")
    validator = Path(__file__).with_name("validate_release_review_evidence.py")
    spec = importlib.util.spec_from_file_location("v025_review_document", validator)
    if spec is None or spec.loader is None:
        raise HistoricalGateError("review evidence parser unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module._load_review_document(path)
    except module.ReleaseReviewEvidenceError as exc:
        raise HistoricalGateError(f"invalid approval document: {exc}") from exc


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HistoricalGateError(f"missing or symlinked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise HistoricalGateError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _payload(manifest: Path, source_sha: str) -> dict[str, dict[str, str]]:
    data = _load(manifest)
    if (
        set(data) != {"schema", "source_sha", "version", "files"}
        or data["schema"] != "gwexpy-v025-release-payload-v1"
        or data["version"] != VERSION
        or data["source_sha"] != source_sha
        or SHA40.fullmatch(source_sha) is None
    ):
        raise HistoricalGateError("payload manifest is not bound to candidate")
    files = data["files"]
    if not isinstance(files, dict) or set(files) != {"wheel", "sdist"}:
        raise HistoricalGateError("payload must have exactly wheel and sdist")
    for kind, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256"}:
            raise HistoricalGateError(f"invalid {kind} payload entry")
        name, digest = entry["name"], entry["sha256"]
        if not isinstance(name, str) or Path(name).name != name or "\\" in name:
            raise HistoricalGateError(f"invalid {kind} filename")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise HistoricalGateError(f"invalid {kind} digest")
        if (
            kind == "wheel"
            and re.fullmatch(r"gwexpy-0\.2\.5-[^-]+-[^-]+-[^-]+\.whl", name) is None
        ):
            raise HistoricalGateError("wrong wheel filename")
        if kind == "sdist" and name != "gwexpy-0.2.5.tar.gz":
            raise HistoricalGateError("wrong sdist filename")
    return files


def _findings(source_root: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    matrix = _load(source_root / MATRIX_REL)
    findings = matrix.get("findings")
    if not isinstance(findings, list) or len(findings) != 74:
        raise HistoricalGateError("historical matrix must contain 74 findings")
    fixed = {
        f["finding_id"] for f in findings if f["post_fix_status"] == "FIX_VERIFIED"
    }
    remaining = [f for f in findings if f["post_fix_status"] == "NOT_REQUALIFIED"]
    behavior = {
        f["finding_id"] for f in remaining if f["baseline_finding_status"] != "BLOCKED"
    }
    blocked = {
        f["finding_id"] for f in remaining if f["baseline_finding_status"] == "BLOCKED"
    }
    if (len(fixed), len(behavior), len(blocked)) != (36, 26, 12):
        raise HistoricalGateError("historical 36+26+12 accounting changed")
    if len(fixed | behavior | blocked) != 74:
        raise HistoricalGateError("historical finding IDs overlap")
    nodes = {
        node
        for f in findings
        if f["finding_id"] in fixed
        for node in f["regression_tests"]
    }
    if len(nodes) != 23:
        raise HistoricalGateError("fixed-defect regression coverage changed")
    return fixed, behavior, blocked, nodes


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdout: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise HistoricalGateError(
            f"command failed ({result.returncode}): {command!r}\n{result.stderr[-6000:]}"
        )
    if stdout is not None:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text(result.stdout, encoding="utf-8")


def _installed(python: Path, artifact: Path) -> None:
    code = """import importlib.metadata, json, pathlib, sys, urllib.parse, gwexpy
assert gwexpy.__version__ == '0.2.5'
assert pathlib.Path(gwexpy.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
assert 'site-packages' in gwexpy.__file__
raw = importlib.metadata.distribution('gwexpy').read_text('direct_url.json')
assert raw is not None
url = json.loads(raw)['url']
assert pathlib.Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).resolve() == pathlib.Path(sys.argv[1]).resolve()
"""
    _run([str(python), "-c", code, str(artifact)], cwd=artifact.parent)


def _probe_commands() -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    """Return the original 17 commands in fixture dependency order."""
    return (
        ("A-NETCDF-PRESENT", "present", "lane-a/netcdf_probe.py", ()),
        ("A-ZARR-PRESENT", "present", "lane-a/zarr_probe.py", ()),
        (
            "A-NETCDF-DTYPE-FOLLOWUP",
            "present",
            "lane-a/heterogeneous_dtype_probe.py",
            ("--out", "evidence/heterogeneous_dtype.jsonl"),
        ),
        (
            "A-ZARR-FOLLOWUP",
            "present",
            "lane-a/zarr_axis_units_multistore_probe.py",
            ("--out", "evidence/zarr_axis_units_multistore.jsonl"),
        ),
        ("A-OPTIONAL-BASE", "base", "lane-a/missing_probe.py", ()),
        (
            "B-TDMS-PRESENT",
            "present",
            "lane-b/lane_b_probe.py",
            ("--family", "tdms", "--out", "repro/lane-b/fixtures/tdms"),
        ),
        (
            "B-TDMS-BASE",
            "base",
            "lane-b/lane_b_probe.py",
            ("--family", "tdms_missing", "--out", "repro/lane-b/fixtures/tdms"),
        ),
        (
            "B-GBD-PRESENT",
            "present",
            "lane-b/lane_b_probe.py",
            ("--family", "gbd", "--out", "repro/lane-b/fixtures/gbd"),
        ),
        (
            "B-GBD-BASE",
            "base",
            "lane-b/lane_b_probe.py",
            ("--family", "gbd", "--out", "repro/lane-b/fixtures/gbd_base"),
        ),
        (
            "B-ATS-PRESENT",
            "present",
            "lane-b/lane_b_probe.py",
            ("--family", "ats", "--out", "repro/lane-b/fixtures/ats"),
        ),
        (
            "B-ATS-BASE",
            "base",
            "lane-b/lane_b_probe.py",
            ("--family", "ats", "--out", "repro/lane-b/fixtures/ats_base"),
        ),
        ("B-MATRIX-METADATA-PRESENT", "present", "lane-b/matrix_metadata.py", ()),
        ("B-GWPY-ORACLE-PRESENT", "present", "lane-b/gwpy_oracle.py", ()),
        ("C-PRESENT", "present", "lane-c/probe.py", ()),
        ("C-BASE", "base", "lane-c/probe.py", ()),
        ("D-PRESENT", "present", "lane-d/d_lane.py", ()),
        ("D-BASE", "base", "lane-d/d_lane.py", ()),
    )


def run_cell(
    source_root: Path,
    payload_dir: Path,
    manifest: Path,
    source_sha: str,
    kind: str,
    work: Path,
) -> None:
    """Install one exact artifact and execute all historical probes and assertions."""
    files = _payload(manifest, source_sha)
    if kind not in files:
        raise HistoricalGateError("unknown distribution kind")
    artifact = payload_dir / files[kind]["name"]
    if _sha(artifact) != files[kind]["sha256"]:
        raise HistoricalGateError("artifact hash differs from payload manifest")
    if work.exists():
        raise HistoricalGateError("fresh work path required")
    _, _, _, nodes = _findings(source_root)
    work.mkdir(parents=True)
    shutil.copytree(source_root / AUDIT_REL / "repro", work / "repro")
    shutil.copytree(source_root / "tests/io", work / "tests/io")
    for mode in ("base", "present"):
        _run([sys.executable, "-m", "venv", str(work / f"{mode}-venv")], cwd=work)
        python = work / f"{mode}-venv/bin/python"
        spec = (
            f"{artifact}[io,netcdf4,zarr,audio]" if mode == "present" else str(artifact)
        )
        packages = [spec, "pytest", "gwpy==4.0.2", "numpy<2"]
        if mode == "present":
            packages.append("obspy<2")
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                *packages,
            ],
            cwd=work,
        )
        _run([str(python), "-m", "pip", "check"], cwd=work)
        _installed(python, artifact)
    audit_matrix = _load(
        source_root / AUDIT_REL / "runtime-characterization-matrix.json"
    )
    if {c["id"] for c in audit_matrix["commands"]} != {c[0] for c in _probe_commands()}:
        raise HistoricalGateError("historical 17-command matrix changed")
    raw = work / "raw"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["GWEXPY_ALLOW_ZARR"] = "1"
    for command_id, mode, script, args in _probe_commands():
        python = work / f"{mode}-venv/bin/python"
        stdout = raw / kind / f"{command_id}.stdout.jsonl"
        _run(
            [str(python), str(work / "repro" / script), *args],
            cwd=work,
            stdout=stdout,
            env=env,
        )
    nc = raw / f"nc-routes-{kind}.jsonl"
    _run(
        [
            str(work / "present-venv/bin/python"),
            str(source_root / PREQUAL_REL / "probe_netcdf_routes.py"),
            "--fixture-root",
            str(work / "repro/lane-a/netcdf"),
        ],
        cwd=work,
        stdout=nc,
        env=env,
    )
    junit = work / "fixed.xml"
    test_nodes = sorted(node.removeprefix("tests/") for node in nodes)
    _run(
        [
            str(work / "present-venv/bin/python"),
            "-m",
            "pytest",
            "--import-mode=importlib",
            "--junitxml",
            str(junit),
            "-q",
            *["tests/" + node for node in test_nodes],
        ],
        cwd=work,
        env=env,
    )
    _check_junit(source_root, junit, nodes)


def _check_junit(source_root: Path, junit: Path, nodes: set[str]) -> None:
    import xml.etree.ElementTree as ET

    root = ET.parse(junit).getroot()
    suites = list(root.iter("testsuite"))
    if len(suites) != 1:
        raise HistoricalGateError("fixed JUnit needs one testsuite")
    suite = suites[0]
    if any(int(suite.attrib[x]) != 0 for x in ("errors", "failures", "skipped")):
        raise HistoricalGateError("fixed regression failed or skipped")
    cases = list(suite.iter("testcase"))
    if len(cases) != int(suite.attrib["tests"]):
        raise HistoricalGateError("fixed JUnit testcase count mismatch")
    observed = set()
    for case in cases:
        if any(case.find(tag) is not None for tag in ("error", "failure", "skipped")):
            raise HistoricalGateError("fixed regression testcase did not pass")
        classname = case.attrib["classname"].split(".")
        if classname[:2] != ["tests", "io"] or len(classname) < 3:
            raise HistoricalGateError("unexpected fixed regression classname")
        observed.add(
            "tests/io/"
            + classname[2]
            + ".py::"
            + "::".join([*classname[3:], case.attrib["name"].split("[", 1)[0]])
        )
    if observed != nodes:
        raise HistoricalGateError(
            "fixed regression node coverage differs from 23 required nodes"
        )


def _check_qualification(
    qualification: Path,
    files: dict[str, dict[str, str]],
    source_sha: str,
    behavior: set[str],
    blocked: set[str],
) -> None:
    qualified = _load(qualification)
    if (
        qualified.get("schema") != "gwexpy-v025-audit-qualification-v1"
        or qualified.get("source_sha") != source_sha
        or qualified.get("artifact_sha256") != {k: files[k]["sha256"] for k in files}
        or qualified.get("status_counts")
        != {"PASS": 26, "FAIL": 0, "NEEDS_HARNESS": 0, "BLOCKED": 12}
        or qualified.get("release_gate") != "OPEN"
    ):
        raise HistoricalGateError("26+12 qualification status mismatch")
    entries = qualified.get("findings")
    if not isinstance(entries, list) or len(entries) != 38:
        raise HistoricalGateError("qualification has incomplete finding list")
    for status, expected in (("PASS", behavior), ("BLOCKED", blocked)):
        selected = [entry for entry in entries if entry.get("status") == status]
        if {entry.get("finding_id") for entry in selected} != expected or len(
            selected
        ) != len(expected):
            raise HistoricalGateError(f"qualification {status} finding IDs mismatch")
        check_status = "PASS" if status == "PASS" else "CHARACTERIZED"
        if any(
            {k: v.get("status") for k, v in entry.get("artifact_checks", {}).items()}
            != {"wheel": check_status, "sdist": check_status}
            for entry in selected
        ):
            raise HistoricalGateError(
                f"qualification {status} artifact assertions incomplete"
            )


def finalize(
    source_root: Path,
    payload_dir: Path,
    manifest: Path,
    source_sha: str,
    wheel_work: Path,
    sdist_work: Path,
    approval: Path,
    output_dir: Path,
) -> None:
    """Replay both retained verifiers before issuing a 74-case GO manifest."""
    if output_dir.exists() or output_dir.is_symlink():
        raise HistoricalGateError("fresh output directory required")
    files = _payload(manifest, source_sha)
    fixed, behavior, blocked, nodes = _findings(source_root)
    output_dir.mkdir(parents=True)
    raw = output_dir / "raw"
    observations = []
    for kind, work in (("wheel", wheel_work), ("sdist", sdist_work)):
        artifact = payload_dir / files[kind]["name"]
        if _sha(artifact) != files[kind]["sha256"]:
            raise HistoricalGateError(f"{kind} artifact changed after probe")
        _check_junit(source_root, work / "fixed.xml", nodes)
        shutil.copy2(work / "fixed.xml", output_dir / f"fixed-{kind}.xml")
        (raw / kind).mkdir(parents=True)
        shutil.copy2(
            work / "raw" / f"nc-routes-{kind}.jsonl", raw / f"nc-routes-{kind}.jsonl"
        )
        for command_id, _, _, _ in _probe_commands():
            source = work / "raw" / kind / f"{command_id}.stdout.jsonl"
            target = raw / kind / source.name
            shutil.copy2(source, target)
            observations.append(
                (kind, command_id, _sha(target), len(target.read_text().splitlines()))
            )
    summary = raw / "qualification-summary.json"
    summary.write_text(
        json.dumps(
            {
                "source_sha": source_sha,
                "wheel_sha256": files["wheel"]["sha256"],
                "sdist_sha256": files["sdist"]["sha256"],
                "observations": [
                    {
                        "kind": kind,
                        "command_id": command,
                        "raw_sha256": digest,
                        "rows": rows,
                    }
                    for kind, command, digest, rows in observations
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    qualification = output_dir / "qualification.json"
    _run(
        [
            sys.executable,
            str(source_root / PREQUAL_REL / "verify_prequalification.py"),
            "--candidate-summary",
            str(summary),
            "--output",
            str(qualification),
            "--expected-source-sha",
            source_sha,
            "--expected-wheel-sha256",
            files["wheel"]["sha256"],
            "--expected-sdist-sha256",
            files["sdist"]["sha256"],
            "--wheel-artifact",
            str(payload_dir / files["wheel"]["name"]),
            "--sdist-artifact",
            str(payload_dir / files["sdist"]["name"]),
        ],
        cwd=output_dir,
    )
    _run(
        [
            sys.executable,
            str(source_root / PREQUAL_REL / "verify_fixed_defects.py"),
            "--wheel-junit",
            str(output_dir / "fixed-wheel.xml"),
            "--sdist-junit",
            str(output_dir / "fixed-sdist.xml"),
        ],
        cwd=output_dir,
    )
    qualified = _load(qualification)
    if (
        qualified.get("schema") != "gwexpy-v025-audit-qualification-v1"
        or qualified.get("source_sha") != source_sha
        or qualified.get("artifact_sha256") != {k: files[k]["sha256"] for k in files}
        or qualified.get("status_counts")
        != {"PASS": 26, "FAIL": 0, "NEEDS_HARNESS": 0, "BLOCKED": 12}
        or qualified.get("release_gate") != "OPEN"
    ):
        raise HistoricalGateError("26+12 qualification status mismatch")
    entries = qualified.get("findings")
    if not isinstance(entries, list) or len(entries) != 38:
        raise HistoricalGateError("qualification has incomplete finding list")
    for status, expected in (("PASS", behavior), ("BLOCKED", blocked)):
        selected = [entry for entry in entries if entry.get("status") == status]
        if {entry.get("finding_id") for entry in selected} != expected or len(
            selected
        ) != len(expected):
            raise HistoricalGateError(f"qualification {status} finding IDs mismatch")
        check_status = "PASS" if status == "PASS" else "CHARACTERIZED"
        if any(
            {k: v.get("status") for k, v in entry.get("artifact_checks", {}).items()}
            != {"wheel": check_status, "sdist": check_status}
            for entry in selected
        ):
            raise HistoricalGateError(
                f"qualification {status} artifact assertions incomplete"
            )
    for kind in ("wheel", "sdist"):
        _write(
            output_dir / f"{kind}.json",
            {
                "schema": CELL_SCHEMA,
                "source_sha": source_sha,
                "distribution": {"kind": kind, "sha256": files[kind]["sha256"]},
                "fixed_finding_ids": sorted(fixed),
                "behavior_finding_ids": sorted(behavior),
                "blocked_finding_ids": sorted(blocked),
                "fixed_junit_sha256": _sha(output_dir / f"fixed-{kind}.xml"),
                "qualification_sha256": _sha(qualification),
            },
        )
    digest = _sha(source_root / DISPOSITION_REL)
    gate = aggregate(manifest, output_dir, approval, source_sha, digest, source_root)
    _write(output_dir / "gate.json", gate)


def aggregate(
    manifest: Path,
    reports: Path,
    approval: Path,
    source_sha: str,
    disposition_digest: str,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless both artifacts account for the same 36+26+12 IDs."""
    files = _payload(manifest, source_sha)
    source_root = source_root or Path(__file__).resolve().parents[2]
    fixed, behavior, blocked, nodes = _findings(source_root)
    human = _approval(approval).get("human_approval")
    if (
        not isinstance(human, dict)
        or human.get("disposition_digest") != disposition_digest
        or SHA256.fullmatch(disposition_digest) is None
    ):
        raise HistoricalGateError("owner approval is not bound to disposition document")
    cell_hashes = {}
    for kind in ("wheel", "sdist"):
        path = reports / f"{kind}.json"
        cell = _load(path)
        if set(cell) != {
            "schema",
            "source_sha",
            "distribution",
            "fixed_finding_ids",
            "behavior_finding_ids",
            "blocked_finding_ids",
            "fixed_junit_sha256",
            "qualification_sha256",
        }:
            raise HistoricalGateError(f"invalid {kind} cell keys")
        if (
            cell["schema"] != CELL_SCHEMA
            or cell["source_sha"] != source_sha
            or cell["distribution"] != {"kind": kind, "sha256": files[kind]["sha256"]}
        ):
            raise HistoricalGateError(f"{kind} cell is not candidate-bound")
        if (
            cell["fixed_finding_ids"],
            cell["behavior_finding_ids"],
            cell["blocked_finding_ids"],
        ) != (sorted(fixed), sorted(behavior), sorted(blocked)):
            raise HistoricalGateError(f"{kind} 36+26+12 finding accounting mismatch")
        for key in ("fixed_junit_sha256", "qualification_sha256"):
            if not isinstance(cell[key], str) or SHA256.fullmatch(cell[key]) is None:
                raise HistoricalGateError(f"invalid {kind} {key}")
        if cell["fixed_junit_sha256"] != _sha(reports / f"fixed-{kind}.xml"):
            raise HistoricalGateError(f"{kind} fixed JUnit hash mismatch")
        if cell["qualification_sha256"] != _sha(reports / "qualification.json"):
            raise HistoricalGateError(f"{kind} qualification hash mismatch")
        _check_junit(source_root, reports / f"fixed-{kind}.xml", nodes)
        cell_hashes[kind] = _sha(path)
    _check_qualification(
        reports / "qualification.json", files, source_sha, behavior, blocked
    )
    return {
        "schema": AGGREGATE_SCHEMA,
        "source_sha": source_sha,
        "artifact_sha256": {k: files[k]["sha256"] for k in files},
        "runtime_pass_count": 62,
        "runtime_finding_ids": sorted(fixed | behavior),
        "disposition_count": 12,
        "disposition_finding_ids": sorted(blocked),
        "disposition_digest": disposition_digest,
        "cell_report_sha256": cell_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--payload-dir", type=Path, required=True)
    run.add_argument("--payload-manifest", type=Path, required=True)
    run.add_argument("--source-sha", required=True)
    run.add_argument("--kind", choices=("wheel", "sdist"), required=True)
    run.add_argument("--work", type=Path, required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--source-root", type=Path, required=True)
    finish.add_argument("--payload-dir", type=Path, required=True)
    finish.add_argument("--payload-manifest", type=Path, required=True)
    finish.add_argument("--source-sha", required=True)
    finish.add_argument("--wheel-work", type=Path, required=True)
    finish.add_argument("--sdist-work", type=Path, required=True)
    finish.add_argument("--approval", type=Path, required=True)
    finish.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run_cell(
                args.source_root,
                args.payload_dir,
                args.payload_manifest,
                args.source_sha,
                args.kind,
                args.work,
            )
        elif args.command == "finalize":
            finalize(
                args.source_root,
                args.payload_dir,
                args.payload_manifest,
                args.source_sha,
                args.wheel_work,
                args.sdist_work,
                args.approval,
                args.output_dir,
            )
    except (HistoricalGateError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"historical gate failed: {exc}\n")


if __name__ == "__main__":
    main()
