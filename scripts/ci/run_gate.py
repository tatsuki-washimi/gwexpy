#!/usr/bin/env python3
"""Run CI gate commands from a local shell.

This script keeps the command surface used by GitHub Actions in one place so
local repro and CI share the same invocation list.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run one command and fail fast."""
    quoted = " ".join(cmd)
    print(f"\n$ {quoted}")
    completed = subprocess.run(cmd, check=False, cwd=cwd)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _aggregate_junit_counts(paths: Iterable[Path]) -> dict[str, int]:
    """Aggregate counters from leaf ``testsuite`` elements in JUnit reports."""
    totals = {name: 0 for name in ("tests", "skipped", "errors", "failures")}
    report_count = 0

    for path in paths:
        report_count += 1
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError, UnicodeError) as exc:
            raise ValueError(f"JUnit report {path} could not be parsed: {exc}") from exc

        if root.tag == "testsuite":
            top_level_suites = [root]
        elif root.tag == "testsuites":
            top_level_suites = [child for child in root if child.tag == "testsuite"]
        else:
            raise ValueError(
                f"JUnit report {path} has unsupported root element {root.tag!r}"
            )

        leaves: list[ET.Element] = []

        def collect_leaves(suite: ET.Element) -> None:
            children = [child for child in suite if child.tag == "testsuite"]
            if children:
                for child in children:
                    collect_leaves(child)
            else:
                leaves.append(suite)

        for suite in top_level_suites:
            collect_leaves(suite)

        if not leaves:
            raise ValueError(f"JUnit report {path} contains no leaf testsuite elements")

        for suite in leaves:
            for name in totals:
                raw_value = suite.attrib.get(name, "0")
                try:
                    value = int(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"JUnit report {path} has non-integer {name}={raw_value!r}"
                    ) from exc
                if value < 0:
                    raise ValueError(f"JUnit report {path} has negative {name}={value}")
                totals[name] += value

    if report_count == 0:
        raise ValueError("no JUnit reports were provided")
    return totals


def _run_strict_junit_gate(
    gate: str,
    junit_path: Path,
    tests: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> None:
    """Run a dedicated pytest selector and fail closed on JUnit counters."""
    if environment is not None:
        os.environ.update(environment)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["pytest", "-v", f"--junit-xml={junit_path}", *tests])

    try:
        counts = _aggregate_junit_counts([junit_path])
    except ValueError as exc:
        raise SystemExit(f"{gate} gate: invalid JUnit report: {exc}") from exc

    if (
        counts["tests"] == 0
        or counts["skipped"]
        or counts["errors"]
        or counts["failures"]
    ):
        raise SystemExit(
            f"{gate} gate: tests={counts['tests']} "
            f"skipped={counts['skipped']} errors={counts['errors']} "
            f"failures={counts['failures']} -- expected tests>0 and "
            "skipped=errors=failures=0"
        )


def run_gate(gate: str, with_fixtures: bool) -> None:
    """Run the command group for a named CI gate."""
    print("=== CI gate start ===")
    print(f"Gate: {gate}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Git root: {Path.cwd()}")
    print(f"with_fixtures: {with_fixtures}")

    if gate == "pr-fast":
        run_cmd(["ruff", "check", "gwexpy", "tests"])
        run_cmd(["python", "scripts/check_forbidden_artifacts.py"])
        run_cmd(
            [
                "mypy",
                "gwexpy",
                "tests/docs/test_tutorial_notebook_quality.py",
                "--ignore-missing-imports",
            ]
        )
        if with_fixtures:
            run_cmd(["python", "tests/fixtures/generate_fixtures.py"])
        run_cmd(
            [
                "pytest",
                "-q",
                "-m",
                "not network and not nds and not root",
                "--ignore=tests/docs/test_docs_notebooks.py",
                "--ignore=tests/gui/",
                "--ignore=tests/nds/",
                "--ignore=tests/io/",
                "--ignore=tests/io_conformance/",
                "--ignore=tests/segments/",
                "--ignore=tests/table/",
                "--ignore=tests/test_geomap.py",
                "--ignore=tests/time/test_time.py",
                "--ignore=tests/test_fitting_highlevel.py",
                "--ignore=tests/timeseries/test_matrix_analysis.py",
                "--ignore=tests/types/test_series_matrix_io.py",
                "tests/",
            ]
        )
        return

    if gate == "io-contract":
        if with_fixtures:
            run_cmd(["python", "tests/fixtures/generate_fixtures.py"])
        run_cmd(
            [
                "pytest",
                "-q",
                "-m",
                "not network and not nds",
                "tests/io/test_io_contract.py",
                "tests/io/test_io_docs_contract_sync.py",
                "tests/io/",
                "tests/segments/",
                "tests/table/",
            ]
        )
        # Wheel/bytecode hygiene is covered by the `validate` job, which runs
        # `python -m build --sdist --wheel` + check_release_artifacts.py after
        # pytest has populated __pycache__ -- a stricter check than a clean
        # build here would be. Keep only the import smoke test in this gate.
        run_cmd(
            [
                "python",
                "-c",
                'import gwexpy\nprint(f"gwexpy version: {gwexpy.__version__}")',
            ],
        )
        return

    if gate == "io-optional":
        run_cmd(
            [
                "pytest",
                "-q",
                "tests/io/test_optional_deps.py",
                "tests/io/test_netcdf4_reader.py",
                "tests/io/test_tdms_reader.py",
                "tests/io/test_audio_metadata.py",
                "tests/io/test_seismic_public_io.py",
            ]
        )
        return

    if gate == "io-conformance":
        run_cmd(["pytest", "-q", "tests/io_conformance"])
        repo_root = Path.cwd().resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tests.io_conformance.contract import load_public_io_contract
        from tests.io_conformance.reporting import summarize_blocking_rows
        from tests.io_conformance.scenarios import expand_contract_scenarios

        contract = load_public_io_contract()
        rows = expand_contract_scenarios(contract)
        summary = summarize_blocking_rows(rows)
        print(summary["blocking_display"])
        return

    if gate == "io-network-backend":
        if with_fixtures:
            run_cmd(["python", "tests/fixtures/generate_fixtures.py"])
        run_cmd(
            [
                "pytest",
                "-q",
                "-m",
                "network or nds",
                "tests/io/",
                "tests/nds/",
                "tests/segments/",
                "tests/timeseries/test_timeseries.py",
            ]
        )
        run_cmd(["pytest", "-q", "tests/io/test_kerberos.py"])
        return

    if gate == "docs-notebook":
        if with_fixtures:
            run_cmd(["python", "tests/fixtures/generate_fixtures.py"])
        os.environ["GWEXPY_RUN_NOTEBOOK_TESTS"] = os.environ.get(
            "GWEXPY_RUN_NOTEBOOK_TESTS",
            "1",
        )
        run_cmd(["pytest", "-q", "tests/docs/test_docs_notebooks.py"])
        return

    if gate == "io-zarr":
        if with_fixtures:
            run_cmd(["python", "tests/fixtures/generate_fixtures.py"])
        os.environ["GWEXPY_ALLOW_ZARR"] = os.environ.get(
            "GWEXPY_ALLOW_ZARR",
            "1",
        )
        run_cmd(["pytest", "-q", "tests/io/test_zarr_reader.py"])
        return

    if gate == "io-gwf":
        _run_strict_junit_gate(
            gate,
            Path("junit/io-gwf.xml"),
            [
                "tests/timeseries/test_gwf_parallel_contract.py",
                "tests/timeseries/test_io_gwf_framel.py",
            ],
        )
        return

    if gate == "io-netcdf":
        _run_strict_junit_gate(
            gate,
            Path("junit/io-netcdf.xml"),
            [
                "tests/io/test_netcdf4_reader.py",
                "tests/timeseries/test_crop_timing_contract.py",
            ],
        )
        return

    if gate == "interop-root":
        _run_strict_junit_gate(
            gate,
            Path("junit/interop-root.xml"),
            [
                "tests/histogram/test_root_interop.py",
                "tests/histogram/test_root_interop_contracts.py",
            ],
            environment={"GWEXPY_RUN_ROOT": "1"},
        )
        return

    if gate == "interop-contract":
        run_cmd(
            [
                "pytest",
                "-q",
                "tests/interop/test_interop_contract.py",
                "tests/interop/test_interop_docs_contract_sync.py",
                "tests/interop/test_mt_mock.py",
            ]
        )
        return

    if gate == "interop-mne":
        # tests/interop/test_interop_mne.py uses `pytest.importorskip("mne")`,
        # so if mne fails to import the whole file silently skips instead of
        # failing. Assert on the JUnit skipped count so that regression does
        # NOT go unnoticed (#493: this gate exists specifically to keep mne
        # coverage off the "runs locally only" list).
        _run_strict_junit_gate(
            gate,
            Path("interop-mne-results.xml"),
            ["tests/interop/test_interop_mne.py"],
        )
        return

    raise SystemExit(f"Unknown gate: {gate}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gate",
        choices=[
            "pr-fast",
            "io-contract",
            "io-conformance",
            "io-optional",
            "io-network-backend",
            "docs-notebook",
            "io-zarr",
            "io-gwf",
            "io-netcdf",
            "interop-contract",
            "interop-mne",
            "interop-root",
        ],
    )
    parser.add_argument(
        "--fixtures",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Control synthetic fixture generation for gates that use it.",
    )
    args = parser.parse_args(argv)
    run_gate(args.gate, args.fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
