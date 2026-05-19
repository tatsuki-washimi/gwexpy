#!/usr/bin/env python3
"""Run CI gate commands from a local shell.

This script keeps the command surface used by GitHub Actions in one place so
local repro and CI share the same invocation list.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run one command and fail fast."""
    quoted = " ".join(cmd)
    print(f"\n$ {quoted}")
    completed = subprocess.run(cmd, check=False, cwd=cwd)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def remove_generated_build_artifacts(repo_root: Path) -> None:
    """Remove generated artifacts that must not leak into wheel builds."""
    shutil.rmtree(repo_root / "build", ignore_errors=True)
    shutil.rmtree(repo_root / "dist", ignore_errors=True)
    for pycache_dir in (repo_root / "gwexpy").rglob("__pycache__"):
        shutil.rmtree(pycache_dir, ignore_errors=True)
    for bytecode in (repo_root / "gwexpy").rglob("*.py[co]"):
        bytecode.unlink(missing_ok=True)


def assert_wheel_has_no_bytecode(repo_root: Path) -> None:
    """Fail if the built wheel contains interpreter cache artifacts."""
    wheels = sorted((repo_root / "dist").glob("*.whl"))
    if not wheels:
        raise SystemExit("No wheel artifact found in dist/.")
    if len(wheels) != 1:
        wheel_list = "\n".join(f"  - {wheel.name}" for wheel in wheels)
        raise SystemExit(f"Expected exactly one wheel artifact in dist/:\n{wheel_list}")

    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        forbidden = [
            name
            for name in archive.namelist()
            if "__pycache__/" in name or name.endswith((".pyc", ".pyo"))
        ]

    if forbidden:
        preview = "\n".join(f"  - {name}" for name in forbidden[:20])
        extra = (
            "" if len(forbidden) <= 20 else f"\n  ... and {len(forbidden) - 20} more"
        )
        raise SystemExit(f"Wheel contains bytecode/cache artifacts:\n{preview}{extra}")


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
        repo_root = Path.cwd().resolve()
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
        remove_generated_build_artifacts(repo_root)
        run_cmd(
            [
                "python",
                "-m",
                "build",
                str(repo_root),
                "--wheel",
                "--no-isolation",
            ],
            cwd=repo_root.parent,
        )
        assert_wheel_has_no_bytecode(repo_root)
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
            "interop-contract",
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
