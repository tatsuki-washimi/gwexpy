"""Contract tests for fail-closed JUnit aggregation in the CI gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from scripts.ci import run_gate

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_report(tmp_path: Path, name: str, xml: str) -> Path:
    """Write synthetic JUnit XML and return its path."""
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return path


def _run_interop_mne_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    xml: str | None,
) -> tuple[list[list[str]], list[list[Path]]]:
    """Run the gate with a synthetic report produced by the test command."""
    monkeypatch.chdir(tmp_path)
    commands: list[list[str]] = []
    aggregate_paths: list[list[Path]] = []
    original_aggregate = run_gate._aggregate_junit_counts

    def fake_run_cmd(
        cmd: list[str],
        *,
        cwd: Path | None = None,
    ) -> None:
        del cwd
        commands.append(cmd)
        assert "--junit-xml=interop-mne-results.xml" in cmd
        if xml is not None:
            Path("interop-mne-results.xml").write_text(xml, encoding="utf-8")

    def recording_aggregate(paths: Iterable[Path]) -> dict[str, int]:
        recorded_paths = list(paths)
        aggregate_paths.append(recorded_paths)
        return original_aggregate(recorded_paths)

    monkeypatch.setattr(run_gate, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(run_gate, "_aggregate_junit_counts", recording_aggregate)
    run_gate.run_gate("interop-mne", with_fixtures=False)
    return commands, aggregate_paths


def test_docs_notebook_gate_derives_environment_in_fresh_clean_subprocess() -> None:
    child_environment = os.environ.copy()
    for variable in ("PYTHONNOUSERSITE", "PYTHONPATH", "PATH"):
        child_environment.pop(variable, None)

    child_script = """
import json
from scripts.ci import run_gate

calls = []

def fake_run_cmd(command, *, cwd=None, environment=None):
    calls.append({
        "command": command,
        "cwd": None if cwd is None else str(cwd),
        "environment": environment,
    })

run_gate.run_cmd = fake_run_cmd
run_gate.run_gate("docs-notebook", with_fixtures=False)
print(json.dumps(calls))
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=REPO_ROOT,
        env=child_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = json.loads(completed.stdout.splitlines()[-1])

    assert len(calls) == 1
    call = calls[0]
    assert call["command"] == [
        "pytest",
        "-q",
        "tests/docs/test_docs_notebooks.py",
    ]
    assert call["cwd"] == str(REPO_ROOT)
    environment = call["environment"]
    assert environment is not None
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"] == str(REPO_ROOT)
    assert environment["PATH"] == os.pathsep.join(
        [str(Path(sys.executable).resolve().parent), os.defpath]
    )
    assert environment["GWEXPY_RUN_NOTEBOOK_TESTS"] == "1"


@pytest.mark.parametrize(
    ("gate", "junit_path", "selected_tests", "required_environment"),
    [
        (
            "io-gwf",
            "junit/io-gwf.xml",
            {
                "tests/timeseries/test_gwf_parallel_contract.py",
                "tests/timeseries/test_io_gwf_framel.py",
            },
            None,
        ),
        (
            "io-netcdf",
            "junit/io-netcdf.xml",
            {
                "tests/io/test_netcdf4_reader.py",
                "tests/timeseries/test_crop_timing_contract.py",
            },
            None,
        ),
        (
            "io-mth5",
            "junit/io-mth5.xml",
            {"tests/io/test_ats_mth5_reader.py"},
            None,
        ),
        (
            "interop-root",
            "junit/interop-root.xml",
            {
                "tests/histogram/test_root_interop.py",
                "tests/histogram/test_root_interop_contracts.py",
            },
            "GWEXPY_RUN_ROOT",
        ),
    ],
)
def test_dedicated_gate_writes_a_strict_junit_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate: str,
    junit_path: str,
    selected_tests: set[str],
    required_environment: str | None,
) -> None:
    """Every dedicated release gate pins its selector and rejects skipped runs."""
    monkeypatch.chdir(tmp_path)
    if required_environment is not None:
        monkeypatch.setenv(required_environment, "0")
    commands: list[list[str]] = []

    def fake_run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
        del cwd
        commands.append(cmd)
        report = next(arg for arg in cmd if arg.startswith("--junit-xml="))
        path = Path(report.split("=", 1)[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '<testsuite name="dedicated" tests="2" skipped="0" errors="0" failures="0" />',
            encoding="utf-8",
        )

    monkeypatch.setattr(run_gate, "run_cmd", fake_run_cmd)
    run_gate.run_gate(gate, with_fixtures=False)

    assert len(commands) == 1
    assert f"--junit-xml={junit_path}" in commands[0]
    assert selected_tests.issubset(commands[0])
    if required_environment is not None:
        assert os.environ[required_environment] == "1"


def test_aggregate_junit_counts_accepts_single_testsuite_root(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        "single.xml",
        '<testsuite name="mne" tests="4" skipped="1" errors="2" failures="1" />',
    )

    assert run_gate._aggregate_junit_counts([report]) == {
        "tests": 4,
        "skipped": 1,
        "errors": 2,
        "failures": 1,
    }


def test_aggregate_junit_counts_sums_leaf_suites_across_files(tmp_path: Path) -> None:
    first = _write_report(
        tmp_path,
        "first.xml",
        '<testsuite name="first" tests="2" skipped="1" errors="0" failures="1" />',
    )
    second = _write_report(
        tmp_path,
        "second.xml",
        """
        <testsuites>
          <testsuite name="second-a" tests="3" skipped="0" errors="1" failures="0" />
          <testsuite name="second-b" tests="4" skipped="2" errors="0" failures="1" />
        </testsuites>
        """,
    )

    assert run_gate._aggregate_junit_counts([first, second]) == {
        "tests": 9,
        "skipped": 3,
        "errors": 1,
        "failures": 2,
    }


def test_aggregate_junit_counts_excludes_nested_parent_totals(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        "nested.xml",
        """
        <testsuite name="root-total" tests="999" skipped="999" errors="999" failures="999">
          <testsuite name="parent-total" tests="100" skipped="100" errors="100" failures="100">
            <testsuite name="leaf-a" tests="2" skipped="1" errors="0" failures="1" />
            <testsuite name="leaf-b" tests="3" skipped="0" errors="1" failures="0" />
          </testsuite>
        </testsuite>
        """,
    )

    assert run_gate._aggregate_junit_counts([report]) == {
        "tests": 5,
        "skipped": 1,
        "errors": 1,
        "failures": 1,
    }


def test_aggregate_junit_counts_rejects_unsupported_xml_root(tmp_path: Path) -> None:
    report = _write_report(tmp_path, "unsupported.xml", "<report />")

    with pytest.raises(ValueError, match="unsupported root element"):
        run_gate._aggregate_junit_counts([report])


def test_aggregate_junit_counts_rejects_testsuites_without_leaf_suites(
    tmp_path: Path,
) -> None:
    report = _write_report(tmp_path, "no-leaves.xml", "<testsuites />")

    with pytest.raises(ValueError, match="no leaf testsuite elements"):
        run_gate._aggregate_junit_counts([report])


def test_aggregate_junit_counts_rejects_non_integer_counter(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        "non-integer.xml",
        '<testsuite name="bad" tests="many" />',
    )

    with pytest.raises(ValueError, match="non-integer tests"):
        run_gate._aggregate_junit_counts([report])


def test_aggregate_junit_counts_rejects_negative_counter(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        "negative.xml",
        '<testsuite name="bad" tests="-1" />',
    )

    with pytest.raises(ValueError, match="negative tests"):
        run_gate._aggregate_junit_counts([report])


def test_aggregate_junit_counts_rejects_empty_report_path_iterable() -> None:
    with pytest.raises(ValueError, match="no JUnit reports were provided"):
        run_gate._aggregate_junit_counts([])


@pytest.mark.parametrize(
    ("name", "xml"),
    [
        ("malformed.xml", "<testsuite"),
        ("missing.xml", None),
    ],
)
def test_aggregate_junit_counts_rejects_invalid_reports(
    tmp_path: Path,
    name: str,
    xml: str | None,
) -> None:
    report = tmp_path / name
    if xml is not None:
        report.write_text(xml, encoding="utf-8")

    with pytest.raises(ValueError, match="JUnit"):
        run_gate._aggregate_junit_counts([report])


def test_interop_mne_gate_rejects_zero_aggregate_tests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="tests=0"):
        _run_interop_mne_gate(
            monkeypatch,
            tmp_path,
            '<testsuite name="empty" tests="0" skipped="0" errors="0" failures="0" />',
        )


def test_interop_mne_gate_accepts_valid_report_and_wires_junit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands, aggregate_paths = _run_interop_mne_gate(
        monkeypatch,
        tmp_path,
        '<testsuite name="valid" tests="2" skipped="0" errors="0" failures="0" />',
    )

    assert commands == [
        [
            "pytest",
            "-v",
            "--junit-xml=interop-mne-results.xml",
            "tests/interop/test_interop_mne.py",
        ]
    ]
    junit_option = next(
        argument for argument in commands[0] if argument.startswith("--junit-xml=")
    )
    assert aggregate_paths == [[Path(junit_option.split("=", 1)[1])]]


def test_interop_mne_gate_rejects_malformed_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="JUnit"):
        _run_interop_mne_gate(monkeypatch, tmp_path, "<testsuite")


def test_interop_mne_gate_rejects_missing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="JUnit"):
        _run_interop_mne_gate(monkeypatch, tmp_path, None)


@pytest.mark.parametrize("counter", ["skipped", "errors", "failures"])
def test_interop_mne_gate_rejects_nonzero_reported_counters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    counter: str,
) -> None:
    counters = {name: 0 for name in ("skipped", "errors", "failures")}
    counters[counter] = 1
    report = (
        '<testsuite name="bad" tests="1" skipped="{skipped}" '
        'errors="{errors}" failures="{failures}" />'
    ).format(**counters)

    with pytest.raises(SystemExit, match=rf"{counter}=1"):
        _run_interop_mne_gate(monkeypatch, tmp_path, report)
