"""Tests for deterministic IO conformance generators."""

from __future__ import annotations

import importlib
import os
import signal
import time
from pathlib import Path

import pytest

from tests.io_conformance import conftest as guard_conftest
from tests.io_conformance.generators import GENERATOR_SPECS, iter_generator_specs

# (spec.name, spec.module_name) for the frozen generator set, in order.  Most
# modules share their spec name; the Zarr generator module is ``zarr_store`` so
# it does not shadow the third-party ``zarr`` package on import.
_FROZEN_GENERATORS = (
    ("csv_txt", "tests.io_conformance.generators.csv_txt"),
    ("audio", "tests.io_conformance.generators.audio"),
    ("hdf5", "tests.io_conformance.generators.hdf5"),
    ("hdf_ndscope", "tests.io_conformance.generators.hdf_ndscope"),
    ("gwf", "tests.io_conformance.generators.gwf"),
    ("sdb", "tests.io_conformance.generators.sdb"),
    ("zarr", "tests.io_conformance.generators.zarr_store"),
)

# Generators whose round-trip is not byte-stable across runs (compared by
# manifest only).
_MANIFEST_ONLY_GENERATORS = frozenset({"gwf"})


def _recorded_pids(pid_dir: Path) -> list[int]:
    return [
        int(path.read_text())
        for path in sorted(pid_dir.glob("*.pid"))
        if path.is_file()
    ]


def _wait_for_pids_to_exit(pids: list[int], timeout: float = 3) -> list[int]:
    deadline = time.monotonic() + timeout
    alive = pids
    while alive and time.monotonic() < deadline:
        next_alive = []
        for pid in alive:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            next_alive.append(pid)
        alive = next_alive
        if alive:
            time.sleep(0.05)
    return alive


def _tear_down_recorded_processes(pid_dir: Path) -> None:
    pids = _recorded_pids(pid_dir)
    parent_pid_path = pid_dir / "parent.pid"
    if parent_pid_path.exists():
        try:
            os.killpg(int(parent_pid_path.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _wait_for_pids_to_exit(pids)


def _assert_pids_are_gone(pids: list[int]) -> None:
    assert not _wait_for_pids_to_exit(pids), "generator processes survived cleanup"
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_generator_smoke_timeout_defaults_are_bounded() -> None:
    """Generator smoke checks have finite default cleanup limits."""
    assert guard_conftest._GENERATOR_SMOKE_TIMEOUT_SECONDS == 60
    assert guard_conftest._GENERATOR_SMOKE_TERMINATION_GRACE_SECONDS == 5


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (("stdout-tail", "stderr-tail"), ("stdout-tail", "stderr-tail")),
        (("-tail", "-tail"), ("stdout-tail", "stderr-tail")),
    ],
    ids=("cumulative-no-duplication", "incremental-no-loss"),
)
def test_bounded_communicate_merges_cumulative_or_incremental_output(
    completed: tuple[str, str], expected: tuple[str, str]
) -> None:
    """A retry preserves timeout output whether communicate returns all or new data."""

    class CompletedProcess:
        def communicate(self, timeout: float) -> tuple[str, str]:
            assert timeout == 0.1
            return completed

    assert guard_conftest._communicate_bounded(
        CompletedProcess(),  # type: ignore[arg-type]
        0.1,
        "stdout",
        "stderr",
    ) == (*expected, True)


def test_pr_fast_excludes_io_conformance_and_dedicated_gate_runs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IO conformance has one dedicated gate and is absent from pr-fast."""
    from scripts.ci import run_gate

    commands: list[list[str]] = []
    monkeypatch.setattr(run_gate, "run_cmd", lambda command: commands.append(command))

    run_gate.run_gate("pr-fast", with_fixtures=False)
    pr_fast_pytest = next(command for command in commands if command[0] == "pytest")
    assert "--ignore=tests/io_conformance/" in pr_fast_pytest

    commands.clear()
    run_gate.run_gate("io-conformance", with_fixtures=False)
    assert commands == [["pytest", "-q", "tests/io_conformance"]]


def test_session_start_runs_source_guard_and_smoke_for_every_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedicated conformance session enforces both generator checks."""
    specs = (
        guard_conftest.GeneratorSpec("first", "first_generator"),
        guard_conftest.GeneratorSpec("second", "second_generator"),
    )
    guarded: list[str] = []
    smoked: list[str] = []
    monkeypatch.setattr(guard_conftest, "iter_generator_specs", lambda: specs)
    monkeypatch.setattr(
        guard_conftest,
        "_guard_generator_source",
        lambda spec: guarded.append(spec.name),
    )
    monkeypatch.setattr(
        guard_conftest, "_run_generator_smoke", lambda spec: smoked.append(spec.name)
    )

    guard_conftest.pytest_sessionstart(None)  # type: ignore[arg-type]

    assert guarded == ["first", "second"]
    assert smoked == ["first", "second"]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_generator_smoke_timeout_kills_the_entire_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hung generator and its SIGTERM-ignoring child are both reaped."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    (tmp_path / "hanging_generator.py").write_text(
        '''
import os
import signal
import subprocess
import sys
import time

def generate(output_dir):
    del output_dir
    pid_dir = os.environ["GENERATOR_PID_DIR"]
    child_code = """
import os
import signal
import time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(os.environ['GENERATOR_PID_DIR'], 'child.pid').write_text(str(os.getpid()))
print('child stdout marker', flush=True)
print('child stderr marker', file=__import__('sys').stderr, flush=True)
Path(os.environ['GENERATOR_PID_DIR'], 'child.ready').touch()
while True:
    time.sleep(1)
"""
    subprocess.Popen([sys.executable, "-c", child_code])
    Path = __import__("pathlib").Path
    Path(pid_dir, "parent.pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("parent stdout marker", flush=True)
    print("parent stderr marker", file=sys.stderr, flush=True)
    deadline = time.monotonic() + 10
    while not Path(pid_dir, "child.ready").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("child readiness timeout")
        time.sleep(0.01)
    while True:
        time.sleep(1)
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERATOR_PID_DIR", str(pid_dir))
    monkeypatch.setattr(guard_conftest, "ROOT", tmp_path)
    monkeypatch.setattr(guard_conftest, "_GENERATOR_SMOKE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(
        guard_conftest, "_GENERATOR_SMOKE_TERMINATION_GRACE_SECONDS", 0.1
    )
    spec = guard_conftest.GeneratorSpec("hanging", "hanging_generator")

    try:
        with pytest.raises(pytest.UsageError) as error:
            guard_conftest._run_generator_smoke(spec)

        message = str(error.value)
        assert "hanging_generator" in message
        assert "2.0" in message
        assert "parent stdout marker" in message
        assert "parent stderr marker" in message
        assert "child stdout marker" in message
        assert "child stderr marker" in message

        _assert_pids_are_gone(_recorded_pids(pid_dir))
    finally:
        _tear_down_recorded_processes(pid_dir)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_generator_smoke_timeout_kills_child_after_parent_closes_pipes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A surviving pipe-less child is killed after its parent exits on SIGTERM."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    (tmp_path / "pipe_closing_generator.py").write_text(
        '''
import os
import subprocess
import sys
import time
from pathlib import Path

def generate(output_dir):
    del output_dir
    child_code = """
import os
import signal
import sys
import time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(os.environ['GENERATOR_PID_DIR'], 'child.pid').write_text(str(os.getpid()))
print('pipe-closing child stdout marker', flush=True)
print('pipe-closing child stderr marker', file=sys.stderr, flush=True)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, sys.stdout.fileno())
os.dup2(devnull, sys.stderr.fileno())
os.close(devnull)
Path(os.environ['GENERATOR_PID_DIR'], 'child.ready').touch()
while True:
    time.sleep(1)
"""
    subprocess.Popen([sys.executable, "-c", child_code])
    Path(os.environ["GENERATOR_PID_DIR"], "parent.pid").write_text(str(os.getpid()))
    print("pipe-closing parent stdout marker", flush=True)
    print("pipe-closing parent stderr marker", file=sys.stderr, flush=True)
    deadline = time.monotonic() + 10
    while not Path(os.environ["GENERATOR_PID_DIR"], "child.ready").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("child readiness timeout")
        time.sleep(0.01)
    while True:
        time.sleep(1)
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERATOR_PID_DIR", str(pid_dir))
    monkeypatch.setattr(guard_conftest, "ROOT", tmp_path)
    monkeypatch.setattr(guard_conftest, "_GENERATOR_SMOKE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(
        guard_conftest, "_GENERATOR_SMOKE_TERMINATION_GRACE_SECONDS", 0.1
    )
    spec = guard_conftest.GeneratorSpec("pipe-closing", "pipe_closing_generator")

    try:
        with pytest.raises(pytest.UsageError) as error:
            guard_conftest._run_generator_smoke(spec)

        message = str(error.value)
        assert "pipe_closing_generator" in message
        assert "2.0" in message
        assert "pipe-closing parent stdout marker" in message
        assert "pipe-closing parent stderr marker" in message
        assert "pipe-closing child stdout marker" in message
        assert "pipe-closing child stderr marker" in message

        _assert_pids_are_gone(_recorded_pids(pid_dir))
    finally:
        _tear_down_recorded_processes(pid_dir)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_generator_smoke_timeout_has_bounded_drain_when_child_escapes_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An escaped child holding pipes cannot make timeout cleanup unbounded."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    (tmp_path / "escaped_child_generator.py").write_text(
        '''
import os
import subprocess
import sys
import time
from pathlib import Path

def generate(output_dir):
    del output_dir
    child_code = """
import os
import signal
import sys
import time
from pathlib import Path
os.setsid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(os.environ['GENERATOR_PID_DIR'], 'escaped-child.pid').write_text(str(os.getpid()))
print('escaped child stdout marker', flush=True)
print('escaped child stderr marker', file=sys.stderr, flush=True)
Path(os.environ['GENERATOR_PID_DIR'], 'escaped-child.ready').touch()
while True:
    time.sleep(1)
"""
    subprocess.Popen([sys.executable, "-c", child_code])
    pid_dir = Path(os.environ["GENERATOR_PID_DIR"])
    (pid_dir / "parent.pid").write_text(str(os.getpid()))
    print("escaped parent stdout marker", flush=True)
    print("escaped parent stderr marker", file=sys.stderr, flush=True)
    deadline = time.monotonic() + 10
    while not (pid_dir / "escaped-child.ready").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("child readiness timeout")
        time.sleep(0.01)
    while True:
        time.sleep(1)
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERATOR_PID_DIR", str(pid_dir))
    monkeypatch.setattr(guard_conftest, "ROOT", tmp_path)
    monkeypatch.setattr(guard_conftest, "_GENERATOR_SMOKE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(
        guard_conftest, "_GENERATOR_SMOKE_TERMINATION_GRACE_SECONDS", 0.1
    )
    spec = guard_conftest.GeneratorSpec("escaped", "escaped_child_generator")

    def fail_if_cleanup_hangs(signum: int, frame: object) -> None:
        del signum, frame
        raise TimeoutError("generator timeout cleanup exceeded its hard bound")

    previous_alarm = signal.signal(signal.SIGALRM, fail_if_cleanup_hangs)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        started = time.monotonic()
        with pytest.raises(pytest.UsageError) as error:
            guard_conftest._run_generator_smoke(spec)
        elapsed = time.monotonic() - started

        assert elapsed < 4
        message = str(error.value)
        assert "escaped_child_generator" in message
        assert "escaped parent stdout marker" in message
        assert "escaped parent stderr marker" in message
        assert "escaped child stdout marker" in message
        assert "escaped child stderr marker" in message
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        _tear_down_recorded_processes(pid_dir)


def test_generator_smoke_timeout_reports_only_bounded_diagnostic_tails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Timeout diagnostics retain trailing sentinels without unbounded output."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    (tmp_path / "noisy_generator.py").write_text(
        """
import os
import sys
import time
from pathlib import Path

def generate(output_dir):
    del output_dir
    Path(os.environ["GENERATOR_PID_DIR"], "parent.pid").write_text(str(os.getpid()))
    print("EARLY_STDOUT_SENTINEL", flush=True)
    print("EARLY_STDERR_SENTINEL", file=sys.stderr, flush=True)
    print("x" * 512 + "TRAILING_STDOUT_SENTINEL", flush=True)
    print("y" * 512 + "TRAILING_STDERR_SENTINEL", file=sys.stderr, flush=True)
    while True:
        time.sleep(1)
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERATOR_PID_DIR", str(pid_dir))
    monkeypatch.setattr(guard_conftest, "ROOT", tmp_path)
    monkeypatch.setattr(guard_conftest, "_GENERATOR_SMOKE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(
        guard_conftest, "_GENERATOR_SMOKE_TERMINATION_GRACE_SECONDS", 0.1
    )
    monkeypatch.setattr(guard_conftest, "_GENERATOR_SMOKE_TAIL_CHARACTERS", 128)
    spec = guard_conftest.GeneratorSpec("noisy", "noisy_generator")

    try:
        with pytest.raises(pytest.UsageError) as error:
            guard_conftest._run_generator_smoke(spec)

        message = str(error.value)
        assert "EARLY_STDOUT_SENTINEL" not in message
        assert "EARLY_STDERR_SENTINEL" not in message
        assert "TRAILING_STDOUT_SENTINEL" in message
        assert "TRAILING_STDERR_SENTINEL" in message
        assert len(message) < 1000
    finally:
        _tear_down_recorded_processes(pid_dir)


def _file_tree(base_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.relative_to(base_dir) for path in base_dir.rglob("*") if path.is_file()
        )
    )


def _read_tree(base_dir: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(base_dir): path.read_bytes()
        for path in sorted(base_dir.rglob("*"))
        if path.is_file()
    }


def test_generator_registry_is_frozen_and_stable() -> None:
    specs = iter_generator_specs()
    assert specs == GENERATOR_SPECS
    assert tuple(spec.name for spec in specs) == tuple(
        name for name, _ in _FROZEN_GENERATORS
    )
    assert tuple(spec.entrypoint for spec in specs) == ("generate",) * len(specs)
    assert tuple(spec.module_name for spec in specs) == tuple(
        module_name for _, module_name in _FROZEN_GENERATORS
    )


@pytest.mark.parametrize("spec", iter_generator_specs(), ids=lambda spec: spec.name)
def test_generators_are_deterministic_and_confined(tmp_path: Path, spec) -> None:
    module = importlib.import_module(spec.module_name)
    entrypoint = getattr(module, spec.entrypoint)

    run_a = tmp_path / f"{spec.name}_a"
    run_b = tmp_path / f"{spec.name}_b"
    try:
        result_a = entrypoint(run_a)
        result_b = entrypoint(run_b)
    except (ImportError, ModuleNotFoundError) as exc:  # optional backend missing
        pytest.skip(f"{spec.name} backend unavailable: {exc}")

    assert isinstance(result_a, dict)
    assert isinstance(result_b, dict)
    assert set(result_a) == set(result_b)

    for output in (run_a, run_b):
        assert output.exists()
        for artifact in output.rglob("*"):
            if artifact.is_file():
                assert artifact.is_relative_to(output)

    for result, output in ((result_a, run_a), (result_b, run_b)):
        for value in result.values():
            artifact_path = Path(value)
            assert artifact_path.is_absolute()
            assert artifact_path.is_relative_to(output)

    assert _file_tree(run_a) == _file_tree(run_b)
    if spec.name in _MANIFEST_ONLY_GENERATORS:
        assert (run_a / "manifest.json").read_bytes() == (
            run_b / "manifest.json"
        ).read_bytes()
        return
    assert _read_tree(run_a) == _read_tree(run_b)


@pytest.mark.parametrize("spec", iter_generator_specs(), ids=lambda spec: spec.name)
def test_generator_source_passes_import_guard(spec) -> None:
    guard_conftest._guard_generator_source(spec)
