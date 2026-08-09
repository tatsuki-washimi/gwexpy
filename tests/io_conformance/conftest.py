"""Import guards and smoke checks for the IO conformance generator harness."""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import pytest

from .generators import GeneratorSpec, iter_generator_specs

# Markers in a generator smoke-check failure that indicate the generator's
# optional *backend* is simply not installed in this environment (as opposed to
# a genuine bug in the generator).  In that case we skip the check with a
# warning instead of aborting the whole session, so backend-less dev
# environments -- and optional-backend generators added for new formats -- do
# not break unrelated conformance tests.  CI installs the backends, so real
# generator regressions there still fail loudly.
# Notably, _BlockGwexpy hook errors ("blocked import: gwexpy") should NOT be
# treated as missing backends; they indicate a real generator violation.
_MISSING_BACKEND_MARKERS = (
    "Missing optional dependency",
    "no GWF API available",
    "please install",
    "is required for",
    "No module named",
)


def _looks_like_missing_backend(stderr: str) -> bool:
    return any(marker in stderr for marker in _MISSING_BACKEND_MARKERS)


ROOT = Path(__file__).resolve().parents[2]
GENERATORS_DIR = Path(__file__).resolve().parent / "generators"
BLOCKED_PREFIX = "gwexpy"
_GENERATOR_SMOKE_TIMEOUT_SECONDS = 60
_GENERATOR_SMOKE_TERMINATION_GRACE_SECONDS = 5
_GENERATOR_SMOKE_TAIL_CHARACTERS = 4000


def _generator_path(spec: GeneratorSpec) -> Path:
    # Derive the file from the module name's last component (not spec.name):
    # the Zarr generator module is ``zarr_store`` to avoid shadowing the
    # third-party ``zarr`` package, while its spec name stays ``zarr``.
    module_leaf = spec.module_name.rsplit(".", 1)[-1]
    return GENERATORS_DIR / f"{module_leaf}.py"


def _process_group_exists(process_group: int) -> bool:
    """Return whether a POSIX process group still has at least one member."""
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _as_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def _merge_captured_output(previous: str, current: str | bytes | None) -> str:
    """Merge cumulative or incremental timeout output without duplication."""
    addition = _as_text(current)
    if not addition or previous.endswith(addition):
        return previous
    if addition.startswith(previous):
        return addition
    overlap_limit = min(len(previous), len(addition))
    for overlap in range(overlap_limit, 0, -1):
        if previous.endswith(addition[:overlap]):
            return previous + addition[overlap:]
    return previous + addition


def _communicate_bounded(
    process: subprocess.Popen[str],
    timeout: float,
    stdout: str,
    stderr: str,
) -> tuple[str, str, bool]:
    """Drain pipes for at most ``timeout``, retaining partial diagnostics."""
    try:
        complete_stdout, complete_stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return (
            _merge_captured_output(stdout, exc.stdout),
            _merge_captured_output(stderr, exc.stderr),
            False,
        )
    return (
        _merge_captured_output(stdout, complete_stdout),
        _merge_captured_output(stderr, complete_stderr),
        True,
    )


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    """Close local pipe readers so escaped descendants cannot block cleanup."""
    for pipe in (process.stdout, process.stderr):
        if pipe is None:
            continue
        try:
            pipe.close()
        except OSError:
            pass


def _reap_process_bounded(process: subprocess.Popen[str], timeout: float) -> None:
    """Wait a bounded time for the direct child, killing it once if necessary."""
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        process.kill()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return


def _terminate_timed_out_process(
    process: subprocess.Popen[str], stdout: str, stderr: str
) -> tuple[str, str]:
    """Terminate a timed-out process tree and drain/reap it within hard bounds."""
    grace = _GENERATOR_SMOKE_TERMINATION_GRACE_SECONDS
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        grace_deadline = time.monotonic() + grace
        stdout, stderr, _ = _communicate_bounded(process, grace, stdout, stderr)
        while _process_group_exists(process.pid) and time.monotonic() < grace_deadline:
            time.sleep(min(0.01, max(0.0, grace_deadline - time.monotonic())))
        if _process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        process.terminate()
        stdout, stderr, terminated = _communicate_bounded(
            process, grace, stdout, stderr
        )
        if not terminated:
            process.kill()

    stdout, stderr, _ = _communicate_bounded(process, grace, stdout, stderr)
    _close_process_pipes(process)
    _reap_process_bounded(process, grace)
    return stdout, stderr


def _parse_generator_source(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_literal_gwexpy_import(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False

    if not node.args:
        return False

    first_arg = node.args[0]
    if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
        return False

    target = first_arg.value
    if target != BLOCKED_PREFIX and not target.startswith(f"{BLOCKED_PREFIX}."):
        return False

    func = node.func
    if isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}:
        return True
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
    ):
        return True
    return False


def _guard_generator_source(spec: GeneratorSpec) -> None:
    path = _generator_path(spec)
    tree = _parse_generator_source(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gwexpy" or alias.name.startswith("gwexpy."):
                    raise pytest.UsageError(
                        f"{path} imports gwexpy directly, which is not allowed"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "gwexpy" or node.module.startswith("gwexpy.")
            ):
                raise pytest.UsageError(
                    f"{path} imports from gwexpy directly, which is not allowed"
                )
        elif _is_literal_gwexpy_import(node):
            raise pytest.UsageError(
                f"{path} uses a literal gwexpy import call, which is not allowed"
            )


def _run_generator_smoke(spec: GeneratorSpec) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT) if not pythonpath else os.pathsep.join((str(ROOT), pythonpath))
    )

    code = textwrap.dedent(
        f"""
        import importlib
        import pathlib
        import sys
        import tempfile

        class _BlockGwexpy:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "gwexpy" or fullname.startswith("gwexpy."):
                    raise ImportError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, _BlockGwexpy())

        module = importlib.import_module({spec.module_name!r})
        entrypoint = getattr(module, {spec.entrypoint!r})
        assert callable(entrypoint)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir) / "out"
            result = entrypoint(output_dir)
            assert output_dir.exists()
            assert any(output_dir.iterdir())
            assert result is None or isinstance(result, dict)
            if isinstance(result, dict):
                for value in result.values():
                    value_path = pathlib.Path(value)
                    assert value_path.is_absolute()
                    assert value_path.is_relative_to(output_dir)
        """
    )

    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=_GENERATOR_SMOKE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _terminate_timed_out_process(
            process,
            _as_text(exc.stdout),
            _as_text(exc.stderr),
        )
        raise pytest.UsageError(
            "IO conformance generator smoke check timed out for "
            f"{spec.module_name} after {_GENERATOR_SMOKE_TIMEOUT_SECONDS}s:\n"
            f"STDOUT (tail):\n{stdout[-_GENERATOR_SMOKE_TAIL_CHARACTERS:]}\n"
            f"STDERR (tail):\n{stderr[-_GENERATOR_SMOKE_TAIL_CHARACTERS:]}"
        )
    if process.returncode != 0:
        if _looks_like_missing_backend(stderr):
            warnings.warn(
                f"Skipping IO conformance generator smoke check for "
                f"{spec.module_name}: optional backend unavailable "
                f"({stderr.strip().splitlines()[-1] if stderr.strip() else 'unknown'}).",
                UserWarning,
                stacklevel=2,
            )
            return
        raise pytest.UsageError(
            "IO conformance generator smoke check failed for "
            f"{spec.module_name}:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail fast if any IO conformance generator violates the import guard."""

    del session
    for spec in iter_generator_specs():
        _guard_generator_source(spec)
        _run_generator_smoke(spec)
