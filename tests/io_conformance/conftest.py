"""Import guards and smoke checks for the IO conformance generator harness."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from .generators import GeneratorSpec, iter_generator_specs

ROOT = Path(__file__).resolve().parents[2]
GENERATORS_DIR = Path(__file__).resolve().parent / "generators"
BLOCKED_PREFIX = "gwexpy"


def _generator_path(spec: GeneratorSpec) -> Path:
    return GENERATORS_DIR / f"{spec.name}.py"


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
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise pytest.UsageError(
            "IO conformance generator smoke check failed for "
            f"{spec.module_name}:\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail fast if any IO conformance generator violates the import guard."""

    del session
    for spec in iter_generator_specs():
        _guard_generator_source(spec)
        _run_generator_smoke(spec)
