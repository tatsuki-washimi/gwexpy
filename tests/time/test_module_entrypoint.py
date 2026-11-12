"""``python -m gwexpy.time`` must run like ``python -m gwpy.time`` (#784)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gwexpy

REPO_ROOT = Path(gwexpy.__file__).resolve().parent.parent


def _run_module(module, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )


def test_entrypoint_module_imports():
    module = importlib.import_module("gwexpy.time.__main__")

    assert callable(module.main)


def test_entrypoint_reuses_the_gwpy_implementation():
    from gwpy.time.__main__ import main as gwpy_main

    module = importlib.import_module("gwexpy.time.__main__")

    assert module.main is gwpy_main


def test_every_exported_name_resolves():
    module = importlib.import_module("gwexpy.time.__main__")

    for name in module.__all__:
        assert hasattr(module, name), name


def test_main_converts_gps_in_process(capsys):
    module = importlib.import_module("gwexpy.time.__main__")

    module.main(["1187008882"])

    assert capsys.readouterr().out.strip() == "2017-08-17 12:41:04.000000 UTC"


@pytest.mark.parametrize(
    "args",
    [("1187008882",), ("2017-08-17", "12:41:04"), ("-f", "%Y", "1187008882")],
    ids=["gps", "datetime", "format"],
)
def test_module_invocation_matches_gwpy(args):
    ours = _run_module("gwexpy.time", *args)
    theirs = _run_module("gwpy.time", *args)

    assert theirs.returncode == 0, theirs.stderr
    assert ours.returncode == 0, ours.stderr
    assert ours.stdout == theirs.stdout
    assert ours.stdout.strip()


def test_module_help_exits_cleanly():
    result = _run_module("gwexpy.time", "--help")

    assert result.returncode == 0, result.stderr
    assert "GPS" in result.stdout
