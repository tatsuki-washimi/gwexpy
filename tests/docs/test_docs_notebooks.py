import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import nbformat
import pytest
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

# Base directory for tutorials
REPO_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = REPO_ROOT / "docs/web/en/user_guide/tutorials"
KERNEL_NAME = "gwexpy-docs-current"

RUN_NOTEBOOK_EXECUTION = (
    os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    or os.environ.get("GWEXPY_RUN_NOTEBOOK_TESTS", "") == "1"
)

pytestmark = pytest.mark.skipif(
    not RUN_NOTEBOOK_EXECUTION,
    reason=(
        "Notebook execution tests are CI-only by default; "
        "set GWEXPY_RUN_NOTEBOOK_TESTS=1 to enable locally."
    ),
)

# List of notebooks to test
NOTEBOOKS = [
    "intro_table.ipynb",
    "intro_noise.ipynb",
    "intro_fitting.ipynb",
    "intro_segment_table.ipynb",
]


def _kernel_spec_argv() -> list[str]:
    return [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]


def _notebook_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["SKIP_MCMC"] = "1"
    return environment


@contextmanager
def _temporary_kernel_manager_class() -> Iterator[type[AsyncKernelManager]]:
    with TemporaryDirectory(prefix="gwexpy-docs-kernel-") as temporary_root:
        kernel_dir = Path(temporary_root) / KERNEL_NAME
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": _kernel_spec_argv(),
                    "display_name": "GWexpy docs (current interpreter)",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        spec_manager = KernelSpecManager(
            kernel_dirs=[temporary_root], ensure_native_kernel=False
        )

        class BoundKernelManager(AsyncKernelManager):
            def __init__(self, *args, **kwargs):
                kwargs["kernel_spec_manager"] = spec_manager
                super().__init__(*args, **kwargs)

        yield BoundKernelManager


def _kernel_validation_source(environment: dict[str, str]) -> str:
    return "\n".join(
        [
            "import os",
            "import sys",
            f"assert sys.executable == {sys.executable!r}",
            "assert sys.version_info[:2] == (3, 11)",
            f"assert os.environ.get('PYTHONNOUSERSITE') == {environment['PYTHONNOUSERSITE']!r}",
            f"assert os.environ.get('PYTHONPATH') == {environment['PYTHONPATH']!r}",
            f"assert os.environ.get('PATH') == {environment['PATH']!r}",
        ]
    )


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebook_execution(nb_name):
    nb_path = NB_DIR / nb_name
    assert nb_path.exists(), f"Notebook {nb_name} not found at {nb_path}"

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    environment = _notebook_environment()
    nb.cells.insert(
        0,
        nbformat.v4.new_code_cell(_kernel_validation_source(environment)),
    )
    try:
        with _temporary_kernel_manager_class() as kernel_manager_class:
            client = NotebookClient(
                nb,
                timeout=300,
                kernel_name=KERNEL_NAME,
                kernel_manager_class=kernel_manager_class,
                resources={"metadata": {"path": str(NB_DIR)}},
            )
            client.execute(env=environment)
    except Exception as e:
        pytest.fail(f"Notebook {nb_name} failed execution: {e}")
