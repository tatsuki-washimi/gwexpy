import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

# Base directory for tutorials
NB_DIR = Path(__file__).parent.parent.parent / "docs/web/en/user_guide/tutorials"

# List of notebooks to test
NOTEBOOKS = [
    "intro_table.ipynb",
    "intro_noise.ipynb",
    "intro_fitting.ipynb",
    "intro_segment_table.ipynb"
]

@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebook_execution(nb_name):
    nb_path = NB_DIR / nb_name
    assert nb_path.exists(), f"Notebook {nb_name} not found at {nb_path}"

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Configure execution
    # We use a 300s timeout as requested
    client = NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': str(NB_DIR)}})

    # Set environment variables if needed (e.g., to skip slow parts)
    os.environ["SKIP_MCMC"] = "1"

    try:
        # Execute the notebook
        client.execute()
    except Exception as e:
        pytest.fail(f"Notebook {nb_name} failed execution: {e}")
