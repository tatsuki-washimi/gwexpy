from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_noise_tutorial_declares_the_packaged_gwinc_dependency() -> None:
    notebook = json.loads(
        (REPO_ROOT / "docs_redesign/tutorials/intro_noise.ipynb").read_text(
            encoding="utf-8"
        )
    )
    first_cell = "".join(notebook["cells"][0]["source"])

    assert "`gwinc` (optional, for detector models)" in first_cell
    assert "pygwinc" not in first_cell


def test_redesign_external_links_use_current_documentation_locations() -> None:
    """Keep links in the redesign source on their verified public locations."""
    source_paths = (
        REPO_ROOT / "docs_redesign/conf.py",
        REPO_ROOT / "docs_redesign/how-to/cli.md",
        REPO_ROOT / "docs_redesign/how-to/case-studies/case_dttxml_calibration.ipynb",
        REPO_ROOT / "gwexpy/interop/openems_.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)

    assert "https://gwpy.github.io/docs/stable/cli/" not in source
    assert "https://gwpy.github.io/docs/stable/" not in source
    assert "https://lscsoft.docs.ligo.org/lalsuite/lal/\n" not in source
    assert (
        "https://docs.ligo.org/lscsoft/lalsuite/lal/group___x_l_a_l_time__c.html"
        in source
    )
    assert "https://dtt.ligo.org/" not in source
    assert "https://openems.de/index.php/HDF5_Field_Dumps.html" not in source
