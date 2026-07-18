from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_docs_redirect_stubs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_docs_redirect_stubs", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_redirect_targets_rejects_missing_published_page(tmp_path: Path):
    module = load_script_module()

    with pytest.raises(FileNotFoundError, match="tutorials/quickstart.html"):
        module.validate_redirect_targets(tmp_path)
