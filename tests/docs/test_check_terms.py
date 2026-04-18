import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "check_terms.py"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_strip_code_keeps_doc_label_but_removes_rst_doc_target():
    module = _load_module("gwexpy_check_terms_doc_target")

    stripped = module.strip_code(
        "- :doc:`BruCo 応用 <../user_guide/tutorials/case_bruco_advanced>`"
    )

    assert "BruCo 応用" in stripped
    assert "case_bruco_advanced" not in stripped


def test_strip_code_removes_url_like_external_link_text():
    module = _load_module("gwexpy_check_terms_external_url")

    stripped = module.strip_code(
        "`gwpy.github.io/docs/stable/ <https://gwpy.github.io/docs/stable/>`_"
    )

    assert "stable" not in stripped
    assert "https://gwpy.github.io/docs/stable/" not in stripped


def test_strip_code_masks_toctree_targets_and_anchor_labels():
    module = _load_module("gwexpy_check_terms_toctree")

    stripped = module.strip_code(
        ".. toctree::\n\n   advanced_bruco\n(validated-ja-adaptive-whitening)=\n"
    )

    assert "advanced_bruco" not in stripped
    assert "validated-ja-adaptive-whitening" not in stripped
