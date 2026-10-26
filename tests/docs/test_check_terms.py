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


def test_term_scan_checks_japanese_po_translations_not_sources_or_fuzzy_entries(
    tmp_path,
):
    module = _load_module("gwexpy_check_terms_po")
    catalog = tmp_path / "guide.po"
    catalog.write_text(
        'msgid "bad variant"\n'
        'msgstr "canonical"\n'
        "\n"
        'msgid "safe source"\n'
        'msgstr "bad variant"\n'
        "\n"
        "#, fuzzy\n"
        'msgid "draft"\n'
        'msgstr "bad variant"\n',
        encoding="utf-8",
    )

    hits = module.scan_po_catalog(catalog, {"canonical": ["bad variant"]})

    assert len(hits) == 1
    assert hits[0]["path"] == catalog
    assert hits[0]["matched"] == "bad variant"


def test_physical_alignment_drift_is_not_misclassified_as_estimation(tmp_path):
    module = _load_module("gwexpy_check_terms_alignment")
    catalog = tmp_path / "guide.po"
    catalog.write_text(
        'msgid "physical drift"\n'
        'msgstr "低周波のアライメント変動"\n'
        "\n"
        'msgid "bad translation"\n'
        'msgstr "整数秒アライメント"\n',
        encoding="utf-8",
    )

    hits = module.scan_po_catalog(catalog, module.load_terms())

    assert [hit["matched"] for hit in hits] == ["アライメント"]


def test_canonical_term_matching_is_case_insensitive(tmp_path):
    module = _load_module("gwexpy_check_terms_canonical_case")
    catalog = tmp_path / "guide.po"
    catalog.write_text(
        'msgid "name"\nmsgstr "BruCo"\n',
        encoding="utf-8",
    )

    hits = module.scan_po_catalog(catalog, {"BruCo": ["Bruco"]})

    assert hits == []
