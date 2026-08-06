"""Regression checks for claims made by the redesigned public website."""

from pathlib import Path

from babel.messages import pofile

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs_redesign"


def _read(relative_path: str) -> str:
    return (DOCS / relative_path).read_text(encoding="utf-8")


def test_redesign_source_notes_are_not_published() -> None:
    """Internal source notes must not become a public README page."""
    conf = _read("conf.py")

    assert '"README.md"' in conf


def test_verification_page_describes_current_redesign_ci() -> None:
    """Keep legacy notebook policy distinct from the redesigned-site build."""
    page = _read("explanation/verification_and_quality.md")

    assert "isolated temporary copy" in page
    assert "MyST-NB cache" in page
    assert "docs/web/{en,ja}" not in page
    assert "executes changed notebooks with `papermill`" not in page


def test_interop_page_does_not_claim_an_unmeasured_priority_ranking() -> None:
    """Interop ordering is architectural guidance, not usage statistics."""
    page = _read("how-to/interop.md")

    assert "What to Prioritize First" not in page
    assert "not a ranking by usage statistics" in page
    assert "## S. Foundation Layer" in page
    assert "pyroomacoustics" in page
    assert "librosa" in page


def test_interop_signal_processing_heading_preserves_legacy_anchor() -> None:
    """Keep the clarified category title without breaking the published anchor."""
    page = _read("how-to/interop.md")
    new_heading = "C. Scientific Computing, Signal Processing, Machine Learning, and Array Backends"
    old_heading = "C. Scientific Computing, Machine Learning, and Array Backends"
    catalog_path = DOCS / "locales/ja/LC_MESSAGES/how-to/interop.po"

    assert f"## {new_heading}" in page
    assert f"## {old_heading}" not in page
    assert "(interop-en-ml-conversion)=" in page

    with catalog_path.open(encoding="utf-8") as stream:
        catalog = pofile.read_po(stream)
    translated = catalog.get(new_heading)
    assert translated is not None
    assert translated.string == "C. 科学計算、信号処理、機械学習、配列バックエンド"
    jump_link = catalog.get(f"[{new_heading}](#interop-en-ml-conversion)")
    assert jump_link is not None
    assert (
        jump_link.string
        == "[C. 科学計算、信号処理、機械学習、配列バックエンド](#interop-en-ml-conversion)"
    )


def test_validation_and_stability_claims_are_scoped_to_implemented_evidence() -> None:
    """Do not turn selected tests or helpers into universal guarantees."""
    validation = _read("explanation/validated_algorithms.md")
    stability = _read("explanation/numerical_stability.md")

    assert "The algorithms listed on this page" in validation
    assert "We generally use $10^{-12}$" not in validation
    assert "All algorithms are verified" not in validation
    assert "`.plot()` or `.spectrogram()`" not in stability
    assert "regardless of input amplitude" not in stability
    assert "`safe_log_scale()`" in stability


def test_citation_and_troubleshooting_have_actionable_public_guidance() -> None:
    """Do not publish placeholder citations or checkout-only commands as PyPI help."""
    citation = _read("about/citation.md")
    troubleshooting = _read("how-to/troubleshooting.md")

    assert "Duncan Macleod et al., gwpy/gwpy: ..." not in citation
    assert "Astropy Collaboration et al., ..." not in citation
    assert "github.com/tatsuki-washimi/gwexpy/security/policy" in troubleshooting
    assert "source checkout" in troubleshooting


def test_architecture_and_physics_pages_describe_observable_behavior() -> None:
    """Avoid unsupported superlatives in public explanatory pages."""
    architecture = _read("explanation/architecture.md")
    physics = _read("explanation/physics_models.md")
    combined = f"{architecture}\n{physics}".lower()

    for claim in (
        "perfectly synchronized",
        "extreme speed",
        "scans thousands of auxiliary channels",
        "optimized for physical data characteristics",
    ):
        assert claim not in combined


def test_corrected_public_pages_have_complete_japanese_catalogs() -> None:
    """Do not expose the corrected claims as English fallbacks in Japanese HTML."""
    catalogs = (
        "explanation/verification_and_quality.po",
        "how-to/interop.po",
        "explanation/validated_algorithms.po",
        "explanation/numerical_stability.po",
        "about/citation.po",
        "how-to/troubleshooting.po",
        "explanation/architecture.po",
        "explanation/physics_models.po",
    )
    locale_root = DOCS / "locales/ja/LC_MESSAGES"

    for relative_path in catalogs:
        with (locale_root / relative_path).open(encoding="utf-8") as stream:
            catalog = pofile.read_po(stream)
        problems = [
            message.id
            for message in catalog
            if message.id and (not message.string or "fuzzy" in message.flags)
        ]
        assert not problems, relative_path
