"""Regression checks for the GWpy 4 proxy migration guidance."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_proxy_migration_guidance_and_navigation_are_published() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    en_guide = (ROOT / "docs/web/en/user_guide/gwexpy_for_gwpy_users_en.md").read_text(
        encoding="utf-8"
    )
    ja_guide = (ROOT / "docs/web/ja/user_guide/gwexpy_for_gwpy_users_ja.md").read_text(
        encoding="utf-8"
    )

    for path in (
        "gwexpy.utils.shell",
        "gwexpy.utils.sphinx",
        "gwexpy.utils.sphinx.ex2rst",
        "gwexpy.utils.sphinx.zenodo",
    ):
        assert path in changelog
        assert path in en_guide
        assert path in ja_guide
    for guide in (en_guide, ja_guide):
        assert "FrameL" in guide
        assert "lazy" in guide.lower() or "遅延" in guide

    en_index = ROOT / "docs/web/en/index.rst"
    en_reference = ROOT / "docs/web/en/reference/index.rst"
    ja_index = ROOT / "docs/web/ja/index.rst"
    ja_reference = ROOT / "docs/web/ja/reference/index.rst"
    assert "user_guide/gwexpy_for_gwpy_users_en" in en_index.read_text()
    assert "../user_guide/gwexpy_for_gwpy_users_en" in en_reference.read_text()
    assert "user_guide/gwexpy_for_gwpy_users_ja" in ja_index.read_text()
    assert "../user_guide/gwexpy_for_gwpy_users_ja" in ja_reference.read_text()
