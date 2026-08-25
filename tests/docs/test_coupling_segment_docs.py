from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_coupling_segment_schema_is_reachable_from_the_public_api_toctree() -> None:
    index = (
        ROOT / "docs" / "web" / "en" / "reference" / "api" / "index.rst"
    ).read_text(encoding="utf-8")
    page = ROOT / "docs" / "web" / "en" / "reference" / "api" / "coupling_segment.rst"

    assert "Coupling segment schema <coupling_segment>" in index
    assert page.is_file()
    assert ".. automodule:: gwexpy.coupling.segment" in page.read_text(encoding="utf-8")
