import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_interop_autosummary_uses_currentmodule_short_names():
    for rel in (
        "docs/web/en/reference/api/interop.rst",
        "docs/web/ja/reference/api/interop.rst",
    ):
        text = (ROOT / rel).read_text()
        assert "gwexpy.interop." not in text


def test_io_format_guides_do_not_start_sections_with_transition():
    for rel in (
        "docs/web/en/user_guide/io_formats.md",
        "docs/web/ja/user_guide/io_formats.md",
    ):
        lines = [line.rstrip() for line in (ROOT / rel).read_text().splitlines()]
        assert lines[3] != "---"


def test_time_frequency_comparison_notebook_has_no_transition_only_markdown_cells():
    nb = json.loads(
        (ROOT / "docs/web/en/user_guide/tutorials/time_frequency_analysis_comparison.ipynb").read_text()
    )
    markdown_cells = [
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in nb["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    assert all(text.strip() != "---" and not text.strip().startswith("---\n") for text in markdown_cells)
