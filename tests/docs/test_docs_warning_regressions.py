import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DOC_SUFFIXES = {".ipynb", ".md", ".rst"}
PUBLIC_DOC_LINK_RE = re.compile(r"\]\(([^)]+\.(?:md|rst)(?:#[^)]*)?)\)")


def _read_notebook(relative_path: str) -> dict:
    path = ROOT / relative_path
    if not path.exists() and "/ja/" in relative_path:
        path = ROOT / relative_path.replace("/ja/", "/en/", 1)
    return json.loads(path.read_text())


def _cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _iter_public_doc_texts():
    for path in sorted((ROOT / "docs" / "web").rglob("*")):
        if path.is_file() and path.suffix in PUBLIC_DOC_SUFFIXES:
            yield path, path.read_text(encoding="utf-8")


def test_interop_autosummary_uses_currentmodule_short_names():
    for rel in (
        "docs/web/en/reference/api/interop.rst",
        "docs/web/ja/reference/api/interop.rst",
    ):
        text = (ROOT / rel).read_text()
        assert "gwexpy.interop." not in text


def test_public_docs_do_not_reference_internal_api_mapping():
    offenders = [
        str(path.relative_to(ROOT))
        for path, text in _iter_public_doc_texts()
        if "API_MAPPING.md" in text
    ]

    assert not offenders, "Internal API_MAPPING.md references found:\n" + "\n".join(offenders)


def test_public_docs_relative_markdown_links_exist():
    offenders: list[str] = []

    for path, text in _iter_public_doc_texts():
        for match in PUBLIC_DOC_LINK_RE.finditer(text):
            raw_target = match.group(1)
            target = raw_target.split("#", 1)[0]
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to((ROOT / "docs" / "web").resolve())
            except ValueError:
                continue
            if not resolved.is_file():
                offenders.append(f"{path.relative_to(ROOT)} -> {raw_target}")

    assert not offenders, "Broken public docs links found:\n" + "\n".join(offenders)


def test_io_format_guides_do_not_start_sections_with_transition():
    for rel in (
        "docs/web/en/user_guide/io_formats.md",
        "docs/web/ja/user_guide/io_formats.md",
    ):
        lines = [line.rstrip() for line in (ROOT / rel).read_text().splitlines()]
        assert lines[3] != "---"


def test_time_frequency_comparison_notebook_has_no_transition_only_markdown_cells():
    nb = _read_notebook("docs/web/en/user_guide/tutorials/time_frequency_analysis_comparison.ipynb")
    markdown_cells = [
        _cell_source(cell) for cell in nb["cells"] if cell.get("cell_type") == "markdown"
    ]
    assert all(text.strip() != "---" and not text.strip().startswith("---\n") for text in markdown_cells)


def test_segment_tutorial_setup_cells_define_segment_table_at_top_level():
    for rel in (
        "docs/web/en/user_guide/tutorials/segment_visualization.ipynb",
        "docs/web/ja/user_guide/tutorials/segment_visualization.ipynb",
        "docs/web/en/user_guide/tutorials/segment_asd_pipeline.ipynb",
        "docs/web/ja/user_guide/tutorials/segment_asd_pipeline.ipynb",
    ):
        nb = _read_notebook(rel)
        setup_cells = [
            _cell_source(cell)
            for cell in nb["cells"]
            if cell.get("cell_type") == "code" and "SegmentTable.from_segments(" in _cell_source(cell)
        ]
        assert len(setup_cells) == 1
        source = setup_cells[0]
        assert "\n    segs =" in source
        assert "\n    st = SegmentTable.from_segments(" in source
        assert "\n    st.add_series_column(" in source
        assert "\n        segs =" not in source
        assert "\n        st = SegmentTable.from_segments(" not in source
        assert "\n        st.add_series_column(" not in source
