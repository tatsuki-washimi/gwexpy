import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_ROOT = ROOT / "docs" / "web"
FORBIDDEN_OUTPUT_PATTERNS = [
    re.compile(r"/home/"),
    re.compile(r"/tmp/"),
    re.compile(r"\bUserWarning\b"),
    re.compile(r"\bDeprecationWarning\b"),
    re.compile(r"\bConvergenceWarning\b"),
]


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _code_cell_source_containing(nb: dict, text: str) -> str:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if text in source:
            return source
    raise AssertionError(f"Could not find code cell containing {text!r}")


@contextmanager
def _pushd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _iter_output_texts(nb: dict):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            chunks: list[str] = []
            text = output.get("text")
            if isinstance(text, list):
                chunks.extend(text)
            elif isinstance(text, str):
                chunks.append(text)
            for mime, payload in output.get("data", {}).items():
                if not mime.startswith("text/"):
                    continue
                if isinstance(payload, list):
                    chunks.extend(payload)
                elif isinstance(payload, str):
                    chunks.append(payload)
            joined = "".join(chunks)
            if joined:
                yield joined


def _markdown_texts(nb: dict) -> list[str]:
    texts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        texts.append("".join(source) if isinstance(source, list) else str(source))
    return texts


def test_tutorial_outputs_do_not_expose_local_paths_or_raw_warnings():
    notebooks = sorted(TUTORIAL_ROOT.glob("*/user_guide/tutorials/*.ipynb"))
    offenders: list[str] = []

    for path in notebooks:
        nb = _read_notebook(path)
        for text in _iter_output_texts(nb):
            hit = next((pat.pattern for pat in FORBIDDEN_OUTPUT_PATTERNS if pat.search(text)), None)
            if hit:
                offenders.append(f"{path.relative_to(ROOT)} -> {hit}")
                break

    assert not offenders, "Forbidden notebook output found:\n" + "\n".join(offenders)


def test_ja_advanced_coupling_mentions_frequency_range_restriction():
    nb = _read_notebook(
        TUTORIAL_ROOT / "ja" / "user_guide" / "tutorials" / "advanced_coupling.ipynb"
    )
    joined = "\n".join(_markdown_texts(nb))
    assert "周波数帯域" in joined or "frange" in joined


def test_ja_case_seismic_obspy_includes_multichannel_section():
    nb = _read_notebook(
        TUTORIAL_ROOT / "ja" / "user_guide" / "tutorials" / "case_seismic_obspy.ipynb"
    )
    joined = "\n".join(_markdown_texts(nb))
    assert "マルチチャンネル" in joined or "3成分" in joined


def test_en_case_arima_burst_search_is_actually_english():
    nb = _read_notebook(
        TUTORIAL_ROOT / "en" / "user_guide" / "tutorials" / "case_arima_burst_search.ipynb"
    )
    first_markdown = _markdown_texts(nb)[0]
    assert "# ARIMA-Based Burst Detection" in first_markdown
    assert "## Introduction" in first_markdown


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/intro_table.ipynb"),
        Path("ja/user_guide/tutorials/intro_table.ipynb"),
    ],
)
def test_intro_table_sample_csv_resolves_from_repo_root(relative_path: Path):
    nb = _read_notebook(TUTORIAL_ROOT / relative_path)
    source = _code_cell_source_containing(nb, "sample_segment_data.csv")

    namespace: dict[str, object] = {}
    with _pushd(ROOT):
        exec(source, namespace)

    assert Path(namespace["sample_csv"]).resolve() == (
        ROOT / "docs" / "_static" / "samples" / "sample_segment_data.csv"
    ).resolve()
