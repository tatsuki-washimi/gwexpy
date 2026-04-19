from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def _listed_case_stems(index_path: str) -> set[str]:
    text = _read_text(index_path)
    stems = set()
    for line in text.splitlines():
        line = line.strip()
        if "case_" not in line or "<" not in line or ">" not in line:
            continue
        stem = line.split("<", 1)[1].split(">", 1)[0]
        if stem.startswith("../user_guide/tutorials/"):
            stem = stem.rsplit("/", 1)[-1]
        stems.add(stem)
    return stems


def _case_notebook_stems(language: str) -> set[str]:
    tutorial_dir = ROOT / "docs" / "web" / language / "user_guide" / "tutorials"
    return {path.stem for path in tutorial_dir.glob("case_*.ipynb")}


def test_tutorial_indexes_do_not_list_case_studies():
    tutorial_indexes = (
        "docs/web/ja/user_guide/tutorials/index.rst",
        "docs/web/en/user_guide/tutorials/index.rst",
    )

    for index_path in tutorial_indexes:
        text = _read_text(index_path)
        assert "ケーススタディ:" not in text
        assert "Case Study:" not in text
        assert "case_" not in text


def test_examples_indexes_are_the_canonical_listing_for_case_studies():
    for language in ("ja", "en"):
        listed = _listed_case_stems(f"docs/web/{language}/examples/index.rst")
        available = _case_notebook_stems(language)
        assert listed == available
