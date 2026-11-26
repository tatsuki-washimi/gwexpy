"""Contract tests for the English and Japanese SegmentTable references."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from gwpy.segments import Segment

from gwexpy.table import SegmentTable

matplotlib.use("Agg")


ROOT = Path(__file__).parents[2]
DOCS = {
    "en": ROOT / "docs/web/en/reference/SegmentTable.md",
    "ja": ROOT / "docs/web/ja/reference/SegmentTable.md",
}


def _read_docs() -> dict[str, str]:
    return {
        language: path.read_text(encoding="utf-8") for language, path in DOCS.items()
    }


def _normalized(document: str) -> str:
    return re.sub(r"\s+", " ", document)


def _fenced_blocks(document: str, language: str) -> list[str]:
    pattern = rf"```{re.escape(language)}\n(.*?)```"
    return re.findall(pattern, document, flags=re.DOTALL)


def _signature_block(document: str) -> str:
    match = re.search(
        r"## (?:Representative signatures|代表的なシグネチャ).*?```text\n(.*?)```",
        document,
        re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _documented_signatures(document: str) -> dict[str, str]:
    lines = [line for line in _signature_block(document).splitlines() if line]
    signatures: dict[str, str] = {}
    for line in lines:
        prefix, separator, _ = line.partition("(")
        assert separator and prefix.startswith("SegmentTable.")
        method = prefix.removeprefix("SegmentTable.")
        assert method not in signatures, f"duplicate documented signature for {method}"
        signatures[method] = line
    return signatures


def _implementation_signature(method: str) -> str:
    signature = inspect.signature(getattr(SegmentTable, method))
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        signature = signature.replace(parameters=parameters[1:])
    return f"SegmentTable.{method}{signature}"


def test_public_api_and_standalone_container_contract() -> None:
    methods = (
        "from_segments",
        "from_table",
        "read_csv",
        "read",
        "write",
        "extract_stat",
        "crop",
        "asd",
        "bandpass",
        "resample",
        "groupby",
        "plot",
        "scatter",
        "hist",
        "segments",
        "overlay",
        "overlay_spectra",
    )
    for method in methods:
        assert callable(getattr(SegmentTable, method, None))

    assert not issubclass(SegmentTable, pd.DataFrame)

    from gwpy.table import Table

    assert not issubclass(SegmentTable, Table)
    assert not hasattr(SegmentTable, "step")
    assert not hasattr(SegmentTable, "bar")


def test_read_is_format_dispatching_factory_and_roundtrips(tmp_path: Path) -> None:
    assert inspect.getattr_static(SegmentTable, "read") is not inspect.getattr_static(
        SegmentTable, "read_csv"
    )
    read_sig = inspect.signature(SegmentTable.read)
    assert list(read_sig.parameters)[:2] == ["source", "format"]

    table = SegmentTable.from_segments([Segment(0, 1), Segment(2, 3)], snr=[4.0, 9.0])
    h5_path = tmp_path / "roundtrip.h5"
    table.write(h5_path)
    loaded_h5 = SegmentTable.read(h5_path)
    assert [loaded_h5.row(i)["span"] for i in range(2)] == [
        Segment(0, 1),
        Segment(2, 3),
    ]
    assert loaded_h5["snr"].to_list() == [4.0, 9.0]

    csv_path = tmp_path / "roundtrip.csv"
    table.write(csv_path)
    loaded_csv = SegmentTable.read(csv_path)
    assert [loaded_csv.row(i)["span"] for i in range(2)] == [
        Segment(0.0, 1.0),
        Segment(2.0, 3.0),
    ]


def test_reference_signatures_match_inspect_and_keep_kwargs() -> None:
    documents = _read_docs()
    methods = (
        "from_segments",
        "from_table",
        "read_csv",
        "read",
        "write",
        "extract_stat",
        "crop",
        "asd",
        "bandpass",
        "resample",
        "groupby",
        "plot",
        "scatter",
        "hist",
        "segments",
        "overlay",
        "overlay_spectra",
    )
    expected = {method: _implementation_signature(method) for method in methods}
    english = _documented_signatures(documents["en"])
    japanese = _documented_signatures(documents["ja"])
    assert english == expected
    assert japanese == expected

    for method in (
        "from_segments",
        "read_csv",
        "read",
        "write",
        "plot",
        "scatter",
        "hist",
        "segments",
        "overlay",
    ):
        assert "**" in expected[method]
    assert "**kwargs" not in expected["overlay_spectra"]


def test_span_contract_is_structurally_documented() -> None:
    documents = _read_docs()
    required_phrases = (
        "gwpy.segments.Segment",
        "(start, end)",
        "Segment(start, end)",
        "[start ... end)",
        "`start`/`end`",
    )
    for document in documents.values():
        assert {phrase for phrase in required_phrases if phrase in document} == set(
            required_phrases
        )


def _read_span_csv(tmp_path: Path, value: str) -> list[Segment]:
    """Write a one-row CSV whose ``span`` cell is *value*, then read it back."""
    import csv

    path = tmp_path / "span.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["span"])
        writer.writerow([value])
    return list(SegmentTable.read_csv(str(path)).to_pandas()["span"])


@pytest.mark.parametrize(
    ("documented_form", "cell", "expected"),
    [
        ("(start, end)", "(0, 1)", Segment(0.0, 1.0)),
        ("Segment(start, end)", "Segment(0, 1)", Segment(0.0, 1.0)),
        ("[start ... end)", "[0 ... 1)", Segment(0.0, 1.0)),
        ("(start, end)", "(0.5, 1.25)", Segment(0.5, 1.25)),
        ("(start, end)", "(-2, -1)", Segment(-2.0, -1.0)),
    ],
)
def test_documented_span_strings_are_actually_parsed(
    tmp_path: Path, documented_form: str, cell: str, expected: Segment
) -> None:
    """Each span string form the reference lists must really round-trip.

    The structural test above only proves the phrases appear in the prose;
    this one proves the parser accepts them, so the two cannot drift apart.
    """
    assert _read_span_csv(tmp_path, cell) == [expected]


@pytest.mark.parametrize("cell", ["(0, 1, 2)", "(0)", "(a, b)", "0 1"])
def test_span_strings_need_exactly_two_numeric_endpoints(
    tmp_path: Path, cell: str
) -> None:
    """The documented 'exactly two numeric endpoints' rule must be enforced."""
    with pytest.raises(ValueError):
        _read_span_csv(tmp_path, cell)


def test_span_columns_are_converted_and_overridable(tmp_path: Path) -> None:
    """Numeric start/end columns become Segments, and span_cols can rename them."""
    default = tmp_path / "default.csv"
    default.write_text("start,end,flag\n0,1,good\n2,3,bad\n", encoding="utf-8")
    assert list(SegmentTable.read_csv(str(default)).to_pandas()["span"]) == [
        Segment(0.0, 1.0),
        Segment(2.0, 3.0),
    ]

    renamed = tmp_path / "renamed.csv"
    renamed.write_text("t0,t1\n0,1\n", encoding="utf-8")
    table = SegmentTable.read_csv(str(renamed), span_cols=("t0", "t1"))
    assert list(table.to_pandas()["span"]) == [Segment(0.0, 1.0)]


def test_constructors_require_segment_objects_as_documented() -> None:
    """The reference says tuples and lists are not accepted here -- verify it."""
    assert list(
        SegmentTable.from_segments([Segment(0, 1), Segment(2, 3)]).to_pandas()["span"]
    ) == [Segment(0, 1), Segment(2, 3)]

    for rejected in ([(0, 1)], [[0, 1]]):
        with pytest.raises(TypeError, match="gwpy.segments.Segment"):
            SegmentTable.from_segments(rejected)

    # Pre-existing Segment values pass through from_table unchanged.
    frame = pd.DataFrame({"span": [Segment(0, 1), Segment(2, 3)]})
    assert list(SegmentTable.from_table(frame).to_pandas()["span"]) == [
        Segment(0, 1),
        Segment(2, 3),
    ]


def test_reference_rejects_old_inheritance_and_stale_claims() -> None:
    documents = _read_docs()
    old_claims = (
        "**Inherits from:**",
        "**継承元:**",
        "extends the standard GWpy/Astropy Table",
        "GWpy / Astropy Table の機能を拡張",
        "has no `write` method",
        "`write` メソッドはありません",
        "an alias for `read_csv`",
        "`read_csv` の別名です",
        "SegmentTable.step",
        "SegmentTable.bar",
    )
    for document in documents.values():
        assert {claim for claim in old_claims if claim in document} == set()


def test_english_and_japanese_cover_the_same_semantic_contract() -> None:
    documents = _read_docs()
    required = {
        "standalone": ("standalone container", "独立したコンテナ"),
        "not_dataframe": (
            "not a `pandas.DataFrame` subclass",
            "`pandas.DataFrame` のサブクラスでも",
        ),
        "not_gwpy_table": (
            "or a `gwpy.table.Table` subclass",
            "`gwpy.table.Table` のサブクラスでもありません",
        ),
        "read_dispatch": (
            "format-dispatching reader",
            "フォーマットを自動判定するファクトリメソッド",
        ),
        "write_hdf5": (
            "`write` persists the table to HDF5",
            "`write` は HDF5",
        ),
        "span": ("[start ... end)", "[start ... end)"),
    }
    normalized = {
        language: _normalized(document) for language, document in documents.items()
    }
    for english, japanese in required.values():
        assert english in normalized["en"]
        assert japanese in normalized["ja"]

    plot_methods = {"plot", "scatter", "hist", "segments", "overlay", "overlay_spectra"}
    headings = {
        "en": ("## Plot helpers", "## Related tutorials"),
        "ja": ("## プロット用ヘルパー", "## 関連チュートリアル"),
    }
    for language, document in documents.items():
        start, end = headings[language]
        plot_section = document.split(start, 1)[1].split(end, 1)[0]
        documented = {
            method for method in plot_methods if f"`{method}`" in plot_section
        }
        assert documented == plot_methods
        assert "SegmentTable.step" not in plot_section
        assert "SegmentTable.bar" not in plot_section
    assert "does not define `step` or `bar` methods." in documents["en"]
    assert (
        "`SegmentTable` に `step` および `bar` メソッドはありません。"
        in documents["ja"]
    )


@pytest.mark.parametrize("language", ["en", "ja"])
def test_all_python_examples_execute_without_display(language: str) -> None:
    document = DOCS[language].read_text(encoding="utf-8")
    namespace = {"__name__": f"segment_table_reference_{language}"}
    blocks = _fenced_blocks(document, "python")
    assert blocks, f"{language} reference must contain an executable Python block"
    for block in blocks:
        tree = ast.parse(block)
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "show"
            for node in ast.walk(tree)
        )
        exec(compile(tree, str(DOCS[language]), "exec"), namespace)


def test_span_example_uses_supported_segment_objects() -> None:
    table = SegmentTable.from_segments([Segment(0, 1), Segment(2, 3)])
    assert len(table) == 2
