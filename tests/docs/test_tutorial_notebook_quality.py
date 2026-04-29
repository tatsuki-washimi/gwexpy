import ast
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import cast

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
FORBIDDEN_PUBLIC_DOC_LINK_PATTERNS = [
    re.compile(r"docs/developers/"),
    re.compile(r"docs_internal/"),
    re.compile(r"API_MAPPING\.md"),
]
STALE_TUTORIAL_CODE_SNIPPETS = [
    "plt.gca().get_images()",
    "plt.gca().collections[-1]",
    "plt.gca().get_children()",
    "hasattr(c, 'get_clim')",
    "spec = data.hht(",
]


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _localized_tutorial_path(relative_path: Path) -> Path:
    path = TUTORIAL_ROOT / relative_path
    if path.exists():
        return path
    parts = relative_path.parts
    if parts and parts[0] == "ja":
        return TUTORIAL_ROOT / Path("en", *parts[1:])
    return path


def _read_tutorial_notebook(relative_path: Path) -> dict:
    return _read_notebook(_localized_tutorial_path(relative_path))


def _public_tutorial_notebooks() -> list[Path]:
    return sorted(TUTORIAL_ROOT.glob("*/user_guide/tutorials/*.ipynb"))


def _public_tutorial_markdown_files() -> list[Path]:
    return sorted(TUTORIAL_ROOT.glob("*/user_guide/tutorials/*.md"))


def _notebook_locale(relative_path: Path) -> str:
    return relative_path.parts[0]


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


def _localized_markdown_texts(nb: dict, locale: str) -> list[str]:
    localized: list[str] = []
    wanted_tag = f"lang-{locale}"
    other_i18n_tags = {"lang-en", "lang-ja"} - {wanted_tag}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []))
        if wanted_tag in tags or not (tags & other_i18n_tags):
            source = cell.get("source", [])
            localized.append(
                "".join(source) if isinstance(source, list) else str(source)
            )
    return localized


def _code_text(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _code_cell_sources(nb: dict) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _call_function_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_mappable_plot_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_function_name(node.func) in {
        "imshow",
        "pcolormesh",
    }


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for element in target.elts for name in _assigned_names(element)}
    return set()


def _explicit_colorbar_mappables_from_plot_assignments(nb: dict) -> list[str]:
    sources = _code_cell_sources(nb)
    parsed_cells: list[ast.Module] = []
    unparsed_colorbar_cells: list[int] = []

    for index, source in enumerate(sources, start=1):
        try:
            parsed_cells.append(ast.parse(source))
        except SyntaxError:
            if "colorbar(" in source:
                unparsed_colorbar_cells.append(index)

    assert not unparsed_colorbar_cells, (
        "Could not parse code cells containing colorbar calls: "
        + ", ".join(str(index) for index in unparsed_colorbar_cells)
    )

    class ColorbarMappableVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.valid_mappables: set[str] = set()
            self.explicit_mappables: list[str] = []
            self.invalid_colorbars: list[str] = []

        def visit_Assign(self, node: ast.Assign) -> None:
            self.visit(node.value)
            target_names = {
                name for target in node.targets for name in _assigned_names(target)
            }
            self._record_assignment(target_names, _is_mappable_plot_call(node.value))

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.annotation is not None:
                self.visit(node.annotation)
            if node.value is None:
                return
            self.visit(node.value)
            self._record_assignment(
                _assigned_names(node.target), _is_mappable_plot_call(node.value)
            )

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            self.visit(node.value)
            self._record_assignment(
                _assigned_names(node.target), _is_mappable_plot_call(node.value)
            )

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self.visit(node.target)
            self.visit(node.value)
            self.valid_mappables.difference_update(_assigned_names(node.target))

        def visit_For(self, node: ast.For) -> None:
            self.visit(node.iter)
            self.valid_mappables.difference_update(_assigned_names(node.target))
            for statement in node.body:
                self.visit(statement)
            for statement in node.orelse:
                self.visit(statement)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.visit(node.iter)
            self.valid_mappables.difference_update(_assigned_names(node.target))
            for statement in node.body:
                self.visit(statement)
            for statement in node.orelse:
                self.visit(statement)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            self.visit(node.iter)
            self.valid_mappables.difference_update(_assigned_names(node.target))
            for condition in node.ifs:
                self.visit(condition)

        def visit_Call(self, node: ast.Call) -> None:
            if not (
                isinstance(node, ast.Call)
                and _call_function_name(node.func) == "colorbar"
            ):
                self.generic_visit(node)
                return

            mappable = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "mappable"
                ),
                None,
            )
            if mappable is None:
                self.invalid_colorbars.append(
                    f"line {node.lineno}: colorbar call lacks mappable="
                )
            elif not isinstance(mappable, ast.Name):
                self.invalid_colorbars.append(
                    f"line {node.lineno}: mappable= is not a simple assigned name"
                )
            elif mappable.id not in self.valid_mappables:
                self.invalid_colorbars.append(
                    f"line {node.lineno}: mappable={mappable.id} is not assigned "
                    "from imshow/pcolormesh"
                )
            else:
                self.explicit_mappables.append(mappable.id)

            self.generic_visit(node)

        def _record_assignment(
            self, target_names: set[str], is_mappable_assignment: bool
        ) -> None:
            if is_mappable_assignment:
                self.valid_mappables.update(target_names)
            else:
                self.valid_mappables.difference_update(target_names)

    visitor = ColorbarMappableVisitor()
    for tree in parsed_cells:
        visitor.visit(tree)

    assert not visitor.invalid_colorbars, "Invalid colorbar calls:\n" + "\n".join(
        visitor.invalid_colorbars
    )
    return visitor.explicit_mappables


def _synthetic_notebook(*sources: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": source,
            }
            for source in sources
        ]
    }


def test_colorbar_mappable_guard_rejects_use_before_assignment():
    nb = _synthetic_notebook(
        "plt.colorbar(mappable=mesh)\n",
        "mesh = ax.pcolormesh(x, y, z)\n",
    )

    with pytest.raises(AssertionError, match="mappable=mesh is not assigned"):
        _explicit_colorbar_mappables_from_plot_assignments(nb)


def test_colorbar_mappable_guard_rejects_stale_reassignment():
    nb = _synthetic_notebook(
        "mesh = ax.pcolormesh(x, y, z)\n",
        "mesh = None\n",
        "plt.colorbar(mappable=mesh)\n",
    )

    with pytest.raises(AssertionError, match="mappable=mesh is not assigned"):
        _explicit_colorbar_mappables_from_plot_assignments(nb)


def test_colorbar_mappable_guard_accepts_current_pcolormesh_assignment():
    nb = _synthetic_notebook(
        "mesh = ax.pcolormesh(x, y, z)\n",
        "plt.colorbar(mappable=mesh)\n",
    )

    assert _explicit_colorbar_mappables_from_plot_assignments(nb) == ["mesh"]


def test_tutorial_outputs_do_not_expose_local_paths_or_raw_warnings():
    notebooks = sorted(TUTORIAL_ROOT.glob("*/user_guide/tutorials/*.ipynb"))
    offenders: list[str] = []

    for path in notebooks:
        nb = _read_notebook(path)
        for text in _iter_output_texts(nb):
            hit = next(
                (pat.pattern for pat in FORBIDDEN_OUTPUT_PATTERNS if pat.search(text)),
                None,
            )
            if hit:
                offenders.append(f"{path.relative_to(ROOT)} -> {hit}")
                break

    assert not offenders, "Forbidden notebook output found:\n" + "\n".join(offenders)


def test_public_tutorial_markdown_does_not_link_internal_docs_surfaces():
    offenders: list[str] = []

    for path in _public_tutorial_notebooks():
        nb = _read_notebook(path)
        for markdown in _markdown_texts(nb):
            hit = next(
                (
                    pattern.pattern
                    for pattern in FORBIDDEN_PUBLIC_DOC_LINK_PATTERNS
                    if pattern.search(markdown)
                ),
                None,
            )
            if hit:
                offenders.append(f"{path.relative_to(ROOT)} -> {hit}")
                break

    for path in _public_tutorial_markdown_files():
        markdown = path.read_text()
        hit = next(
            (
                pattern.pattern
                for pattern in FORBIDDEN_PUBLIC_DOC_LINK_PATTERNS
                if pattern.search(markdown)
            ),
            None,
        )
        if hit:
            offenders.append(f"{path.relative_to(ROOT)} -> {hit}")

    assert not offenders, "Forbidden internal-doc link found:\n" + "\n".join(offenders)


def test_public_tutorial_code_does_not_use_known_stale_hht_or_colorbar_patterns():
    offenders: list[str] = []

    for path in _public_tutorial_notebooks():
        joined = _code_text(_read_notebook(path))
        hit = next(
            (snippet for snippet in STALE_TUTORIAL_CODE_SNIPPETS if snippet in joined),
            None,
        )
        if hit:
            offenders.append(f"{path.relative_to(ROOT)} -> {hit}")

    for path in _public_tutorial_markdown_files():
        markdown = path.read_text()
        hit = next(
            (
                snippet
                for snippet in STALE_TUTORIAL_CODE_SNIPPETS
                if snippet in markdown
            ),
            None,
        )
        if hit:
            offenders.append(f"{path.relative_to(ROOT)} -> {hit}")

    assert not offenders, "Stale tutorial pattern found:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/intro_interop.ipynb"),
        Path("ja/user_guide/tutorials/intro_interop.ipynb"),
    ],
)
def test_intro_interop_uses_explicit_axes_for_pandas_plot(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    source = _code_cell_source_containing(nb, 's_pd = ts.to_pandas(index="datetime")')

    assert "fig, ax = plt.subplots()" in source
    assert 's_pd.plot(ax=ax, title="Pandas Series")' in source
    assert "plt.close(fig)" in source


def test_intro_interop_explains_all_extra_scope():
    nb = _read_tutorial_notebook(Path("en/user_guide/tutorials/intro_interop.ipynb"))
    markdown = " ".join("\n".join(_markdown_texts(nb)).split())

    assert "`gwexpy[all]` installs the declared GWexpy extras" in markdown
    assert "does not install every public interop backend" in markdown


def test_example_intro_interop_uses_explicit_axes_for_pandas_plot():
    nb = _read_notebook(ROOT / "examples" / "basic-new-methods" / "intro_Interop.ipynb")
    source = _code_cell_source_containing(nb, 's_pd = ts.to_pandas(index="datetime")')

    assert "fig, ax = plt.subplots()" in source
    assert 's_pd.plot(ax=ax, title="Pandas Series")' in source
    assert "plt.close(fig)" in source


def test_ja_advanced_coupling_mentions_frequency_range_restriction():
    relative_path = Path("ja/user_guide/tutorials/advanced_coupling.ipynb")
    nb = _read_tutorial_notebook(relative_path)
    joined = "\n".join(_localized_markdown_texts(nb, _notebook_locale(relative_path)))
    assert "周波数帯域" in joined or "frange" in joined


def test_ja_case_seismic_obspy_includes_multichannel_section():
    relative_path = Path("ja/user_guide/tutorials/case_seismic_obspy.ipynb")
    nb = _read_tutorial_notebook(relative_path)
    joined = "\n".join(_localized_markdown_texts(nb, _notebook_locale(relative_path)))
    assert "マルチチャンネル" in joined or "3成分" in joined


def test_en_case_arima_burst_search_is_actually_english():
    relative_path = Path("en/user_guide/tutorials/case_arima_burst_search.ipynb")
    nb = _read_tutorial_notebook(relative_path)
    first_markdown = _localized_markdown_texts(nb, _notebook_locale(relative_path))[0]
    assert "# ARIMA-Based Burst Detection" in first_markdown
    assert "## Introduction" in first_markdown


def test_en_case_arima_burst_search_has_markdown_sections_not_code():
    nb = _read_tutorial_notebook(
        Path("en/user_guide/tutorials/case_arima_burst_search.ipynb")
    )
    code_texts = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    assert all("[![Open In Colab]" not in text for text in code_texts)
    assert all(
        not text.lstrip().startswith("## 1. Generate detector noise")
        for text in code_texts
    )


def test_advanced_arima_notebooks_are_tagged_ci_heavy():
    for relative_path in (
        Path("en/user_guide/tutorials/advanced_bruco.ipynb"),
        Path("en/user_guide/tutorials/advanced_arima.ipynb"),
        Path("en/user_guide/tutorials/advanced_correlation.ipynb"),
        Path("en/user_guide/tutorials/advanced_fitting.ipynb"),
        Path("en/user_guide/tutorials/advanced_peak_tracking.ipynb"),
        Path("en/user_guide/tutorials/advanced_spectrogram_processing.ipynb"),
        Path("en/user_guide/tutorials/case_bootstrap_gls_fitting.ipynb"),
        Path("en/user_guide/tutorials/case_gbd_format.ipynb"),
        Path("en/user_guide/tutorials/case_transfer_function.ipynb"),
        Path("en/user_guide/tutorials/intro_interop.ipynb"),
        Path("en/user_guide/tutorials/intro_plotting.ipynb"),
        Path("en/user_guide/tutorials/intro_timeseries.ipynb"),
        Path("en/user_guide/tutorials/matrix_frequencyseries.ipynb"),
        Path("en/user_guide/tutorials/matrix_spectrogram.ipynb"),
        Path("en/user_guide/tutorials/matrix_timeseries.ipynb"),
        Path("en/user_guide/tutorials/rayleigh_gauch_tutorial.ipynb"),
        Path("en/user_guide/tutorials/advanced_decomposition.ipynb"),
    ):
        nb = _read_tutorial_notebook(relative_path)
        tags = nb.get("cells", [{}])[0].get("metadata", {}).get("tags", [])
        assert "ci-heavy" in tags


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/intro_table.ipynb"),
        Path("ja/user_guide/tutorials/intro_table.ipynb"),
    ],
)
def test_intro_table_sample_csv_resolves_from_repo_root(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    source = _code_cell_source_containing(nb, "sample_segment_data.csv")

    namespace: dict[str, object] = {}
    with _pushd(ROOT):
        exec(source, namespace)

    sample_csv = cast(Path, namespace["sample_csv"])
    assert (
        sample_csv.resolve()
        == (ROOT / "docs" / "_static" / "samples" / "sample_segment_data.csv").resolve()
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/case_bootstrap_gls_fitting.ipynb"),
        Path("ja/user_guide/tutorials/case_bootstrap_gls_fitting.ipynb"),
    ],
)
def test_case_bootstrap_gls_fitting_uses_explicit_mappables_for_colorbars(
    relative_path: Path,
):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "plt.gca().get_images()" not in joined
    assert "plt.gca().collections[-1]" not in joined
    assert "plt.colorbar(mappable=im, ax=ax1" in joined
    assert "plt.colorbar(mappable=im3, ax=ax3" in joined
    assert "plt.colorbar(mappable=im4, ax=ax4" in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/advanced_hht.ipynb"),
        Path("ja/user_guide/tutorials/advanced_hht.ipynb"),
    ],
)
def test_advanced_hht_uses_explicit_mappables_for_colorbars(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "plt.gca().get_images()" not in joined
    assert "plt.gca().collections[-1]" not in joined
    assert 'plt.colorbar(mappable=mesh, ax=ax1, label="Power")' in joined
    assert "sc = None" in joined
    assert "if sc is not None:" in joined
    assert "cbar = plt.colorbar(mappable=sc, ax=ax2)" in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/advanced_hht.ipynb"),
        Path("ja/user_guide/tutorials/advanced_hht.ipynb"),
    ],
)
def test_advanced_hht_spectrogram_example_calls_hht_on_timeseries(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "spec = data.hht(" not in joined
    assert "spec = ts_norm.hht(" in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/advanced_control_basics.ipynb"),
        Path("ja/user_guide/tutorials/advanced_control_basics.ipynb"),
        Path("en/user_guide/tutorials/advanced_control_discretization.ipynb"),
        Path("ja/user_guide/tutorials/advanced_control_discretization.ipynb"),
        Path("en/user_guide/tutorials/advanced_control_modeling.ipynb"),
        Path("ja/user_guide/tutorials/advanced_control_modeling.ipynb"),
        Path("en/user_guide/tutorials/case_coupling_analysis.ipynb"),
        Path("ja/user_guide/tutorials/case_coupling_analysis.ipynb"),
        Path("en/user_guide/tutorials/case_lockin_detection.ipynb"),
        Path("ja/user_guide/tutorials/case_lockin_detection.ipynb"),
        Path("en/user_guide/tutorials/case_signal_extraction.ipynb"),
        Path("ja/user_guide/tutorials/case_signal_extraction.ipynb"),
        Path("en/user_guide/tutorials/case_wiener_filter.ipynb"),
        Path("ja/user_guide/tutorials/case_wiener_filter.ipynb"),
        Path("ja/user_guide/tutorials/advanced_hht.ipynb"),
    ],
)
def test_non_fitting_tutorials_keep_source_notebooks_clean(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    assert not any(
        cell.get("cell_type") == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
        for cell in nb.get("cells", [])
    ), f"Expected clean committed notebook source in {relative_path}"


def test_ja_advanced_hht_keeps_note_in_markdown_not_code():
    relative_path = Path("ja/user_guide/tutorials/advanced_hht.ipynb")
    nb = _read_tutorial_notebook(relative_path)
    first_code = next(
        cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"
    )

    first_code_source = "".join(first_code.get("source", []))
    first_markdown_source = _localized_markdown_texts(
        nb, _notebook_locale(relative_path)
    )[0]

    assert "ワークフロー重視" not in first_code_source
    assert "ワークフロー重視" in first_markdown_source


def test_ja_advanced_hht_spectrogram_cell_keeps_inline_kwargs():
    nb = _read_tutorial_notebook(Path("ja/user_guide/tutorials/advanced_hht.ipynb"))
    joined = _code_text(nb)

    assert "emd_kwargs=emd_kwargs" not in joined
    assert "hilbert_kwargs=hilbert_kwargs" not in joined
    assert '"eemd_trials": 10' in joined
    assert '"pad": 200' in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/intro_frequencyseries.ipynb"),
        Path("ja/user_guide/tutorials/intro_frequencyseries.ipynb"),
    ],
)
def test_intro_frequencyseries_avoids_slow_plot_wrappers(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "ts.plot(title=ts.name)" not in joined
    assert "red_ts.plot(" not in joined
    assert "ax.plot(ts.times.value, ts.value" in joined
    assert "axes[1].plot(red_ts.times.value, red_ts.value" in joined


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/case_seismic_obspy.ipynb"),
        Path("ja/user_guide/tutorials/case_seismic_obspy.ipynb"),
    ],
)
def test_case_seismic_obspy_avoids_slow_plot_wrappers(relative_path: Path):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "ts_seismic.plot(" not in joined
    assert "ax.plot(ts_seismic.times.value, ts_seismic.value" in joined
    assert "plot = sg.plot()" not in joined
    assert "mesh = ax.pcolormesh(" in joined


@pytest.mark.parametrize(
    ("relative_path", "minimum_explicit_colorbars"),
    [
        (
            Path("en/user_guide/tutorials/advanced_correlation.ipynb"),
            1,
        ),
        (
            Path("en/user_guide/tutorials/time_frequency_analysis_comparison.ipynb"),
            2,
        ),
        (
            Path("en/user_guide/tutorials/case_gbd_format.ipynb"),
            1,
        ),
        (
            Path("en/user_guide/tutorials/rayleigh_gauch_tutorial.ipynb"),
            3,
        ),
    ],
)
def test_public_tutorial_colorbars_use_explicit_mappables(
    relative_path: Path,
    minimum_explicit_colorbars: int,
):
    nb = _read_tutorial_notebook(relative_path)
    joined = _code_text(nb)

    assert "plt.gca().get_images()" not in joined
    assert "plt.gca().collections[-1]" not in joined
    assert "plt.gca().get_children()" not in joined
    assert "hasattr(c, 'get_clim')" not in joined

    explicit_mappables = _explicit_colorbar_mappables_from_plot_assignments(nb)
    assert len(explicit_mappables) >= minimum_explicit_colorbars


def test_case_gbd_format_spectrogram_uses_auto_gps_xscale():
    nb = _read_tutorial_notebook(Path("en/user_guide/tutorials/case_gbd_format.ipynb"))
    source = _code_cell_source_containing(nb, "ts_ch0.spectrogram")
    tree = ast.parse(source)

    assert "sg = ts_ch0.spectrogram" in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_xscale"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "auto-gps"
        for node in ast.walk(tree)
    )
