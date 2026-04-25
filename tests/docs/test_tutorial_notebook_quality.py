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
            localized.append("".join(source) if isinstance(source, list) else str(source))
    return localized


def _code_text(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


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


def test_public_tutorials_do_not_reference_internal_api_mapping():
    notebooks = sorted(TUTORIAL_ROOT.glob("*/user_guide/tutorials/*.ipynb"))
    offenders: list[str] = []

    for path in notebooks:
        nb = _read_notebook(path)
        joined = "\n".join(_markdown_texts(nb))
        if "API_MAPPING.md" in joined:
            offenders.append(str(path.relative_to(ROOT)))

    assert not offenders, "Internal API_MAPPING.md references found:\n" + "\n".join(offenders)


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
    nb = _read_tutorial_notebook(Path("en/user_guide/tutorials/case_arima_burst_search.ipynb"))
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
    assert sample_csv.resolve() == (
        ROOT / "docs" / "_static" / "samples" / "sample_segment_data.csv"
    ).resolve()


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
    assert "plt.colorbar(mappable=mesh, ax=ax1, label=\"Power\")" in joined
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
    first_code = next(cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code")

    first_code_source = "".join(first_code.get("source", []))
    first_markdown_source = _localized_markdown_texts(nb, _notebook_locale(relative_path))[0]

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
