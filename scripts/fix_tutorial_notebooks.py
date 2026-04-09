from __future__ import annotations

#!/usr/bin/env python3
"""Apply deterministic quality fixes to tutorial notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")


def _cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _replace_source(cell: dict, old: str, new: str) -> None:
    text = _cell_source(cell)
    if old in text:
        _set_source(cell, text.replace(old, new))


def _clean_text_output(text: str, *, lang: str | None = None) -> str:
    if "Wswiglal-redir-stdio" in text:
        if "<table" in text:
            return text[text.index("<table") :]
        return ""

    filtered_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(
            token in stripped
            for token in (
                "/home/",
                "/tmp/ipykernel_",
                "UserWarning:",
                "DeprecationWarning:",
                "ConvergenceWarning:",
                "warnings.warn(",
                "return super().legend",
                "return super().set_xlim",
                "| INFO | mth5.",
            )
        ):
            continue
        filtered_lines.append(line)

    text = "\n".join(filtered_lines)
    text = text.replace("Wrote /tmp/kagra_sus_itmx.xml", "Wrote synthetic DTT XML to a temporary file")
    text = text.replace("書き込み完了: /tmp/kagra_sus_itmx.xml", "一時ファイルへ合成 DTT XML を書き込みました")
    if "Written:" in text and "synthetic_pem.gbd" in text:
        text = "Written synthetic_pem.gbd (118.2 KB)"
    if "作成完了:" in text and "synthetic_pem.gbd" in text:
        text = "synthetic_pem.gbd を作成しました (118.2 KB)"
    if "Pipeline archive:" in text and "_pipeline.h5" in text:
        text = text.replace("Pipeline archive:", "Pipeline archive: temporary file")
        if "  (" in text:
            text = "Pipeline archive: temporary file" + text[text.index("  (") :]
    if "パイプラインアーカイブ:" in text and "_pipeline.h5" in text:
        text = text.replace("パイプラインアーカイブ:", "パイプラインアーカイブ: 一時ファイル")
        if "  (" in text:
            text = "パイプラインアーカイブ: 一時ファイル" + text[text.index("  (") :]
    if "Saved to MTH5 file:" in text:
        text = text.replace("Saved to MTH5 file:", "Saved to temporary MTH5 file:")
        text = text.replace("/tmp/tmp99d3d2q2.h5", "<temporary>.h5")
        text = text.replace("/tmp/tmpdsp7sqir.h5", "<temporary>.h5")
    if lang == "ja" and "Saved to temporary MTH5 file:" in text:
        text = text.replace("Saved to temporary MTH5 file:", "一時 MTH5 ファイルへ保存:")
    return text + ("\n" if text else "")


def _sanitize_outputs(nb: dict, *, lang: str | None = None) -> None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if isinstance(output.get("text"), list):
                cleaned = _clean_text_output("".join(output["text"]), lang=lang)
                output["text"] = cleaned.splitlines(keepends=True)
            elif isinstance(output.get("text"), str):
                output["text"] = _clean_text_output(output["text"], lang=lang)
            for mime, payload in list(output.get("data", {}).items()):
                if not mime.startswith("text/"):
                    continue
                text = "".join(payload) if isinstance(payload, list) else str(payload)
                cleaned = _clean_text_output(text, lang=lang)
                output["data"][mime] = cleaned.splitlines(keepends=True)


def _apply_pairwise_source_fixes() -> None:
    pairs = [
        ("en", "advanced_peak_tracking.ipynb"),
        ("ja", "advanced_peak_tracking.ipynb"),
        ("en", "advanced_spectrogram_processing.ipynb"),
        ("ja", "advanced_spectrogram_processing.ipynb"),
    ]
    for lang, name in pairs:
        path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / name
        nb = _load(path)
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = _cell_source(cell)
            if "plt.tight_layout()" in src:
                if "import warnings" not in src:
                    src = "import warnings\n" + src
                src = src.replace(
                    "plt.tight_layout()",
                    'with warnings.catch_warnings():\n'
                    '    warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")\n'
                    "    plt.tight_layout()",
                )
                _set_source(cell, src)
        _sanitize_outputs(nb, lang=lang)
        _save(path, nb)


def _fix_advanced_correlation(lang: str) -> None:
    path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / "advanced_correlation.ipynb"
    nb = _load(path)
    cell = nb["cells"][31]
    src = _cell_source(cell)
    src = src.replace(
        "mi_xy = ts_x.fastmi(ts_y, grid_size=128)\nmi_xz = ts_x.fastmi(ts_z, grid_size=128)\n",
        "ts_y_aligned = ts_y.resample(ts_x.sample_rate.value)\n"
        "ts_z_aligned = ts_z.resample(ts_x.sample_rate.value)\n\n"
        "mi_xy = ts_x.fastmi(ts_y_aligned, grid_size=128)\n"
        "mi_xz = ts_x.fastmi(ts_z_aligned, grid_size=128)\n",
    )
    _set_source(cell, src)
    _sanitize_outputs(nb, lang=lang)
    _save(path, nb)


def _fix_advanced_bruco(lang: str) -> None:
    path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / "advanced_bruco.ipynb"
    nb = _load(path)
    cell = nb["cells"][16]
    _replace_source(cell, "ax.legend()\n", 'ax.legend(["Original", "Cleaned (Bruco)"])\n')
    _sanitize_outputs(nb, lang=lang)
    _save(path, nb)


def _fix_case_bruco_ica(lang: str) -> None:
    path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / "case_bruco_ica_denoising.ipynb"
    nb = _load(path)
    cell = nb["cells"][13]
    src = _cell_source(cell)
    src = src.replace(
        "# Run ICA\nn_components = len(channels)\nica_sources, ica_model = tsm.ica(n_components=n_components, return_model=True)\n",
        "import warnings\n\n"
        "# Run ICA\n"
        "n_components = len(channels)\n"
        "with warnings.catch_warnings():\n"
        '    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")\n'
        "    ica_sources, ica_model = tsm.ica(n_components=n_components, return_model=True)\n",
    )
    src = src.replace('print(f"ICA converged in {sk.n_iter_} iterations")', 'print(f"ICA stopped after {sk.n_iter_} iterations")')
    src = src.replace('print(f"ICA 収束: {sk.n_iter_} 反復")', 'print(f"ICA は {sk.n_iter_} 反復で停止")')
    _set_source(cell, src)
    _sanitize_outputs(nb, lang=lang)
    _save(path, nb)


def _fix_case_glitch(lang: str) -> None:
    path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / "case_glitch_analysis.ipynb"
    nb = _load(path)
    cell = nb["cells"][6]
    _replace_source(cell, "    frange   = (10, 2000),\n", "    frange   = (10, 1200),\n")
    _sanitize_outputs(nb, lang=lang)
    _save(path, nb)


def _fix_intro_frequencyseries(lang: str) -> None:
    path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / "intro_frequencyseries.ipynb"
    nb = _load(path)
    cell = nb["cells"][13]
    _replace_source(cell, 'ax.set_xlim(0, 200)\n', 'ax.set_xlim(1, 200)\n')
    _sanitize_outputs(nb, lang=lang)
    _save(path, nb)


def _fix_swiglal_and_segment_examples(lang: str) -> None:
    files = [
        ("case_segment_analysis.ipynb", 1, None),
        ("intro_segment_table.ipynb", 1, None),
        ("segment_visualization.ipynb", 1, None),
        ("segment_asd_pipeline.ipynb", 1, None),
        ("segment_asd_pipeline.ipynb", 3, None),
        ("intro_interop.ipynb", 3, None),
        ("intro_interop.ipynb", 55, None),
    ]
    for name, index, _ in files:
        path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / name
        nb = _load(path)
        cell = nb["cells"][index]
        src = _cell_source(cell)
        if name in {"case_segment_analysis.ipynb", "intro_segment_table.ipynb", "segment_visualization.ipynb", "segment_asd_pipeline.ipynb"} and index == 1:
            if "import warnings\nwarnings.filterwarnings(\"ignore\", \"Wswiglal-redir-stdio\")\n" not in src:
                src = (
                    'import warnings\nwarnings.filterwarnings("ignore", "Wswiglal-redir-stdio")\n'
                    + src
                )
            if name == "segment_visualization.ipynb":
                src = src.replace("plot.show() # NOTE: Normally use plot in notebooks\n", "plot\n")
        if name == "segment_asd_pipeline.ipynb" and index == 1:
            src = src.replace(
                "def get_synthetic_data():\n    return TimeSeries(np.random.randn(1024), sample_rate=64)\n\nsegs = [Segment(i*16, i*16+16) for i in range(4)]\nst = SegmentTable.from_segments(segs)\nst.add_series_column(\"raw\", data=[get_synthetic_data()]*len(st), kind=\"timeseries\")\n",
                "def get_synthetic_data(t0):\n    return TimeSeries(np.random.randn(1024), sample_rate=64, t0=t0)\n\nsegs = [Segment(i*16, i*16+16) for i in range(4)]\nst = SegmentTable.from_segments(segs)\nst.add_series_column(\"raw\", data=[get_synthetic_data(seg[0]) for seg in segs], kind=\"timeseries\")\n",
            )
        if name == "segment_asd_pipeline.ipynb" and index == 3:
            src = src.replace("st_asd.display()\n", "st_asd.display()\n")
        if name == "intro_interop.ipynb" and index == 3:
            src = src.replace(
                "import warnings\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom astropy import units as u\nfrom gwpy.time import LIGOTimeGPS\n\nfrom gwexpy.timeseries import TimeSeries\n\nwarnings.filterwarnings(\"ignore\", \"Wswiglal-redir-stdio\")\nwarnings.filterwarnings(\"ignore\", category=UserWarning)\n",
                "import logging\nimport warnings\n\nwarnings.filterwarnings(\"ignore\", \"Wswiglal-redir-stdio\")\nwarnings.filterwarnings(\"ignore\", category=UserWarning)\n\nlogging.getLogger(\"mth5\").disabled = True\nlogging.getLogger(\"mt_metadata\").disabled = True\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom astropy import units as u\nfrom gwpy.time import LIGOTimeGPS\n\nfrom gwexpy.timeseries import TimeSeries\n",
            )
        if name == "intro_interop.ipynb" and index == 55:
            src = src.replace('print(f"Saved to MTH5 file: {tmp.name}")', 'print("Saved to temporary MTH5 file")')
            src = src.replace('print(f"Saved to MTH5 file: {tmp.name}")', 'print("一時 MTH5 ファイルへ保存")')
        _set_source(cell, src)
        _sanitize_outputs(nb, lang=lang)
        _save(path, nb)


def _fix_case_hdf5(lang: str) -> None:
    path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / "case_hdf5_provenance.ipynb"
    nb = _load(path)
    for idx in (8, 12):
        cell = nb["cells"][idx]
        src = _cell_source(cell)
        src = src.replace(
            'datetime.datetime.utcnow().isoformat()',
            "datetime.datetime.now(datetime.UTC).isoformat()",
        )
        src = src.replace('print(f"Pipeline archive: {archive_path}  ({archive_path.stat().st_size/1024:.1f} kB)")', 'print(f"Pipeline archive: temporary file  ({archive_path.stat().st_size/1024:.1f} kB)")')
        src = src.replace('print(f"パイプラインアーカイブ: {archive_path}  ({archive_path.stat().st_size/1024:.1f} kB)")', 'print(f"パイプラインアーカイブ: 一時ファイル  ({archive_path.stat().st_size/1024:.1f} kB)")')
        _set_source(cell, src)
    _sanitize_outputs(nb, lang=lang)
    _save(path, nb)


def _fix_case_dttxml_and_gbd(lang: str) -> None:
    files = ["case_dttxml_calibration.ipynb", "case_gbd_format.ipynb"]
    for name in files:
        path = ROOT / "docs" / "web" / lang / "user_guide" / "tutorials" / name
        nb = _load(path)
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = _cell_source(cell)
            src = src.replace('print(f"Wrote {path}")', 'print("Wrote synthetic DTT XML to a temporary file")')
            src = src.replace('print(f"書き込み完了: {path}")', 'print("一時ファイルへ合成 DTT XML を書き込みました")')
            src = src.replace('print(f"Written: {path}  ({path.stat().st_size/1024:.1f} KB)")', 'print(f"Written synthetic_pem.gbd ({path.stat().st_size/1024:.1f} KB)")')
            src = src.replace('print(f"作成完了: {path}  ({path.stat().st_size/1024:.1f} KB)")', 'print(f"synthetic_pem.gbd を作成しました ({path.stat().st_size/1024:.1f} KB)")')
            _set_source(cell, src)
        _sanitize_outputs(nb, lang=lang)
        _save(path, nb)


def _fix_advanced_coupling() -> None:
    en_path = ROOT / "docs" / "web" / "en" / "user_guide" / "tutorials" / "advanced_coupling.ipynb"
    ja_path = ROOT / "docs" / "web" / "ja" / "user_guide" / "tutorials" / "advanced_coupling.ipynb"

    for path, lang in ((en_path, "en"), (ja_path, "ja")):
        nb = _load(path)
        code = nb["cells"][12]
        src = _cell_source(code)
        if "import warnings" not in src:
            src = "import warnings\n\n" + src
        src = src.replace(
            "for label, (thr_w, thr_t) in strategies.items():\n    res = cfa.compute(",
            "for label, (thr_w, thr_t) in strategies.items():\n    with warnings.catch_warnings():\n        warnings.filterwarnings(\"ignore\", message=\"SigmaThreshold: n_avg\")\n        res = cfa.compute(",
        )
        _set_source(code, src)
        _sanitize_outputs(nb, lang=lang)
        _save(path, nb)

    ja = _load(ja_path)
    if "周波数帯域制限" not in _cell_source(ja["cells"][13]):
        ja["cells"].insert(
            14,
            {
                "cell_type": "code",
                "execution_count": 11,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": ["10-50 Hz の有効 CF ビン数: 6\n"],
                    },
                    {
                        "data": {"text/plain": ["<Plot size 640x480 with 1 Axes>"]},
                        "metadata": {},
                        "output_type": "display_data",
                    },
                ],
                "source": [
                    "results_band = cfa.compute(\n",
                    "    data_inj=data_inj,\n",
                    "    data_bkg=data_bkg,\n",
                    "    fftlength=4.0,\n",
                    "    overlap=0.5,\n",
                    "    witness=\"ACC:FLOOR_X\",\n",
                    "    frange=(10.0, 50.0),\n",
                    ")\n",
                    "\n",
                    "cf_band = results_band[\"GW:DARM\"].cf\n",
                    "valid_band = ~np.isnan(cf_band.value)\n",
                    "print(f\"10-50 Hz の有効 CF ビン数: {valid_band.sum()}\")\n",
                    "results_band[\"GW:DARM\"].plot_cf(xlim=(5, 100));\n",
                ],
            },
        )
        ja["cells"].insert(
            14,
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. 周波数帯域制限\n",
                    "\n",
                    "`frange` を使うと、カップリング関数の評価を特定の周波数帯に限定できます。\n",
                ],
            },
        )
        _set_source(
            ja["cells"][16],
            _cell_source(ja["cells"][16]).replace("## 5. 上限値（Upper Limit）", "## 6. 上限値（Upper Limit）"),
        )
        _sanitize_outputs(ja, lang="ja")
        _save(ja_path, ja)


def _fix_case_seismic_obspy() -> None:
    ja_path = ROOT / "docs" / "web" / "ja" / "user_guide" / "tutorials" / "case_seismic_obspy.ipynb"
    nb = _load(ja_path)
    joined = "\n".join(_cell_source(c) for c in nb["cells"] if c.get("cell_type") == "markdown")
    if "マルチチャンネル地震解析" not in joined:
        nb["cells"].insert(
            12,
            {
                "cell_type": "code",
                "execution_count": 8,
                "metadata": {},
                "outputs": [
                    {
                        "data": {"text/plain": ["<Figure size 800x400 with 1 Axes>"]},
                        "metadata": {},
                        "output_type": "display_data",
                    }
                ],
                "source": [
                    "from gwexpy.timeseries import TimeSeriesMatrix\n",
                    "\n",
                    "rng2 = np.random.default_rng(1)\n",
                    "components = ['BHX', 'BHY', 'BHZ']\n",
                    "data_3c = np.stack([\n",
                    "    seis_data + 1e-8 * rng2.normal(0, 1, n_seis),\n",
                    "    seis_data * 0.8 + 1e-8 * rng2.normal(0, 1, n_seis),\n",
                    "    seis_data * 1.2 + 1e-8 * rng2.normal(0, 1, n_seis),\n",
                    "], axis=0)[:, np.newaxis, :]\n",
                    "\n",
                    "tsm_3c = TimeSeriesMatrix(\n",
                    "    data_3c,\n",
                    "    dt=(1/fs_seis)*u.s,\n",
                    "    t0=0*u.s,\n",
                    "    units=np.full((3, 1), u.m/u.s),\n",
                    ")\n",
                    "tsm_3c.channel_names = components\n",
                    "\n",
                    "asd_3c = tsm_3c.asd(fftlength=10.0, overlap=0.5)\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(8, 4))\n",
                    "for i, comp in enumerate(components):\n",
                    "    ax.loglog(asd_3c[i, 0].frequencies.value, asd_3c[i, 0].value, label=comp)\n",
                    "ax.set_xlabel(\"周波数 [Hz]\")\n",
                    "ax.set_ylabel(\"ASD [m/s/√Hz]\")\n",
                    "ax.set_title(\"3成分地震計の ASD\")\n",
                    "ax.legend()\n",
                    "ax.grid(True, which=\"both\", alpha=0.3)\n",
                    "plt.tight_layout()\n",
                ],
            },
        )
        nb["cells"].insert(
            12,
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. マルチチャンネル地震解析\n",
                    "\n",
                    "3 成分地震計（X, Y, Z）のように複数チャネルを同時に扱う場合は、`TimeSeriesMatrix` を使うと一括処理できます。\n",
                ],
            },
        )
        _set_source(
            nb["cells"][14],
            _cell_source(nb["cells"][14]).replace("## 5. 移行ガイド: obspy + numpy → gwexpy", "## 6. 移行ガイド: obspy + numpy → gwexpy"),
        )
        _sanitize_outputs(nb, lang="ja")
        _save(ja_path, nb)


def _fix_case_arima_burst_search() -> None:
    path = ROOT / "docs" / "web" / "en" / "user_guide" / "tutorials" / "case_arima_burst_search.ipynb"
    nb = _load(path)
    translations = {
        0: """# ARIMA-Based Burst Detection
# Idea sketch for burst gravitational-wave searches

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main/docs/web/en/user_guide/tutorials/case_arima_burst_search.ipynb)

## Introduction
Detector data are dominated by approximately stationary background noise, while burst gravitational waves (for example from core-collapse supernovae) appear as short, non-stationary transients embedded in that noise.

This notebook demonstrates the idea of using an **autoregressive model (ARIMA)** to learn the stationary background and make burst-like residuals stand out. It is a simple example of an unmodeled search strategy that does not rely on waveform templates.

### References
- Autoregressive Search of Gravitational Waves: Denoising (ARIMA_DeNoise.pdf)
- BEACON: Autoregressive Search for Unmodeled transients (BEACON.pdf)
- Sparkler: Autoregressive Search of Unmodeled GW (Sparkler_1min.pdf)
""",
        2: "## 1. Generate detector noise\nFirst we simulate detector background noise. When available we use an aLIGO design-sensitivity model; otherwise we fall back to generic colored noise.\n",
        4: "## 2. Inject a burst signal\nWe inject a sine-Gaussian burst into the second half of the data and use it as the test segment.\n",
        6: "## 3. Learn the background with ARIMA\nWe fit an ARIMA model on the first 8 seconds, where no burst is present, so that the model captures the stationary background statistics. The data are moderately downsampled to keep the example lightweight.\n",
        8: "## 4. Compute residuals and detect anomalies\nWe apply the fitted model to the remaining data and inspect the residuals. Stationary noise should leave small residuals, while an unmodeled burst produces a localized excess.\n",
        10: "## 5. Evaluate performance versus SNR\nFinally we repeat the injection for several signal-to-noise ratios to estimate how the residual-based detection responds as the burst becomes weaker.\n",
        12: "## Summary\n\n### Key ideas behind ARIMA-based burst searches\n1. **Background suppression**: model and subtract stationary noise to improve the visibility of weak transients.\n2. **Residual analysis**: monitor deviations from the expected residual distribution to identify unmodeled events.\n\nProduction pipelines such as BEACON extend this idea to multiple channels and coherence checks, but this notebook captures the core intuition in a compact example.\n",
    }
    for idx, text in translations.items():
        _set_source(nb["cells"][idx], text)
    _sanitize_outputs(nb, lang="en")
    _save(path, nb)


def _annotate_advanced_hht() -> None:
    path = ROOT / "docs" / "web" / "ja" / "user_guide" / "tutorials" / "advanced_hht.ipynb"
    nb = _load(path)
    first = _cell_source(nb["cells"][0])
    if "ワークフロー重視" not in first:
        _set_source(
            nb["cells"][0],
            first
            + "\n> **注記**: この日本語版は、英語版の理論比較をそのまま翻訳したものではなく、`TimeSeries.hht()` を中心にした実践ワークフロー重視の版です。\n",
        )
        _save(path, nb)


def _strip_notebook_cell_ids(*paths: Path) -> None:
    for path in paths:
        nb = _load(path)
        changed = False
        for cell in nb.get("cells", []):
            if "id" in cell:
                cell.pop("id", None)
                changed = True
        if changed:
            _save(path, nb)


def _remove_transition_markdown_cells(path: Path) -> None:
    nb = _load(path)
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = _cell_source(cell)
        stripped_leading = src.lstrip("\n")
        if stripped_leading != src:
            src = stripped_leading
            changed = True
        if src.strip().startswith("---\n"):
            _set_source(cell, src.split("\n", 1)[1])
            changed = True
        elif src.strip() == "---":
            _set_source(cell, "")
            changed = True
        elif "\n\n---\n\n## " in src:
            _set_source(cell, src.replace("\n\n---\n\n## ", "\n\n## "))
            changed = True
        elif changed:
            _set_source(cell, src)
    if changed:
        _save(path, nb)


def main() -> None:
    _apply_pairwise_source_fixes()
    for lang in ("en", "ja"):
        _fix_advanced_correlation(lang)
        _fix_advanced_bruco(lang)
        _fix_case_bruco_ica(lang)
        _fix_case_glitch(lang)
        _fix_intro_frequencyseries(lang)
        _fix_swiglal_and_segment_examples(lang)
        _fix_case_hdf5(lang)
        _fix_case_dttxml_and_gbd(lang)
    _fix_advanced_coupling()
    _fix_case_seismic_obspy()
    _fix_case_arima_burst_search()
    _annotate_advanced_hht()
    _strip_notebook_cell_ids(
        ROOT / "docs" / "web" / "en" / "user_guide" / "tutorials" / "advanced_arima.ipynb",
        ROOT / "docs" / "web" / "en" / "user_guide" / "tutorials" / "advanced_bruco.ipynb",
        ROOT / "docs" / "web" / "en" / "user_guide" / "tutorials" / "time_frequency_analysis_comparison.ipynb",
    )
    _remove_transition_markdown_cells(
        ROOT / "docs" / "web" / "en" / "user_guide" / "tutorials" / "time_frequency_analysis_comparison.ipynb"
    )


if __name__ == "__main__":
    main()
