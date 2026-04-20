"""Extend advanced_fitting.ipynb (EN + JA) with Lorentzian / Voigt spectral-line
fitting sections.

Usage:
    python scripts/extend_fitting_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# New cells — English version
# ---------------------------------------------------------------------------
EN_NEW_CELLS = [
    # ── Section header ──────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "source": (
            "## 7. Spectral Line Fitting: Lorentzian and Voigt Profiles\n"
            "\n"
            "In gravitational-wave detector commissioning, narrowband resonances (mechanical\n"
            "modes, violin modes, line injections) appear as sharp peaks in ASD spectra.  \n"
            "Three profiles are commonly used:\n"
            "\n"
            "| Model | When to use |\n"
            "|-------|-------------|\n"
            "| **Gaussian** | Doppler- or pressure-broadened lines; truly symmetric noise peaks |\n"
            "| **Lorentzian** | Resonances with a single loss mechanism (Q-limited linewidth) |\n"
            "| **Voigt** | Composite broadening (Gaussian instrument resolution + Lorentzian damping) |\n"
            "\n"
            "All three are available in `gwexpy.fitting.models` and can be referenced by string."
        ),
    },
    # ── Lorentzian fit ──────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "source": (
            "### 7a. Lorentzian (HWHM parameterisation)\n"
            "\n"
            "$$f(x) = \\frac{A\\,\\gamma^2}{(x - x_0)^2 + \\gamma^2}$$\n"
            "\n"
            "- `A` : peak amplitude\n"
            "- `x0`: centre frequency\n"
            "- `gamma`: half-width at half-maximum (HWHM)"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "from gwexpy.fitting.models import lorentzian, voigt\n"
            "\n"
            "# ── Synthetic ASD with a Lorentzian resonance peak ──────────────\n"
            "np.random.seed(0)\n"
            "freqs = np.linspace(90, 130, 400)   # Hz, around a 110 Hz mode\n"
            "TRUE_A, TRUE_X0, TRUE_GAMMA = 50.0, 110.0, 0.8\n"
            "\n"
            "y_lorenz = lorentzian(freqs, TRUE_A, TRUE_X0, TRUE_GAMMA)\n"
            "noise = np.random.normal(0, 0.5, len(freqs))\n"
            "y_data = y_lorenz + noise + 1.0   # +1 flat background\n"
            "\n"
            "fs_lorenz = FrequencySeries(y_data, frequencies=freqs,\n"
            "                           name='ASD with Lorentzian peak', unit='1/rtHz')\n"
            "\n"
            "# ── Fit ─────────────────────────────────────────────────────────\n"
            "res_lorenz = fs_lorenz.fit(\n"
            "    'lorentzian',\n"
            "    p0={'A': 30.0, 'x0': 112.0, 'gamma': 1.5},\n"
            "    sigma=0.5,\n"
            ")\n"
            "\n"
            "print(f\"True  A={TRUE_A}, x0={TRUE_X0}, gamma={TRUE_GAMMA}\")\n"
            "print(f\"Fit   A={res_lorenz.params['A']:.2f} ± {res_lorenz.errors['A']:.2f}\")\n"
            "print(f\"      x0={res_lorenz.params['x0']:.3f} ± {res_lorenz.errors['x0']:.3f}\")\n"
            "print(f\"      gamma={res_lorenz.params['gamma']:.3f} ± {res_lorenz.errors['gamma']:.3f}\")\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "res_lorenz.plot(ax=ax, label='Lorentzian fit')\n"
            "ax.set_xlabel('Frequency [Hz]')\n"
            "ax.set_ylabel('ASD [1/√Hz]')\n"
            "ax.set_title('Lorentzian Spectral Line Fit')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    },
    # ── Lorentzian Q-factor ─────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "source": (
            "### 7b. Lorentzian Q-factor parameterisation\n"
            "\n"
            "For resonators it is often more natural to express the linewidth via the\n"
            "quality factor $Q = x_0 / (2\\gamma)$:\n"
            "\n"
            "$$f(x) = \\frac{A\\,\\gamma^2}{(x - x_0)^2 + \\gamma^2}, \\quad \\gamma = \\frac{x_0}{2Q}$$\n"
            "\n"
            "Use `'lorentzian_q'` to fit directly in terms of `(A, x0, Q)`."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "TRUE_Q = TRUE_X0 / (2 * TRUE_GAMMA)   # ~68.75\n"
            "\n"
            "res_q = fs_lorenz.fit(\n"
            "    'lorentzian_q',\n"
            "    p0={'A': 30.0, 'x0': 112.0, 'Q': 50.0},\n"
            "    sigma=0.5,\n"
            ")\n"
            "\n"
            "print(f\"True  Q = {TRUE_Q:.2f}\")\n"
            "print(f\"Fit   Q = {res_q.params['Q']:.2f} ± {res_q.errors['Q']:.2f}\")\n"
            "print(f\"      x0 = {res_q.params['x0']:.3f} ± {res_q.errors['x0']:.3f} Hz\")\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "res_q.plot(ax=ax, label=f\"Q = {res_q.params['Q']:.1f}\")\n"
            "ax.set_xlabel('Frequency [Hz]')\n"
            "ax.set_ylabel('ASD [1/√Hz]')\n"
            "ax.set_title('Lorentzian Q-factor Fit')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    },
    # ── Voigt profile ────────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "source": (
            "### 7c. Voigt Profile\n"
            "\n"
            "The Voigt profile is the convolution of a Gaussian (width $\\sigma$) and a Lorentzian\n"
            "(HWHM $\\gamma$).  It is computed via the **Faddeeva function** for efficiency.\n"
            "\n"
            "Use it when the line shows both instrument-resolution broadening (Gaussian) and\n"
            "intrinsic damping (Lorentzian)."
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ── Synthetic data with mixed broadening ─────────────────────────\n"
            "TRUE_SIGMA, TRUE_GAMMA_V = 0.5, 0.4\n"
            "y_voigt = voigt(freqs, A=40.0, x0=110.0, sigma=TRUE_SIGMA, gamma=TRUE_GAMMA_V)\n"
            "y_data_v = y_voigt + np.random.normal(0, 0.4, len(freqs)) + 0.5\n"
            "\n"
            "fs_voigt = FrequencySeries(y_data_v, frequencies=freqs,\n"
            "                          name='ASD with Voigt peak', unit='1/rtHz')\n"
            "\n"
            "# Fit Lorentzian (wrong model) for comparison\n"
            "res_l = fs_voigt.fit('lorentzian', p0={'A': 30.0, 'x0': 111.0, 'gamma': 0.8},\n"
            "                     sigma=0.4)\n"
            "\n"
            "# Fit Voigt (correct model)\n"
            "res_v = fs_voigt.fit('voigt',\n"
            "                     p0={'A': 30.0, 'x0': 111.0, 'sigma': 0.6, 'gamma': 0.5},\n"
            "                     sigma=0.4)\n"
            "\n"
            "print(f\"Lorentzian  chi2/ndof = {res_l.chi2:.1f}/{res_l.ndof}\")\n"
            "print(f\"Voigt       chi2/ndof = {res_v.chi2:.1f}/{res_v.ndof}\")\n"
            "print(f\"Recovered sigma={res_v.params['sigma']:.3f} (true {TRUE_SIGMA}),\"\n"
            "      f\" gamma={res_v.params['gamma']:.3f} (true {TRUE_GAMMA_V})\")\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "res_l.plot(ax=axes[0], label='Lorentzian fit')\n"
            "axes[0].set_title(f\"Lorentzian  χ²/ndof = {res_l.chi2:.0f}/{res_l.ndof}\")\n"
            "axes[0].set_xlabel('Frequency [Hz]')\n"
            "axes[0].legend()\n"
            "\n"
            "res_v.plot(ax=axes[1], label='Voigt fit')\n"
            "axes[1].set_title(f\"Voigt  χ²/ndof = {res_v.chi2:.0f}/{res_v.ndof}\")\n"
            "axes[1].set_xlabel('Frequency [Hz]')\n"
            "axes[1].legend()\n"
            "\n"
            "plt.suptitle('Model Comparison: Lorentzian vs Voigt', fontsize=13)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    },
]

# ---------------------------------------------------------------------------
# New cells — Japanese version
# ---------------------------------------------------------------------------
JA_NEW_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "## 7. スペクトル線フィッティング：ローレンツ関数と Voigt プロファイル\n"
            "\n"
            "重力波検出器のコミッショニングでは、機械共振（バイオリンモード、ペンジュラムモード等）や\n"
            "注入ライン信号が ASD スペクトル上に鋭いピークとして現れます。  \n"
            "以下の 3 つのプロファイルが広く使われます：\n"
            "\n"
            "| モデル | 使いどき |\n"
            "|--------|----------|\n"
            "| **Gaussian** | ドップラー・圧力広がり；対称なノイズピーク |\n"
            "| **Lorentzian** | 単一損失機構による共振（Q 値で幅が決まる） |\n"
            "| **Voigt** | ガウス（計器分解能）＋ローレンツ（減衰）の複合広がり |\n"
            "\n"
            "いずれも `gwexpy.fitting.models` に実装済みで、文字列名での指定が可能です。"
        ),
    },
    {
        "cell_type": "markdown",
        "source": (
            "### 7a. ローレンツ関数（HWHM パラメータ化）\n"
            "\n"
            "$$f(x) = \\frac{A\\,\\gamma^2}{(x - x_0)^2 + \\gamma^2}$$\n"
            "\n"
            "- `A` : ピーク振幅\n"
            "- `x0`: 中心周波数\n"
            "- `gamma`: 半値半幅 (HWHM)"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "from gwexpy.fitting.models import lorentzian, voigt\n"
            "\n"
            "# ── ローレンツ共振ピークを含む合成 ASD データ ──────────────────\n"
            "np.random.seed(0)\n"
            "freqs = np.linspace(90, 130, 400)   # Hz（110 Hz モード付近）\n"
            "TRUE_A, TRUE_X0, TRUE_GAMMA = 50.0, 110.0, 0.8\n"
            "\n"
            "y_lorenz = lorentzian(freqs, TRUE_A, TRUE_X0, TRUE_GAMMA)\n"
            "noise = np.random.normal(0, 0.5, len(freqs))\n"
            "y_data = y_lorenz + noise + 1.0   # +1 フラット背景\n"
            "\n"
            "fs_lorenz = FrequencySeries(y_data, frequencies=freqs,\n"
            "                           name='ASD with Lorentzian peak', unit='1/rtHz')\n"
            "\n"
            "# ── フィット ─────────────────────────────────────────────────────\n"
            "res_lorenz = fs_lorenz.fit(\n"
            "    'lorentzian',\n"
            "    p0={'A': 30.0, 'x0': 112.0, 'gamma': 1.5},\n"
            "    sigma=0.5,\n"
            ")\n"
            "\n"
            "print(f\"真値  A={TRUE_A}, x0={TRUE_X0}, gamma={TRUE_GAMMA}\")\n"
            "print(f\"推定  A={res_lorenz.params['A']:.2f} ± {res_lorenz.errors['A']:.2f}\")\n"
            "print(f\"      x0={res_lorenz.params['x0']:.3f} ± {res_lorenz.errors['x0']:.3f}\")\n"
            "print(f\"      gamma={res_lorenz.params['gamma']:.3f} ± {res_lorenz.errors['gamma']:.3f}\")\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "res_lorenz.plot(ax=ax, label='ローレンツフィット')\n"
            "ax.set_xlabel('周波数 [Hz]')\n"
            "ax.set_ylabel('ASD [1/√Hz]')\n"
            "ax.set_title('ローレンツ スペクトル線フィット')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    },
    {
        "cell_type": "markdown",
        "source": (
            "### 7b. ローレンツ Q 値パラメータ化\n"
            "\n"
            "共振器では線幅を品質係数 $Q = x_0 / (2\\gamma)$ で表すと直感的です：\n"
            "\n"
            "$$\\gamma = \\frac{x_0}{2Q}$$\n"
            "\n"
            "`'lorentzian_q'` を使うと `(A, x0, Q)` を直接フィットできます。"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "TRUE_Q = TRUE_X0 / (2 * TRUE_GAMMA)   # ~68.75\n"
            "\n"
            "res_q = fs_lorenz.fit(\n"
            "    'lorentzian_q',\n"
            "    p0={'A': 30.0, 'x0': 112.0, 'Q': 50.0},\n"
            "    sigma=0.5,\n"
            ")\n"
            "\n"
            "print(f\"真値  Q = {TRUE_Q:.2f}\")\n"
            "print(f\"推定  Q = {res_q.params['Q']:.2f} ± {res_q.errors['Q']:.2f}\")\n"
            "print(f\"      x0 = {res_q.params['x0']:.3f} ± {res_q.errors['x0']:.3f} Hz\")\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "res_q.plot(ax=ax, label=f\"Q = {res_q.params['Q']:.1f}\")\n"
            "ax.set_xlabel('周波数 [Hz]')\n"
            "ax.set_ylabel('ASD [1/√Hz]')\n"
            "ax.set_title('ローレンツ Q 値フィット')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    },
    {
        "cell_type": "markdown",
        "source": (
            "### 7c. Voigt プロファイル\n"
            "\n"
            "Voigt プロファイルはガウス関数（幅 $\\sigma$）とローレンツ関数（HWHM $\\gamma$）の畳み込みで、\n"
            "Faddeeva 関数を使って高速計算されます。  \n"
            "計器分解能（ガウス）と内部減衰（ローレンツ）が共存するラインに使用します。"
        ),
    },
    {
        "cell_type": "code",
        "source": (
            "# ── 複合広がりを持つ合成データ ──────────────────────────────────\n"
            "TRUE_SIGMA, TRUE_GAMMA_V = 0.5, 0.4\n"
            "y_voigt = voigt(freqs, A=40.0, x0=110.0, sigma=TRUE_SIGMA, gamma=TRUE_GAMMA_V)\n"
            "y_data_v = y_voigt + np.random.normal(0, 0.4, len(freqs)) + 0.5\n"
            "\n"
            "fs_voigt = FrequencySeries(y_data_v, frequencies=freqs,\n"
            "                          name='ASD with Voigt peak', unit='1/rtHz')\n"
            "\n"
            "# ローレンツ（誤ったモデル）で比較\n"
            "res_l = fs_voigt.fit('lorentzian', p0={'A': 30.0, 'x0': 111.0, 'gamma': 0.8},\n"
            "                     sigma=0.4)\n"
            "\n"
            "# Voigt（正しいモデル）でフィット\n"
            "res_v = fs_voigt.fit('voigt',\n"
            "                     p0={'A': 30.0, 'x0': 111.0, 'sigma': 0.6, 'gamma': 0.5},\n"
            "                     sigma=0.4)\n"
            "\n"
            "print(f\"ローレンツ  chi2/ndof = {res_l.chi2:.1f}/{res_l.ndof}\")\n"
            "print(f\"Voigt       chi2/ndof = {res_v.chi2:.1f}/{res_v.ndof}\")\n"
            "print(f\"推定  sigma={res_v.params['sigma']:.3f}（真値 {TRUE_SIGMA}）,\"\n"
            "      f\" gamma={res_v.params['gamma']:.3f}（真値 {TRUE_GAMMA_V}）\")\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "res_l.plot(ax=axes[0], label='ローレンツフィット')\n"
            "axes[0].set_title(f\"ローレンツ  χ²/ndof = {res_l.chi2:.0f}/{res_l.ndof}\")\n"
            "axes[0].set_xlabel('周波数 [Hz]')\n"
            "axes[0].legend()\n"
            "\n"
            "res_v.plot(ax=axes[1], label='Voigt フィット')\n"
            "axes[1].set_title(f\"Voigt  χ²/ndof = {res_v.chi2:.0f}/{res_v.ndof}\")\n"
            "axes[1].set_xlabel('周波数 [Hz]')\n"
            "axes[1].legend()\n"
            "\n"
            "plt.suptitle('モデル比較: ローレンツ vs Voigt', fontsize=13)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper to build a notebook cell dict
# ---------------------------------------------------------------------------
def _make_cell(cell_type: str, source: str) -> dict:
    cell: dict = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def extend_notebook(nb_path: Path, new_cells_spec: list[dict]) -> None:
    nb = json.loads(nb_path.read_text())
    for spec in new_cells_spec:
        nb["cells"].append(_make_cell(spec["cell_type"], spec["source"]))
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"Extended: {nb_path}  (+{len(new_cells_spec)} cells)")


if __name__ == "__main__":
    en_path = REPO / "docs/web/en/user_guide/tutorials/advanced_fitting.ipynb"
    ja_path = REPO / "docs/web/ja/user_guide/tutorials/advanced_fitting.ipynb"

    extend_notebook(en_path, EN_NEW_CELLS)
    extend_notebook(ja_path, JA_NEW_CELLS)

    print("Done — Lorentzian/Voigt sections added to both EN and JA fitting notebooks.")
