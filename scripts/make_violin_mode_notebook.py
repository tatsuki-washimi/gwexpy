"""
Generate case_violin_mode.ipynb (EN + JA).

Covers:
  - Violin mode physics (f_n, Q value, FWHM, ring-down time)
  - Synthetic ASD generation with lorentzian_q
  - Single-mode Lorentzian Q fit using FrequencySeries.fit()
  - Multi-mode batch processing
  - Time-variation tracking via spectrogram + argmax

Usage:
    python scripts/make_violin_mode_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).parent.parent
COLAB_BASE = (
    "https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main"
    "/docs/web/en/user_guide/tutorials"
)

# ============================================================
# Shared code cells
# ============================================================

SETUP_CODE = """\
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks as sp_find_peaks

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.fitting.models import lorentzian_q
from gwexpy.timeseries import TimeSeries

plt.rcParams["figure.figsize"] = (10, 5)
"""

MOCK_ASD_CODE = """\
rng = np.random.default_rng(42)

# ── Frequency axis: 100–450 Hz, df = 0.01 Hz ─────────────────────────────────
df    = 0.01   # Hz resolution
freqs = np.arange(100, 450, df)

# ── Flat background ASD ───────────────────────────────────────────────────────
BACKGROUND = 3e-22   # strain / sqrt(Hz)
background = np.ones_like(freqs) * BACKGROUND

# ── Violin modes (using lorentzian_q for both generation and fitting) ─────────
MODES = {
    "1st violin": {"f0": 170.0, "A": 5e-21, "Q": 1.0e4},
    "2nd violin": {"f0": 340.0, "A": 2e-21, "Q": 8.0e3},
}

asd_data = background.copy()
for cfg in MODES.values():
    asd_data += lorentzian_q(freqs, A=cfg["A"], x0=cfg["f0"], Q=cfg["Q"])

# Add small Gaussian noise
asd_data += rng.normal(0, BACKGROUND * 0.05, len(freqs))
asd_data  = np.clip(asd_data, 0, None)

asd = FrequencySeries(asd_data, frequencies=freqs,
                      unit="strain/rtHz", name="Synthetic DARM ASD")

print("Frequency range:", freqs[0], "–", freqs[-1], "Hz")
print("Modes inserted:")
for name, cfg in MODES.items():
    FWHM_true = cfg["f0"] / cfg["Q"]
    print(f"  {name}: f0={cfg['f0']} Hz,  Q={cfg['Q']:.0e},  FWHM={FWHM_true*1000:.1f} mHz")

fig, ax = plt.subplots(figsize=(12, 4))
asd.plot(ax=ax)
ax.set_yscale("log")
ax.set_title("Synthetic ASD with violin modes")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [strain/\\u221aHz]")
plt.tight_layout()
plt.show()
"""

SINGLE_FIT_CODE = """\
# ── Crop around 1st violin mode ───────────────────────────────────────────────
BAND   = 0.5    # ± Hz around the expected mode
f_nom  = 170.0  # nominal frequency
asd_1st = asd.crop(f_nom - BAND, f_nom + BAND)

# ── Lorentzian Q fit ──────────────────────────────────────────────────────────
result_1 = asd_1st.fit(
    "lorentzian_q",
    p0={"A": 3e-21, "x0": 170.5, "Q": 5e3},
    sigma=BACKGROUND * 0.1,
)

print("=== 1st Violin Mode Fit ===")
print(f"  f0    = {result_1.params['x0']:.4f} \\u00b1 {result_1.errors['x0']:.4f}  Hz")
print(f"  Q     = {result_1.params['Q']:.2e} \\u00b1 {result_1.errors['Q']:.2e}")
print(f"  A     = {result_1.params['A']:.2e} \\u00b1 {result_1.errors['A']:.2e}")
print(f"  chi2/ndof = {result_1.chi2:.1f} / {result_1.ndof}")

fig, ax = plt.subplots(figsize=(10, 4))
result_1.plot(ax=ax, label="Lorentzian Q fit")
ax.set_yscale("log")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [strain/\\u221aHz]")
ax.set_title("1st Violin Mode — Lorentzian Q Fit")
ax.legend()
plt.tight_layout()
plt.show()
"""

PARAMS_CODE = """\
# ── Extract physical parameters ───────────────────────────────────────────────
f0 = float(result_1.params["x0"])
Q  = float(result_1.params["Q"])
A  = float(result_1.params["A"])

FWHM  = f0 / Q              # Full Width at Half Maximum  [Hz]
tau   = Q / (np.pi * f0)    # Ring-down time constant  [s]
gamma = FWHM / 2             # HWHM  [Hz]

print(f"Center frequency  f0   = {f0:.4f} Hz")
print(f"Quality factor    Q    = {Q:.3e}")
print(f"FWHM (linewidth)       = {FWHM * 1e3:.3f} mHz")
print(f"Ring-down time    tau  = {tau:.2f} s  ({tau / 60:.2f} min)")

# ── Reconstruct model over a wider band for visualisation ────────────────────
asd_wide = asd.crop(165, 175)
f_plot   = asd_wide.frequencies.value
model    = lorentzian_q(f_plot, A=A, x0=f0, Q=Q) + BACKGROUND

fig, ax = plt.subplots(figsize=(10, 4))
asd_wide.plot(ax=ax, label="ASD data", alpha=0.6)
ax.semilogy(f_plot, model, "r-", lw=2,
            label=f"Fit: Q={Q:.1e}, FWHM={FWHM*1e3:.1f} mHz")
ax.set_yscale("log")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [strain/\\u221aHz]")
ax.set_title("1st Violin Mode — Fit Parameters")
ax.legend()
plt.tight_layout()
plt.show()
"""

BATCH_CODE = """\
# ── Batch-fit all violin modes ────────────────────────────────────────────────
FIT_CONFIG = {
    "1st violin": {"band": (169.5, 170.5), "p0": {"A": 3e-21, "x0": 170.0, "Q": 5e3}},
    "2nd violin": {"band": (339.5, 340.5), "p0": {"A": 1e-21, "x0": 340.0, "Q": 4e3}},
}

fit_results = {}
for mode_name, cfg in FIT_CONFIG.items():
    seg = asd.crop(*cfg["band"])
    res = seg.fit("lorentzian_q", p0=cfg["p0"], sigma=BACKGROUND * 0.1)
    fit_results[mode_name] = res

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"{'Mode':<14} {'f0 [Hz]':>10} {'Q':>10} {'FWHM [mHz]':>12} {'tau [s]':>10}")
print("-" * 60)
for mode_name, res in fit_results.items():
    f0_f   = float(res.params["x0"])
    Q_f    = float(res.params["Q"])
    FWHM_f = f0_f / Q_f
    tau_f  = Q_f / (np.pi * f0_f)
    print(f"{mode_name:<14} {f0_f:>10.3f} {Q_f:>10.2e}"
          f" {FWHM_f * 1e3:>12.2f} {tau_f:>10.1f}")

# ── Side-by-side plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (mode_name, res) in zip(axes, fit_results.items()):
    res.plot(ax=ax)
    ax.set_yscale("log")
    f0_f = float(res.params["x0"])
    Q_f  = float(res.params["Q"])
    ax.set_title(f"{mode_name}: f0={f0_f:.2f} Hz, Q={Q_f:.1e}")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("ASD")
plt.suptitle("Violin Mode Batch Fits", fontsize=13)
plt.tight_layout()
plt.show()
"""

TIME_VARIATION_CODE = """\
# ── Generate 60-s time-varying violin signal ──────────────────────────────────
DURATION = 60
FS_VIOL  = 4096
t_viol   = np.arange(0, DURATION, 1.0 / FS_VIOL)

# Linear drift: f0 = 170.0 → 170.02 Hz over 60 s (temperature effect)
f0_drift = 170.0 + (0.02 / DURATION) * t_viol
phi_viol = 2 * np.pi * np.cumsum(f0_drift) / FS_VIOL
sig_viol = 1e-20 * np.sin(phi_viol)
noise_v  = rng.normal(0, 3e-22, len(t_viol))

ts_viol   = TimeSeries(sig_viol + noise_v, dt=1.0 / FS_VIOL,
                       name="DARM_violin", unit="strain")

# ── Spectrogram ───────────────────────────────────────────────────────────────
spec_viol = ts_viol.spectrogram2(4.0, overlap=2.0)
times_v   = spec_viol.times.value
freqs_v   = spec_viol.frequencies.value

# ── Track via argmax in band ──────────────────────────────────────────────────
TRACK_BAND = (169.8, 170.3)
band_mask  = (freqs_v >= TRACK_BAND[0]) & (freqs_v <= TRACK_BAND[1])
track_viol = np.full(spec_viol.shape[0], np.nan)

for t_idx in range(spec_viol.shape[0]):
    row_band = spec_viol.value[t_idx, band_mask]
    if row_band.max() > 0:
        track_viol[t_idx] = freqs_v[band_mask][np.argmax(row_band)]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

spec_viol.plot(ax=axes[0], norm="log")
axes[0].set_ylim(*TRACK_BAND)
axes[0].set_title("Spectrogram around 1st violin mode")
axes[0].colorbar(label="Power")
axes[0].plot(times_v, track_viol, "r.", ms=4, label="argmax track")
axes[0].legend(loc="upper right", fontsize=9)

mask_v = ~np.isnan(track_viol)
axes[1].plot(times_v[mask_v], track_viol[mask_v], "o-", ms=3, color="tab:orange")
axes[1].set_ylabel("Frequency [Hz]")
axes[1].set_xlabel("Time [s]")
axes[1].set_title("Tracked violin mode frequency vs time")

plt.tight_layout()
plt.show()

drift_total = np.nanmax(track_viol) - np.nanmin(track_viol)
print(f"Total frequency drift: {drift_total * 1e3:.2f} mHz over {DURATION} s")
"""

SUMMARY_CODE = """\
print("=" * 62)
print("Violin Mode Analysis — Key Parameters")
print("-" * 62)
rows = [
    ("f_n (n-th mode)",   "n / (2L) * sqrt(T / rho*A)"),
    ("Q value",           "f0 / FWHM  =  f0 * pi * tau"),
    ("FWHM (linewidth)",  "f0 / Q                [Hz]"),
    ("Ring-down tau",     "Q / (pi * f0)         [s]"),
    ("HWHM gamma",        "FWHM / 2              [Hz]"),
]
for name, formula in rows:
    print(f"  {name:<22} {formula}")
print("=" * 62)
print()
print("gwexpy API used:")
print("  FrequencySeries.fit('lorentzian_q', p0=...)  -- Q-value fit")
print("  lorentzian_q(freqs, A, x0, Q)               -- model evaluation")
print("  ts.spectrogram2(fftlen, overlap)             -- time tracking")
"""

# ============================================================
# English cells
# ============================================================
EN_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "# Violin Mode Analysis\n"
            "\n"
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({COLAB_BASE}/case_violin_mode.ipynb)\n"
            "\n"
            "In gravitational-wave detectors, suspension fibres holding the mirrors\n"
            "resonate as stretched strings. Their modes — **violin modes** — appear as\n"
            "sharp Lorentzian peaks in the strain ASD.\n"
            "\n"
            "This tutorial shows how to:\n"
            "1. Model violin-mode peaks with `lorentzian_q`\n"
            "2. Fit an ASD to extract **Q value, FWHM, and ring-down time**\n"
            "3. Batch-process multiple harmonics\n"
            "4. Track a slowly drifting mode frequency through time\n"
            "\n"
            "**Prerequisites:** [Advanced Fitting](advanced_fitting.ipynb)"
        ),
    },
    {"cell_type": "code", "source": SETUP_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 1. Physics Background\n"
            "\n"
            "The $n$-th resonant frequency of a fibre (length $L$, tension $T$,\n"
            "density $\\rho$, cross-section $A$):\n"
            "\n"
            "$$f_n = \\frac{n}{2L}\\sqrt{\\frac{T}{\\rho A}}$$\n"
            "\n"
            "KAGRA typical values: 1st violin ~170–190 Hz, 2nd ~340–380 Hz.\n"
            "\n"
            "Each mode is a Lorentzian with quality factor $Q = f_0/\\text{FWHM}$:\n"
            "\n"
            "$$\\text{ASD}(f) \\approx \\frac{A\\,\\gamma^2}{(f-f_0)^2 + \\gamma^2},"
            "\\quad \\gamma = \\frac{f_0}{2Q}$$\n"
            "\n"
            "Ring-down time constant: $\\tau = Q / (\\pi f_0)$"
        ),
    },
    {
        "cell_type": "markdown",
        "source": (
            "## 2. Synthetic ASD Data\n"
            "\n"
            "Generate a synthetic ASD with two violin modes on a flat background\n"
            "using `lorentzian_q`.  The same function is used for fitting,\n"
            "so the model is self-consistent."
        ),
    },
    {"cell_type": "code", "source": MOCK_ASD_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 3. Single-Mode Fit\n"
            "\n"
            "Crop the ASD to a narrow band around the 1st violin mode and\n"
            "fit with `'lorentzian_q'`.  Initial guesses (`p0`) should be\n"
            "within a factor ~2 of the true values."
        ),
    },
    {"cell_type": "code", "source": SINGLE_FIT_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 4. Extracting Physical Parameters\n"
            "\n"
            "From the fit parameters $(f_0, Q)$ derive all relevant quantities:"
        ),
    },
    {"cell_type": "code", "source": PARAMS_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 5. Multi-Mode Batch Processing\n"
            "\n"
            "Loop over a configuration dictionary to fit each harmonic\n"
            "and collect results in a summary table."
        ),
    },
    {"cell_type": "code", "source": BATCH_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 6. Time-Variation Tracking\n"
            "\n"
            "Violin-mode frequencies drift slowly with temperature (~mHz/°C).\n"
            "Monitor by computing a spectrogram and tracking the peak position\n"
            "with `argmax` inside a narrow frequency band."
        ),
    },
    {"cell_type": "code", "source": TIME_VARIATION_CODE},
    {"cell_type": "markdown", "source": "## 7. Summary"},
    {"cell_type": "code", "source": SUMMARY_CODE},
]

# ============================================================
# Japanese cells
# ============================================================
JA_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "# バイオリンモード解析\n"
            "\n"
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({COLAB_BASE}/case_violin_mode.ipynb)\n"
            "\n"
            "重力波検出器では鏡を吊るす懸架ファイバーが弦楽器の弦のように振動し、\n"
            "**バイオリンモード**と呼ばれる鋭いローレンツピークが strain ASD 上に現れます。\n"
            "\n"
            "このチュートリアルでは以下を解説します：\n"
            "1. `lorentzian_q` でバイオリンモードをモデル化\n"
            "2. ASD フィットから **Q 値・FWHM・リングダウン時間** を抽出\n"
            "3. 複数の倍音をバッチ処理\n"
            "4. ゆっくりドリフトするモード周波数の時間追跡\n"
            "\n"
            "**前提:** [Advanced Fitting](advanced_fitting.ipynb)"
        ),
    },
    {"cell_type": "code", "source": SETUP_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 1. 物理的背景\n"
            "\n"
            "長さ $L$、張力 $T$、密度 $\\rho$、断面積 $A$ のファイバーの $n$ 次共振周波数：\n"
            "\n"
            "$$f_n = \\frac{n}{2L}\\sqrt{\\frac{T}{\\rho A}}$$\n"
            "\n"
            "KAGRA 代表値：1st バイオリン ~170–190 Hz、2nd ~340–380 Hz。\n"
            "\n"
            "各モードは品質係数 $Q = f_0/\\text{FWHM}$ のローレンツ関数：\n"
            "\n"
            "$$\\text{ASD}(f) \\approx \\frac{A\\,\\gamma^2}{(f-f_0)^2 + \\gamma^2},"
            "\\quad \\gamma = \\frac{f_0}{2Q}$$\n"
            "\n"
            "リングダウン時定数：$\\tau = Q / (\\pi f_0)$"
        ),
    },
    {
        "cell_type": "markdown",
        "source": (
            "## 2. 合成 ASD データの準備\n"
            "\n"
            "`gwexpy.fitting.models.lorentzian_q` を使って、フラット背景に 2 本のバイオリンモードを\n"
            "含む合成 ASD を生成します。フィットにも同じ関数を使うためモデルが自己整合します。"
        ),
    },
    {"cell_type": "code", "source": MOCK_ASD_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 3. 単一モードフィット\n"
            "\n"
            "1st バイオリンモード付近の狭帯域に ASD をクロップし、\n"
            "`'lorentzian_q'` でフィットします。初期値 (`p0`) は真値の\n"
            "~2 倍以内が目安です。"
        ),
    },
    {"cell_type": "code", "source": SINGLE_FIT_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 4. 物理パラメータの抽出\n"
            "\n"
            "フィットパラメータ $(f_0, Q)$ からすべての関連量を導出します："
        ),
    },
    {"cell_type": "code", "source": PARAMS_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 5. 複数モードのバッチ処理\n"
            "\n"
            "設定辞書をループして各倍音をフィットし、サマリーテーブルにまとめます。"
        ),
    },
    {"cell_type": "code", "source": BATCH_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 6. 時間変動の追跡\n"
            "\n"
            "バイオリンモード周波数は温度とともにゆっくりドリフトします（~mHz/°C）。\n"
            "スペクトログラムを計算し、狭帯域内の `argmax` でピーク位置を追跡します。"
        ),
    },
    {"cell_type": "code", "source": TIME_VARIATION_CODE},
    {"cell_type": "markdown", "source": "## 7. まとめ"},
    {"cell_type": "code", "source": SUMMARY_CODE},
]

# ============================================================
# Builder
# ============================================================
def _cell(cell_type: str, source: str) -> dict:
    c: dict = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def make_nb(cells_spec: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "cells": [_cell(s["cell_type"], s["source"]) for s in cells_spec],
    }


if __name__ == "__main__":
    out_en = REPO / "docs/web/en/user_guide/tutorials/case_violin_mode.ipynb"
    out_ja = REPO / "docs/web/ja/user_guide/tutorials/case_violin_mode.ipynb"

    out_en.write_text(json.dumps(make_nb(EN_CELLS), ensure_ascii=False, indent=1) + "\n")
    out_ja.write_text(json.dumps(make_nb(JA_CELLS), ensure_ascii=False, indent=1) + "\n")

    print(f"Created: {out_en}  ({len(EN_CELLS)} cells)")
    print(f"Created: {out_ja}  ({len(JA_CELLS)} cells)")
    print("Done.")
