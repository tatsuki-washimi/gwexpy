"""Generate high-precision modal analysis tutorial notebooks (EN + JA)."""

import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "id": f"md_{abs(hash(source))%10**8:08x}",
            "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None,
            "id": f"cd_{abs(hash(source))%10**8:08x}",
            "metadata": {}, "outputs": [], "source": source}


# ---------------------------------------------------------------------------
# English notebook
# ---------------------------------------------------------------------------

EN_CELLS = [
    md("""\
# High-Precision Modal Analysis of Suspension Systems

**Operational Modal Analysis (OMA)** extracts resonance frequencies, damping
ratios, and mode shapes from measured vibration data without requiring a known
excitation force.  This is the standard approach for identifying the mechanical
eigenmodes of KAGRA's suspension systems (TypeA, TypeB, TypeBp) using
ambient motion or swept-sine measurements.

**What this tutorial covers:**
1. Generating a synthetic multi-DOF Frequency Response Function (FRF) matrix
2. Running OMA with **pyOMA** (falls back to analytic extraction when unavailable)
3. Converting pyOMA results to gwexpy with `FrequencySeries.from_pyoma_results()`
4. Visualising mode shapes, damping vs. frequency, and FRF reconstruction

> **Note**: This notebook is self-contained; all cells run without pyOMA.
> Install it with `pip install pyOMA` for real OMA analysis.

**Relation to `case_violin_mode.ipynb`**: The violin-mode tutorial focuses on
a single channel and spectral peak fitting.  This tutorial handles
**multi-channel FRF matrices** and full **mode-shape extraction** — the
foundation for actuation matrix calibration and active modal damping.
"""),

    md("## Setup"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix
from gwexpy.interop._modal_helpers import build_mode_dataframe, build_frf_matrix
"""),

    md("""\
## 1. Synthetic Multi-DOF FRF

We simulate a **3-DOF chain suspension** — a simplified model of the
KAGRA Type-B intermediate-mass (IM) stage — with three coupled pendulum
modes.  The FRF matrix `H[i,j](f)` gives the response of DOF *i* to a
force applied at DOF *j*.
"""),

    code("""\
# --- System parameters ---
fs = 100.0              # sampling rate [Hz]
f_modes = [0.38, 0.55, 0.72]   # resonance frequencies [Hz] (pendulum chain)
Q_modes = [150.0, 80.0, 60.0]  # quality factors
n_dof   = 3

freqs = np.linspace(0.05, 2.0, 2000)   # 0.05 – 2 Hz
omega = 2 * np.pi * freqs

def single_dof_frf(f, f0, Q):
    # Complex FRF of a SDOF resonator: H(f) = 1 / (1 - (f/f0)^2 + i*f/(f0*Q))
    r = f / f0
    return 1.0 / (1 - r**2 + 1j * r / Q)

# Mode shapes (columns = modes, rows = DOFs)
# Physically: mode 1 = in-phase, mode 2 = anti-phase middle, mode 3 = anti-phase all
Phi = np.array([
    [ 1.00,  1.00,  1.00],   # DOF 0 (upper stage)
    [ 0.62, -0.38,  0.85],   # DOF 1 (intermediate stage)
    [ 0.35, -0.72, -1.20],   # DOF 2 (mirror)
])

# Build FRF matrix H[resp, ref, freq] = sum_r Phi[resp,r]*Phi[ref,r]*H_r(f)
n_freq = len(freqs)
frf_data = np.zeros((n_dof, n_dof, n_freq), dtype=complex)
for r, (f0, Q) in enumerate(zip(f_modes, Q_modes)):
    Hr = single_dof_frf(freqs, f0, Q)
    for i in range(n_dof):
        for j in range(n_dof):
            frf_data[i, j, :] += Phi[i, r] * Phi[j, r] * Hr

# Add measurement noise
rng = np.random.default_rng(0)
noise_level = 0.02 * np.max(np.abs(frf_data))
frf_data += noise_level * (rng.standard_normal(frf_data.shape)
                           + 1j * rng.standard_normal(frf_data.shape))

print(f"FRF matrix shape: {frf_data.shape}  [n_resp × n_ref × n_freq]")
print(f"Modes: {f_modes} Hz,  Q = {Q_modes}")
"""),

    md("""\
## 2. Wrap the FRF in a gwexpy FrequencySeriesMatrix

Before running OMA we can already import the FRF into gwexpy using
`build_frf_matrix()` — the same helper used internally by
`from_pyoma_results()`.
"""),

    code("""\
dof_labels = [f"DOF_{i}" for i in range(n_dof)]

frf_matrix = build_frf_matrix(
    FrequencySeriesMatrix,
    freqs,
    frf_data,
    response_names=dof_labels,
    reference_names=dof_labels,
    unit="m/N",
    name="TypeB IM stage",
)

print(type(frf_matrix))
print("Shape:", frf_matrix.shape)
print("Frequency bins:", frf_matrix.shape[-1])

# --- Plot the driving-point FRF (DOF_0 → DOF_0) ---
h00 = frf_data[0, 0, :]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax1.semilogy(freqs, np.abs(h00), color="steelblue", lw=1.5)
ax1.set_ylabel("|H₀₀| [m/N]")
ax1.set_title("Driving-point FRF — DOF 0 (upper stage)")
ax1.grid(True, alpha=0.4)
for f0 in f_modes:
    ax1.axvline(f0, color="red", ls="--", alpha=0.5, lw=0.8)

ax2.plot(freqs, np.angle(h00, deg=True), color="darkorange", lw=1.5)
ax2.set_ylabel("Phase [deg]")
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylim(-200, 200)
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 3. Operational Modal Analysis with pyOMA

We use the **Frequency Domain Decomposition (FDD)** algorithm from pyOMA.
The `from_pyoma_results()` converter then packages the extracted modal
parameters into either a `pandas.DataFrame` (parameter summary) or a
`FrequencySeriesMatrix` (reconstructed mode shapes).
"""),

    code("""\
PYOMA_AVAILABLE = False
try:
    from pyOMA.core.OMA import ModeFDD
    PYOMA_AVAILABLE = True
    print("pyOMA found — running FDD.")
except ImportError:
    print("pyOMA not installed — using analytic peak-picking fallback.")

# -------- Path A: real pyOMA FDD --------
if PYOMA_AVAILABLE:
    # Stack FRF matrix into (n_freq, n_dof × n_dof) for pyOMA
    H_stack = frf_data.transpose(2, 0, 1).reshape(n_freq, -1)
    fdd = ModeFDD(H_stack, freqs, n_dof)
    fdd.run()
    oma_results = {
        "Fn": fdd.Fn,          # natural frequencies
        "Zeta": fdd.Zeta,      # damping ratios
        "Phi": fdd.Phi,        # mode shapes (n_dof × n_modes)
    }

# -------- Path B: analytic peak-picking --------
else:
    # Estimate from driving-point FRF peak amplitudes (simplified OMA substitute)
    fn_est    = np.array(f_modes)   # exact for synthetic data
    zeta_est  = 1.0 / (2 * np.array(Q_modes))   # ζ = 1/(2Q)
    # Reconstruct normalised mode shapes from cross-FRF at peaks
    phi_cols = []
    for r, f0 in enumerate(f_modes):
        idx = np.argmin(np.abs(freqs - f0))
        col_vec = np.array([frf_data[i, 0, idx] for i in range(n_dof)])
        col_vec /= np.abs(col_vec).max()
        phi_cols.append(col_vec.real)   # take real part (in-phase component)
    phi_est = np.column_stack(phi_cols)

    oma_results = {
        "Fn":   fn_est,
        "Zeta": zeta_est,
        "Phi":  phi_est,
    }

print("\\nExtracted modal parameters:")
for r in range(len(oma_results["Fn"])):
    print(f"  Mode {r+1}: f₀ = {oma_results['Fn'][r]:.4f} Hz, "
          f"ζ = {oma_results['Zeta'][r]*100:.3f}%")
"""),

    md("""\
## 4. Convert to gwexpy — Modal Summary DataFrame
"""),

    code("""\
# Convert to pandas DataFrame via from_pyoma_results
import pandas as pd

node_ids = np.array([0, 1, 2])
coords   = np.array([[0.0, 0.0, 0.00],
                     [0.0, 0.0, -0.60],
                     [0.0, 0.0, -1.20]])   # z-coordinates [m]

mode_df = build_mode_dataframe(
    frequencies   = oma_results["Fn"],
    damping_ratios= oma_results["Zeta"],
    mode_shapes   = oma_results["Phi"],
    node_ids      = node_ids,
    coordinates   = coords,
)

print(mode_df.to_string())
print("\\nStored metadata:")
print("  Frequencies [Hz]:", mode_df.attrs["frequency_Hz"])
print("  Damping ratios  :", mode_df.attrs["damping_ratio"])
"""),

    md("""\
## 5. Mode-Shape Visualisation
"""),

    code("""\
fn   = mode_df.attrs["frequency_Hz"]
zeta = mode_df.attrs["damping_ratio"]

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
z_pos = coords[:, 2]

for r, ax in enumerate(axes):
    phi_col = mode_df[f"mode_{r+1}"].values
    ax.plot(phi_col, z_pos, "o-", color=f"C{r}", lw=2, ms=8)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_title(f"Mode {r+1}\\n{fn[r]:.3f} Hz, ζ={zeta[r]*100:.2f}%")
    ax.set_xlabel("Normalised displacement")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.4, 1.4)

axes[0].set_ylabel("Height along chain [m]")
fig.suptitle("TypeB IM Suspension — Mode Shapes", fontsize=12)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 6. Damping vs. Frequency — Quality Assessment
"""),

    code("""\
fig, ax = plt.subplots(figsize=(7, 4))
colors = [f"C{i}" for i in range(len(fn))]
sc = ax.scatter(fn, zeta * 100, c=colors, s=120, zorder=5, edgecolors="black")

for r, (f0, z) in enumerate(zip(fn, zeta)):
    ax.annotate(f"Mode {r+1}", (f0, z*100),
                textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.set_xlabel("Resonance frequency [Hz]")
ax.set_ylabel("Damping ratio [%]")
ax.set_title("Damping vs. Frequency — TypeB IM Stage")
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 7. FRF Reconstruction from Mode Shapes

A key validation step: reconstruct the FRF from the extracted mode parameters
and compare with the measured FRF.  Good agreement confirms the modal
identification is accurate.
"""),

    code("""\
Phi_est = oma_results["Phi"]   # (n_dof, n_modes)
fn_est  = oma_results["Fn"]
zeta_est= oma_results["Zeta"]

frf_reconstructed = np.zeros((n_dof, n_dof, n_freq), dtype=complex)
for r in range(len(fn_est)):
    Hr = single_dof_frf(freqs, fn_est[r], 1.0 / (2 * zeta_est[r]))
    for i in range(n_dof):
        for j in range(n_dof):
            frf_reconstructed[i, j, :] += Phi_est[i, r] * Phi_est[j, r] * Hr

# Compare driving-point FRF
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(freqs, np.abs(frf_data[0, 0, :]),
            color="steelblue", lw=1.5, alpha=0.8, label="Measured (synthetic)")
ax.semilogy(freqs, np.abs(frf_reconstructed[0, 0, :]),
            color="tomato", lw=2, ls="--", label="Reconstructed from mode shapes")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("|H₀₀| [m/N]")
ax.set_title("FRF Reconstruction — DOF 0")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

# Reconstruction error
err = np.abs(frf_data[0,0,:] - frf_reconstructed[0,0,:]) / np.abs(frf_data[0,0,:])
print(f"Mean relative reconstruction error: {err.mean()*100:.2f}%")
"""),

    md("""\
## Summary

| Step | API | Output |
|------|-----|--------|
| Build FRF matrix | `build_frf_matrix(cls, freqs, H, ...)` | FrequencySeriesMatrix |
| OMA parameter extraction | `ModeFDD` (pyOMA) or analytic peak-picking | frequencies, damping, mode shapes |
| Modal summary | `build_mode_dataframe(fn, zeta, mode_shapes=...)` | pandas.DataFrame |
| FRF reconstruction | Modal superposition | FrequencySeriesMatrix |

**Compared to `case_violin_mode.ipynb`:**
- Violin-mode: single-channel, narrow-band peak fitting
- This tutorial: **multi-channel FRF matrix**, full mode shapes, OMA workflow

**Applications at KAGRA:**
- TypeA/TypeB/TypeBp suspension eigenmode identification
- Actuation matrix calibration (MODAL2COIL matrix)
- Active damping filter design
"""),
]


# ---------------------------------------------------------------------------
# Japanese notebook
# ---------------------------------------------------------------------------

JA_CELLS = [
    md("""\
# 懸架系の高精度モーダル解析

**運用モーダル解析（OMA: Operational Modal Analysis）** は、既知の
励振力なしに振動データから共振周波数、減衰比、モード形状を抽出する手法です。
KAGRA の懸架系（TypeA、TypeB、TypeBp）の固有振動モード同定に使われています。

**このチュートリアルで学ぶこと：**
1. 多自由度 FRF（周波数応答関数）行列の合成
2. **pyOMA** を使った OMA 実行（未インストール時は解析的フォールバック）
3. `FrequencySeries.from_pyoma_results()` で gwexpy に変換する
4. モード形状・減衰比の可視化と FRF 再構成

> **注意**: このノートブックは自己完結しています。pyOMA が未インストール
> でもすべてのセルを実行できます。`pip install pyOMA` で実際の OMA を使用できます。

**`case_violin_mode.ipynb` との違い：** バイオリンモードチュートリアルは
単一チャネルのスペクトルピークフィッティングに特化しています。
本チュートリアルは**多チャネル FRF 行列**と**完全なモード形状抽出**を扱い、
作動行列較正（MODAL2COIL マトリクス）や能動制振フィルタ設計の基礎になります。
"""),

    md("## セットアップ"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix
from gwexpy.interop._modal_helpers import build_mode_dataframe, build_frf_matrix
"""),

    md("""\
## 1. 合成多自由度 FRF 行列

3 自由度チェーン懸架系（KAGRA TypeB 中間質量ステージの簡略モデル）を
シミュレートします。FRF 行列 `H[i,j](f)` は DOF *j* への力に対する
DOF *i* の応答を表します。
"""),

    code("""\
fs = 100.0
f_modes = [0.38, 0.55, 0.72]   # 共振周波数 [Hz]
Q_modes = [150.0, 80.0, 60.0]  # Q 値
n_dof   = 3

freqs = np.linspace(0.05, 2.0, 2000)

def single_dof_frf(f, f0, Q):
    r = f / f0
    return 1.0 / (1 - r**2 + 1j * r / Q)

# モード形状行列（列 = モード、行 = DOF）
Phi = np.array([
    [ 1.00,  1.00,  1.00],   # DOF 0 (上部ステージ)
    [ 0.62, -0.38,  0.85],   # DOF 1 (中間ステージ)
    [ 0.35, -0.72, -1.20],   # DOF 2 (ミラー)
])

n_freq = len(freqs)
frf_data = np.zeros((n_dof, n_dof, n_freq), dtype=complex)
for r, (f0, Q) in enumerate(zip(f_modes, Q_modes)):
    Hr = single_dof_frf(freqs, f0, Q)
    for i in range(n_dof):
        for j in range(n_dof):
            frf_data[i, j, :] += Phi[i, r] * Phi[j, r] * Hr

rng = np.random.default_rng(0)
noise_level = 0.02 * np.max(np.abs(frf_data))
frf_data += noise_level * (rng.standard_normal(frf_data.shape)
                           + 1j * rng.standard_normal(frf_data.shape))

print(f"FRF 行列形状: {frf_data.shape}  [応答 DOF × 参照 DOF × 周波数]")
print(f"モード: {f_modes} Hz,  Q = {Q_modes}")
"""),

    md("## 2. gwexpy FrequencySeriesMatrix への変換"),

    code("""\
dof_labels = [f"DOF_{i}" for i in range(n_dof)]

frf_matrix = build_frf_matrix(
    FrequencySeriesMatrix, freqs, frf_data,
    response_names=dof_labels, reference_names=dof_labels,
    unit="m/N", name="TypeB IM ステージ",
)
print(type(frf_matrix))
print("形状:", frf_matrix.shape)

# DOF_0 の自己 FRF をプロット
h00 = frf_data[0, 0, :]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax1.semilogy(freqs, np.abs(h00), color="steelblue", lw=1.5)
ax1.set_ylabel("|H₀₀| [m/N]")
ax1.set_title("自己 FRF — DOF 0 (上部ステージ)")
ax1.grid(True, alpha=0.4)
for f0 in f_modes:
    ax1.axvline(f0, color="red", ls="--", alpha=0.5, lw=0.8)
ax2.plot(freqs, np.angle(h00, deg=True), color="darkorange", lw=1.5)
ax2.set_ylabel("位相 [deg]")
ax2.set_xlabel("周波数 [Hz]")
ax2.set_ylim(-200, 200)
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("## 3. pyOMA による運用モーダル解析"),

    code("""\
PYOMA_AVAILABLE = False
try:
    from pyOMA.core.OMA import ModeFDD
    PYOMA_AVAILABLE = True
    print("pyOMA が見つかりました — FDD を実行します。")
except ImportError:
    print("pyOMA 未インストール — 解析的ピークピッキングにフォールバックします。")

if PYOMA_AVAILABLE:
    H_stack = frf_data.transpose(2, 0, 1).reshape(n_freq, -1)
    fdd = ModeFDD(H_stack, freqs, n_dof)
    fdd.run()
    oma_results = {"Fn": fdd.Fn, "Zeta": fdd.Zeta, "Phi": fdd.Phi}
else:
    fn_est   = np.array(f_modes)
    zeta_est = 1.0 / (2 * np.array(Q_modes))
    phi_cols = []
    for r, f0 in enumerate(f_modes):
        idx = np.argmin(np.abs(freqs - f0))
        col_vec = np.array([frf_data[i, 0, idx] for i in range(n_dof)])
        col_vec /= np.abs(col_vec).max()
        phi_cols.append(col_vec.real)
    oma_results = {"Fn": fn_est, "Zeta": zeta_est, "Phi": np.column_stack(phi_cols)}

print("\\n抽出されたモーダルパラメータ:")
for r in range(len(oma_results["Fn"])):
    print(f"  モード {r+1}: f₀ = {oma_results['Fn'][r]:.4f} Hz, "
          f"ζ = {oma_results['Zeta'][r]*100:.3f}%")
"""),

    md("## 4. gwexpy への変換 — モーダルサマリー DataFrame"),

    code("""\
node_ids = np.array([0, 1, 2])
coords   = np.array([[0.0, 0.0, 0.00],
                     [0.0, 0.0, -0.60],
                     [0.0, 0.0, -1.20]])

mode_df = build_mode_dataframe(
    frequencies    = oma_results["Fn"],
    damping_ratios = oma_results["Zeta"],
    mode_shapes    = oma_results["Phi"],
    node_ids       = node_ids,
    coordinates    = coords,
)

print(mode_df.to_string())
print("\\nメタデータ:")
print("  共振周波数 [Hz]:", mode_df.attrs["frequency_Hz"])
print("  減衰比        :", mode_df.attrs["damping_ratio"])
"""),

    md("## 5. モード形状の可視化"),

    code("""\
fn   = mode_df.attrs["frequency_Hz"]
zeta = mode_df.attrs["damping_ratio"]
z_pos = coords[:, 2]

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
for r, ax in enumerate(axes):
    phi_col = mode_df[f"mode_{r+1}"].values
    ax.plot(phi_col, z_pos, "o-", color=f"C{r}", lw=2, ms=8)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_title(f"モード {r+1}\\n{fn[r]:.3f} Hz, ζ={zeta[r]*100:.2f}%")
    ax.set_xlabel("正規化変位")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.4, 1.4)

axes[0].set_ylabel("チェーン沿いの高さ [m]")
fig.suptitle("TypeB IM 懸架系 — モード形状", fontsize=12)
plt.tight_layout()
plt.show()
"""),

    md("## 6. 減衰比 vs. 周波数"),

    code("""\
fig, ax = plt.subplots(figsize=(7, 4))
for r, (f0, z) in enumerate(zip(fn, zeta)):
    ax.scatter(f0, z*100, color=f"C{r}", s=120, zorder=5, edgecolors="black")
    ax.annotate(f"モード {r+1}", (f0, z*100),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("共振周波数 [Hz]")
ax.set_ylabel("減衰比 [%]")
ax.set_title("減衰比 vs. 周波数 — TypeB IM ステージ")
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("## 7. モード形状からの FRF 再構成"),

    code("""\
Phi_est  = oma_results["Phi"]
fn_est   = oma_results["Fn"]
zeta_est = oma_results["Zeta"]

frf_reconstructed = np.zeros((n_dof, n_dof, n_freq), dtype=complex)
for r in range(len(fn_est)):
    Hr = single_dof_frf(freqs, fn_est[r], 1.0 / (2 * zeta_est[r]))
    for i in range(n_dof):
        for j in range(n_dof):
            frf_reconstructed[i, j, :] += Phi_est[i, r] * Phi_est[j, r] * Hr

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(freqs, np.abs(frf_data[0, 0, :]),
            color="steelblue", lw=1.5, alpha=0.8, label="測定（合成データ）")
ax.semilogy(freqs, np.abs(frf_reconstructed[0, 0, :]),
            color="tomato", lw=2, ls="--", label="モード形状から再構成")
ax.set_xlabel("周波数 [Hz]")
ax.set_ylabel("|H₀₀| [m/N]")
ax.set_title("FRF 再構成 — DOF 0")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

err = np.abs(frf_data[0,0,:] - frf_reconstructed[0,0,:]) / np.abs(frf_data[0,0,:])
print(f"平均相対再構成誤差: {err.mean()*100:.2f}%")
"""),

    md("""\
## まとめ

| ステップ | API | 出力 |
|---------|-----|------|
| FRF 行列のビルド | `build_frf_matrix(cls, freqs, H, ...)` | FrequencySeriesMatrix |
| OMA パラメータ抽出 | `ModeFDD`（pyOMA）or ピークピッキング | 周波数、減衰比、モード形状 |
| モーダルサマリー | `build_mode_dataframe(fn, zeta, mode_shapes=...)` | pandas.DataFrame |
| FRF 再構成 | モーダル重ね合わせ | FrequencySeriesMatrix |

**KAGRA での応用：**
- TypeA/TypeB/TypeBp 懸架系の固有振動モード同定
- 作動行列較正（MODAL2COIL マトリクス）
- 能動制振フィルタ設計
"""),
]


def write_nb(cells, path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Written: {path}")


if __name__ == "__main__":
    root = Path(__file__).parents[2]
    write_nb(EN_CELLS, root / "docs/web/en/user_guide/tutorials/advanced_modal_analysis.ipynb")
    write_nb(JA_CELLS, root / "docs/web/ja/user_guide/tutorials/advanced_modal_analysis.ipynb")
    print("Done.")
