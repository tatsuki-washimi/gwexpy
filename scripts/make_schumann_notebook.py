"""Generate Schumann resonance analysis tutorial notebooks (EN + JA).

Usage:
    python scripts/make_schumann_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_nb(cells: list[tuple[str, str]]) -> dict:
    """Build a minimal nbformat v4 notebook from a list of (cell_type, source) tuples."""
    nb_cells = []
    for i, (ctype, src) in enumerate(cells):
        src = src.strip("\n")
        if ctype == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "id": f"cell-{i}",
                "metadata": {},
                "source": src,
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "id": f"cell-{i}",
                "metadata": {},
                "outputs": [],
                "source": src,
            })
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": nb_cells,
    }


# ---------------------------------------------------------------------------
# English cells
# ---------------------------------------------------------------------------

EN_CELLS: list[tuple[str, str]] = [
    # ── Title ──────────────────────────────────────────────────────────────
    ("markdown", """\
# Schumann Resonance Analysis in GW Detectors

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main/docs/web/en/user_guide/tutorials/case_schumann_resonance.ipynb)

**Schumann resonances** are global electromagnetic resonances excited by lightning
discharges in the Earth–ionosphere cavity.  They appear as narrow spectral peaks at
approximately 7.83, 14.3, 20.8, and 26.4 Hz — squarely in the sensitive band of
second-generation gravitational-wave detectors such as KAGRA.

This notebook demonstrates an end-to-end characterisation workflow using gwexpy:

1. **Bootstrap PSD** (`bootstrap_spectrogram`) — robust spectral estimation with
   asymmetric confidence intervals
2. **Lorentzian Q-factor fit** (`fit_series` + `lorentzian_q`) — measure resonance
   frequency, quality factor, and amplitude of each mode
3. **Covariance structure** (`BifrequencyMap`) — visualise inter-bin correlations
   returned by the bootstrap; used automatically for GLS fitting
4. **Temporal tracking** — monitor mode amplitude over the observation window

> **Prerequisites**:
> - [Spectrogram basics](intro_spectrogram.ipynb)
> - [Advanced Fitting](advanced_fitting.ipynb)
> - [Bootstrap PSD & GLS Fitting](case_bootstrap_gls_fitting.ipynb)
"""),

    # ── Setup ──────────────────────────────────────────────────────────────
    ("markdown", "## Setup"),
    ("code", """\
# ruff: noqa: I001
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from gwexpy.fitting import fit_series
from gwexpy.fitting.models import lorentzian_q
from gwexpy.frequencyseries import BifrequencyMap  # noqa: F401 (shown for clarity)
from gwexpy.spectral import bootstrap_spectrogram
from gwexpy.timeseries import TimeSeries
"""),

    # ── Mock data ──────────────────────────────────────────────────────────
    ("markdown", """\
## 1. Mock Data: Magnetometer with Schumann Resonances

We simulate a single-axis magnetometer (ELF band) measuring Earth's background
electromagnetic field.  The four Schumann resonances are generated as **narrowband
noise with Lorentzian spectral shape** (characterised by frequency and Q-factor),
superimposed on a broadband noise floor.

| Mode | Frequency | Q | Peak ASD |
|------|-----------|---|----------|
| SR1  | 7.83 Hz   | 5.0 | 4.0 nT/√Hz |
| SR2  | 14.3 Hz   | 4.5 | 2.5 nT/√Hz |
| SR3  | 20.8 Hz   | 4.0 | 1.5 nT/√Hz |
| SR4  | 26.4 Hz   | 3.5 | 1.0 nT/√Hz |
"""),
    ("code", """\
rng = np.random.default_rng(42)
fs = 512        # sample rate [Hz]
T  = 300.0      # duration [s]
n  = int(fs * T)

# Schumann resonance parameters (Earth-ionosphere cavity modes)
SR_FREQS = [7.83, 14.3, 20.8, 26.4]   # Hz — first 4 modes
SR_Q     = [5.0,  4.5,  4.0,  3.5]    # quality factors
SR_AMP   = [4.0,  2.5,  1.5,  1.0]    # peak ASD [nT/√Hz]
NOISE_FLOOR = 0.3                       # broadband floor [nT/√Hz]

# ── Frequency-domain synthesis ──────────────────────────────────────────────
f = np.fft.rfftfreq(n, d=1.0 / fs)   # one-sided frequency axis [Hz]

# Build target ASD [nT/√Hz] as sum of Lorentzians + floor
asd_target = np.full_like(f, NOISE_FLOOR)
for f0, q, A in zip(SR_FREQS, SR_Q, SR_AMP):
    gamma = f0 / (2.0 * q)            # half-width at half-maximum [Hz]
    asd_target += A * gamma / np.sqrt((f - f0) ** 2 + gamma ** 2)

# Convert ASD to rfft amplitudes so that Welch PSD ≈ asd_target²
# For one-sided PSD: S(f) = 2·|X[k]|² / (N·fs)  ⟹  |X[k]| = asd·√(N·fs/2)
amp = asd_target * np.sqrt(n * fs / 2.0)
amp[0] = 0.0  # zero DC component

Z = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(2)
x = np.fft.irfft(amp * Z, n=n)

mag = TimeSeries(x, dt=1.0 / fs, unit=u.nT,
                 name="K1:PEM-MAG_EXV_EAST_X_DQ", t0=0)
print(f"Duration: {T:.0f} s | fs: {fs} Hz | N: {n:,}")
"""),

    # ── Quick-look ASD ─────────────────────────────────────────────────────
    ("code", """\
# Quick-look ASD — verify Schumann peaks are present
fig, ax = plt.subplots(figsize=(10, 4))
asd_raw = mag.asd(fftlength=16.0, overlap=8.0)
ax.semilogy(asd_raw.frequencies.value, asd_raw.value, lw=0.8,
            color='steelblue', label='Single ASD estimate (16 s FFT)')

for f0, lab in zip(SR_FREQS, ['SR1', 'SR2', 'SR3', 'SR4']):
    ax.axvline(f0, color='red', ls='--', lw=0.8, alpha=0.7)
    ax.text(f0 + 0.15, 0.35, lab, color='red', fontsize=8)

ax.set_xlim(4, 35)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('ASD [nT/√Hz]')
ax.set_title('Magnetometer ASD — four Schumann resonance peaks visible')
ax.legend()
plt.tight_layout()
plt.show()
"""),

    # ── Bootstrap PSD ──────────────────────────────────────────────────────
    ("markdown", """\
## 2. Bootstrap PSD Estimation

`bootstrap_spectrogram` resamples the spectrogram's time columns to produce a
**robust PSD estimate** (median or mean) together with asymmetric confidence
intervals.  When `return_map=True` it also returns the **covariance `BifrequencyMap`**
`cov_map(f1, f2)` — a 2-D matrix quantifying correlations between frequency bins,
which is used in the GLS fitting step.
"""),
    ("code", """\
# Compute spectrogram: 16 s FFT segments, 50 % Hann-window overlap
spec = mag.spectrogram2(fftlength=16.0, overlap=8.0, window='hann')
print(f"Spectrogram shape: {spec.shape}  (n_times × n_freqs)")

# Bootstrap resampling — returns (PSD, covariance BifrequencyMap)
psd_boot, cov_map = bootstrap_spectrogram(
    spec,
    n_boot=500,
    method='median',
    ci=0.68,
    fftlength=16.0,
    overlap=8.0,
    return_map=True,
    ignore_nan=True,
)
print(f"Bootstrap PSD shape : {psd_boot.shape}")
print(f"Covariance map shape: {cov_map.shape}  ← BifrequencyMap(f1, f2)")
"""),
    ("code", """\
# Plot bootstrap PSD with 1-σ confidence band derived from cov_map diagonal
fig, ax = plt.subplots(figsize=(10, 4))

f_psd = psd_boot.frequencies.value
y_psd = psd_boot.value

# Diagonal of covariance = variance per frequency bin
diag = cov_map.diagonal(method='mean')
y_var = np.interp(f_psd, diag.frequencies.value, np.abs(diag.value))
y_lo = np.sqrt(np.maximum(y_psd - np.sqrt(y_var), 1e-12))
y_hi = np.sqrt(y_psd + np.sqrt(y_var))

ax.semilogy(f_psd, np.sqrt(y_psd), lw=1.5, color='steelblue',
            label='Bootstrap median ASD')
ax.fill_between(f_psd, y_lo, y_hi, alpha=0.25, color='steelblue', label='±1σ')

for f0 in SR_FREQS:
    ax.axvline(f0, color='red', ls='--', lw=0.7, alpha=0.6)

ax.set_xlim(4, 35)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('ASD [nT/√Hz]')
ax.set_title('Bootstrap PSD — median ± 1σ confidence band')
ax.legend()
plt.tight_layout()
plt.show()
"""),

    # ── Lorentzian fitting ─────────────────────────────────────────────────
    ("markdown", """\
## 3. Lorentzian Q-Factor Fitting

Each Schumann mode is modelled as a **Lorentzian peak** parameterised by Q-factor:

$$S(f) = \\frac{A\\,\\gamma^2}{(f - f_0)^2 + \\gamma^2}, \\quad \\gamma = \\frac{f_0}{2Q}$$

`fit_series` minimises a **Generalised Least Squares** (GLS) cost by passing the
covariance BifrequencyMap directly via `cov=cov_map`.  This accounts for
correlations between overlapping FFT segments and gives correct parameter
uncertainties.
"""),
    ("code", """\
# Fit each Schumann mode independently in a dedicated frequency window
fit_ranges = [(6.0, 10.5), (11.5, 17.5), (17.5, 24.5), (23.0, 30.5)]

fit_results = []
for i, (f0, q0, A0, (flo, fhi)) in enumerate(
        zip(SR_FREQS, SR_Q, SR_AMP, fit_ranges)):
    result = fit_series(
        psd_boot,
        lorentzian_q,
        x_range=(flo, fhi),
        cov=cov_map,
        p0={'A': A0 ** 2, 'x0': f0, 'Q': q0},
        limits={'A': (0, 500), 'x0': (flo, fhi), 'Q': (1.0, 100.0)},
    )
    fit_results.append(result)
    p, e = result.params, result.errors
    print(
        f"SR{i + 1}: f0 = {p['x0']:.3f} ± {e['x0']:.3f} Hz  |"
        f"  Q = {p['Q']:.2f} ± {e['Q']:.2f}  |"
        f"  A = {p['A']:.4f} ± {e['A']:.4f} nT²/Hz"
    )
"""),
    ("code", """\
# Plot fitted Lorentzians over the bootstrap PSD
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
plot_ranges = [(5.5, 11.0), (11.5, 17.5), (17.5, 24.5), (22.5, 30.5)]
labels = ['SR1 (7.83 Hz)', 'SR2 (14.3 Hz)', 'SR3 (20.8 Hz)', 'SR4 (26.4 Hz)']

for ax, result, fr, color, label in zip(axes, fit_results, plot_ranges, colors, labels):
    psd_crop = psd_boot.crop(*fr)
    f_crop = psd_crop.frequencies.value
    y_crop = psd_crop.value

    ax.semilogy(f_crop, y_crop, '.', color=color, ms=2.5, label='Bootstrap PSD')

    f_fine = np.linspace(*fr, 300)
    ax.semilogy(f_fine, lorentzian_q(f_fine, **result.params),
                'k-', lw=1.5, label='Lorentzian fit')

    p = result.params
    ax.set_title(f"{label}\\nf₀={p['x0']:.3f} Hz, Q={p['Q']:.2f}", fontsize=9)
    ax.set_xlabel('Frequency [Hz]')
    ax.legend(fontsize=7)

axes[0].set_ylabel('PSD [nT²/Hz]')
plt.suptitle('GLS Lorentzian fits to Schumann resonance modes (cov_map used)', y=1.01)
plt.tight_layout()
plt.show()
"""),

    # ── BifrequencyMap ─────────────────────────────────────────────────────
    ("markdown", """\
## 4. Covariance Structure via BifrequencyMap

The bootstrap covariance map `cov_map(f1, f2)` returned by `bootstrap_spectrogram`
encodes **how strongly spectral estimates at different frequencies are correlated**.
For magnetometer data with narrow resonances, we expect:

- **High covariance along the diagonal** (`f1 ≈ f2`) at each resonance frequency
  → adjacent bins share correlated noise from overlapping FFT windows.
- **Off-diagonal structure** near resonances → the peak broadens beyond one bin.

`BifrequencyMap` methods used here:
- `.diagonal(method='mean')` — returns the variance profile (diagonal of the matrix)
- `.get_slice(at=f0, axis='f1')` — returns a 1-D covariance slice at fixed f1
"""),
    ("code", """\
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: 2-D covariance map (4–35 Hz sub-region) ─────────────────────────
f_lo, f_hi = 4.0, 35.0
f_vals = cov_map.frequency1.value
mask = (f_vals >= f_lo) & (f_vals <= f_hi)
cov_sub = np.abs(cov_map.value[np.ix_(mask, mask)])
f_sub   = f_vals[mask]

im = axes[0].pcolormesh(f_sub, f_sub, cov_sub,
                         norm=LogNorm(vmin=cov_sub[cov_sub > 0].min(), vmax=cov_sub.max()),
                         cmap='inferno', shading='auto')
fig.colorbar(im, ax=axes[0], label='|Covariance| [nT⁴/Hz²]')

for f0 in SR_FREQS:
    axes[0].axvline(f0, color='cyan', lw=0.6, ls='--')
    axes[0].axhline(f0, color='cyan', lw=0.6, ls='--')

axes[0].set_xlabel('f₁ [Hz]')
axes[0].set_ylabel('f₂ [Hz]')
axes[0].set_title('Bootstrap Covariance Map (4–35 Hz)')

# ── Right: diagonal = variance per bin ────────────────────────────────────
diag = cov_map.diagonal(method='mean')
f_d = diag.frequencies.value
m_d = (f_d >= f_lo) & (f_d <= f_hi)
axes[1].semilogy(f_d[m_d], np.abs(diag.value[m_d]), lw=1.2, color='steelblue')

for f0 in SR_FREQS:
    axes[1].axvline(f0, color='red', ls='--', lw=0.7)

axes[1].set_xlim(f_lo, f_hi)
axes[1].set_xlabel('Frequency [Hz]')
axes[1].set_ylabel('Diagonal variance [nT⁴/Hz²]')
axes[1].set_title('Diagonal of Covariance Map (≈ PSD² / N_bootstrap)')

plt.tight_layout()
plt.show()
"""),
    ("code", """\
# Covariance slice at SR1 (7.83 Hz) — shows spectral width of the resonance
cov_slice = cov_map.get_slice(at=7.83, axis='f1')
f_sl = cov_slice.frequencies.value
m_sl = (f_sl >= f_lo) & (f_sl <= f_hi)

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(f_sl[m_sl], np.abs(cov_slice.value[m_sl]), lw=1.2, color='darkorange')
ax.axvline(7.83, color='red', ls='--', lw=1, label='SR1 (7.83 Hz)')
for f0 in SR_FREQS[1:]:
    ax.axvline(f0, color='gray', ls=':', lw=0.7)

ax.set_xlim(f_lo, f_hi)
ax.set_xlabel('Frequency f₂ [Hz]')
ax.set_ylabel('|Cov(f₁=7.83 Hz, f₂)| [nT⁴/Hz²]')
ax.set_title('Covariance slice at SR1 — peak width reflects resonance bandwidth')
ax.legend()
plt.tight_layout()
plt.show()
"""),

    # ── Temporal tracking ──────────────────────────────────────────────────
    ("markdown", """\
## 5. Temporal Amplitude Tracking

Schumann resonance amplitude varies with global lightning activity (diurnal cycle,
seasonal effects).  We extract the band-averaged power around each mode from the
spectrogram and convert to an **ASD time series** to monitor evolution.
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(10, 4))
times_s = spec.times.value
colors_t = ['steelblue', 'darkorange', 'seagreen', 'crimson']

for f0, color, label in zip(SR_FREQS, colors_t,
                             ['SR1 7.83 Hz', 'SR2 14.3 Hz',
                              'SR3 20.8 Hz', 'SR4 26.4 Hz']):
    # Narrow-band power (±0.5 Hz window around each mode)
    spec_band = spec.crop_frequencies(f0 - 0.5, f0 + 0.5)
    amp_t = np.sqrt(spec_band.value.mean(axis=1))
    ax.plot(times_s, amp_t, lw=0.8, color=color, label=label)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Mean ASD in ±0.5 Hz band [nT/√Hz]')
ax.set_title('Schumann resonance amplitude evolution (300 s mock observation)')
ax.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()
"""),

    # ── Summary ────────────────────────────────────────────────────────────
    ("markdown", """\
## Summary

| Step | Tool | Output |
|------|------|--------|
| Robust PSD | `bootstrap_spectrogram(return_map=True)` | Median PSD + 1σ band |
| Uncertainty model | `BifrequencyMap` (from bootstrap) | Inter-bin covariance matrix |
| Peak characterisation | `fit_series(lorentzian_q, cov=cov_map)` | f₀, Q, amplitude per mode |
| Temporal monitoring | `Spectrogram.crop_frequencies()` | Amplitude time series |

### Key takeaways

- **Bootstrap resampling** gives reliable uncertainty estimates for narrow spectral
  peaks where single-segment estimates are noisy.
- **BifrequencyMap covariance** is essential for proper GLS fitting: passing `cov=cov_map`
  to `fit_series` automatically accounts for inter-bin correlations.
- The `lorentzian_q` parameterisation directly gives the physically meaningful **Q-factor**
  of each Schumann mode.
- The full pipeline applies unchanged to real KAGRA PEM magnetometer data.

### Next steps

- Load real data: `TimeSeries.read('K1:PEM-MAG_EXV_EAST_X_DQ', start, end)`.
- Compute multi-channel Schumann coherence with `BifrequencyMap.propagate()` to
  estimate the magnetic coupling contribution to DARM noise.
- Track daily/seasonal variations by looping over GPS time segments.
"""),
]


# ---------------------------------------------------------------------------
# Japanese cells
# ---------------------------------------------------------------------------

JA_CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# GW 検出器におけるシューマン共鳴解析

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main/docs/web/ja/user_guide/tutorials/case_schumann_resonance.ipynb)

**シューマン共鳴**は、地球と電離層のキャビティ内で落雷によって励起されるグローバルな電磁共鳴です。
約 7.83、14.3、20.8、26.4 Hz に鋭いスペクトルピークとして現れ、KAGRA などの第 2 世代重力波検出器の
感度帯域に直接重なります。

このノートブックでは gwexpy を用いたエンドツーエンドの特性評価ワークフローを示します:

1. **ブートストラップ PSD** (`bootstrap_spectrogram`) — 非対称信頼区間付きのロバストなスペクトル推定
2. **ローレンツ Q ファクターフィット** (`fit_series` + `lorentzian_q`) — 各モードの共鳴周波数・Q 値・振幅を計測
3. **共分散構造** (`BifrequencyMap`) — ブートストラップが返す周波数ビン間相関を可視化し、GLS フィットに利用
4. **時間追跡** — 観測ウィンドウ内でのモード振幅の時間変化を監視

> **前提知識**:
> - [スペクトログラム基礎](intro_spectrogram.ipynb)
> - [フィッティング上級編](advanced_fitting.ipynb)
> - [ブートストラップ PSD & GLS フィット](case_bootstrap_gls_fitting.ipynb)
"""),

    ("markdown", "## セットアップ"),
    ("code", """\
# ruff: noqa: I001
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from gwexpy.fitting import fit_series
from gwexpy.fitting.models import lorentzian_q
from gwexpy.frequencyseries import BifrequencyMap  # noqa: F401 (参照用)
from gwexpy.spectral import bootstrap_spectrogram
from gwexpy.timeseries import TimeSeries
"""),

    ("markdown", """\
## 1. モックデータ: シューマン共鳴を含む磁力計データ

地球の背景電磁場を測定する 1 軸磁力計 (ELF 帯) をシミュレートします。
4 つのシューマン共鳴を **ローレンツ型スペクトル形状の狭帯域ノイズ** として生成し、
広帯域ノイズフロアに重ね合わせます。

| モード | 周波数 | Q 値 | ピーク ASD |
|--------|--------|------|-----------|
| SR1    | 7.83 Hz | 5.0 | 4.0 nT/√Hz |
| SR2    | 14.3 Hz | 4.5 | 2.5 nT/√Hz |
| SR3    | 20.8 Hz | 4.0 | 1.5 nT/√Hz |
| SR4    | 26.4 Hz | 3.5 | 1.0 nT/√Hz |
"""),
    ("code", """\
rng = np.random.default_rng(42)
fs = 512        # サンプリングレート [Hz]
T  = 300.0      # 継続時間 [s]
n  = int(fs * T)

# シューマン共鳴パラメータ (地球-電離層キャビティモード)
SR_FREQS = [7.83, 14.3, 20.8, 26.4]   # Hz — 第 1〜4 モード
SR_Q     = [5.0,  4.5,  4.0,  3.5]    # Q 値
SR_AMP   = [4.0,  2.5,  1.5,  1.0]    # ピーク ASD [nT/√Hz]
NOISE_FLOOR = 0.3                       # 広帯域フロア [nT/√Hz]

# ── 周波数領域合成 ───────────────────────────────────────────────────────────
f = np.fft.rfftfreq(n, d=1.0 / fs)   # 片側周波数軸 [Hz]

# 目標 ASD をローレンツ和 + フロアで構築
asd_target = np.full_like(f, NOISE_FLOOR)
for f0, q, A in zip(SR_FREQS, SR_Q, SR_AMP):
    gamma = f0 / (2.0 * q)            # 半値半幅 [Hz]
    asd_target += A * gamma / np.sqrt((f - f0) ** 2 + gamma ** 2)

# rfft 振幅に変換: Welch PSD ≈ asd_target² となるよう正規化
amp = asd_target * np.sqrt(n * fs / 2.0)
amp[0] = 0.0  # DC 成分をゼロに

Z = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(2)
x = np.fft.irfft(amp * Z, n=n)

mag = TimeSeries(x, dt=1.0 / fs, unit=u.nT,
                 name="K1:PEM-MAG_EXV_EAST_X_DQ", t0=0)
print(f"継続時間: {T:.0f} s | fs: {fs} Hz | N: {n:,}")
"""),

    ("code", """\
# クイック確認: ASD でシューマン共鳴ピークを確認
fig, ax = plt.subplots(figsize=(10, 4))
asd_raw = mag.asd(fftlength=16.0, overlap=8.0)
ax.semilogy(asd_raw.frequencies.value, asd_raw.value, lw=0.8,
            color='steelblue', label='単一 ASD 推定 (16 s FFT)')

for f0, lab in zip(SR_FREQS, ['SR1', 'SR2', 'SR3', 'SR4']):
    ax.axvline(f0, color='red', ls='--', lw=0.8, alpha=0.7)
    ax.text(f0 + 0.15, 0.35, lab, color='red', fontsize=8)

ax.set_xlim(4, 35)
ax.set_xlabel('周波数 [Hz]')
ax.set_ylabel('ASD [nT/√Hz]')
ax.set_title('磁力計 ASD — シューマン共鳴の 4 ピークが確認できる')
ax.legend()
plt.tight_layout()
plt.show()
"""),

    ("markdown", """\
## 2. ブートストラップ PSD 推定

`bootstrap_spectrogram` はスペクトログラムの時間カラムをリサンプリングし、
**ロバストな PSD 推定値** (中央値または平均値) と非対称信頼区間を提供します。
`return_map=True` にすると、周波数ビン間の相関を定量化する **共分散 `BifrequencyMap`**
`cov_map(f1, f2)` も返します。これは次のステップの GLS フィットに使用されます。
"""),
    ("code", """\
# スペクトログラム計算: 16 s FFT、50 % ハン窓オーバーラップ
spec = mag.spectrogram2(fftlength=16.0, overlap=8.0, window='hann')
print(f"スペクトログラム shape: {spec.shape}  (n_times × n_freqs)")

# ブートストラップ — (PSD, 共分散 BifrequencyMap) を返す
psd_boot, cov_map = bootstrap_spectrogram(
    spec,
    n_boot=500,
    method='median',
    ci=0.68,
    fftlength=16.0,
    overlap=8.0,
    return_map=True,
    ignore_nan=True,
)
print(f"ブートストラップ PSD shape : {psd_boot.shape}")
print(f"共分散マップ shape: {cov_map.shape}  ← BifrequencyMap(f1, f2)")
"""),
    ("code", """\
# ブートストラップ PSD と ±1σ 信頼区間をプロット
fig, ax = plt.subplots(figsize=(10, 4))

f_psd = psd_boot.frequencies.value
y_psd = psd_boot.value

diag = cov_map.diagonal(method='mean')
y_var = np.interp(f_psd, diag.frequencies.value, np.abs(diag.value))
y_lo = np.sqrt(np.maximum(y_psd - np.sqrt(y_var), 1e-12))
y_hi = np.sqrt(y_psd + np.sqrt(y_var))

ax.semilogy(f_psd, np.sqrt(y_psd), lw=1.5, color='steelblue',
            label='ブートストラップ中央値 ASD')
ax.fill_between(f_psd, y_lo, y_hi, alpha=0.25, color='steelblue', label='±1σ')

for f0 in SR_FREQS:
    ax.axvline(f0, color='red', ls='--', lw=0.7, alpha=0.6)

ax.set_xlim(4, 35)
ax.set_xlabel('周波数 [Hz]')
ax.set_ylabel('ASD [nT/√Hz]')
ax.set_title('ブートストラップ PSD — 中央値 ± 1σ 信頼区間')
ax.legend()
plt.tight_layout()
plt.show()
"""),

    ("markdown", """\
## 3. ローレンツ Q ファクターフィット

各シューマンモードを **Q ファクター型ローレンツ分布** でモデル化します:

$$S(f) = \\frac{A\\,\\gamma^2}{(f - f_0)^2 + \\gamma^2}, \\quad \\gamma = \\frac{f_0}{2Q}$$

`fit_series` に `cov=cov_map` を渡すことで、オーバーラップ FFT セグメント間の
相関を考慮した **一般化最小二乗法 (GLS)** でフィットを行い、正しいパラメータ誤差が得られます。
"""),
    ("code", """\
# 各シューマンモードを専用の周波数ウィンドウで個別にフィット
fit_ranges = [(6.0, 10.5), (11.5, 17.5), (17.5, 24.5), (23.0, 30.5)]

fit_results = []
for i, (f0, q0, A0, (flo, fhi)) in enumerate(
        zip(SR_FREQS, SR_Q, SR_AMP, fit_ranges)):
    result = fit_series(
        psd_boot,
        lorentzian_q,
        x_range=(flo, fhi),
        cov=cov_map,
        p0={'A': A0 ** 2, 'x0': f0, 'Q': q0},
        limits={'A': (0, 500), 'x0': (flo, fhi), 'Q': (1.0, 100.0)},
    )
    fit_results.append(result)
    p, e = result.params, result.errors
    print(
        f"SR{i + 1}: f0 = {p['x0']:.3f} ± {e['x0']:.3f} Hz  |"
        f"  Q = {p['Q']:.2f} ± {e['Q']:.2f}  |"
        f"  A = {p['A']:.4f} ± {e['A']:.4f} nT²/Hz"
    )
"""),
    ("code", """\
# フィット済みローレンツ曲線をブートストラップ PSD に重ねてプロット
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
plot_ranges = [(5.5, 11.0), (11.5, 17.5), (17.5, 24.5), (22.5, 30.5)]
labels = ['SR1 (7.83 Hz)', 'SR2 (14.3 Hz)', 'SR3 (20.8 Hz)', 'SR4 (26.4 Hz)']

for ax, result, fr, color, label in zip(axes, fit_results, plot_ranges, colors, labels):
    psd_crop = psd_boot.crop(*fr)
    f_crop = psd_crop.frequencies.value
    y_crop = psd_crop.value

    ax.semilogy(f_crop, y_crop, '.', color=color, ms=2.5, label='ブートストラップ PSD')

    f_fine = np.linspace(*fr, 300)
    ax.semilogy(f_fine, lorentzian_q(f_fine, **result.params),
                'k-', lw=1.5, label='ローレンツフィット')

    p = result.params
    ax.set_title(f"{label}\\nf₀={p['x0']:.3f} Hz, Q={p['Q']:.2f}", fontsize=9)
    ax.set_xlabel('周波数 [Hz]')
    ax.legend(fontsize=7)

axes[0].set_ylabel('PSD [nT²/Hz]')
plt.suptitle('シューマン共鳴モードへの GLS ローレンツフィット (cov_map 使用)', y=1.01)
plt.tight_layout()
plt.show()
"""),

    ("markdown", """\
## 4. BifrequencyMap による共分散構造の解析

`bootstrap_spectrogram` が返す共分散マップ `cov_map(f1, f2)` は、
**異なる周波数のスペクトル推定値がどの程度相関しているか** を表します。
狭帯域共鳴を持つ磁力計データでは以下が期待されます:

- **共鳴周波数における対角線沿いの高共分散** (`f1 ≈ f2`) — オーバーラップ FFT ウィンドウから生じる隣接ビン間の相関
- **共鳴付近のオフ対角構造** — ピークが 1 ビンを超えて広がることを反映

ここで使用する `BifrequencyMap` メソッド:
- `.diagonal(method='mean')` — 分散プロファイルを返す (行列の対角線)
- `.get_slice(at=f0, axis='f1')` — f1 固定での 1 次元共分散スライスを返す
"""),
    ("code", """\
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── 左: 2-D 共分散マップ (4〜35 Hz) ──────────────────────────────────────
f_lo, f_hi = 4.0, 35.0
f_vals = cov_map.frequency1.value
mask = (f_vals >= f_lo) & (f_vals <= f_hi)
cov_sub = np.abs(cov_map.value[np.ix_(mask, mask)])
f_sub   = f_vals[mask]

im = axes[0].pcolormesh(f_sub, f_sub, cov_sub,
                         norm=LogNorm(vmin=cov_sub[cov_sub > 0].min(), vmax=cov_sub.max()),
                         cmap='inferno', shading='auto')
fig.colorbar(im, ax=axes[0], label='|共分散| [nT⁴/Hz²]')

for f0 in SR_FREQS:
    axes[0].axvline(f0, color='cyan', lw=0.6, ls='--')
    axes[0].axhline(f0, color='cyan', lw=0.6, ls='--')

axes[0].set_xlabel('f₁ [Hz]')
axes[0].set_ylabel('f₂ [Hz]')
axes[0].set_title('ブートストラップ共分散マップ (4〜35 Hz)')

# ── 右: 対角線 = ビン毎の分散 ────────────────────────────────────────────
diag = cov_map.diagonal(method='mean')
f_d = diag.frequencies.value
m_d = (f_d >= f_lo) & (f_d <= f_hi)
axes[1].semilogy(f_d[m_d], np.abs(diag.value[m_d]), lw=1.2, color='steelblue')

for f0 in SR_FREQS:
    axes[1].axvline(f0, color='red', ls='--', lw=0.7)

axes[1].set_xlim(f_lo, f_hi)
axes[1].set_xlabel('周波数 [Hz]')
axes[1].set_ylabel('対角分散 [nT⁴/Hz²]')
axes[1].set_title('共分散マップの対角線 (≈ PSD² / N_bootstrap)')

plt.tight_layout()
plt.show()
"""),
    ("code", """\
# SR1 (7.83 Hz) における共分散スライス — 共鳴の周波数幅を反映
cov_slice = cov_map.get_slice(at=7.83, axis='f1')
f_sl = cov_slice.frequencies.value
m_sl = (f_sl >= f_lo) & (f_sl <= f_hi)

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(f_sl[m_sl], np.abs(cov_slice.value[m_sl]), lw=1.2, color='darkorange')
ax.axvline(7.83, color='red', ls='--', lw=1, label='SR1 (7.83 Hz)')
for f0 in SR_FREQS[1:]:
    ax.axvline(f0, color='gray', ls=':', lw=0.7)

ax.set_xlim(f_lo, f_hi)
ax.set_xlabel('周波数 f₂ [Hz]')
ax.set_ylabel('|Cov(f₁=7.83 Hz, f₂)| [nT⁴/Hz²]')
ax.set_title('SR1 における共分散スライス — ピーク幅が共鳴の帯域幅を反映')
ax.legend()
plt.tight_layout()
plt.show()
"""),

    ("markdown", """\
## 5. 振幅の時間追跡

シューマン共鳴の振幅は世界的な落雷活動に応じて変動します (日変化・季節変化)。
スペクトログラムから各モード周辺の帯域内平均パワーを抽出し、
**ASD 時系列**に変換して時間変化を監視します。
"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(10, 4))
times_s = spec.times.value
colors_t = ['steelblue', 'darkorange', 'seagreen', 'crimson']

for f0, color, label in zip(SR_FREQS, colors_t,
                             ['SR1 7.83 Hz', 'SR2 14.3 Hz',
                              'SR3 20.8 Hz', 'SR4 26.4 Hz']):
    # ±0.5 Hz 帯域内の狭帯域パワー
    spec_band = spec.crop_frequencies(f0 - 0.5, f0 + 0.5)
    amp_t = np.sqrt(spec_band.value.mean(axis=1))
    ax.plot(times_s, amp_t, lw=0.8, color=color, label=label)

ax.set_xlabel('時間 [s]')
ax.set_ylabel('帯域内平均 ASD [nT/√Hz]')
ax.set_title('シューマン共鳴振幅の時間変化 (300 s モック観測)')
ax.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()
"""),

    ("markdown", """\
## まとめ

| ステップ | ツール | 出力 |
|---------|--------|------|
| ロバスト PSD | `bootstrap_spectrogram(return_map=True)` | 中央値 PSD + 1σ 帯 |
| 不確かさモデル | `BifrequencyMap` (ブートストラップから) | 周波数ビン間共分散行列 |
| ピーク特性評価 | `fit_series(lorentzian_q, cov=cov_map)` | 各モードの f₀、Q、振幅 |
| 時間監視 | `Spectrogram.crop_frequencies()` | 振幅時系列 |

### ポイント

- **ブートストラップリサンプリング**は、単一セグメント推定がノイジーな狭帯域ピークに対して
  信頼できる不確かさ推定を与えます。
- **BifrequencyMap 共分散**は GLS フィットに不可欠です。`fit_series` に `cov=cov_map` を
  渡すことで、周波数ビン間の相関が自動的に考慮されます。
- `lorentzian_q` パラメタライゼーションにより、各シューマンモードの物理的に意味のある
  **Q ファクター**が直接得られます。
- このワークフローは実際の KAGRA PEM 磁力計データにそのまま適用できます。

### 次のステップ

- 実データの読み込み: `TimeSeries.read('K1:PEM-MAG_EXV_EAST_X_DQ', start, end)`.
- `BifrequencyMap.propagate()` と組み合わせて、DARM ノイズへの磁気結合寄与を推定する。
- GPS 時間セグメントをループして、シューマン共鳴パラメータの日変化・季節変化を追跡する。
"""),
]


# ---------------------------------------------------------------------------
# Generate notebooks
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    out_en = repo_root / "docs/web/en/user_guide/tutorials/case_schumann_resonance.ipynb"
    out_ja = repo_root / "docs/web/ja/user_guide/tutorials/case_schumann_resonance.ipynb"

    for path, cells in [(out_en, EN_CELLS), (out_ja, JA_CELLS)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        nb = make_nb(cells)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
        print(f"Written: {path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
