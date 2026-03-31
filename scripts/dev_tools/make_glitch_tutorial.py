"""Generate glitch analysis (Q-transform/Omega-scan) tutorial notebooks (EN + JA)."""

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
# Glitch Analysis: Q-Transform and Omega-Scan

**Glitches** are short-duration, non-Gaussian noise transients that contaminate
gravitational-wave data.  The **Q-transform** (or Omega scan) is the standard
time-frequency representation used to characterise and classify glitches at all
major detector sites.

The Q-transform tiles the time-frequency plane with constant-Q (constant
bandwidth-to-frequency ratio) windows and measures the normalised energy in each
tile.  A glitch appears as a cluster of tiles with anomalously high SNR.

**What this tutorial covers:**
1. Injecting synthetic glitches into a noise background
2. Computing the Q-transform with `q_scan()` and `QTiling`
3. Visualising the time-frequency map (Omega scan)
4. Extracting peak SNR, time, and frequency from `QGram`
5. Characterising glitch morphology (duration, bandwidth, Q)

**Related tutorials:**
- `advanced_peak_detection.ipynb` — spectral line detection
- `advanced_peak_tracking.ipynb` — tracking lines over time
- `case_bruco_advanced.ipynb` — coherence-based noise coupling
"""),

    md("## Setup"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt

from gwexpy.timeseries import TimeSeries
from gwexpy.signal.qtransform import q_scan, QTiling, DEFAULT_QRANGE
from gwexpy.signal import whiten
from gwexpy.statistics.rayleigh_test import rayleigh_test
"""),

    md("""\
## 1. Synthetic Data with Injected Glitches

We create a 4 s segment of KAGRA-like DARM noise with three injected glitches
representative of common morphologies seen in practice:

| Glitch | Morphology | Typical origin |
|--------|-----------|----------------|
| Blip | Short broadband burst | Unknown / scattered light |
| Scattered light | Arch-shaped frequency sweep | Mirror motion + scattering |
| Koi fish | Low-frequency with harmonic | Suspension resonance |
"""),

    code("""\
fs   = 4096.0    # sample rate [Hz]
T    = 4.0       # segment duration [s]
N    = int(T * fs)
t    = np.arange(N) / fs
t0   = 1_300_000_000
rng  = np.random.default_rng(7)

# --- Gaussian noise coloured to LIGO-like O3 sensitivity ---
freqs_n = np.fft.rfftfreq(N, 1.0 / fs)[1:]
# Simplified ASD: f^{-2} below 30 Hz, flat above (units: normalised)
asd = np.where(freqs_n < 30, (30.0 / freqs_n)**2, 1.0)
noise_fft = asd * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(freqs_n)))
noise_fft = np.concatenate([[0.0], noise_fft])
noise = np.fft.irfft(noise_fft, n=N)
noise /= noise.std()   # normalise to unit RMS

# --- Glitch 1: Blip at t=1.0 s, ~100 Hz, Q~8 ---
t_blip = 1.0
f_blip, Q_blip = 100.0, 8.0
tau_blip = Q_blip / (np.pi * f_blip)
envelope_blip = np.exp(-0.5 * ((t - t_blip) / tau_blip)**2)
glitch_blip = 15.0 * envelope_blip * np.sin(2*np.pi*f_blip*(t - t_blip))

# --- Glitch 2: Scattered-light arch at t=2.0–2.5 s ---
# Frequency sweeps as f(t) = f0 * |sin(pi*t/P)|  (mirror pendulum)
t_sl = np.linspace(1.8, 2.6, 500)
f_sl = 50.0 * np.abs(np.sin(np.pi * (t_sl - 1.8) / 0.8))
phi_sl = 2*np.pi * np.cumsum(f_sl) / fs * (t_sl[1] - t_sl[0]) * fs
glitch_sl = np.zeros(N)
idx_sl = ((t_sl - t[0]) * fs).astype(int)
idx_sl = idx_sl[(idx_sl >= 0) & (idx_sl < N)]
glitch_sl[idx_sl[:len(phi_sl[:len(idx_sl)])]] = 12.0 * np.sin(phi_sl[:len(idx_sl)])

# --- Glitch 3: Koi-fish (low-freq + harmonics) at t=3.0 s ---
t_koi = 3.0
f_koi, tau_koi = 20.0, 0.08
env_koi = np.exp(-((t - t_koi) / tau_koi)**2)
glitch_koi = (8.0 * env_koi * np.sin(2*np.pi*f_koi*t) +
              4.0 * env_koi * np.sin(2*np.pi*2*f_koi*t) +
              2.0 * env_koi * np.sin(2*np.pi*3*f_koi*t))

signal = noise + glitch_blip + glitch_sl + glitch_koi
ts = TimeSeries(signal, t0=t0, sample_rate=fs, name="K1:LSC-DARM_OUT_DQ")

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t, signal, lw=0.5, color="steelblue", alpha=0.8)
ax.axvline(t_blip, color="red",   ls="--", lw=0.8, label="Blip")
ax.axvline(2.2,    color="orange", ls="--", lw=0.8, label="Scattered light")
ax.axvline(t_koi,  color="green",  ls="--", lw=0.8, label="Koi fish")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalised strain")
ax.set_title("4 s DARM segment with injected glitches")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 2. Q-Transform (Omega Scan)

`q_scan()` searches over a range of Q values and returns the tile with the
highest normalised energy together with the full `QGram` object for plotting.

Key parameters:
- `qrange` — range of Q values to search (default: 4–64)
- `frange` — frequency range in Hz
- `mismatch` — fractional tile overlap (smaller = finer grid, slower)
"""),

    code("""\
# Whiten the data first (Q-transform assumes white noise for SNR normalisation)
ts_white = whiten(ts, fftlength=1.0, overlap=0.5)

# Run Q-scan over the full segment
qgram, far = q_scan(
    ts_white,
    qrange   = (4, 64),
    frange   = (10, 2000),
    mismatch = 0.2,
)

print("Q-scan result:")
print(f"  Peak SNR      : {qgram.peak['energy']:.1f}")
print(f"  Peak time     : t0 + {qgram.peak['time'] - t0:.3f} s")
print(f"  Peak frequency: {qgram.peak['frequency']:.1f} Hz")
print(f"  Peak Q        : {qgram.peak.get('q', 'N/A')}")
print(f"  Estimated FAR : {far:.2e} Hz")
"""),

    md("""\
## 3. Time-Frequency Map (Omega Scan Plot)

`QGram.interpolate()` resamples the irregular Q tiles onto a regular
time-frequency grid suitable for imshow.
"""),

    code("""\
# Interpolate to a regular (time, freq) grid
# duration and sampling control the output resolution
qscan_interp = qgram.interpolate(dt=1.0/fs, df=1.0)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(
    qscan_interp.T,
    origin="lower",
    aspect="auto",
    extent=[t[0], t[-1], 10, 2000],
    vmin=0, vmax=25,
    cmap="viridis",
)
ax.set_yscale("log")
ax.set_xlabel("Time [s relative to epoch]")
ax.set_ylabel("Frequency [Hz]")
ax.set_title("Omega Scan — 4 s DARM segment")

cb = plt.colorbar(im, ax=ax)
cb.set_label("Normalised energy (SNR²)")

# Annotate glitch locations
ax.axvline(t_blip, color="red",   ls="--", lw=1, alpha=0.7, label="Blip")
ax.axvline(2.2,    color="orange", ls="--", lw=1, alpha=0.7, label="Scattered light")
ax.axvline(t_koi,  color="lime",   ls="--", lw=1, alpha=0.7, label="Koi fish")
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.show()
"""),

    md("""\
## 4. Glitch Characterisation with QTiling

`QTiling` lets us search a specific Q plane and extract glitch parameters
(peak time, frequency, duration, bandwidth) for each candidate event.
"""),

    code("""\
# Search around each expected glitch location
glitch_windows = [
    ("Blip",            0.8,  1.2,  50,  500, (4,  20)),
    ("Scattered light", 1.7,  2.7,  10,  200, (4,  16)),
    ("Koi fish",        2.8,  3.3,  10,  150, (16, 64)),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, t_lo, t_hi, f_lo, f_hi, qr) in zip(axes, glitch_windows):
    # Extract sub-segment
    i0, i1 = int(t_lo*fs), int(t_hi*fs)
    ts_seg = TimeSeries(ts_white.value[i0:i1], t0=t0+t_lo,
                        sample_rate=fs, name=name)

    qg, _ = q_scan(ts_seg, qrange=qr, frange=(f_lo, f_hi), mismatch=0.15)
    qi = qg.interpolate(dt=2.0/fs, df=2.0)

    ax.imshow(qi.T, origin="lower", aspect="auto", cmap="inferno",
              extent=[t_lo, t_hi, f_lo, f_hi], vmin=0, vmax=20)
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"{name}\\nPeak SNR={qg.peak['energy']:.1f}, "
                 f"f={qg.peak['frequency']:.0f} Hz")

plt.suptitle("Glitch Morphology Comparison", fontsize=12)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 5. Statistical Significance — Rayleigh Test

The `rayleigh_test` measures how non-Gaussian the time-frequency energy
distribution is.  A p-value below 0.01 in a quiet background suggests
the data segment contains a significant non-Gaussian transient.
"""),

    code("""\
# Compare a clean segment vs. a glitchy segment
ts_clean = TimeSeries(noise, t0=t0, sample_rate=fs, name="clean")
ts_clean_w = whiten(ts_clean, fftlength=1.0)
ts_glitch_w = ts_white

qg_clean,  _ = q_scan(ts_clean_w,  qrange=(4,64), frange=(20,500))
qg_glitch, _ = q_scan(ts_glitch_w, qrange=(4,64), frange=(20,500))

# Rayleigh test on normalised energies across tiles
# (simulated as white chi-squared background)
energies_clean  = np.clip(qg_clean.value.ravel(),  0, None)
energies_glitch = np.clip(qg_glitch.value.ravel(), 0, None)

r_clean  = rayleigh_test(energies_clean[energies_clean  > 0])
r_glitch = rayleigh_test(energies_glitch[energies_glitch > 0])

print(f"Clean segment  — Rayleigh statistic: {r_clean.statistic:.4f}, "
      f"p-value: {r_clean.pvalue:.4f}")
print(f"Glitchy segment — Rayleigh statistic: {r_glitch.statistic:.4f}, "
      f"p-value: {r_glitch.pvalue:.6f}")
print()
if r_glitch.pvalue < 0.01:
    print("Non-Gaussian transient detected (p < 0.01)")
else:
    print("Segment appears Gaussian at 1% level")
"""),

    md("""\
## Summary

| Step | API | Output |
|------|-----|--------|
| Whiten data | `whiten(ts, fftlength, overlap)` | Whitened TimeSeries |
| Q-scan | `q_scan(ts, qrange, frange, mismatch)` | QGram, FAR |
| Plot Omega scan | `qgram.interpolate(dt, df)` | 2D energy array |
| Glitch parameters | `qgram.peak` | time, frequency, SNR, Q |
| Statistical test | `rayleigh_test(energies)` | statistic, p-value |

**Glitch morphology guide:**

| Type | Frequency sweep | Duration | Q range |
|------|----------------|---------|---------|
| Blip | None | < 10 ms | 4–20 |
| Scattered light | Arch / inverted arch | 0.1–1 s | 4–16 |
| Koi fish | Low-freq + harmonics | 50–200 ms | 16–64 |
| Whistle | Monotone sweep | 0.1–2 s | 20–100 |

**Tip**: For production glitch cataloguing, use `q_scan` with the default
`qrange=(4,64)` and then manually inspect tiles with SNR > 8.
"""),
]


# ---------------------------------------------------------------------------
# Japanese notebook
# ---------------------------------------------------------------------------

JA_CELLS = [
    md("""\
# グリッチ詳細解析：Q-変換と Omega スキャン

**グリッチ**は重力波データを汚染する短時間・非ガウス性ノイズ過渡現象です。
**Q-変換**（Omega スキャン）は全主要検出器サイトでグリッチの特性評価と
分類に使われる標準的な時間周波数表現です。

Q-変換は一定Q（一定の帯域幅対周波数比）ウィンドウで時間周波数平面を
タイル化し、各タイルの正規化エネルギーを測定します。
グリッチは異常に高い SNR を持つタイルのクラスターとして現れます。

**このチュートリアルで学ぶこと：**
1. ノイズ背景に合成グリッチを注入する
2. `q_scan()` と `QTiling` で Q-変換を計算する
3. 時間周波数マップ（Omega スキャン）を可視化する
4. `QGram` からピーク SNR・時刻・周波数を抽出する
5. グリッチの形態（継続時間、帯域幅、Q）を特定する
"""),

    md("## セットアップ"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt

from gwexpy.timeseries import TimeSeries
from gwexpy.signal.qtransform import q_scan, QTiling, DEFAULT_QRANGE
from gwexpy.signal import whiten
from gwexpy.statistics.rayleigh_test import rayleigh_test
"""),

    md("""\
## 1. グリッチを含む合成データの生成

実際に観測される典型的なグリッチ3種類を合成します：

| グリッチ | 形態 | 典型的な原因 |
|---------|-----|------------|
| Blip | 短時間広帯域バースト | 不明 / 散乱光 |
| 散乱光 | アーチ状周波数スウィープ | ミラー運動 + 散乱 |
| 鯉（Koi fish）| 低周波 + 高調波 | 懸架共振 |
"""),

    code("""\
fs   = 4096.0
T    = 4.0
N    = int(T * fs)
t    = np.arange(N) / fs
t0   = 1_300_000_000
rng  = np.random.default_rng(7)

freqs_n = np.fft.rfftfreq(N, 1.0 / fs)[1:]
asd = np.where(freqs_n < 30, (30.0 / freqs_n)**2, 1.0)
noise_fft = asd * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(freqs_n)))
noise_fft = np.concatenate([[0.0], noise_fft])
noise = np.fft.irfft(noise_fft, n=N)
noise /= noise.std()

# Blip グリッチ (t=1.0 s)
t_blip = 1.0
f_blip, Q_blip = 100.0, 8.0
tau_blip = Q_blip / (np.pi * f_blip)
envelope_blip = np.exp(-0.5 * ((t - t_blip) / tau_blip)**2)
glitch_blip = 15.0 * envelope_blip * np.sin(2*np.pi*f_blip*(t - t_blip))

# 散乱光グリッチ (t=1.8–2.6 s)
t_sl = np.linspace(1.8, 2.6, 500)
f_sl = 50.0 * np.abs(np.sin(np.pi * (t_sl - 1.8) / 0.8))
phi_sl = 2*np.pi * np.cumsum(f_sl) / fs * (t_sl[1] - t_sl[0]) * fs
glitch_sl = np.zeros(N)
idx_sl = ((t_sl - t[0]) * fs).astype(int)
idx_sl = idx_sl[(idx_sl >= 0) & (idx_sl < N)]
glitch_sl[idx_sl[:len(phi_sl[:len(idx_sl)])]] = 12.0 * np.sin(phi_sl[:len(idx_sl)])

# 鯉グリッチ (t=3.0 s)
t_koi = 3.0
f_koi, tau_koi = 20.0, 0.08
env_koi = np.exp(-((t - t_koi) / tau_koi)**2)
glitch_koi = (8.0 * env_koi * np.sin(2*np.pi*f_koi*t) +
              4.0 * env_koi * np.sin(2*np.pi*2*f_koi*t) +
              2.0 * env_koi * np.sin(2*np.pi*3*f_koi*t))

signal = noise + glitch_blip + glitch_sl + glitch_koi
ts = TimeSeries(signal, t0=t0, sample_rate=fs, name="K1:LSC-DARM_OUT_DQ")

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t, signal, lw=0.5, color="steelblue", alpha=0.8)
ax.axvline(t_blip, color="red",   ls="--", lw=0.8, label="Blip")
ax.axvline(2.2,    color="orange", ls="--", lw=0.8, label="散乱光")
ax.axvline(t_koi,  color="green",  ls="--", lw=0.8, label="鯉")
ax.set_xlabel("時間 [s]")
ax.set_ylabel("正規化歪み")
ax.set_title("グリッチを含む 4 s DARM セグメント")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 2. Q-変換（Omega スキャン）

`q_scan()` は Q 値の範囲を探索し、最高正規化エネルギーを持つタイルと
プロット用の完全な `QGram` オブジェクトを返します。

主要パラメータ：
- `qrange` — 探索する Q 値の範囲（デフォルト: 4〜64）
- `frange` — 周波数範囲 [Hz]
- `mismatch` — タイルオーバーラップの割合（小さいほど細かいグリッド）
"""),

    code("""\
ts_white = whiten(ts, fftlength=1.0, overlap=0.5)

qgram, far = q_scan(
    ts_white,
    qrange   = (4, 64),
    frange   = (10, 2000),
    mismatch = 0.2,
)

print("Q-スキャン結果:")
print(f"  ピーク SNR      : {qgram.peak['energy']:.1f}")
print(f"  ピーク時刻      : t0 + {qgram.peak['time'] - t0:.3f} s")
print(f"  ピーク周波数    : {qgram.peak['frequency']:.1f} Hz")
print(f"  推定 FAR        : {far:.2e} Hz")
"""),

    md("## 3. 時間周波数マップ（Omega スキャンプロット）"),

    code("""\
qscan_interp = qgram.interpolate(dt=1.0/fs, df=1.0)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(
    qscan_interp.T, origin="lower", aspect="auto", cmap="viridis",
    extent=[t[0], t[-1], 10, 2000], vmin=0, vmax=25,
)
ax.set_yscale("log")
ax.set_xlabel("時間 [s（エポックからの相対時刻）]")
ax.set_ylabel("周波数 [Hz]")
ax.set_title("Omega スキャン — 4 s DARM セグメント")

cb = plt.colorbar(im, ax=ax)
cb.set_label("正規化エネルギー（SNR²）")

ax.axvline(t_blip, color="red",   ls="--", lw=1, alpha=0.7, label="Blip")
ax.axvline(2.2,    color="orange", ls="--", lw=1, alpha=0.7, label="散乱光")
ax.axvline(t_koi,  color="lime",   ls="--", lw=1, alpha=0.7, label="鯉")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.show()
"""),

    md("## 4. QTiling によるグリッチ特性評価"),

    code("""\
glitch_windows = [
    ("Blip",   0.8, 1.2,  50,  500, (4,  20)),
    ("散乱光", 1.7, 2.7,  10,  200, (4,  16)),
    ("鯉",     2.8, 3.3,  10,  150, (16, 64)),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, t_lo, t_hi, f_lo, f_hi, qr) in zip(axes, glitch_windows):
    i0, i1 = int(t_lo*fs), int(t_hi*fs)
    ts_seg = TimeSeries(ts_white.value[i0:i1], t0=t0+t_lo,
                        sample_rate=fs, name=name)
    qg, _ = q_scan(ts_seg, qrange=qr, frange=(f_lo, f_hi), mismatch=0.15)
    qi = qg.interpolate(dt=2.0/fs, df=2.0)

    ax.imshow(qi.T, origin="lower", aspect="auto", cmap="inferno",
              extent=[t_lo, t_hi, f_lo, f_hi], vmin=0, vmax=20)
    ax.set_yscale("log")
    ax.set_xlabel("時間 [s]")
    ax.set_ylabel("周波数 [Hz]")
    ax.set_title(f"{name}\\nピーク SNR={qg.peak['energy']:.1f}, "
                 f"f={qg.peak['frequency']:.0f} Hz")

plt.suptitle("グリッチ形態比較", fontsize=12)
plt.tight_layout()
plt.show()
"""),

    md("## 5. 統計的有意性 — Rayleigh 検定"),

    code("""\
ts_clean = TimeSeries(noise, t0=t0, sample_rate=fs, name="clean")
ts_clean_w = whiten(ts_clean, fftlength=1.0)

qg_clean,  _ = q_scan(ts_clean_w,  qrange=(4,64), frange=(20,500))
qg_glitch, _ = q_scan(ts_white,    qrange=(4,64), frange=(20,500))

energies_clean  = np.clip(qg_clean.value.ravel(),  0, None)
energies_glitch = np.clip(qg_glitch.value.ravel(), 0, None)

r_clean  = rayleigh_test(energies_clean[energies_clean   > 0])
r_glitch = rayleigh_test(energies_glitch[energies_glitch > 0])

print(f"クリーンセグメント — Rayleigh 統計量: {r_clean.statistic:.4f}, "
      f"p 値: {r_clean.pvalue:.4f}")
print(f"グリッチセグメント — Rayleigh 統計量: {r_glitch.statistic:.4f}, "
      f"p 値: {r_glitch.pvalue:.6f}")
if r_glitch.pvalue < 0.01:
    print("非ガウス性過渡現象を検出（p < 0.01）")
else:
    print("1% 水準でガウス分布と判定")
"""),

    md("""\
## まとめ

| ステップ | API | 出力 |
|---------|-----|------|
| データのホワイトニング | `whiten(ts, fftlength, overlap)` | ホワイトニング済み TimeSeries |
| Q スキャン | `q_scan(ts, qrange, frange, mismatch)` | QGram, FAR |
| Omega スキャンプロット | `qgram.interpolate(dt, df)` | 2D エネルギー配列 |
| グリッチパラメータ | `qgram.peak` | 時刻、周波数、SNR、Q |
| 統計検定 | `rayleigh_test(energies)` | 統計量、p 値 |

**グリッチ形態ガイド：**

| 種類 | 周波数スウィープ | 継続時間 | Q 範囲 |
|------|----------------|---------|-------|
| Blip | なし | < 10 ms | 4–20 |
| 散乱光 | アーチ状 | 0.1–1 s | 4–16 |
| 鯉 | 低周波 + 高調波 | 50–200 ms | 16–64 |
| ホイッスル | 単調スウィープ | 0.1–2 s | 20–100 |
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
    write_nb(EN_CELLS, root / "docs/web/en/user_guide/tutorials/case_glitch_analysis.ipynb")
    write_nb(JA_CELLS, root / "docs/web/ja/user_guide/tutorials/case_glitch_analysis.ipynb")
    print("Done.")
