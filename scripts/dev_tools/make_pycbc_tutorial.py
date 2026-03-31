"""Generate PyCBC interoperability tutorial notebooks (EN + JA)."""

import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "id": f"md_{abs(hash(source))%10**8:08x}",
            "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None,
            "id": f"cd_{abs(hash(source))%10**8:08x}",
            "metadata": {}, "outputs": [], "source": source}


EN_CELLS = [
    md("""\
# PyCBC Interoperability: From gwexpy Preprocessing to GW Search

[PyCBC](https://pycbc.org/) is a Python toolkit for gravitational-wave
data analysis, including matched-filter searches for compact binary
coalescences (CBC).  gwexpy provides **bidirectional converters** between
its own data types and PyCBC's `TimeSeries` / `FrequencySeries`.

This enables a natural workflow where:
- **gwexpy** handles data loading, preprocessing, and noise characterisation
- **PyCBC** performs the matched-filter search or parameter estimation
- **gwexpy** post-processes and visualises the search output

**What this tutorial covers:**
1. Converting a gwexpy `TimeSeries` to PyCBC and back
2. Converting a gwexpy `FrequencySeries` (ASD) to a PyCBC PSD
3. Full preprocessing pipeline: conditioning → matched filter (simulated CBC)
4. Visualising matched-filter SNR time series with gwexpy

> **Note**: This notebook works without a PyCBC installation.
> When PyCBC is absent the matched-filter step uses a simplified
> analytic substitute so every cell still executes.
> Install PyCBC with `pip install pycbc` for real searches.
"""),

    md("## Setup"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt

from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.interop.pycbc_ import (
    to_pycbc_timeseries, from_pycbc_timeseries,
    to_pycbc_frequencyseries, from_pycbc_frequencyseries,
)

PYCBC_AVAILABLE = False
try:
    import pycbc
    import pycbc.types
    import pycbc.filter
    import pycbc.waveform
    import pycbc.psd
    PYCBC_AVAILABLE = True
    print(f"PyCBC {pycbc.__version__} found — running real matched filter.")
except ImportError:
    print("PyCBC not installed — using analytic fallback for matched filter.")
"""),

    md("""\
## 1. Synthetic DARM Data with an Injected CBC Signal

We create a 64 s segment of coloured noise and inject a synthetic
binary neutron star (BNS) inspiral — a chirp signal that sweeps from
~30 Hz to ~1000 Hz in the last few seconds before merger.
"""),

    code("""\
fs   = 4096.0
T    = 64.0
N    = int(T * fs)
t0   = 1_300_000_000   # GPS
rng  = np.random.default_rng(0)
t    = np.arange(N) / fs

# --- Coloured noise (LIGO-like O3 sensitivity) ---
freqs_n = np.fft.rfftfreq(N, 1.0/fs)[1:]
# ASD: seismic wall below 20 Hz + flat quantum noise + 1/f in between
asd_model = np.where(freqs_n < 20, (20/freqs_n)**4,
            np.where(freqs_n < 200, (200/freqs_n)**0.5, 1.0))
fft = asd_model * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(freqs_n)))
noise = np.fft.irfft(np.concatenate([[0.0], fft]), n=N)
noise *= 1e-23   # strain-like amplitude

# --- Synthetic BNS chirp (analytical post-Newtonian 0-th order) ---
# h(t) ~ A(t) * cos(phi(t))  where phi ~ (t_c - t)^(5/8)
M_chirp = 1.2   # chirp mass [Msun]  -> sets chirp duration
G_c3 = 4.926e-6   # G*Msun/c^3 [s]
eta = 0.25        # symmetric mass ratio (equal masses)
tc  = T - 0.5    # merger time [s]
tau = np.maximum(tc - t, 1e-4)

# Instantaneous frequency and phase (PN 0th order)
f_gw  = (5.0/(256.0 * np.pi**(8.0/3))) ** (3.0/8) * (G_c3 * M_chirp)**(-5.0/8) * tau**(-3.0/8)
phi_gw = -2.0 * (tau / (5.0 * G_c3 * M_chirp))**(5.0/8) / np.pi

# Amplitude ~ tau^(-1/4)
amp_gw = 1e-22 * (G_c3 * M_chirp / tau)**(1.0/4)

# Only keep the inspiral up to fISCO ~ 1500 Hz
mask = f_gw < 1500.0
chirp = np.where(mask, amp_gw * np.cos(phi_gw), 0.0)

ts_strain = TimeSeries(noise + chirp, t0=t0, sample_rate=fs,
                       name="K1:LSC-DARM_OUT_DQ", unit="strain")

print(f"Chirp starts at t = {t[mask][0]:.1f} s  "
      f"(freq = {f_gw[mask][0]:.1f} Hz)")
print(f"Chirp ends   at t = {tc:.1f} s  (merger)")

# Quick plot
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t[-int(3*fs):], ts_strain.value[-int(3*fs):], lw=0.5, color="steelblue")
ax.set_xlabel("Time [s from t0]")
ax.set_ylabel("Strain")
ax.set_title("Last 3 s: DARM noise + BNS inspiral injection")
plt.tight_layout()
plt.show()
"""),

    md("""\
## 2. gwexpy → PyCBC Conversion

`to_pycbc_timeseries()` maps a gwexpy `TimeSeries` to a
`pycbc.types.TimeSeries`.  Metadata (t0, dt, unit) is preserved.
"""),

    code("""\
if PYCBC_AVAILABLE:
    # --- Convert to PyCBC ---
    ts_pycbc = to_pycbc_timeseries(ts_strain)
    print(f"pycbc.TimeSeries:")
    print(f"  delta_t   : {ts_pycbc.delta_t}")
    print(f"  start_time: {float(ts_pycbc.start_time):.1f}")
    print(f"  len       : {len(ts_pycbc)}")

    # --- Convert back to gwexpy ---
    ts_back = from_pycbc_timeseries(TimeSeries, ts_pycbc)
    print(f"\\nRoundtrip max error: "
          f"{np.max(np.abs(ts_strain.value - ts_back.value)):.2e}")
else:
    print("PyCBC not available — conversion demo skipped.")
    print("The to/from functions accept pycbc.types.TimeSeries objects.")
    ts_back = ts_strain   # identity for subsequent cells
"""),

    md("""\
## 3. PSD / ASD Conversion

A matched filter requires a **noise PSD** to whiten the data.
We estimate the ASD from the gwexpy `TimeSeries` and convert it to
a PyCBC `FrequencySeries` for use as the matched-filter PSD.
"""),

    code("""\
# Estimate ASD from gwexpy (use a quiet segment before the chirp)
ts_quiet = TimeSeries(ts_strain.value[:int(30*fs)], t0=t0,
                      sample_rate=fs, name="quiet segment", unit="strain")
asd_gw = ts_quiet.asd(fftlength=4.0, method="median")

print(f"gwexpy ASD: {len(asd_gw)} bins, df={asd_gw.df.value:.4f} Hz")

if PYCBC_AVAILABLE:
    # PyCBC expects PSD (not ASD); convert ASD -> PSD FrequencySeries
    fs_pycbc = to_pycbc_frequencyseries(asd_gw)
    psd_pycbc = pycbc.types.FrequencySeries(
        fs_pycbc.numpy()**2, delta_f=fs_pycbc.delta_f,
        epoch=fs_pycbc.epoch,
    )
    print(f"PyCBC PSD: {len(psd_pycbc)} bins, df={psd_pycbc.delta_f:.4f} Hz")
else:
    print("PyCBC not available — PSD conversion demo skipped.")
"""),

    md("""\
## 4. Matched-Filter Search (BNS Template)

A matched filter computes the cross-correlation between the data and a
template waveform, normalised by the noise PSD.  The peak of the
normalised SNR time series $\\rho(t)$ indicates the merger time.

$$\\rho(t) = \\frac{4}{\\sigma} \\, \\text{Re} \\int_0^\\infty \\frac{\\tilde{d}(f)\\,\\tilde{h}^*(f)}{S_n(f)} e^{2\\pi i f t} \\, df$$
"""),

    code("""\
if PYCBC_AVAILABLE:
    # Generate BNS template (equal mass, m1=m2=1.4 Msun)
    hp, hc = pycbc.waveform.get_td_waveform(
        approximant="TaylorT4",
        mass1=1.4, mass2=1.4,
        delta_t=1.0/fs,
        f_lower=30.0,
    )

    # Matched filter
    snr_pycbc = pycbc.filter.matched_filter(
        hp.to_frequencyseries(delta_f=1.0/T),
        to_pycbc_timeseries(ts_strain).to_frequencyseries(delta_f=1.0/T),
        psd=psd_pycbc,
        low_frequency_cutoff=30.0,
    )

    # Convert SNR back to gwexpy
    snr_ts = from_pycbc_timeseries(TimeSeries, snr_pycbc.real())
    snr_ts.name = "SNR (BNS template)"

else:
    # Analytic substitute: cross-correlate data with the injected chirp
    chirp_ts = TimeSeries(chirp, t0=t0, sample_rate=fs, name="template")
    # Simple FFT cross-correlation, normalised to peak ~ 10
    xcorr = np.fft.irfft(
        np.fft.rfft(ts_strain.value) * np.conj(np.fft.rfft(chirp_ts.value))
    )
    xcorr_norm = xcorr / (np.std(xcorr[:int(20*fs)]) + 1e-50)
    snr_ts = TimeSeries(np.abs(xcorr_norm), t0=t0, sample_rate=fs,
                        name="SNR (analytic xcorr)")

print(f"SNR time series: {len(snr_ts)} samples")
print(f"Peak SNR: {snr_ts.value.max():.1f}  "
      f"at t = {t[snr_ts.value.argmax()]:.2f} s  "
      f"(true merger: {tc:.2f} s)")
"""),

    md("""\
## 5. Visualise the SNR Time Series

Plot the matched-filter SNR as a function of time, with the injected
merger time marked.
"""),

    code("""\
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Raw strain (last 5 s)
win = slice(-int(5*fs), None)
axes[0].plot(t[win], ts_strain.value[win] * 1e22, lw=0.5, color="steelblue")
axes[0].set_ylabel("Strain [×10⁻²²]")
axes[0].set_title("DARM strain + BNS injection")
axes[0].grid(True, alpha=0.4)
axes[0].axvline(tc, color="red", ls="--", lw=1.5, label="Merger time")
axes[0].legend(fontsize=9)

# SNR
snr_plot = snr_ts.value[win] if hasattr(snr_ts, "value") else snr_ts[win]
axes[1].plot(t[win], snr_plot, lw=1.2, color="darkorange")
axes[1].axhline(8.0, color="gray", ls="--", lw=1, label="SNR = 8 threshold")
axes[1].axvline(tc, color="red", ls="--", lw=1.5)
axes[1].set_ylabel("Matched-filter SNR")
axes[1].set_xlabel(f"Time since GPS {t0} [s]")
axes[1].set_title("Matched-filter SNR (BNS template)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()
"""),

    md("""\
## 6. Conversion Reference

| gwexpy object | To PyCBC | From PyCBC |
|---------------|---------|------------|
| `TimeSeries` | `to_pycbc_timeseries(ts)` | `from_pycbc_timeseries(TimeSeries, pycbc_ts)` |
| `FrequencySeries` | `to_pycbc_frequencyseries(fs)` | `from_pycbc_frequencyseries(FrequencySeries, pycbc_fs)` |

**Preserved metadata:**

| Attribute | gwexpy | PyCBC |
|-----------|--------|-------|
| Sample rate | `ts.sample_rate` | `1 / pycbc_ts.delta_t` |
| Start time | `ts.t0.value` | `float(pycbc_ts.start_time)` |
| Frequency resolution | `fs.df.value` | `pycbc_fs.delta_f` |

**Typical workflow:**
1. Load / preprocess data with gwexpy (`read`, `whiten`, `crop`)
2. Estimate PSD with gwexpy (`ts.asd()`)
3. Convert to PyCBC for matched filter or PE
4. Convert SNR time series back to gwexpy for visualisation and archiving
"""),
]


# ---------------------------------------------------------------------------
# Japanese notebook
# ---------------------------------------------------------------------------

JA_CELLS = [
    md("""\
# PyCBC 連携：gwexpy 前処理から重力波探索まで

[PyCBC](https://pycbc.org/) は連星合体（CBC）のマッチトフィルター探索や
パラメータ推定を行う重力波データ解析ツールキットです。
gwexpy は自身のデータ型と PyCBC の `TimeSeries` / `FrequencySeries` 間の
**双方向コンバータ**を提供します。

これにより、以下のような自然なワークフローが実現します：
- **gwexpy**: データ読み込み、前処理、ノイズ特性評価
- **PyCBC**: マッチトフィルター探索やパラメータ推定
- **gwexpy**: 探索結果の後処理と可視化

**このチュートリアルで学ぶこと：**
1. gwexpy `TimeSeries` を PyCBC に変換して戻す
2. gwexpy `FrequencySeries`（ASD）を PyCBC PSD に変換する
3. 完全な前処理パイプライン：コンディショニング → マッチトフィルター
4. gwexpy で SNR 時系列を可視化する

> **注意**: PyCBC がインストールされていなくても動作します。
> `pip install pycbc` でインストールすれば実際の探索を実行できます。
"""),

    md("## セットアップ"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt

from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.interop.pycbc_ import (
    to_pycbc_timeseries, from_pycbc_timeseries,
    to_pycbc_frequencyseries, from_pycbc_frequencyseries,
)

PYCBC_AVAILABLE = False
try:
    import pycbc
    import pycbc.types
    import pycbc.filter
    import pycbc.waveform
    import pycbc.psd
    PYCBC_AVAILABLE = True
    print(f"PyCBC {pycbc.__version__} が見つかりました — 実際のマッチトフィルターを実行します。")
except ImportError:
    print("PyCBC 未インストール — 解析的フォールバックを使用します。")
"""),

    md("""\
## 1. CBC 信号を注入した合成 DARM データ

64 s のカラードノイズに合成 BNS (連星中性子星) インスパイラルを注入します。
チャープ信号は合体直前の数秒で ~30 Hz から ~1000 Hz に掃引します。
"""),

    code("""\
fs   = 4096.0
T    = 64.0
N    = int(T * fs)
t0   = 1_300_000_000
rng  = np.random.default_rng(0)
t    = np.arange(N) / fs

freqs_n = np.fft.rfftfreq(N, 1.0/fs)[1:]
asd_model = np.where(freqs_n < 20, (20/freqs_n)**4,
            np.where(freqs_n < 200, (200/freqs_n)**0.5, 1.0))
fft = asd_model * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(freqs_n)))
noise = np.fft.irfft(np.concatenate([[0.0], fft]), n=N) * 1e-23

M_chirp = 1.2
G_c3 = 4.926e-6
tc   = T - 0.5
tau  = np.maximum(tc - t, 1e-4)
f_gw = (5.0/(256.0 * np.pi**(8.0/3)))**(3.0/8) * (G_c3 * M_chirp)**(-5.0/8) * tau**(-3.0/8)
phi_gw = -2.0 * (tau / (5.0 * G_c3 * M_chirp))**(5.0/8) / np.pi
amp_gw = 1e-22 * (G_c3 * M_chirp / tau)**(1.0/4)
mask = f_gw < 1500.0
chirp = np.where(mask, amp_gw * np.cos(phi_gw), 0.0)

ts_strain = TimeSeries(noise + chirp, t0=t0, sample_rate=fs,
                       name="K1:LSC-DARM_OUT_DQ", unit="strain")

print(f"チャープ開始: t = {t[mask][0]:.1f} s  (f = {f_gw[mask][0]:.1f} Hz)")
print(f"合体時刻    : t = {tc:.1f} s")

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t[-int(3*fs):], ts_strain.value[-int(3*fs):], lw=0.5, color="steelblue")
ax.set_xlabel("GPS からの経過時間 [s]")
ax.set_ylabel("歪み")
ax.set_title("直前 3 s: DARM ノイズ + BNS インスパイラル注入")
plt.tight_layout()
plt.show()
"""),

    md("## 2. gwexpy → PyCBC 変換"),

    code("""\
if PYCBC_AVAILABLE:
    ts_pycbc = to_pycbc_timeseries(ts_strain)
    print(f"pycbc.TimeSeries:")
    print(f"  delta_t   : {ts_pycbc.delta_t}")
    print(f"  start_time: {float(ts_pycbc.start_time):.1f}")
    print(f"  len       : {len(ts_pycbc)}")

    ts_back = from_pycbc_timeseries(TimeSeries, ts_pycbc)
    print(f"\\n往復変換の最大誤差: "
          f"{np.max(np.abs(ts_strain.value - ts_back.value)):.2e}")
else:
    print("PyCBC 未インストール — 変換デモをスキップします。")
    ts_back = ts_strain
"""),

    md("## 3. PSD / ASD 変換"),

    code("""\
ts_quiet = TimeSeries(ts_strain.value[:int(30*fs)], t0=t0,
                      sample_rate=fs, name="静穏セグメント", unit="strain")
asd_gw = ts_quiet.asd(fftlength=4.0, method="median")
print(f"gwexpy ASD: {len(asd_gw)} ビン, df={asd_gw.df.value:.4f} Hz")

if PYCBC_AVAILABLE:
    fs_pycbc = to_pycbc_frequencyseries(asd_gw)
    psd_pycbc = pycbc.types.FrequencySeries(
        fs_pycbc.numpy()**2, delta_f=fs_pycbc.delta_f,
        epoch=fs_pycbc.epoch,
    )
    print(f"PyCBC PSD: {len(psd_pycbc)} ビン, df={psd_pycbc.delta_f:.4f} Hz")
else:
    print("PyCBC 未インストール — PSD 変換デモをスキップします。")
"""),

    md("""\
## 4. マッチトフィルター探索（BNS テンプレート）

マッチトフィルターはデータとテンプレート波形の相互相関を
ノイズ PSD で正規化して計算します。
正規化 SNR 時系列 $\\rho(t)$ のピークが合体時刻を示します。
"""),

    code("""\
if PYCBC_AVAILABLE:
    hp, hc = pycbc.waveform.get_td_waveform(
        approximant="TaylorT4",
        mass1=1.4, mass2=1.4,
        delta_t=1.0/fs,
        f_lower=30.0,
    )
    snr_pycbc = pycbc.filter.matched_filter(
        hp.to_frequencyseries(delta_f=1.0/T),
        to_pycbc_timeseries(ts_strain).to_frequencyseries(delta_f=1.0/T),
        psd=psd_pycbc,
        low_frequency_cutoff=30.0,
    )
    snr_ts = from_pycbc_timeseries(TimeSeries, snr_pycbc.real())
    snr_ts.name = "SNR (BNS テンプレート)"
else:
    chirp_ts = TimeSeries(chirp, t0=t0, sample_rate=fs, name="テンプレート")
    xcorr = np.fft.irfft(
        np.fft.rfft(ts_strain.value) * np.conj(np.fft.rfft(chirp_ts.value))
    )
    xcorr_norm = xcorr / (np.std(xcorr[:int(20*fs)]) + 1e-50)
    snr_ts = TimeSeries(np.abs(xcorr_norm), t0=t0, sample_rate=fs,
                        name="SNR（解析的相互相関）")

print(f"SNR 時系列: {len(snr_ts)} サンプル")
print(f"ピーク SNR: {snr_ts.value.max():.1f}  "
      f"at t = {t[snr_ts.value.argmax()]:.2f} s  "
      f"（真の合体時刻: {tc:.2f} s）")
"""),

    md("## 5. SNR 時系列の可視化"),

    code("""\
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

win = slice(-int(5*fs), None)
axes[0].plot(t[win], ts_strain.value[win] * 1e22, lw=0.5, color="steelblue")
axes[0].set_ylabel("歪み [×10⁻²²]")
axes[0].set_title("DARM 歪み + BNS 注入")
axes[0].grid(True, alpha=0.4)
axes[0].axvline(tc, color="red", ls="--", lw=1.5, label="合体時刻")
axes[0].legend(fontsize=9)

snr_plot = snr_ts.value[win]
axes[1].plot(t[win], snr_plot, lw=1.2, color="darkorange")
axes[1].axhline(8.0, color="gray", ls="--", lw=1, label="SNR = 8 閾値")
axes[1].axvline(tc, color="red", ls="--", lw=1.5)
axes[1].set_ylabel("マッチトフィルター SNR")
axes[1].set_xlabel(f"GPS {t0} からの経過時間 [s]")
axes[1].set_title("マッチトフィルター SNR（BNS テンプレート）")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 変換リファレンス

| gwexpy オブジェクト | → PyCBC | ← PyCBC |
|------------------|--------|---------|
| `TimeSeries` | `to_pycbc_timeseries(ts)` | `from_pycbc_timeseries(TimeSeries, pycbc_ts)` |
| `FrequencySeries` | `to_pycbc_frequencyseries(fs)` | `from_pycbc_frequencyseries(FrequencySeries, pycbc_fs)` |

**典型的なワークフロー：**
1. gwexpy でデータを読み込み・前処理（`read`、`whiten`、`crop`）
2. gwexpy で PSD を推定（`ts.asd()`）
3. PyCBC に変換してマッチトフィルター or PE を実行
4. SNR 時系列を gwexpy に戻して可視化・アーカイブ
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
    write_nb(EN_CELLS, root / "docs/web/en/user_guide/tutorials/case_pycbc_search.ipynb")
    write_nb(JA_CELLS, root / "docs/web/ja/user_guide/tutorials/case_pycbc_search.ipynb")
    print("Done.")
