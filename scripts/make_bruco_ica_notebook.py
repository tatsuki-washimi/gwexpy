"""Generate case_bruco_ica_denoising.ipynb (EN + JA)."""

import json
from pathlib import Path


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


# ---------------------------------------------------------------------------
# English notebook
# ---------------------------------------------------------------------------
EN_CELLS = [
    md("""\
# Bruco + ICA End-to-End Noise Reduction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main/docs/web/en/user_guide/tutorials/case_bruco_ica_denoising.ipynb)

This notebook demonstrates an end-to-end noise reduction workflow used in real interferometer commissioning:

1. **Bruco** – brute-force coherence scan to identify the most correlated auxiliary channels.
2. **ICA** – Independent Component Analysis to separate noise sources.
3. **Noise subtraction** – remove noise contributions from the DARM channel and compare ASDs.

This reproduces the workflow used in O4b commissioning (e.g., DARM 116 Hz line investigation).

> **Prerequisites**: Familiarity with
> - [Bruco tutorial](advanced_bruco.ipynb) — Bruco basics
> - [PCA/ICA tutorial](advanced_decomposition.ipynb) — decomposition basics
"""),

    md("## Setup"),

    code("""\
# ruff: noqa: I001
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from gwexpy.analysis import Bruco
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
"""),

    md("""\
## 1. Mock Data Generation

We simulate a scenario where the DARM channel contains:
- Broadband Gaussian noise (the "signal" floor)
- A 116 Hz line noise leaking from environmental sensors

Four PEM (Physical Environment Monitor) auxiliary channels each contain varying amounts
of the same 116 Hz line.  This mirrors the real O4b commissioning case.
"""),

    code("""\
rng = np.random.default_rng(42)

fs = 512        # sample rate [Hz]
T  = 64.0       # duration [s]
n  = int(fs * T)
t  = np.arange(n) / fs

FREQ_LINE = 116.0  # Hz — the line noise we want to remove

# ----- Independent sources -----
src_line  = np.sin(2 * np.pi * FREQ_LINE * t)   # coherent line noise
src_broad = rng.normal(0, 1, n)                  # broadband floor noise

# ----- DARM: broadband + line leakage -----
DARM_LINE_COUPLING = 0.5
darm_noise = src_broad + DARM_LINE_COUPLING * src_line

target = TimeSeries(
    darm_noise, dt=1 / fs, unit=u.dimensionless_unscaled,
    name="K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ", t0=0,
)

# ----- 4 auxiliary channels (different line content) -----
aux_configs = {
    "K1:PEM-ACC_PSL_TABLE_PSL1_Y": (0.9, 0.1),  # 90% line, 10% noise
    "K1:PEM-ACC_PSL_TABLE_PSL2_X": (0.7, 0.3),
    "K1:PEM-MIC_PSL_TABLE_PSL1_Z": (0.5, 0.5),
    "K1:PEM-MIC_PSL_TABLE_PSL2_Z": (0.2, 0.8),  # mostly noise
}

aux_dict = TimeSeriesDict({
    name: TimeSeries(
        a_line * src_line + a_noise * rng.normal(0, 1, n),
        dt=1 / fs, unit=u.dimensionless_unscaled, name=name, t0=0,
    )
    for name, (a_line, a_noise) in aux_configs.items()
})

print(f"DARM   sample rate: {target.sample_rate}")
print(f"Aux channels: {list(aux_dict.keys())}")
"""),

    code("""\
# Visualize the line noise in the ASD before cleaning
fig, ax = plt.subplots(figsize=(10, 4))
asd = target.asd(fftlength=4, overlap=2)
ax.loglog(asd.frequencies.value, asd.value, label="DARM (original)", color="steelblue")

for name, ts in aux_dict.items():
    a = ts.asd(fftlength=4, overlap=2)
    ax.loglog(a.frequencies.value, a.value, alpha=0.5, lw=0.8, label=name.split(":")[-1])

ax.axvline(FREQ_LINE, color="red", ls="--", label=f"{FREQ_LINE} Hz line")
ax.set_xlim(1, fs / 2)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [1/√Hz]")
ax.set_title("ASD before cleaning — line noise visible at 116 Hz")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 2. Bruco: Identify Correlated Auxiliary Channels

`Bruco.compute()` scans all auxiliary channels and returns the top-*N* most coherent
channels at each frequency.  This is the **first stage** of the pipeline:
identify *which* sensors see the same noise as DARM.
"""),

    code("""\
bruco = Bruco(target_channel=target.name, aux_channels=[])

result = bruco.compute(
    fftlength=4.0,
    overlap=2.0,
    target_data=target,
    aux_data=aux_dict,
    top_n=4,         # keep top-4 channels per frequency bin
)

print("Bruco scan complete.")
print(f"Result type: {type(result)}")
"""),

    code("""\
# Show top channels near the 116 Hz line
df = result.to_dataframe(ranks=[0])
df_line = (
    df[df["frequency"].between(114, 118)]
    .sort_values("coherence", ascending=False)
    .dropna(subset=["channel"])
    .head(10)
    .reset_index(drop=True)
)
print("Top correlated channels near 116 Hz:")
df_line
"""),

    code("""\
# Coherence projection plot — shows which channels dominate at each frequency
result.plot_projection(coherence_threshold=0.3)
plt.xlim(1, fs / 2)
plt.axvline(FREQ_LINE, color="red", ls="--", lw=1)
plt.title("Bruco noise projection")
plt.show()
"""),

    md("""\
## 3. ICA: Separate Noise Sources

From the Bruco scan we know which channels are highly coherent with DARM near 116 Hz.
We now stack these channels into a `TimeSeriesMatrix` and apply **ICA** to unmix the
underlying independent sources.

The ICA model gives us a *mixing matrix* **A** such that:
```
X = S · A^T   (X: observed channels, S: independent sources)
```
We can then subtract the noise-source contributions from DARM.
"""),

    code("""\
# Pick the top-2 channels from Bruco result
TOP_CHANNELS = df_line["channel"].dropna().unique()[:2].tolist()
print("Selected channels for ICA:", TOP_CHANNELS)

# Stack DARM + top channels into a TimeSeriesMatrix  (shape: n_ch × 1 × n_samples)
channels = [target] + [aux_dict[ch] for ch in TOP_CHANNELS]
data_3d = np.stack([ch.value for ch in channels], axis=0)[:, np.newaxis, :]

tsm = TimeSeriesMatrix(
    data_3d, dt=1 / fs, unit=u.dimensionless_unscaled, t0=0,
)
print(f"TimeSeriesMatrix shape: {tsm.shape}  (n_channels × 1 × n_samples)")
"""),

    code("""\
# Run ICA
n_components = len(channels)
ica_sources, ica_model = tsm.ica(n_components=n_components, return_model=True)

sk = ica_model.sklearn_model
print(f"ICA converged in {sk.n_iter_} iterations")
print(f"Mixing matrix A shape: {sk.mixing_.shape}")  # (n_channels, n_components)

# Visualize ICA sources in frequency domain
fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4), sharey=True)
for k in range(n_components):
    src_ts = TimeSeries(
        ica_sources.value[k, 0, :], dt=1 / fs,
        unit=u.dimensionless_unscaled, t0=0,
    )
    asd_k = src_ts.asd(fftlength=4, overlap=2)
    axes[k].semilogy(asd_k.frequencies.value, asd_k.value)
    axes[k].axvline(FREQ_LINE, color="red", ls="--", lw=0.8)
    axes[k].set_title(f"ICA source #{k}")
    axes[k].set_xlim(1, fs / 2)
    axes[k].set_xlabel("Frequency [Hz]")

axes[0].set_ylabel("ASD [arb/√Hz]")
plt.suptitle("ICA independent components — one should peak at 116 Hz")
plt.tight_layout()
plt.show()
"""),

    md("""\
## 4. Noise Subtraction and ASD Comparison

Identify which ICA components contain the 116 Hz line (by inspecting the ASD),
then subtract their contribution from DARM using the mixing matrix.

```
DARM_clean = DARM - Σ_k  A[0, k] · S_k(t)     (summed over noise components k)
```
"""),

    code("""\
# Identify the noise component: the one with the largest ASD at FREQ_LINE
fftlength = 4.0
overlap   = 2.0
freqs = np.fft.rfftfreq(int(fftlength * fs), d=1 / fs)
line_bin = np.argmin(np.abs(freqs - FREQ_LINE))

A = sk.mixing_           # (n_channels, n_components)
X_raw = data_3d[:, 0, :].T   # (n_samples, n_channels)
S = sk.transform(X_raw)       # (n_samples, n_components)

# ASD of each component at the line frequency
asd_at_line = []
for k in range(n_components):
    s_ts = TimeSeries(S[:, k], dt=1 / fs, unit=u.dimensionless_unscaled, t0=0)
    asd_k = s_ts.asd(fftlength=fftlength, overlap=overlap)
    val = float(np.interp(FREQ_LINE, asd_k.frequencies.value, asd_k.value))
    asd_at_line.append(val)

noise_component = int(np.argmax(asd_at_line))
print(f"Dominant noise component: #{noise_component}  (ASD at {FREQ_LINE} Hz = {asd_at_line[noise_component]:.4f})")

# Subtract noise component from DARM (channel index 0)
darm_clean = X_raw[:, 0] - A[0, noise_component] * S[:, noise_component]
target_clean = TimeSeries(
    darm_clean, dt=1 / fs, unit=u.dimensionless_unscaled,
    name="DARM_cleaned", t0=0,
)
"""),

    code("""\
# Compare original vs cleaned ASD
asd_orig  = target.asd(fftlength=fftlength, overlap=overlap)
asd_clean = target_clean.asd(fftlength=fftlength, overlap=overlap)

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(asd_orig.frequencies.value,  asd_orig.value,  label="DARM original",  color="steelblue", lw=1.2)
ax.loglog(asd_clean.frequencies.value, asd_clean.value, label="DARM cleaned",   color="orangered",  lw=1.2)
ax.axvline(FREQ_LINE, color="gray", ls="--", lw=0.8, label=f"{FREQ_LINE} Hz")

# Suppression factor at line frequency
ratio = float(np.interp(FREQ_LINE, asd_orig.frequencies.value, asd_orig.value)) / \
        float(np.interp(FREQ_LINE, asd_clean.frequencies.value, asd_clean.value))
ax.set_xlim(50, 200)
ax.set_ylim(1e-4, 10)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [1/√Hz]")
ax.set_title(f"Noise reduction at {FREQ_LINE} Hz: ×{ratio:.1f} suppression")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Suppression factor at {FREQ_LINE} Hz: {ratio:.2f}×")
"""),

    md("""\
## Summary

| Step | Tool | Output |
|------|------|--------|
| 1. Coherence scan | `Bruco.compute()` | Top correlated channels at each frequency |
| 2. Source separation | `TimeSeriesMatrix.ica()` | Independent components + mixing matrix |
| 3. Noise subtraction | Mixing matrix algebra | Cleaned DARM channel |

### Key takeaways

- **Bruco** efficiently narrows down thousands of channels to the few that matter.
- **ICA** goes beyond coherence: it separates *independent* source contributions even when
  multiple sensors share a common noise.
- The mixing matrix `A` provides a direct subtraction formula without time-domain filtering.

### Next steps

- Replace mock data with real NDS/GWF data (`TimeSeries.read()` or NDS2).
- Run on longer segments and cross-validate with Bruco's projection estimate.
- Combine with `Spectrogram.normalize()` (SNR spectrogram) to track the line over time.
"""),
]

# ---------------------------------------------------------------------------
# Japanese notebook
# ---------------------------------------------------------------------------
JA_CELLS = [
    md("""\
# Bruco + ICA エンドツーエンド ノイズ削減

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main/docs/web/ja/user_guide/tutorials/case_bruco_ica_denoising.ipynb)

このノートブックでは、実際の干渉計コミッショニングで使われる **エンドツーエンドのノイズ削減ワークフロー** を示します：

1. **Bruco** — ブルートフォースコヒーレンス スキャンで最相関の補助チャンネルを特定
2. **ICA** — 独立成分分析でノイズ源を分離
3. **ノイズ差し引き** — DARM からノイズ寄与を除去し、ASD を比較

これは O4b コミッショニング（DARM 116 Hz ライン調査など）で使われたワークフローを再現しています。

> **前提知識**：
> - [Bruco チュートリアル](advanced_bruco.ipynb) — Bruco の基礎
> - [PCA/ICA チュートリアル](advanced_decomposition.ipynb) — 分解手法の基礎
"""),

    md("## セットアップ"),

    code("""\
# ruff: noqa: I001
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from gwexpy.analysis import Bruco
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
"""),

    md("""\
## 1. モックデータの生成

DARM チャンネルに以下が混入するシナリオをシミュレートします：
- 広帯域ガウスノイズ（バックグラウンド雑音フロア）
- 環境センサーから漏れた 116 Hz ラインノイズ

4 つの PEM（Physical Environment Monitor）補助チャンネルは、それぞれ異なる割合で同じ 116 Hz ラインを含みます。
これは O4b コミッショニングの実際のケースを模倣しています。
"""),

    code("""\
rng = np.random.default_rng(42)

fs = 512        # サンプルレート [Hz]
T  = 64.0       # データ長 [s]
n  = int(fs * T)
t  = np.arange(n) / fs

FREQ_LINE = 116.0  # Hz — 除去したいラインノイズの周波数

# ----- 独立ノイズ源 -----
src_line  = np.sin(2 * np.pi * FREQ_LINE * t)   # コヒーレントなラインノイズ
src_broad = rng.normal(0, 1, n)                  # 広帯域フロアノイズ

# ----- DARM: 広帯域 + ラインノイズ漏れ -----
DARM_LINE_COUPLING = 0.5
darm_noise = src_broad + DARM_LINE_COUPLING * src_line

target = TimeSeries(
    darm_noise, dt=1 / fs, unit=u.dimensionless_unscaled,
    name="K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ", t0=0,
)

# ----- 4 つの補助チャンネル（ライン含有率が異なる）-----
aux_configs = {
    "K1:PEM-ACC_PSL_TABLE_PSL1_Y": (0.9, 0.1),  # 90% ライン、10% ノイズ
    "K1:PEM-ACC_PSL_TABLE_PSL2_X": (0.7, 0.3),
    "K1:PEM-MIC_PSL_TABLE_PSL1_Z": (0.5, 0.5),
    "K1:PEM-MIC_PSL_TABLE_PSL2_Z": (0.2, 0.8),  # ほぼノイズ
}

aux_dict = TimeSeriesDict({
    name: TimeSeries(
        a_line * src_line + a_noise * rng.normal(0, 1, n),
        dt=1 / fs, unit=u.dimensionless_unscaled, name=name, t0=0,
    )
    for name, (a_line, a_noise) in aux_configs.items()
})

print(f"DARM   サンプルレート: {target.sample_rate}")
print(f"補助チャンネル: {list(aux_dict.keys())}")
"""),

    code("""\
# クリーニング前の ASD で 116 Hz ラインを可視化
fig, ax = plt.subplots(figsize=(10, 4))
asd = target.asd(fftlength=4, overlap=2)
ax.loglog(asd.frequencies.value, asd.value, label="DARM（元）", color="steelblue")

for name, ts in aux_dict.items():
    a = ts.asd(fftlength=4, overlap=2)
    ax.loglog(a.frequencies.value, a.value, alpha=0.5, lw=0.8, label=name.split(":")[-1])

ax.axvline(FREQ_LINE, color="red", ls="--", label=f"{FREQ_LINE} Hz ライン")
ax.set_xlim(1, fs / 2)
ax.set_xlabel("周波数 [Hz]")
ax.set_ylabel("ASD [1/√Hz]")
ax.set_title("クリーニング前の ASD — 116 Hz にラインノイズが見える")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 2. Bruco: 相関補助チャンネルの特定

`Bruco.compute()` は全補助チャンネルをスキャンし、各周波数ビンで上位 *N* 件の最相関チャンネルを返します。
これがパイプラインの **第 1 ステージ** です：DARM と同じノイズを観測しているセンサーを特定します。
"""),

    code("""\
bruco = Bruco(target_channel=target.name, aux_channels=[])

result = bruco.compute(
    fftlength=4.0,
    overlap=2.0,
    target_data=target,
    aux_data=aux_dict,
    top_n=4,         # 各周波数ビンで上位 4 チャンネルを保持
)

print("Bruco スキャン完了")
print(f"結果タイプ: {type(result)}")
"""),

    code("""\
# 116 Hz ライン付近の上位チャンネルを表示
df = result.to_dataframe(ranks=[0])
df_line = (
    df[df["frequency"].between(114, 118)]
    .sort_values("coherence", ascending=False)
    .dropna(subset=["channel"])
    .head(10)
    .reset_index(drop=True)
)
print("116 Hz 付近の最相関チャンネル:")
df_line
"""),

    code("""\
# コヒーレンス投影プロット
result.plot_projection(coherence_threshold=0.3)
plt.xlim(1, fs / 2)
plt.axvline(FREQ_LINE, color="red", ls="--", lw=1)
plt.title("Bruco ノイズ投影")
plt.show()
"""),

    md("""\
## 3. ICA: ノイズ源の分離

Bruco スキャンから、DARM と 116 Hz 付近で高いコヒーレンスを持つチャンネルが分かりました。
次にこれらのチャンネルを `TimeSeriesMatrix` に積み上げ、**ICA** を適用して
独立したノイズ源を分離します。

ICA モデルは混合行列 **A** を求めます：
```
X = S · A^T   （X: 観測チャンネル、S: 独立成分）
```
この行列を使って DARM からノイズ成分の寄与を差し引けます。
"""),

    code("""\
# Bruco 結果から上位 2 チャンネルを選択
TOP_CHANNELS = df_line["channel"].dropna().unique()[:2].tolist()
print("ICA に使用するチャンネル:", TOP_CHANNELS)

# DARM + 上位チャンネルを TimeSeriesMatrix に積み上げ（shape: n_ch × 1 × n_samples）
channels = [target] + [aux_dict[ch] for ch in TOP_CHANNELS]
data_3d = np.stack([ch.value for ch in channels], axis=0)[:, np.newaxis, :]

tsm = TimeSeriesMatrix(
    data_3d, dt=1 / fs, unit=u.dimensionless_unscaled, t0=0,
)
print(f"TimeSeriesMatrix の shape: {tsm.shape}  (チャンネル数 × 1 × サンプル数)")
"""),

    code("""\
# ICA の実行
n_components = len(channels)
ica_sources, ica_model = tsm.ica(n_components=n_components, return_model=True)

sk = ica_model.sklearn_model
print(f"ICA 収束: {sk.n_iter_} 反復")
print(f"混合行列 A の shape: {sk.mixing_.shape}")  # (n_channels, n_components)

# 各 ICA 成分の ASD を周波数領域で可視化
fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4), sharey=True)
for k in range(n_components):
    src_ts = TimeSeries(
        ica_sources.value[k, 0, :], dt=1 / fs,
        unit=u.dimensionless_unscaled, t0=0,
    )
    asd_k = src_ts.asd(fftlength=4, overlap=2)
    axes[k].semilogy(asd_k.frequencies.value, asd_k.value)
    axes[k].axvline(FREQ_LINE, color="red", ls="--", lw=0.8)
    axes[k].set_title(f"ICA 成分 #{k}")
    axes[k].set_xlim(1, fs / 2)
    axes[k].set_xlabel("周波数 [Hz]")

axes[0].set_ylabel("ASD [任意単位/√Hz]")
plt.suptitle("ICA 独立成分 — 1 つが 116 Hz にピークを持つはず")
plt.tight_layout()
plt.show()
"""),

    md("""\
## 4. ノイズ差し引きと ASD 比較

116 Hz ラインを含む ICA 成分を特定し（ASD を確認して選択）、
混合行列を使って DARM からその寄与を差し引きます。

```
DARM_clean = DARM - Σ_k  A[0, k] · S_k(t)     （ノイズ成分 k の和）
```
"""),

    code("""\
# ノイズ成分の特定：FREQ_LINE で ASD が最大の成分を選ぶ
fftlength = 4.0
overlap   = 2.0
freqs = np.fft.rfftfreq(int(fftlength * fs), d=1 / fs)

A = sk.mixing_           # (n_channels, n_components)
X_raw = data_3d[:, 0, :].T   # (n_samples, n_channels)
S = sk.transform(X_raw)       # (n_samples, n_components)

# 各成分のライン周波数での ASD を計算
asd_at_line = []
for k in range(n_components):
    s_ts = TimeSeries(S[:, k], dt=1 / fs, unit=u.dimensionless_unscaled, t0=0)
    asd_k = s_ts.asd(fftlength=fftlength, overlap=overlap)
    val = float(np.interp(FREQ_LINE, asd_k.frequencies.value, asd_k.value))
    asd_at_line.append(val)

noise_component = int(np.argmax(asd_at_line))
print(f"主ノイズ成分: #{noise_component}  ({FREQ_LINE} Hz での ASD = {asd_at_line[noise_component]:.4f})")

# DARM（チャンネル index 0）からノイズ成分を差し引く
darm_clean = X_raw[:, 0] - A[0, noise_component] * S[:, noise_component]
target_clean = TimeSeries(
    darm_clean, dt=1 / fs, unit=u.dimensionless_unscaled,
    name="DARM_cleaned", t0=0,
)
"""),

    code("""\
# 元の DARM とクリーン DARM の ASD を比較
asd_orig  = target.asd(fftlength=fftlength, overlap=overlap)
asd_clean = target_clean.asd(fftlength=fftlength, overlap=overlap)

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(asd_orig.frequencies.value,  asd_orig.value,  label="DARM 元データ", color="steelblue", lw=1.2)
ax.loglog(asd_clean.frequencies.value, asd_clean.value, label="DARM クリーン", color="orangered",  lw=1.2)
ax.axvline(FREQ_LINE, color="gray", ls="--", lw=0.8, label=f"{FREQ_LINE} Hz")

# 抑制率を計算
ratio = float(np.interp(FREQ_LINE, asd_orig.frequencies.value, asd_orig.value)) / \
        float(np.interp(FREQ_LINE, asd_clean.frequencies.value, asd_clean.value))
ax.set_xlim(50, 200)
ax.set_ylim(1e-4, 10)
ax.set_xlabel("周波数 [Hz]")
ax.set_ylabel("ASD [1/√Hz]")
ax.set_title(f"{FREQ_LINE} Hz でのノイズ削減: {ratio:.1f} 倍抑制")
ax.legend()
plt.tight_layout()
plt.show()

print(f"{FREQ_LINE} Hz での抑制率: {ratio:.2f} 倍")
"""),

    md("""\
## まとめ

| ステップ | ツール | 出力 |
|---------|--------|------|
| 1. コヒーレンス スキャン | `Bruco.compute()` | 各周波数での最相関チャンネル |
| 2. ノイズ源分離 | `TimeSeriesMatrix.ica()` | 独立成分 + 混合行列 |
| 3. ノイズ差し引き | 混合行列の代数演算 | クリーン DARM チャンネル |

### ポイント

- **Bruco** が数千チャンネルを効率的に絞り込み、重要な少数チャンネルを特定します。
- **ICA** はコヒーレンスを超えて、複数センサーが共通ノイズを共有する場合でも *独立* な寄与を分離します。
- 混合行列 `A` が、時間領域フィルタリングなしに直接差し引き式を与えます。

### 次のステップ

- モックデータを実データ（`TimeSeries.read()` や NDS2）に置き換える。
- より長いセグメントで解析し、Bruco の投影推定と照合する。
- `Spectrogram.normalize()`（SNR スペクトログラム）と組み合わせて、ラインの時間変化を追跡する。
"""),
]


def make_nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 2,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "cells": cells,
    }


en_path = Path("docs/web/en/user_guide/tutorials/case_bruco_ica_denoising.ipynb")
ja_path = Path("docs/web/ja/user_guide/tutorials/case_bruco_ica_denoising.ipynb")

en_path.write_text(json.dumps(make_nb(EN_CELLS), indent=1, ensure_ascii=False))
ja_path.write_text(json.dumps(make_nb(JA_CELLS), indent=1, ensure_ascii=False))

print(f"Created: {en_path}")
print(f"Created: {ja_path}")
