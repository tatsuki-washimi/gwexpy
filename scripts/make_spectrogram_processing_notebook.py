"""Generate advanced_spectrogram_processing.ipynb (EN + JA).

Covers:
  - Spectrogram.normalize()  – SNR, median, mean, percentile, reference
  - Spectrogram.clean()      – threshold, rolling_median, line_removal, combined

Usage:
    python scripts/make_spectrogram_processing_notebook.py
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
# Shared code cells (language-agnostic)
# ============================================================

SETUP_CODE = """\
import matplotlib.pyplot as plt
import numpy as np

from gwexpy.timeseries import TimeSeries

plt.rcParams["figure.figsize"] = (12, 4)
"""

MOCK_DATA_CODE = """\
# ── Reproducible seed ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

DURATION  = 120    # seconds
FS        = 512    # Hz
FFTLEN    = 4.0    # s
STRIDE    = 2.0    # s

t = np.arange(0, DURATION, 1 / FS)

# Base: flat noise (simulate whitened strain)
base_noise = rng.normal(0, 1.0, len(t))

# Add persistent narrowband line at 60 Hz (power line)
line60 = 3.0 * np.sin(2 * np.pi * 60 * t)

# Add slowly drifting broadband noise floor (non-stationary trend)
trend = 1.0 + 0.5 * np.sin(2 * np.pi * t / DURATION)

# Add a few transient glitches (excess-power bursts)
glitch_times = [15.0, 45.0, 90.0]
glitch_signal = np.zeros_like(t)
for gt in glitch_times:
    idx = int(gt * FS)
    width = int(0.05 * FS)
    glitch_signal[idx : idx + width] += rng.normal(0, 8.0, width)

strain = (base_noise * trend) + line60 + glitch_signal

ts = TimeSeries(strain, dt=1.0 / FS, name="STRAIN", unit="strain")
spec = ts.spectrogram(STRIDE, fftlength=FFTLEN, overlap=FFTLEN / 2)
print("Spectrogram shape (time × freq):", spec.shape)
print("Time bins:", spec.shape[0], " | Freq bins:", spec.shape[1])
"""

BASELINE_PLOT_CODE = """\
fig, ax = plt.subplots(figsize=(12, 4))
spec.plot(ax=ax, norm="log")
ax.set_title("Raw spectrogram")
ax.colorbar(label="Power [strain²/Hz]")
plt.tight_layout()
plt.show()
"""

NORMALIZE_SNR_CODE = """\
# ── SNR normalization ─────────────────────────────────────────────────────────
# Each time slice is divided by the median PSD along the time axis.
# Result: dimensionless SNR² (≈ 1 for stationary background).
spec_snr = spec.normalize(method="snr")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
spec.plot(ax=axes[0], norm="log")
axes[0].set_title("Raw spectrogram")
axes[0].colorbar(label="Power [strain²/Hz]")

spec_snr.plot(ax=axes[1], norm="log", vmin=0.1, vmax=100)
axes[1].set_title("SNR spectrogram  (normalize='snr')")
axes[1].colorbar(label="SNR² [dimensionless]")

plt.tight_layout()
plt.show()
"""

NORMALIZE_METHODS_CODE = """\
# ── Compare normalization methods ─────────────────────────────────────────────
methods = ["median", "mean", "percentile"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, m in zip(axes, methods):
    kw = {"percentile": 75.0} if m == "percentile" else {}
    spec_n = spec.normalize(method=m, **kw)
    spec_n.plot(ax=ax, norm="log", vmin=0.1, vmax=100)
    label = f"percentile=75" if m == "percentile" else ""
    ax.set_title(f"method='{m}' {label}")
    ax.colorbar(label="SNR² []")
plt.suptitle("Normalization methods comparison", fontsize=13)
plt.tight_layout()
plt.show()
"""

NORMALIZE_REF_CODE = """\
# ── Reference normalization ───────────────────────────────────────────────────
# Use the first 30 s as a quiet reference segment.
quiet_ts = ts.crop(0, 30)
ref_spec = quiet_ts.spectrogram(STRIDE, fftlength=FFTLEN, overlap=FFTLEN / 2)
reference_psd = np.median(ref_spec.value, axis=0)   # median over the 30-s window

spec_ref = spec.normalize(method="reference", reference=reference_psd)

fig, ax = plt.subplots(figsize=(12, 4))
spec_ref.plot(ax=ax, norm="log", vmin=0.1, vmax=100)
ax.set_title("Reference-normalized spectrogram (30-s quiet baseline)")
ax.colorbar(label="SNR² []")
plt.tight_layout()
plt.show()
"""

CLEAN_THRESHOLD_CODE = """\
# ── threshold cleaning ────────────────────────────────────────────────────────
# Pixels exceeding  median + threshold × MAD  are replaced.
spec_thr, mask = spec_snr.clean(method="threshold", threshold=5.0, return_mask=True)

print(f"Flagged pixels: {mask.sum()} / {mask.size}"
      f"  ({100 * mask.mean():.2f} %)")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
spec_snr.plot(ax=axes[0], norm="log", vmin=0.1, vmax=100)
axes[0].set_title("SNR spectrogram (before threshold clean)")
axes[0].colorbar(label="SNR²")

spec_thr.plot(ax=axes[1], norm="log", vmin=0.1, vmax=100)
axes[1].set_title("After threshold clean  (threshold=5 MAD)")
axes[1].colorbar(label="SNR²")

plt.tight_layout()
plt.show()
"""

CLEAN_ROLLING_CODE = """\
# ── rolling-median cleaning ───────────────────────────────────────────────────
# Divide by a rolling median along the time axis to remove slow trends.
spec_roll = spec.clean(method="rolling_median", window_size=10)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
spec.plot(ax=axes[0], norm="log")
axes[0].set_title("Raw spectrogram")
axes[0].colorbar(label="Power")

spec_roll.plot(ax=axes[1], norm="log")
axes[1].set_title("After rolling-median detrend  (window=10 bins)")
axes[1].colorbar(label="Normalised power []")

plt.tight_layout()
plt.show()
"""

CLEAN_LINE_CODE = """\
# ── line-removal cleaning ─────────────────────────────────────────────────────
# Detect persistent narrowband lines and replace with column median.
spec_noline, line_indices = spec_snr.clean(
    method="line_removal",
    persistence_threshold=0.8,
    amplitude_threshold=3.0,
    return_mask=True,
)

# line_indices is a bool mask here; retrieve actual frequency values
freq_axis = spec.frequencies.value
if hasattr(line_indices, "sum"):   # ndarray mask
    flagged_freqs = freq_axis[np.any(line_indices, axis=0)]
else:
    flagged_freqs = freq_axis[line_indices]
print("Detected line frequencies [Hz]:", np.round(flagged_freqs, 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
spec_snr.plot(ax=axes[0], norm="log", vmin=0.1, vmax=100)
axes[0].set_title("SNR spectrogram (with 60 Hz power line)")
axes[0].colorbar(label="SNR²")

spec_noline.plot(ax=axes[1], norm="log", vmin=0.1, vmax=100)
axes[1].set_title("After line-removal clean")
axes[1].colorbar(label="SNR²")

plt.tight_layout()
plt.show()
"""

CLEAN_COMBINED_CODE = """\
# ── combined pipeline ─────────────────────────────────────────────────────────
# threshold → rolling_median → line_removal in one call.
spec_clean = spec.clean(
    method="combined",
    threshold=5.0,
    window_size=10,
    persistence_threshold=0.8,
    amplitude_threshold=3.0,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
spec.plot(ax=axes[0], norm="log")
axes[0].set_title("Raw spectrogram")
axes[0].colorbar(label="Power")

spec_clean.plot(ax=axes[1], norm="log")
axes[1].set_title("Fully cleaned spectrogram  (method='combined')")
axes[1].colorbar(label="Normalised power []")

plt.tight_layout()
plt.show()
"""

SUMMARY_TABLE_CODE = """\
print("=" * 62)
print(f"{'Method':<22} {'Description':<38}")
print("-" * 62)
rows = [
    ("normalize('snr')",       "÷ median PSD  →  SNR² map"),
    ("normalize('median')",    "÷ median PSD  (alias for snr)"),
    ("normalize('mean')",      "÷ mean PSD"),
    ("normalize('percentile')","÷ Nth-percentile PSD"),
    ("normalize('reference')", "÷ user-supplied reference spectrum"),
    ("clean('threshold')",     "Replace MAD-outlier pixels"),
    ("clean('rolling_median')","Divide by rolling-median trend"),
    ("clean('line_removal')",  "Remove persistent narrowband lines"),
    ("clean('combined')",      "threshold → rolling_median → line_removal"),
]
for m, d in rows:
    print(f"  {m:<20} {d}")
print("=" * 62)
"""

# ============================================================
# English cells
# ============================================================
EN_CELLS = [
    # 0 – title
    {
        "cell_type": "markdown",
        "source": (
            "# Spectrogram Processing: Normalization and Cleaning\n"
            "\n"
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({COLAB_BASE}/advanced_spectrogram_processing.ipynb)\n"
            "\n"
            "This tutorial demonstrates two groups of spectrogram post-processing tools\n"
            "available in **gwexpy**:\n"
            "\n"
            "| API | Purpose |\n"
            "|-----|---------|\n"
            "| `Spectrogram.normalize()` | Convert raw power to SNR or relative units |\n"
            "| `Spectrogram.clean()` | Remove glitches, trends, and persistent lines |\n"
            "\n"
            "**Prerequisites:** [Spectrogram Basics](intro_spectrogram.ipynb)"
        ),
    },
    # 1 – setup
    {"cell_type": "code", "source": SETUP_CODE},
    # 2 – mock data
    {
        "cell_type": "markdown",
        "source": (
            "## 1. Synthetic Data\n"
            "\n"
            "We build a 2-minute strain time series containing:\n"
            "\n"
            "- **Stationary Gaussian noise** with a slowly drifting amplitude (non-stationarity)\n"
            "- **Persistent narrowband line** at 60 Hz (power-line coupling)\n"
            "- **Transient glitches** at 15 s, 45 s, and 90 s\n"
        ),
    },
    {"cell_type": "code", "source": MOCK_DATA_CODE},
    # 3 – baseline plot
    {
        "cell_type": "markdown",
        "source": (
            "## 2. Baseline Spectrogram\n"
            "\n"
            "The raw spectrogram already shows glitches (bright vertical stripes),\n"
            "the 60 Hz line (bright horizontal stripe), and the time-varying noise floor."
        ),
    },
    {"cell_type": "code", "source": BASELINE_PLOT_CODE},
    # 4 – normalize snr
    {
        "cell_type": "markdown",
        "source": (
            "## 3. Normalization\n"
            "\n"
            "### 3a. SNR Spectrogram (`method='snr'`)\n"
            "\n"
            "The most common normalization for transient searches: divide each time slice by\n"
            "the **median PSD** along the time axis.  \n"
            "Result is a dimensionless **SNR²** map where the stationary background ≈ 1."
        ),
    },
    {"cell_type": "code", "source": NORMALIZE_SNR_CODE},
    # 5 – normalize methods comparison
    {
        "cell_type": "markdown",
        "source": (
            "### 3b. Other Normalization Methods\n"
            "\n"
            "| `method=` | Denominator |\n"
            "|-----------|-------------|\n"
            "| `'median'` | Median PSD per frequency bin (identical to `'snr'`) |\n"
            "| `'mean'` | Mean PSD per frequency bin |\n"
            "| `'percentile'` | Nth-percentile PSD (controlled by `percentile=`) |\n"
        ),
    },
    {"cell_type": "code", "source": NORMALIZE_METHODS_CODE},
    # 6 – reference normalization
    {
        "cell_type": "markdown",
        "source": (
            "### 3c. Reference Normalization (`method='reference'`)\n"
            "\n"
            "Supply your own reference spectrum — useful when you want to compare\n"
            "against a specific quiet period or an independently measured noise floor."
        ),
    },
    {"cell_type": "code", "source": NORMALIZE_REF_CODE},
    # 7 – clean threshold
    {
        "cell_type": "markdown",
        "source": (
            "## 4. Cleaning\n"
            "\n"
            "### 4a. Threshold Cleaning (`method='threshold'`)\n"
            "\n"
            "Pixels that exceed `median + threshold × MAD` (per frequency bin) are\n"
            "flagged as outliers and replaced.  Replacement strategy is controlled by\n"
            "`fill=` (`'median'`, `'nan'`, `'zero'`, or `'interpolate'`).\n"
            "\n"
            "This removes short-duration **glitches** that appear as bright vertical streaks."
        ),
    },
    {"cell_type": "code", "source": CLEAN_THRESHOLD_CODE},
    # 8 – rolling median
    {
        "cell_type": "markdown",
        "source": (
            "### 4b. Rolling-Median Detrending (`method='rolling_median'`)\n"
            "\n"
            "Divide each column by a running median along the time axis.  \n"
            "This removes **slow non-stationary trends** without distorting short-duration features."
        ),
    },
    {"cell_type": "code", "source": CLEAN_ROLLING_CODE},
    # 9 – line removal
    {
        "cell_type": "markdown",
        "source": (
            "### 4c. Persistent Line Removal (`method='line_removal'`)\n"
            "\n"
            "A frequency bin is flagged as a persistent line if it exceeds\n"
            "`amplitude_threshold × global_median` for more than\n"
            "`persistence_threshold` fraction of time bins.  \n"
            "Detected bins are replaced with their time-median."
        ),
    },
    {"cell_type": "code", "source": CLEAN_LINE_CODE},
    # 10 – combined
    {
        "cell_type": "markdown",
        "source": (
            "### 4d. Full Cleaning Pipeline (`method='combined'`)\n"
            "\n"
            "Runs threshold → rolling-median → line-removal in sequence.  \n"
            "Recommended as a one-stop solution for commissioning-style data quality checks."
        ),
    },
    {"cell_type": "code", "source": CLEAN_COMBINED_CODE},
    # 11 – summary
    {
        "cell_type": "markdown",
        "source": (
            "## 5. Summary\n"
            "\n"
            "All methods and their purposes:"
        ),
    },
    {"cell_type": "code", "source": SUMMARY_TABLE_CODE},
]

# ============================================================
# Japanese cells
# ============================================================
JA_CELLS = [
    # 0 – title
    {
        "cell_type": "markdown",
        "source": (
            "# スペクトログラム処理：正規化とクリーニング\n"
            "\n"
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({COLAB_BASE}/advanced_spectrogram_processing.ipynb)\n"
            "\n"
            "このチュートリアルでは **gwexpy** が提供するスペクトログラム後処理ツール 2 種類を解説します：\n"
            "\n"
            "| API | 目的 |\n"
            "|-----|------|\n"
            "| `Spectrogram.normalize()` | 生パワーを SNR や相対単位に変換 |\n"
            "| `Spectrogram.clean()` | グリッチ・トレンド・持続的ラインの除去 |\n"
            "\n"
            "**前提:** [スペクトログラム入門](intro_spectrogram.ipynb)"
        ),
    },
    # 1 – setup
    {"cell_type": "code", "source": SETUP_CODE},
    # 2 – mock data
    {
        "cell_type": "markdown",
        "source": (
            "## 1. 合成データの準備\n"
            "\n"
            "以下を含む 2 分間の strain 時系列を生成します：\n"
            "\n"
            "- **定常ガウスノイズ**（ゆっくり変化する振幅 → 非定常）\n"
            "- **60 Hz 持続ライン**（電源ハム）\n"
            "- **過渡グリッチ**（15 s, 45 s, 90 s）\n"
        ),
    },
    {"cell_type": "code", "source": MOCK_DATA_CODE},
    # 3 – baseline
    {
        "cell_type": "markdown",
        "source": (
            "## 2. ベースライン スペクトログラム\n"
            "\n"
            "生スペクトログラムには、グリッチ（明るい縦縞）、60 Hz ライン（明るい横縞）、\n"
            "時間変動するノイズフロアがすでに見えています。"
        ),
    },
    {"cell_type": "code", "source": BASELINE_PLOT_CODE},
    # 4 – normalize snr
    {
        "cell_type": "markdown",
        "source": (
            "## 3. 正規化\n"
            "\n"
            "### 3a. SNR スペクトログラム (`method='snr'`)\n"
            "\n"
            "過渡現象探索で最もよく使われる正規化：各時刻スライスを時間軸方向の\n"
            "**中央値 PSD** で割ります。  \n"
            "結果は無次元の **SNR²** マップで、定常背景 ≈ 1 になります。"
        ),
    },
    {"cell_type": "code", "source": NORMALIZE_SNR_CODE},
    # 5 – normalize methods
    {
        "cell_type": "markdown",
        "source": (
            "### 3b. その他の正規化メソッド\n"
            "\n"
            "| `method=` | 分母 |\n"
            "|-----------|------|\n"
            "| `'median'` | 周波数ビン毎の中央値 PSD（`'snr'` と同一） |\n"
            "| `'mean'` | 周波数ビン毎の平均 PSD |\n"
            "| `'percentile'` | 第 N パーセンタイル PSD（`percentile=` で指定） |\n"
        ),
    },
    {"cell_type": "code", "source": NORMALIZE_METHODS_CODE},
    # 6 – reference
    {
        "cell_type": "markdown",
        "source": (
            "### 3c. 参照スペクトル正規化 (`method='reference'`)\n"
            "\n"
            "独自の参照スペクトルを渡して正規化します。  \n"
            "特定の静粛期間や独立計測したノイズフロアとの比較に有用です。"
        ),
    },
    {"cell_type": "code", "source": NORMALIZE_REF_CODE},
    # 7 – clean threshold
    {
        "cell_type": "markdown",
        "source": (
            "## 4. クリーニング\n"
            "\n"
            "### 4a. 閾値クリーニング (`method='threshold'`)\n"
            "\n"
            "周波数ビン毎に `中央値 + threshold × MAD` を超えるピクセルを外れ値として検出し、\n"
            "`fill=` で指定した方法で置換します。  \n"
            "短時間の**グリッチ**（明るい縦縞）を除去するのに適しています。"
        ),
    },
    {"cell_type": "code", "source": CLEAN_THRESHOLD_CODE},
    # 8 – rolling median
    {
        "cell_type": "markdown",
        "source": (
            "### 4b. ローリング中央値デトレンド (`method='rolling_median'`)\n"
            "\n"
            "時間軸に沿ったローリング中央値で各列を割ります。  \n"
            "短時間特徴を歪めずに**ゆっくりとした非定常トレンド**を除去します。"
        ),
    },
    {"cell_type": "code", "source": CLEAN_ROLLING_CODE},
    # 9 – line removal
    {
        "cell_type": "markdown",
        "source": (
            "### 4c. 持続ライン除去 (`method='line_removal'`)\n"
            "\n"
            "時間ビンの `persistence_threshold` 割以上で `amplitude_threshold × 全体中央値` を\n"
            "超える周波数ビンを持続ラインとして検出し、時間方向の中央値で置換します。"
        ),
    },
    {"cell_type": "code", "source": CLEAN_LINE_CODE},
    # 10 – combined
    {
        "cell_type": "markdown",
        "source": (
            "### 4d. 完全クリーニングパイプライン (`method='combined'`)\n"
            "\n"
            "threshold → rolling-median → line-removal を順に適用します。  \n"
            "コミッショニングスタイルのデータ品質チェックに推奨のワンストップ解法です。"
        ),
    },
    {"cell_type": "code", "source": CLEAN_COMBINED_CODE},
    # 11 – summary
    {
        "cell_type": "markdown",
        "source": (
            "## 5. まとめ\n"
            "\n"
            "全メソッドと目的の一覧："
        ),
    },
    {"cell_type": "code", "source": SUMMARY_TABLE_CODE},
]


# ============================================================
# Notebook builder
# ============================================================
def _cell(cell_type: str, source: str, lang: str = "python") -> dict:
    c: dict = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def make_nb(cells_spec: list[dict]) -> dict:
    cells = [_cell(s["cell_type"], s["source"]) for s in cells_spec]
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
        "cells": cells,
    }


if __name__ == "__main__":
    out_en = (
        REPO / "docs/web/en/user_guide/tutorials/advanced_spectrogram_processing.ipynb"
    )
    out_ja = (
        REPO / "docs/web/ja/user_guide/tutorials/advanced_spectrogram_processing.ipynb"
    )

    out_en.write_text(json.dumps(make_nb(EN_CELLS), ensure_ascii=False, indent=1) + "\n")
    out_ja.write_text(json.dumps(make_nb(JA_CELLS), ensure_ascii=False, indent=1) + "\n")

    print(f"Created: {out_en}")
    print(f"Created: {out_ja}")
    print("Done.")
