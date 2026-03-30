"""
Generate advanced_peak_tracking.ipynb (EN + JA).

Covers:
  - spectrogram2() per-frame PSD
  - scipy.signal.find_peaks on each time frame
  - nearest-neighbor line tracking
  - multi-line tracking with overlay plot
  - frequency drift TimeSeries visualization

Usage:
    python scripts/make_peak_tracking_notebook.py
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

from gwexpy.timeseries import TimeSeries

plt.rcParams["figure.figsize"] = (12, 4)
"""

MOCK_DATA_CODE = """\
rng = np.random.default_rng(0)

DURATION = 120   # s
FS       = 512   # Hz
t = np.arange(0, DURATION, 1 / FS)

# ── Line 1: linear drift 58 → 62 Hz ──────────────────────────────────────────
f1_t   = 58.0 + (4.0 / DURATION) * t          # frequency as a function of t
phi1   = 2 * np.pi * np.cumsum(f1_t) / FS      # instantaneous phase
s1     = 0.5 * np.sin(phi1)

# ── Line 2: 120 Hz with slow sinusoidal wobble ±0.5 Hz ───────────────────────
f2_t   = 120.0 + 0.5 * np.sin(2 * np.pi * 0.03 * t)
phi2   = 2 * np.pi * np.cumsum(f2_t) / FS
s2     = 0.3 * np.sin(phi2)

# ── Background Gaussian noise ────────────────────────────────────────────────
noise = rng.normal(0, 0.05, len(t))

strain = s1 + s2 + noise
ts = TimeSeries(strain, dt=1.0 / FS, name="STRAIN", unit="strain")

print(f"Duration : {DURATION} s   |  Sample rate: {FS} Hz")
print(f"Line 1   : 58 → 62 Hz (linear drift)")
print(f"Line 2   : 120 Hz ± 0.5 Hz (slow wobble)")
"""

SPEC_CODE = """\
FFTLEN  = 4.0   # FFT length [s]
OVERLAP = 2.0   # overlap [s]

spec = ts.spectrogram2(FFTLEN, overlap=OVERLAP)
print("Spectrogram shape (time × freq):", spec.shape)
print(f"Time bins : {spec.shape[0]}  |  Freq bins: {spec.shape[1]}")
print(f"Freq resolution: {spec.df.value:.4f} Hz")

fig, ax = plt.subplots(figsize=(12, 4))
spec.plot(ax=ax, norm="log")
ax.set_ylim(0, 200)
ax.set_title("Raw spectrogram (0–200 Hz)")
ax.colorbar(label="Power [strain²/Hz]")
plt.tight_layout()
plt.show()
"""

SINGLE_FRAME_CODE = """\
# ── Inspect one time frame ────────────────────────────────────────────────────
T_IDX = spec.shape[0] // 4          # pick a time index near the 30-s mark
row   = spec.value[T_IDX, :]        # PSD at this time
freqs = spec.frequencies.value      # frequency axis [Hz]

peaks_idx, props = sp_find_peaks(
    row,
    height=row.max() * 0.02,        # at least 2 % of frame maximum
    distance=int(5 / spec.df.value),# minimum 5 Hz separation
)

print(f"Time index {T_IDX}  (t ≈ {spec.times.value[T_IDX]:.1f} s)")
print(f"Detected peaks at: {freqs[peaks_idx].round(2)} Hz")

fig, ax = plt.subplots(figsize=(10, 3))
ax.semilogy(freqs, row, lw=0.8, label="PSD")
ax.semilogy(freqs[peaks_idx], row[peaks_idx], "rv", ms=8, label="peaks")
ax.set_xlim(40, 160)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Power [strain²/Hz]")
ax.set_title(f"Single frame at t ≈ {spec.times.value[T_IDX]:.1f} s")
ax.legend()
plt.tight_layout()
plt.show()
"""

SINGLE_TRACK_CODE = """\
# ── Nearest-neighbor single-line tracker ─────────────────────────────────────
def track_line(spec, seed_freq, max_jump=2.0, min_height_frac=0.01):
    \"\"\"
    Track one spectral line through a spectrogram using nearest-neighbor search.

    Parameters
    ----------
    spec            : gwexpy Spectrogram
    seed_freq       : float  — starting frequency estimate [Hz]
    max_jump        : float  — maximum allowed Hz jump between frames
    min_height_frac : float  — minimum height as fraction of frame peak

    Returns
    -------
    track : ndarray, shape (n_times,)
        Tracked frequency at each time bin (NaN = no detection).
    \"\"\"
    freqs   = spec.frequencies.value
    n_times = spec.shape[0]
    track   = np.full(n_times, np.nan)
    current = float(seed_freq)

    for t in range(n_times):
        row = spec.value[t, :]
        thr = row.max() * min_height_frac
        idx, _ = sp_find_peaks(row, height=thr,
                               distance=int(2 / spec.df.value))
        if len(idx) == 0:
            continue
        pf      = freqs[idx]
        nearest = int(np.argmin(np.abs(pf - current)))
        if np.abs(pf[nearest] - current) < max_jump:
            track[t] = pf[nearest]
            current   = pf[nearest]   # propagate seed to next frame

    return track


track_line1 = track_line(spec, seed_freq=58.5)
times = spec.times.value

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

spec.plot(ax=axes[0], norm="log")
axes[0].set_ylim(40, 80)
axes[0].set_title("Spectrogram (40–80 Hz)")
axes[0].colorbar(label="Power")

mask = ~np.isnan(track_line1)
axes[1].plot(times[mask], track_line1[mask], "o-", ms=3, color="tab:orange",
             label="tracked frequency")
axes[1].axhline(58, ls="--", color="gray", lw=0.8, label="start 58 Hz")
axes[1].axhline(62, ls="--", color="gray", lw=0.8, label="end 62 Hz")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Frequency [Hz]")
axes[1].set_title("Tracked Line 1 frequency vs time")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()

valid = ~np.isnan(track_line1)
print(f"Detection rate: {valid.sum()}/{len(track_line1)} frames "
      f"({100 * valid.mean():.0f} %)")
print(f"Freq range: {np.nanmin(track_line1):.2f} – {np.nanmax(track_line1):.2f} Hz")
"""

MULTI_TRACK_CODE = """\
# ── Track both lines simultaneously ──────────────────────────────────────────
SEEDS = {
    "Line1 (58→62 Hz)": 58.5,
    "Line2 (120 Hz)":   120.0,
}

tracks = {name: track_line(spec, seed, max_jump=2.5)
          for name, seed in SEEDS.items()}

# ── Overlay on spectrogram ────────────────────────────────────────────────────
colors = ["tab:red", "tab:cyan"]

fig, ax = plt.subplots(figsize=(12, 5))
spec.plot(ax=ax, norm="log")
ax.set_ylim(40, 150)
ax.colorbar(label="Power [strain²/Hz]")

for (name, trk), color in zip(tracks.items(), colors):
    mask = ~np.isnan(trk)
    ax.plot(times[mask], trk[mask], "o", ms=3, color=color,
            label=name, alpha=0.85)

ax.set_title("Multi-line tracking overlay on spectrogram")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()
"""

DRIFT_VIS_CODE = """\
# ── Build TimeSeries from tracking results and plot frequency evolution ────────
fig, axes = plt.subplots(len(tracks), 1, figsize=(12, 5), sharex=True)

for ax, (name, trk) in zip(axes, tracks.items()):
    mask    = ~np.isnan(trk)
    t_valid = times[mask]
    f_valid = trk[mask]

    ts_freq = TimeSeries(f_valid, times=t_valid, name=name, unit="Hz")
    ts_freq.plot(ax=ax)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(name)

axes[-1].set_xlabel("Time [s]")
plt.suptitle("Spectral Line Frequency vs Time", fontsize=13)
plt.tight_layout()
plt.show()

# ── Drift statistics ─────────────────────────────────────────────────────────
print(f"{'Line':<25} {'min [Hz]':>10} {'max [Hz]':>10} {'drift [Hz]':>12}")
print("-" * 60)
for name, trk in tracks.items():
    lo, hi = np.nanmin(trk), np.nanmax(trk)
    print(f"{name:<25} {lo:>10.3f} {hi:>10.3f} {hi - lo:>12.3f}")
"""

SUMMARY_CODE = """\
print("=" * 60)
print("Peak / Line Tracking — Key Steps")
print("-" * 60)
steps = [
    ("1. spectrogram2()",       "Compute time-frequency power map"),
    ("2. sp_find_peaks(row)",   "Detect peaks in each time frame"),
    ("3. nearest-neighbour",    "Connect peaks across frames by min-dist"),
    ("4. NaN on miss",          "Mark undetected frames as NaN"),
    ("5. TimeSeries(track)",    "Build frequency-vs-time TimeSeries"),
]
for s, d in steps:
    print(f"  {s:<25} {d}")
print("=" * 60)
"""

# ============================================================
# English cells
# ============================================================
EN_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "# Peak / Line Time Tracking\n"
            "\n"
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({COLAB_BASE}/advanced_peak_tracking.ipynb)\n"
            "\n"
            "This tutorial extends [advanced_peak_detection.ipynb](advanced_peak_detection.ipynb)\n"
            "by showing how to **track spectral lines through time** using a spectrogram.\n"
            "\n"
            "Real-world examples:\n"
            "- Power-line harmonics drifting with mains frequency\n"
            "- Violin-mode resonances shifting with temperature\n"
            "- Calibration lines with intentional frequency modulation\n"
            "\n"
            "**Strategy**: compute a spectrogram → detect peaks per frame\n"
            "→ connect with *nearest-neighbour* algorithm."
        ),
    },
    {"cell_type": "code", "source": SETUP_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 1. Synthetic Data\n"
            "\n"
            "Two spectral lines embedded in white Gaussian noise:\n"
            "\n"
            "| Line | Behaviour | Amplitude |\n"
            "|------|-----------|----------|\n"
            "| 1 | 58 → 62 Hz linear drift | 0.5 |\n"
            "| 2 | 120 Hz ± 0.5 Hz sinusoidal wobble | 0.3 |"
        ),
    },
    {"cell_type": "code", "source": MOCK_DATA_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 2. Spectrogram Overview\n"
            "\n"
            "`ts.spectrogram2(fftlength, overlap)` returns a time–frequency power map.\n"
            "The diagonal streak (Line 1 drift) and slightly wavy horizontal stripe\n"
            "(Line 2 wobble) are already visible."
        ),
    },
    {"cell_type": "code", "source": SPEC_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 3. Frame-by-Frame Peak Detection\n"
            "\n"
            "`spec.value[t, :]` is the 1-D PSD array at time index `t`.\n"
            "Pass it to `scipy.signal.find_peaks` with a height threshold\n"
            "and a minimum-distance constraint (in frequency bins)."
        ),
    },
    {"cell_type": "code", "source": SINGLE_FRAME_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 4. Single-Line Tracking  (nearest-neighbour)\n"
            "\n"
            "**Algorithm**\n"
            "\n"
            "```\n"
            "seed = initial frequency estimate\n"
            "for each time frame t:\n"
            "    detect all peaks in PSD(t)\n"
            "    find the peak nearest to current seed\n"
            "    if distance < max_jump:\n"
            "        record that frequency\n"
            "        update seed ← new frequency   # propagate\n"
            "    else:\n"
            "        record NaN                     # no detection\n"
            "```\n"
            "\n"
            "Updating the seed at each step lets the tracker follow a\n"
            "slowly drifting line rather than locking to the initial frequency."
        ),
    },
    {"cell_type": "code", "source": SINGLE_TRACK_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 5. Multi-Line Tracking\n"
            "\n"
            "Run one tracker per seed frequency and overlay results on\n"
            "the spectrogram for visual verification."
        ),
    },
    {"cell_type": "code", "source": MULTI_TRACK_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 6. Frequency Drift Visualisation\n"
            "\n"
            "Wrap each track in a `TimeSeries` to plot frequency evolution\n"
            "and compute basic drift statistics."
        ),
    },
    {"cell_type": "code", "source": DRIFT_VIS_CODE},
    {"cell_type": "code", "source": SUMMARY_CODE},
]

# ============================================================
# Japanese cells
# ============================================================
JA_CELLS = [
    {
        "cell_type": "markdown",
        "source": (
            "# ピーク/スペクトル線の時間追跡\n"
            "\n"
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({COLAB_BASE}/advanced_peak_tracking.ipynb)\n"
            "\n"
            "このチュートリアルは [advanced_peak_detection.ipynb](advanced_peak_detection.ipynb) の続編です。\n"
            "スペクトログラムを使って**スペクトル線を時間方向に追跡**する方法を解説します。\n"
            "\n"
            "典型的な応用例：\n"
            "- 商用電源周波数変動に伴うハム波のドリフト\n"
            "- バイオリンモード共振周波数の温度依存ドリフト\n"
            "- 周波数変調されたキャリブレーション注入ライン\n"
            "\n"
            "**戦略**: スペクトログラムを計算 → 各フレームでピーク検出\n"
            "→ *最近傍法* で時刻間を接続。"
        ),
    },
    {"cell_type": "code", "source": SETUP_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 1. 合成データの準備\n"
            "\n"
            "白色ガウスノイズに埋め込まれた 2 本のスペクトル線：\n"
            "\n"
            "| ライン | 挙動 | 振幅 |\n"
            "|--------|------|------|\n"
            "| 1 | 58 → 62 Hz 線形ドリフト | 0.5 |\n"
            "| 2 | 120 Hz ± 0.5 Hz 正弦波ウォブル | 0.3 |"
        ),
    },
    {"cell_type": "code", "source": MOCK_DATA_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 2. スペクトログラム概観\n"
            "\n"
            "`ts.spectrogram2(fftlength, overlap)` で時間-周波数パワーマップを取得します。\n"
            "斜めのストリーク（ライン 1 のドリフト）とわずかに波打つ水平ストライプ\n"
            "（ライン 2 のウォブル）がすでに見えています。"
        ),
    },
    {"cell_type": "code", "source": SPEC_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 3. フレーム毎のピーク検出\n"
            "\n"
            "`spec.value[t, :]` は時刻インデックス `t` の 1 次元 PSD 配列です。\n"
            "高さ閾値と最小周波数間隔制約を指定して `scipy.signal.find_peaks` に渡します。"
        ),
    },
    {"cell_type": "code", "source": SINGLE_FRAME_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 4. 単一ライン追跡（最近傍法）\n"
            "\n"
            "**アルゴリズム**\n"
            "\n"
            "```\n"
            "seed = 初期周波数の推定値\n"
            "for 各時刻フレーム t:\n"
            "    PSD(t) の全ピークを検出\n"
            "    現在の seed に最も近いピークを探す\n"
            "    if 距離 < max_jump:\n"
            "        その周波数を記録\n"
            "        seed ← 新しい周波数  # 次フレームへ伝播\n"
            "    else:\n"
            "        NaN を記録          # 未検出\n"
            "```\n"
            "\n"
            "各ステップで seed を更新することが重要です。\n"
            "これによりゆっくりドリフトするラインを追いかけられます。"
        ),
    },
    {"cell_type": "code", "source": SINGLE_TRACK_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 5. 複数ライン同時追跡\n"
            "\n"
            "seed 周波数ごとにトラッカーを走らせ、結果をスペクトログラムに\n"
            "オーバーレイして視覚的に確認します。"
        ),
    },
    {"cell_type": "code", "source": MULTI_TRACK_CODE},
    {
        "cell_type": "markdown",
        "source": (
            "## 6. 周波数ドリフトの可視化\n"
            "\n"
            "各トラック結果を `TimeSeries` でラップして周波数変化をプロットし、\n"
            "ドリフト統計を計算します。"
        ),
    },
    {"cell_type": "code", "source": DRIFT_VIS_CODE},
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
    out_en = REPO / "docs/web/en/user_guide/tutorials/advanced_peak_tracking.ipynb"
    out_ja = REPO / "docs/web/ja/user_guide/tutorials/advanced_peak_tracking.ipynb"

    out_en.write_text(json.dumps(make_nb(EN_CELLS), ensure_ascii=False, indent=1) + "\n")
    out_ja.write_text(json.dumps(make_nb(JA_CELLS), ensure_ascii=False, indent=1) + "\n")

    print(f"Created: {out_en}  ({len(EN_CELLS)} cells)")
    print(f"Created: {out_ja}  ({len(JA_CELLS)} cells)")
    print("Done.")
