"""
Generate hero_plot.png for the GWexpy gateway landing page.

Output: docs/_static/images/hero_plot.png
  - 800 x 450 px, PNG, white background
  - Shows FrequencySeriesMatrix multi-channel frequency response (colour-mapped)

Usage:
    python scripts/generate_hero_plot.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── repo root を sys.path に追加 ─────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gwexpy import FrequencySeriesMatrix  # noqa: E402

OUTPUT = ROOT / "docs/_static/images/hero_plot.png"
DPI = 100
WIDTH_PX, HEIGHT_PX = 800, 450


def _make_synthetic_fsm() -> FrequencySeriesMatrix:
    """Synthetic multi-channel FrequencySeriesMatrix (Lorentzian peaks)."""
    from gwexpy.frequencyseries import FrequencySeries

    rng = np.random.default_rng(42)
    freqs = np.linspace(1, 512, 2048)
    n_ch = 6
    channels = [f"H1:STRAIN_CH{i + 1}" for i in range(n_ch)]

    data = np.zeros((n_ch, len(freqs)))
    base_freqs = [50, 120, 200, 280, 350, 440]
    for i, f0 in enumerate(base_freqs):
        q = 20 + i * 5
        peak = 1.0 / (1 + ((freqs - f0) / (f0 / q)) ** 2)
        noise_floor = 10 ** (-3 + rng.normal(0, 0.05, len(freqs)))
        data[i] = peak + noise_floor

    series_list = [
        FrequencySeries(data[i], frequencies=freqs, name=ch, unit="strain/rtHz")
        for i, ch in enumerate(channels)
    ]
    return FrequencySeriesMatrix(series_list)


def main() -> None:
    fsm = _make_synthetic_fsm()

    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)

    cmap = plt.get_cmap("viridis")
    n_ch = len(fsm.channel_names)
    for i, name in enumerate(fsm.channel_names):
        color = cmap(i / max(n_ch - 1, 1))
        # FSM[i] は shape (1, n_freq) なので .squeeze() で平坦化
        data_i = fsm[i].squeeze()
        ax.semilogy(
            data_i.frequencies.value,
            np.abs(data_i.value),
            color=color,
            linewidth=1.4,
            alpha=0.9,
            label=name,
        )

    ax.set_xlabel("Frequency [Hz]", fontsize=12)
    ax.set_ylabel("Amplitude [strain/√Hz]", fontsize=12)
    ax.set_title(
        "GWexpy — FrequencySeriesMatrix: Multi-channel Frequency Response",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, 512)

    # カラーバー（チャンネル軸のグラデーションを表示）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_ch - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Channel index", fontsize=10)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT}  ({OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
