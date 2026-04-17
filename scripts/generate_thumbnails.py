"""
Generate three Visual-Examples thumbnail images for the GWexpy hub pages.

Outputs (all 400×300 px, PNG, white background, DPI≥100):
  docs/_static/images/case_noise_budget_thumb.png
  docs/_static/images/case_transfer_function_thumb.png
  docs/_static/images/case_active_damping_thumb.png

Usage:
    python scripts/generate_thumbnails.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGES = ROOT / "docs/_static/images"
DPI = 100
W, H = 4.0, 3.0  # inches → 400×300 px at DPI=100


# ─────────────────────────────────────────────────────────────────────────────
# 1. Noise Budget (BrUCo スタイル)
# ─────────────────────────────────────────────────────────────────────────────

def _noise_budget_thumb() -> None:
    rng = np.random.default_rng(0)
    freqs = np.logspace(0, 2.7, 400)  # 1 – 500 Hz

    total = 1e-19 / freqs**0.8 + 1e-21
    seismic = 1e-19 / freqs**1.5 + rng.uniform(0, 5e-22, len(freqs))
    thermal = np.full_like(freqs, 3e-21) + rng.uniform(0, 1e-22, len(freqs))
    quantum = 1e-22 * freqs**0.5 + rng.uniform(0, 5e-23, len(freqs))

    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    ax.loglog(freqs, total, "k-", lw=2, label="Total")
    ax.loglog(freqs, seismic, color="#e74c3c", lw=1.5, label="Seismic")
    ax.loglog(freqs, thermal, color="#3498db", lw=1.5, label="Thermal")
    ax.loglog(freqs, quantum, color="#2ecc71", lw=1.5, label="Quantum")
    ax.set_xlabel("Frequency [Hz]", fontsize=9)
    ax.set_ylabel("Strain [1/√Hz]", fontsize=9)
    ax.set_title("Noise Budget (BrUCo)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(1, 500)
    fig.tight_layout()
    out = IMAGES / "case_noise_budget_thumb.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transfer Function (FrequencySeriesMatrix スタイル)
# ─────────────────────────────────────────────────────────────────────────────

def _transfer_function_thumb() -> None:
    rng = np.random.default_rng(1)
    freqs = np.logspace(0, 3, 500)

    # 2次共振系の伝達関数
    f0, q = 100.0, 8.0
    tf = 1.0 / (1 - (freqs / f0) ** 2 + 1j * freqs / (f0 * q))
    amp = np.abs(tf) * (1 + rng.normal(0, 0.02, len(freqs)))
    phase_deg = np.angle(tf, deg=True) + rng.normal(0, 1, len(freqs))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, H), dpi=DPI, sharex=True)
    ax1.semilogx(freqs, 20 * np.log10(amp), color="#2980b9", lw=1.5)
    ax1.set_ylabel("Magnitude [dB]", fontsize=8)
    ax1.set_title("Transfer Function (FrequencySeriesMatrix)", fontsize=9, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.25)

    ax2.semilogx(freqs, phase_deg, color="#e67e22", lw=1.5)
    ax2.set_ylabel("Phase [deg]", fontsize=8)
    ax2.set_xlabel("Frequency [Hz]", fontsize=8)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.set_xlim(1, 1000)

    fig.tight_layout()
    out = IMAGES / "case_transfer_function_thumb.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Active Damping (Before / After スタイル)
# ─────────────────────────────────────────────────────────────────────────────

def _active_damping_thumb() -> None:
    rng = np.random.default_rng(2)
    t = np.linspace(0, 2.0, 4000)
    f_vio = 3.5  # Hz — violin mode

    before = np.exp(-0.5 * t) * np.cos(2 * np.pi * f_vio * t)
    before += rng.normal(0, 0.04, len(t))

    after = np.exp(-8.0 * t) * np.cos(2 * np.pi * f_vio * t)
    after += rng.normal(0, 0.04, len(t))

    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    ax.plot(t, before, color="#e74c3c", lw=1.2, alpha=0.85, label="Before (free)")
    ax.plot(t, after, color="#2ecc71", lw=1.5, label="After (damped)")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Displacement [a.u.]", fontsize=9)
    ax.set_title("Active Damping — Before / After", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 2.0)

    fig.tight_layout()
    out = IMAGES / "case_active_damping_thumb.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    _noise_budget_thumb()
    _transfer_function_thumb()
    _active_damping_thumb()
    print("All thumbnails generated.")


if __name__ == "__main__":
    main()
