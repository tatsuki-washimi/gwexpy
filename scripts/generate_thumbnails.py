"""Generate three Visual Examples thumbnail images for the GWexpy hub pages.

Outputs (all 400 x 300 px, PNG, white background, DPI >= 100):
  docs/_static/images/case_noise_budget_thumb.png
  docs/_static/images/case_transfer_function_thumb.png
  docs/_static/images/case_active_damping_thumb.png

Usage:
    python scripts/generate_thumbnails.py
"""

from __future__ import annotations

import io
import pathlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGES = ROOT / "docs/_static/images"
DPI = 100
WIDTH_PX = 400
HEIGHT_PX = 300
FIGSIZE = (WIDTH_PX / DPI, HEIGHT_PX / DPI)


def _save_rgb_figure(fig: plt.Figure, output: Path) -> None:
    """Save an exact-size RGB PNG with an opaque white background."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor="white")
    buffer.seek(0)

    image = Image.open(buffer).convert("RGBA")
    if image.size != (WIDTH_PX, HEIGHT_PX):
        image = image.resize((WIDTH_PX, HEIGHT_PX), Image.Resampling.LANCZOS)

    background = Image.new("RGB", image.size, "white")
    background.paste(image, mask=image.getchannel("A"))
    background.save(output, format="PNG", dpi=(DPI, DPI))


def _noise_budget_thumb(output_dir: Path) -> Path:
    rng = np.random.default_rng(0)
    freqs = np.logspace(0, 2.7, 360)

    total = 1.2e-19 / freqs**0.82 + 1.3e-21
    seismic = 1.4e-19 / freqs**1.5 + rng.uniform(0, 6e-22, len(freqs))
    thermal = np.full_like(freqs, 3.5e-21) + rng.uniform(0, 2e-22, len(freqs))
    quantum = 1.1e-22 * freqs**0.5 + rng.uniform(0, 6e-23, len(freqs))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.loglog(freqs, total, color="black", lw=2.0, label="Total")
    ax.loglog(freqs, seismic, color="#d1495b", lw=1.4, label="Seismic")
    ax.loglog(freqs, thermal, color="#00798c", lw=1.4, label="Thermal")
    ax.loglog(freqs, quantum, color="#edae49", lw=1.4, label="Quantum")
    ax.set_xlim(1, 500)
    ax.set_xlabel("Frequency [Hz]", fontsize=8.5)
    ax.set_ylabel("Strain [1/√Hz]", fontsize=8.5)
    ax.set_title("BrUCo Noise Budget", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.92)

    output = output_dir / "case_noise_budget_thumb.png"
    _save_rgb_figure(fig, output)
    plt.close(fig)
    return output


def _transfer_function_thumb(output_dir: Path) -> Path:
    freqs = np.logspace(0, 3, 420)
    centers = [45.0, 90.0, 180.0]
    q_values = [5.0, 7.0, 10.0]
    transfer_rows = []

    for center, q_value in zip(centers, q_values, strict=True):
        response = 1.0 / (1 - (freqs / center) ** 2 + 1j * freqs / (center * q_value))
        transfer_rows.append(response)

    transfer = np.asarray(transfer_rows)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(transfer), 1.0e-8))
    phase_deg = np.angle(transfer[1], deg=True)

    fig, (ax_mag, ax_phase) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        dpi=DPI,
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.08},
    )
    image = ax_mag.imshow(
        magnitude_db,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[freqs[0], freqs[-1], 0, transfer.shape[0]],
    )
    ax_mag.set_xscale("log")
    ax_mag.set_ylabel("Channel", fontsize=8)
    ax_mag.set_yticks(np.arange(transfer.shape[0]) + 0.5)
    ax_mag.set_yticklabels(["A", "B", "C"], fontsize=7.5)
    ax_mag.set_title(
        "FrequencySeriesMatrix Transfer Function",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )
    colorbar = fig.colorbar(image, ax=ax_mag, pad=0.01, fraction=0.04)
    colorbar.ax.tick_params(labelsize=6.5)
    colorbar.set_label("Magnitude [dB]", fontsize=7)

    ax_phase.semilogx(freqs, phase_deg, color="#d1495b", lw=1.6)
    ax_phase.axhline(0.0, color="black", lw=0.6, ls="--", alpha=0.6)
    ax_phase.set_xlabel("Frequency [Hz]", fontsize=8)
    ax_phase.set_ylabel("Phase [deg]", fontsize=8)
    ax_phase.grid(True, which="both", alpha=0.22)
    ax_phase.tick_params(labelsize=7.5)

    output = output_dir / "case_transfer_function_thumb.png"
    _save_rgb_figure(fig, output)
    plt.close(fig)
    return output


def _active_damping_thumb(output_dir: Path) -> Path:
    rng = np.random.default_rng(2)
    time = np.linspace(0, 2.0, 1800)
    violin_hz = 3.5

    before = np.exp(-0.55 * time) * np.cos(2 * np.pi * violin_hz * time)
    before += rng.normal(0, 0.03, len(time))

    after = np.exp(-4.0 * time) * np.cos(2 * np.pi * violin_hz * time)
    after += rng.normal(0, 0.02, len(time))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(time, before, color="#d1495b", lw=1.0, alpha=0.8, label="Before")
    ax.plot(time, after, color="#2a9d8f", lw=1.5, label="After")
    ax.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.7)
    ax.set_xlim(0, 2.0)
    ax.set_xlabel("Time [s]", fontsize=8.5)
    ax.set_ylabel("Displacement [a.u.]", fontsize=8.5)
    ax.set_title(
        "Active Damping: Before / After",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)

    output = output_dir / "case_active_damping_thumb.png"
    _save_rgb_figure(fig, output)
    plt.close(fig)
    return output


def generate_all_thumbnails(output_dir: Path | str = IMAGES) -> dict[str, str]:
    """Generate all thumbnail images and return their output paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "case_noise_budget_thumb.png": str(_noise_budget_thumb(output_dir)),
        "case_transfer_function_thumb.png": str(_transfer_function_thumb(output_dir)),
        "case_active_damping_thumb.png": str(_active_damping_thumb(output_dir)),
    }
    return outputs


def main() -> None:
    """Generate the default thumbnail set and print their output paths."""
    outputs = generate_all_thumbnails()
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
