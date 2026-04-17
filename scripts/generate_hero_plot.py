"""Generate hero_plot.png for the GWexpy gateway landing page.

Output: docs/_static/images/hero_plot.png
  - 800 x 450 px, PNG, white background
  - Shows FrequencySeriesMatrix matrix data plus a fitted channel response

Usage:
    python scripts/generate_hero_plot.py
"""

from __future__ import annotations

import io
import pathlib
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gwexpy  # noqa: E402
from gwexpy import FrequencySeriesMatrix  # noqa: E402

OUTPUT = ROOT / "docs/_static/images/hero_plot.png"
DPI = 100
WIDTH_PX, HEIGHT_PX = 800, 450


def resonant_background(
    frequency_hz: np.ndarray,
    background: float,
    slope: float,
    amplitude: float,
    center_hz: float,
    gamma_hz: float,
) -> np.ndarray:
    """Power-law background plus Lorentzian resonance."""
    safe_frequency = np.maximum(np.asarray(frequency_hz, dtype=float), 1.0)
    background_term = background * (safe_frequency / 80.0) ** slope
    resonance = amplitude / (1.0 + ((safe_frequency - center_hz) / gamma_hz) ** 2)
    return background_term + resonance


def _save_rgb_figure(
    fig: plt.Figure,
    output: Path,
    *,
    width_px: int,
    height_px: int,
    dpi: int,
) -> None:
    """Save a Matplotlib figure as an opaque RGB PNG with exact pixel size."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)

    image = Image.open(buffer).convert("RGBA")
    if image.size != (width_px, height_px):
        image = image.resize((width_px, height_px), Image.Resampling.LANCZOS)

    background = Image.new("RGB", image.size, "white")
    background.paste(image, mask=image.getchannel("A"))
    output.parent.mkdir(parents=True, exist_ok=True)
    background.save(output, format="PNG", dpi=(dpi, dpi))


def build_frequencyseries_matrix(seed: int = 42) -> FrequencySeriesMatrix:
    """Create a synthetic multi-channel matrix with shared resonant structure."""
    rng = np.random.default_rng(seed)
    freqs = np.linspace(8.0, 512.0, 900)
    channel_count = 6
    data = np.empty((channel_count, 1, freqs.size), dtype=float)
    channel_names = []

    for idx in range(channel_count):
        center = 58.0 + 58.0 * idx
        gamma = 5.0 + 0.75 * idx
        slope = -0.38 - 0.04 * idx
        background = 1.4e-3 * (1.0 + 0.16 * idx)
        amplitude = 0.82 + 0.18 * idx

        profile = resonant_background(freqs, background, slope, amplitude, center, gamma)
        ripple = 0.014 * np.sin(freqs / (10.0 + idx))
        jitter = rng.normal(0.0, 6.0e-3, freqs.size)
        data[idx, 0] = np.maximum(profile * (1.0 + ripple + jitter), 1.0e-6)
        channel_names.append(f"CH{idx + 1}")

    return FrequencySeriesMatrix(
        data,
        frequencies=freqs,
        unit="strain/rtHz",
        channel_names=channel_names,
        rows={f"row{idx}": {"name": name} for idx, name in enumerate(channel_names)},
        cols={"response": {"name": "response"}},
        name="Synthetic FrequencySeriesMatrix",
    )


def build_hero_figure() -> tuple[plt.Figure, Mapping[str, float | bool]]:
    """Build the hero figure and return fit verification metadata."""
    gwexpy.register_all()
    fsm = build_frequencyseries_matrix()

    highlighted_index = 2
    series = fsm[highlighted_index, 0]
    sigma = np.full(series.shape, 0.03, dtype=float)
    fit_result = series.fit(
        resonant_background,
        x_range=(80.0, 240.0),
        sigma=sigma,
        p0={
            "background": 2.0e-3,
            "slope": -0.5,
            "amplitude": 1.0,
            "center_hz": 174.0,
            "gamma_hz": 6.0,
        },
    )

    fig, (ax_matrix, ax_fit) = plt.subplots(
        2,
        1,
        figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI),
        dpi=DPI,
        gridspec_kw={"height_ratios": [1.05, 1.4], "hspace": 0.18},
    )

    matrix_data = np.log10(np.maximum(np.squeeze(fsm[:, 0, :].value, axis=1), 1.0e-12))
    heatmap = ax_matrix.imshow(
        matrix_data,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[series.frequencies.value[0], series.frequencies.value[-1], 0, fsm.shape[0]],
    )
    ax_matrix.set_ylabel("Matrix Channel")
    ax_matrix.set_yticks(np.arange(fsm.shape[0]) + 0.5)
    ax_matrix.set_yticklabels(list(fsm.channel_names))
    ax_matrix.set_title(
        "FrequencySeriesMatrix + Channel Fit",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    colorbar = fig.colorbar(heatmap, ax=ax_matrix, pad=0.01, fraction=0.04)
    colorbar.set_label("log10 amplitude", fontsize=9)

    cmap = plt.get_cmap("viridis")
    x_values = series.frequencies.value
    for idx, name in enumerate(fsm.channel_names):
        color = cmap(idx / max(fsm.shape[0] - 1, 1))
        alpha = 0.96 if idx == highlighted_index else 0.35
        linewidth = 2.0 if idx == highlighted_index else 1.1
        ax_fit.semilogy(
            x_values,
            fsm[idx, 0].value,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=name if idx == highlighted_index else None,
        )

    fit_curve = fit_result.model(x_values).value
    ax_fit.semilogy(
        x_values,
        fit_curve,
        color="#d62728",
        linewidth=2.3,
        label="Lorentzian-style fit",
    )
    ax_fit.axvspan(80.0, 240.0, color="#d62728", alpha=0.06)
    ax_fit.set_xlim(8, 512)
    ax_fit.set_xlabel("Frequency [Hz]")
    ax_fit.set_ylabel("Amplitude [strain/√Hz]")
    ax_fit.grid(True, which="both", alpha=0.2)
    ax_fit.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_fit.text(
        0.014,
        0.05,
        (
            f"Matrix.fit(): {hasattr(fsm, 'fit')}\n"
            f"Selected series fit center: {float(fit_result.params['center_hz']):.1f} Hz\n"
            f"Reduced χ²: {fit_result.reduced_chi2:.2f}"
        ),
        transform=ax_fit.transAxes,
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92},
    )

    metadata = {
        "frequencyseriesmatrix_has_fit": hasattr(fsm, "fit"),
        "fit_target_has_fit": hasattr(series, "fit"),
        "fit_center_hz": float(fit_result.params["center_hz"]),
        "fit_gamma_hz": float(fit_result.params["gamma_hz"]),
    }
    return fig, metadata


def generate_hero_plot(output: Path | str = OUTPUT) -> dict[str, float | bool | str]:
    """Generate the hero image and return verification metadata."""
    output_path = Path(output)
    fig, metadata = build_hero_figure()
    try:
        _save_rgb_figure(
            fig,
            output_path,
            width_px=WIDTH_PX,
            height_px=HEIGHT_PX,
            dpi=DPI,
        )
    finally:
        plt.close(fig)

    return {
        **metadata,
        "output": str(output_path),
        "width_px": WIDTH_PX,
        "height_px": HEIGHT_PX,
        "dpi": DPI,
    }


def main() -> None:
    """Generate the default hero image and print a short report."""
    report = generate_hero_plot()
    print(
        "Saved hero plot:",
        report["output"],
        f"({report['width_px']}x{report['height_px']} px, {report['dpi']} DPI)",
    )
    print(
        "FrequencySeriesMatrix.fit() available:",
        report["frequencyseriesmatrix_has_fit"],
        "| extracted FrequencySeries.fit() available:",
        report["fit_target_has_fit"],
    )


if __name__ == "__main__":
    main()
