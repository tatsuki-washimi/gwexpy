from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_ROOT = ROOT / "docs" / "web"

pytest.importorskip("iminuit")

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _code_cell_source(nb: dict, index: int) -> str:
    cell = nb["cells"][index]
    assert cell["cell_type"] == "code"
    return "".join(cell.get("source", []))


def _gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def test_advanced_fitting_gaussian_examples_recover_visible_peak():
    rng = np.random.default_rng(42)
    x = np.linspace(-5, 5, 100)
    y = _gaussian(x, a=10.0, mu=0.5, sigma=1.2) + rng.normal(0, 0.8, len(x))
    series = FrequencySeries(y, frequencies=x)

    limits = {"mu": (-2.0, 2.0), "sigma": (0.2, 3.0)}

    result_callable = series.fit(
        _gaussian,
        sigma=0.8,
        p0={"a": 8.0, "mu": 0.0, "sigma": 1.0},
        limits=limits,
    )
    result_string = series.fit(
        "gaus",
        sigma=0.8,
        p0={"A": 8.0, "mu": 0.0, "sigma": 1.0},
        x_range=(-2.5, 3.5),
        limits=limits,
    )

    assert abs(result_callable.params["mu"] - 0.5) < 0.15
    assert abs(result_callable.params["sigma"] - 1.2) < 0.15
    assert abs(result_string.params["mu"] - 0.5) < 0.15
    assert abs(result_string.params["sigma"] - 1.2) < 0.15


def test_case_violin_mode_tracking_recovers_injected_drift_within_20_percent():
    rng = np.random.default_rng(42)

    duration = 120
    sample_rate = 4096
    injected_drift_hz = 0.08

    t = np.arange(0, duration, 1.0 / sample_rate)
    f0_drift = 170.0 + (injected_drift_hz / duration) * t
    phase = 2 * np.pi * np.cumsum(f0_drift) / sample_rate
    signal = 1e-20 * np.sin(phase)
    noise = rng.normal(0, 3e-22, len(t))

    series = TimeSeries(signal + noise, dt=1.0 / sample_rate, unit="strain")
    spec = series.spectrogram2(16.0, overlap=8.0)

    frequencies = spec.frequencies.value
    band_mask = (frequencies >= 169.9) & (frequencies <= 170.2)
    band_freqs = frequencies[band_mask]
    track = np.full(spec.shape[0], np.nan)

    for row_idx in range(spec.shape[0]):
        row_band = spec.value[row_idx, band_mask]
        peak_idx = int(np.argmax(row_band))
        peak_freq = band_freqs[peak_idx]

        if 0 < peak_idx < len(band_freqs) - 1:
            y0 = row_band[peak_idx - 1]
            y1 = row_band[peak_idx]
            y2 = row_band[peak_idx + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            peak_freq = peak_freq + delta * (band_freqs[1] - band_freqs[0])

        track[row_idx] = peak_freq

    recovered_drift_hz = float(np.nanmax(track) - np.nanmin(track))
    relative_error = abs(recovered_drift_hz - injected_drift_hz) / injected_drift_hz

    assert recovered_drift_hz > 0
    assert relative_error <= 0.2


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/advanced_fitting.ipynb"),
        Path("ja/user_guide/tutorials/advanced_fitting.ipynb"),
    ],
)
def test_advanced_fitting_notebooks_include_fit_guardrails(relative_path: Path):
    nb = _read_notebook(TUTORIAL_ROOT / relative_path)

    fit_cell = _code_cell_source(nb, 6)
    string_fit_cell = _code_cell_source(nb, 10)

    assert '"mu": (-2.0, 2.0)' in fit_cell
    assert '"sigma": (0.2, 3.0)' in fit_cell
    assert 'x_range=(-2.5, 3.5)' in string_fit_cell
    assert '"mu": (-2.0, 2.0)' in string_fit_cell
    assert '"sigma": (0.2, 3.0)' in string_fit_cell


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/case_bootstrap_gls_fitting.ipynb"),
        Path("ja/user_guide/tutorials/case_bootstrap_gls_fitting.ipynb"),
    ],
)
def test_case_bootstrap_gls_fitting_notebooks_define_peak_center_and_q_limits(
    relative_path: Path,
):
    nb = _read_notebook(TUTORIAL_ROOT / relative_path)
    bounds_cell = _code_cell_source(nb, 28)

    assert '"f0": (95.0, 105.0)' in bounds_cell
    assert '"Q": (8.0, 35.0)' in bounds_cell
    assert '"alpha": (-2.0, 0.0)' in bounds_cell


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("en/user_guide/tutorials/case_violin_mode.ipynb"),
        Path("ja/user_guide/tutorials/case_violin_mode.ipynb"),
    ],
)
def test_case_violin_mode_notebooks_document_resolution_guardrail(relative_path: Path):
    nb = _read_notebook(TUTORIAL_ROOT / relative_path)
    markdown = "".join(nb["cells"][12].get("source", []))
    code = _code_cell_source(nb, 13)

    assert "sub-bin" in markdown or "サブビン" in markdown
    assert "fftlength = 16.0" in code
    assert "injected_drift_hz = 0.08" in code
    assert "recovered_drift_hz" in code
    assert "relative_error" in code
