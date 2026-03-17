"""Smoke tests for manuscript-facing reproduction paths."""

from __future__ import annotations

import numpy as np

from gwexpy.timeseries import TimeSeries, TimeSeriesList, TimeSeriesMatrix


def test_timeserieslist_to_matrix_and_coherence_ranking_smoke() -> None:
    """The manuscript-facing list->matrix->ranking path should work end-to-end."""
    rng = np.random.default_rng(1234)
    sr = 512
    duration = 8
    n = sr * duration
    t = np.arange(n) / sr

    common = np.sin(2 * np.pi * 50 * t)
    target = TimeSeries(common + 0.05 * rng.standard_normal(n), t0=0, sample_rate=sr, name="target")
    corr = TimeSeries(common + 0.15 * rng.standard_normal(n), t0=0, sample_rate=sr, name="corr")
    noise = TimeSeries(rng.standard_normal(n), t0=0, sample_rate=sr, name="noise")

    matrix = TimeSeriesList([target, corr, noise]).to_matrix()
    assert isinstance(matrix, TimeSeriesMatrix)

    result = matrix.coherence_ranking(
        target="target",
        band=(10, 100),
        top_n=2,
        fftlength=2.0,
        overlap=1.0,
        parallel=1,
    )

    top = result.topk(n=2)
    assert top[0] == "corr"
    assert "noise" in top
