"""Tests for TimeSeriesMatrix.coherence_ranking()."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

from gwexpy.timeseries import TimeSeriesMatrix


def _make_timeseries(signal: np.ndarray, name: str, sr: int = 512) -> TimeSeries:
    return TimeSeries(signal, t0=0, sample_rate=sr, name=name)


@pytest.fixture()
def coherence_matrix() -> TimeSeriesMatrix:
    """3-channel matrix: target, correlated aux, and pure noise."""
    rng = np.random.default_rng(0)
    sr = 512
    duration = 16  # seconds – enough for fftlength=2.0
    n = sr * duration
    t = np.arange(n) / sr

    common = np.sin(2 * np.pi * 50 * t)
    target = _make_timeseries(common + 0.05 * rng.standard_normal(n), "target", sr)
    corr = _make_timeseries(common + 0.2 * rng.standard_normal(n), "corr", sr)
    noise = _make_timeseries(rng.standard_normal(n), "noise", sr)

    return TimeSeriesMatrix([[target, corr, noise]])


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_matrix_has_coherence_ranking_attribute() -> None:
    assert hasattr(TimeSeriesMatrix, "coherence_ranking")


def test_matrix_shape(coherence_matrix: TimeSeriesMatrix) -> None:
    assert coherence_matrix.shape[1] == 3  # 3 channels in 1 row


# ---------------------------------------------------------------------------
# coherence_ranking
# ---------------------------------------------------------------------------


def test_coherence_ranking_returns_bruco_result(
    coherence_matrix: TimeSeriesMatrix,
) -> None:
    from gwexpy.analysis.bruco import BrucoResult

    result = coherence_matrix.coherence_ranking(
        target="target", top_n=2, fftlength=2.0, overlap=1.0, parallel=1
    )
    assert isinstance(result, BrucoResult)


def test_coherence_ranking_corr_ranks_first(
    coherence_matrix: TimeSeriesMatrix,
) -> None:
    """The correlated channel should be ranked above the noise channel."""
    result = coherence_matrix.coherence_ranking(
        target="target", top_n=2, fftlength=2.0, overlap=1.0, parallel=1
    )
    top = result.topk(n=2)
    assert top[0] == "corr", f"Expected 'corr' as top channel, got {top}"


def test_coherence_ranking_band(coherence_matrix: TimeSeriesMatrix) -> None:
    """Band-limited ranking should still identify corr as top channel near 50 Hz."""
    result = coherence_matrix.coherence_ranking(
        target="target", top_n=2, fftlength=2.0, overlap=1.0, parallel=1
    )
    top_band = result.topk(n=1, band=(40.0, 60.0))
    assert top_band == ["corr"]


def test_coherence_ranking_unknown_target_raises(
    coherence_matrix: TimeSeriesMatrix,
) -> None:
    with pytest.raises(KeyError, match="not found in matrix"):
        coherence_matrix.coherence_ranking(target="no_such_channel")


def test_coherence_ranking_plot_ranked(coherence_matrix: TimeSeriesMatrix) -> None:
    result = coherence_matrix.coherence_ranking(
        target="target", top_n=2, fftlength=2.0, overlap=1.0, parallel=1
    )
    fig = result.plot_ranked(top_k=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# TimeSeriesMatrix constructor with list-of-lists (used in paper Listing 3)
# ---------------------------------------------------------------------------


def test_matrix_constructor_list_of_lists() -> None:
    """TimeSeriesMatrix([[t1, t2, t3]]) should create a 1×3 matrix."""
    rng = np.random.default_rng(1)
    sr = 256
    n = sr * 4
    t1 = _make_timeseries(rng.standard_normal(n), "ch1", sr)
    t2 = _make_timeseries(rng.standard_normal(n), "ch2", sr)
    t3 = _make_timeseries(rng.standard_normal(n), "ch3", sr)

    matrix = TimeSeriesMatrix([[t1, t2, t3]])
    assert matrix.shape[1] == 3
    names = matrix.channel_names
    assert "ch1" in names
    assert "ch2" in names
    assert "ch3" in names
