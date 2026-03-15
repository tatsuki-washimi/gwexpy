"""Tests for BrucoResult.topk(), plot_ranked(), and band-aware get_ranked_channels."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gwexpy.analysis.bruco import BrucoResult


@pytest.fixture()
def simple_result() -> BrucoResult:
    """BrucoResult with 3 channels across 50 frequency bins."""
    freqs = np.linspace(1.0, 100.0, 50)
    target_psd = np.ones(50)
    res = BrucoResult(freqs, "target", target_psd, top_n=3)
    # ch_high: high coherence 0.9 across all bins
    # ch_mid:  mid coherence 0.5 across all bins
    # ch_low:  low coherence 0.2 across all bins
    cohs = np.vstack(
        [
            np.full(50, 0.9),  # ch_high
            np.full(50, 0.5),  # ch_mid
            np.full(50, 0.2),  # ch_low
        ]
    )
    res.update_batch(["ch_high", "ch_mid", "ch_low"], cohs)
    return res


@pytest.fixture()
def banded_result() -> BrucoResult:
    """BrucoResult where channel dominance varies by frequency band.

    - ch_low_freq:  coherence 0.9 only below 50 Hz
    - ch_high_freq: coherence 0.9 only above 50 Hz
    """
    freqs = np.linspace(1.0, 100.0, 100)
    target_psd = np.ones(100)
    res = BrucoResult(freqs, "target", target_psd, top_n=2)

    coh_low = np.where(freqs < 50.0, 0.9, 0.1)
    coh_high = np.where(freqs >= 50.0, 0.9, 0.1)
    cohs = np.vstack([coh_low, coh_high])
    res.update_batch(["ch_low_freq", "ch_high_freq"], cohs)
    return res


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------


def test_topk_returns_list(simple_result: BrucoResult) -> None:
    top = simple_result.topk(n=2)
    assert isinstance(top, list)
    assert len(top) == 2


def test_topk_order(simple_result: BrucoResult) -> None:
    """Most coherent channel should rank first."""
    top = simple_result.topk(n=3)
    assert top[0] == "ch_high"
    assert top[1] == "ch_mid"
    assert top[2] == "ch_low"


def test_topk_n_limit(simple_result: BrucoResult) -> None:
    top = simple_result.topk(n=1)
    assert len(top) == 1
    assert top[0] == "ch_high"


def test_topk_no_band_equals_get_ranked(simple_result: BrucoResult) -> None:
    """topk without band should match get_ranked_channels."""
    assert simple_result.topk(n=3) == simple_result.get_ranked_channels(limit=3)


# ---------------------------------------------------------------------------
# band-limited topk
# ---------------------------------------------------------------------------


def test_topk_with_band_low(banded_result: BrucoResult) -> None:
    """Band < 50 Hz → ch_low_freq should win."""
    top = banded_result.topk(n=1, band=(1.0, 49.0))
    assert top[0] == "ch_low_freq"


def test_topk_with_band_high(banded_result: BrucoResult) -> None:
    """Band > 50 Hz → ch_high_freq should win."""
    top = banded_result.topk(n=1, band=(51.0, 100.0))
    assert top[0] == "ch_high_freq"


def test_topk_band_out_of_range(simple_result: BrucoResult) -> None:
    """A band entirely outside the frequency range returns empty."""
    top = simple_result.topk(n=3, band=(1000.0, 2000.0))
    assert top == []


# ---------------------------------------------------------------------------
# get_ranked_channels band parameter (direct)
# ---------------------------------------------------------------------------


def test_get_ranked_channels_no_band(simple_result: BrucoResult) -> None:
    ranked = simple_result.get_ranked_channels(limit=2)
    assert ranked == ["ch_high", "ch_mid"]


def test_get_ranked_channels_with_band(banded_result: BrucoResult) -> None:
    ranked_low = banded_result.get_ranked_channels(limit=1, band=(1.0, 49.0))
    assert ranked_low == ["ch_low_freq"]
    ranked_high = banded_result.get_ranked_channels(limit=1, band=(51.0, 100.0))
    assert ranked_high == ["ch_high_freq"]


# ---------------------------------------------------------------------------
# plot_ranked
# ---------------------------------------------------------------------------


def test_plot_ranked_returns_figure(simple_result: BrucoResult) -> None:
    fig = simple_result.plot_ranked(top_k=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ranked_with_band(banded_result: BrucoResult) -> None:
    fig = banded_result.plot_ranked(top_k=1, band=(1.0, 49.0))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ranked_top_k_0(simple_result: BrucoResult) -> None:
    """top_k=0 should produce a figure with no plotted channels."""
    fig = simple_result.plot_ranked(top_k=0)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
