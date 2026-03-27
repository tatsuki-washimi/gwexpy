"""Tests for BrucoResult and helper functions in gwexpy/analysis/bruco.py."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from gwexpy.analysis.bruco import (
    BrucoResult,
    _auto_block_size,
    _resolve_block_size,
)


# ---------------------------------------------------------------------------
# _auto_block_size
# ---------------------------------------------------------------------------


class TestAutoBlockSize:
    def test_zero_bins_returns_default(self):
        result = _auto_block_size(0, 5)
        assert result == 256  # _BRUCO_BLOCK_SIZE_DEFAULT

    def test_negative_bins_returns_default(self):
        result = _auto_block_size(-1, 5)
        assert result == 256

    def test_very_large_bins_returns_min(self):
        # With very large n_bins, max_cols will be tiny → clamped to MIN
        result = _auto_block_size(10_000_000, 5)
        assert result == 16  # _BRUCO_BLOCK_SIZE_MIN

    def test_small_bins_returns_max(self):
        # With very small n_bins, result should be <= MAX
        result = _auto_block_size(1, 1)
        assert result <= 1024  # _BRUCO_BLOCK_SIZE_MAX
        assert result >= 16

    def test_reasonable_bins(self):
        result = _auto_block_size(100, 5)
        assert 16 <= result <= 1024


# ---------------------------------------------------------------------------
# _resolve_block_size
# ---------------------------------------------------------------------------


class TestResolveBlockSize:
    def test_none_falls_back_to_env_default(self):
        result = _resolve_block_size(None, 100, 5)
        assert result >= 1

    def test_auto_string(self):
        result = _resolve_block_size("auto", 100, 5)
        assert result >= 16

    def test_integer(self):
        result = _resolve_block_size(64, 100, 5)
        assert result == 64

    def test_string_integer(self):
        result = _resolve_block_size("32", 100, 5)
        assert result == 32

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            _resolve_block_size(0, 100, 5)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _resolve_block_size(-1, 100, 5)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="block_size must be"):
            _resolve_block_size("not_a_number", 100, 5)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="block_size must be"):
            _resolve_block_size(3.14, 100, 5)


# ---------------------------------------------------------------------------
# BrucoResult - helpers
# ---------------------------------------------------------------------------


def _make_result(n_bins=10, top_n=3):
    freqs = np.linspace(10, 100, n_bins)
    target_psd = np.ones(n_bins)
    return BrucoResult(freqs, "Target", target_psd, top_n=top_n)


def _make_result_with_data(n_bins=10, top_n=3):
    res = _make_result(n_bins, top_n)
    names = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    coh = rng.uniform(0, 1, (len(names), n_bins))
    coh[0] = 0.9  # A is always high
    coh[1] = 0.5
    res.update_batch(names, coh)
    return res


# ---------------------------------------------------------------------------
# BrucoResult - initialization
# ---------------------------------------------------------------------------


class TestBrucoResultInit:
    def test_basic_construction(self):
        res = _make_result()
        assert res.n_bins == 10
        assert res.top_n == 3
        assert res.target_name == "Target"

    def test_top_n_less_than_1_raises(self):
        with pytest.raises(ValueError, match="top_n"):
            BrucoResult(np.ones(5), "T", np.ones(5), top_n=0)

    def test_mismatched_freq_spectrum_raises(self):
        with pytest.raises(ValueError):
            BrucoResult(np.ones(5), "T", np.ones(6))

    def test_with_metadata(self):
        res = BrucoResult(np.ones(5), "T", np.ones(5), metadata={"run": "test"})
        assert res.metadata["run"] == "test"

    def test_block_size_auto(self):
        res = BrucoResult(np.ones(5), "T", np.ones(5), block_size="auto")
        assert res.block_size >= 16


# ---------------------------------------------------------------------------
# BrucoResult - update_batch
# ---------------------------------------------------------------------------


class TestUpdateBatch:
    def test_empty_batch_no_op(self):
        res = _make_result()
        old = res.top_coherence.copy()
        res.update_batch([], np.zeros((0, 10)))
        np.testing.assert_array_equal(res.top_coherence, old)

    def test_single_channel_update(self):
        res = _make_result(n_bins=5, top_n=2)
        coh = np.array([[0.8, 0.7, 0.6, 0.5, 0.4]])
        res.update_batch(["X"], coh)
        assert res.top_channels[0, 0] == "X"
        assert res.top_coherence[0, 0] == pytest.approx(0.8)

    def test_nan_values_treated_as_zero(self):
        """NaN coherence values are converted to 0.0; bins with 0-coh may not update."""
        res = _make_result(n_bins=3, top_n=1)
        coh = np.array([[np.nan, 0.5, 0.3]])
        res.update_batch(["A"], coh)
        # Bin 1 (coh=0.5) and Bin 2 (coh=0.3) should be filled
        assert res.top_channels[1, 0] == "A"
        assert res.top_coherence[1, 0] == pytest.approx(0.5)
        # Bin 0 (NaN→0): tied with existing 0.0 slots; may not be updated
        # Just verify coherence is 0.0
        assert res.top_coherence[0, 0] == pytest.approx(0.0)

    def test_multiple_batches_accumulate(self):
        res = _make_result(n_bins=5, top_n=2)
        res.update_batch(["A"], np.array([[0.5, 0.4, 0.3, 0.2, 0.1]]))
        res.update_batch(["B"], np.array([[0.9, 0.1, 0.1, 0.1, 0.1]]))
        # Bin 0: B(0.9) > A(0.5)
        assert res.top_channels[0, 0] == "B"
        assert res.top_channels[0, 1] == "A"

    def test_wrong_ndim_raises(self):
        res = _make_result()
        with pytest.raises(ValueError, match="2D"):
            res.update_batch(["A"], np.ones(10))

    def test_wrong_rows_raises(self):
        res = _make_result()
        with pytest.raises(ValueError):
            res.update_batch(["A", "B"], np.ones((3, 10)))  # 2 names, 3 rows

    def test_wrong_cols_raises(self):
        res = _make_result(n_bins=10)
        with pytest.raises(ValueError):
            res.update_batch(["A"], np.ones((1, 5)))  # wrong n_bins


# ---------------------------------------------------------------------------
# BrucoResult - get_noise_projection
# ---------------------------------------------------------------------------


class TestGetNoiseProjection:
    def test_rank_zero(self):
        res = _make_result_with_data()
        proj, coh = res.get_noise_projection(0)
        assert len(proj) == 10
        assert len(coh) == 10

    def test_rank_out_of_range_raises(self):
        res = _make_result(top_n=3)
        with pytest.raises(ValueError, match="Rank"):
            res.get_noise_projection(5)

    def test_negative_rank_raises(self):
        res = _make_result(top_n=3)
        with pytest.raises(ValueError):
            res.get_noise_projection(-1)

    def test_psd_mode(self):
        res = _make_result_with_data()
        proj_asd, _ = res.get_noise_projection(0, asd=True)
        proj_psd, coh = res.get_noise_projection(0, asd=False)
        # PSD should be ASD squared (up to floating point)
        np.testing.assert_allclose(proj_psd, proj_asd**2, rtol=1e-5)

    def test_coherence_threshold(self):
        res = _make_result(n_bins=5, top_n=1)
        coh = np.array([[0.1, 0.9, 0.3, 0.7, 0.05]])
        res.update_batch(["A"], coh)
        proj, _ = res.get_noise_projection(0, coherence_threshold=0.5)
        # Bins below threshold (coh < 0.5) should be NaN
        assert np.isnan(proj[0])  # 0.1 < 0.5
        assert not np.isnan(proj[1])  # 0.9 >= 0.5


# ---------------------------------------------------------------------------
# BrucoResult - projection_for_channel
# ---------------------------------------------------------------------------


class TestProjectionForChannel:
    def test_present_channel(self):
        res = _make_result(n_bins=5, top_n=2)
        res.update_batch(["A"], np.full((1, 5), 0.5))
        proj = res.projection_for_channel("A")
        assert len(proj) == 5
        assert np.all(proj > 0)

    def test_absent_channel_zeros(self):
        res = _make_result(n_bins=5)
        proj = res.projection_for_channel("NotPresent")
        np.testing.assert_array_equal(proj, 0.0)


# ---------------------------------------------------------------------------
# BrucoResult - dominant_channel
# ---------------------------------------------------------------------------


class TestDominantChannel:
    def test_dominant_channel_returns_most_frequent(self):
        res = _make_result(n_bins=6, top_n=1)
        # A appears in 4 out of 6 bins, B in 2
        names = ["A", "B"]
        coh = np.array([[0.9, 0.8, 0.85, 0.7, 0.2, 0.15], [0.1, 0.1, 0.1, 0.1, 0.9, 0.9]])
        res.update_batch(names, coh)
        dom = res.dominant_channel(0)
        assert dom == "A"

    def test_dominant_channel_no_data_returns_none(self):
        res = _make_result()
        assert res.dominant_channel(0) is None

    def test_dominant_channel_out_of_range_raises(self):
        res = _make_result(top_n=2)
        with pytest.raises(ValueError):
            res.dominant_channel(5)


# ---------------------------------------------------------------------------
# BrucoResult - get_ranked_channels / topk
# ---------------------------------------------------------------------------


class TestGetRankedChannels:
    def test_returns_list(self):
        res = _make_result_with_data()
        channels = res.get_ranked_channels(limit=2)
        assert isinstance(channels, list)
        assert len(channels) <= 2

    def test_empty_result_returns_empty(self):
        res = _make_result()
        assert res.get_ranked_channels() == []

    def test_band_filter(self):
        res = _make_result_with_data(n_bins=20, top_n=3)
        channels = res.get_ranked_channels(band=(20.0, 50.0))
        assert isinstance(channels, list)

    def test_topk_alias(self):
        res = _make_result_with_data()
        assert res.topk(n=2) == res.get_ranked_channels(limit=2)

    def test_topk_with_band(self):
        res = _make_result_with_data(n_bins=20, top_n=3)
        result = res.topk(n=2, band=(20.0, 50.0))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# BrucoResult - coherence_for_channel
# ---------------------------------------------------------------------------


class TestCoherenceForChannel:
    def test_returns_nans_when_absent(self):
        res = _make_result(n_bins=5)
        coh = res.coherence_for_channel("NonExistent")
        assert np.all(np.isnan(coh))

    def test_present_channel_has_values(self):
        res = _make_result(n_bins=5, top_n=1)
        res.update_batch(["A"], np.full((1, 5), 0.64))
        coh = res.coherence_for_channel("A", asd=True)
        # asd=True → sqrt(0.64) = 0.8
        np.testing.assert_allclose(coh, 0.8, rtol=1e-5)

    def test_psd_mode(self):
        res = _make_result(n_bins=5, top_n=1)
        res.update_batch(["A"], np.full((1, 5), 0.64))
        coh = res.coherence_for_channel("A", asd=False)
        np.testing.assert_allclose(coh, 0.64, rtol=1e-5)


# ---------------------------------------------------------------------------
# BrucoResult - to_dataframe
# ---------------------------------------------------------------------------


class TestToDataframe:
    def test_returns_dataframe(self):
        import pandas as pd
        res = _make_result_with_data()
        df = res.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "frequency" in df.columns
        assert "channel" in df.columns
        assert "coherence" in df.columns

    def test_empty_result_returns_empty_df(self):
        import pandas as pd
        res = _make_result()
        df = res.to_dataframe(ranks=[99])  # out of range → filtered
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_stride(self):
        import pandas as pd
        res = _make_result_with_data(n_bins=10)
        df_full = res.to_dataframe(ranks=[0])
        df_stride = res.to_dataframe(ranks=[0], stride=2)
        assert len(df_stride) == len(df_full) // 2


# ---------------------------------------------------------------------------
# BrucoResult - plot methods
# ---------------------------------------------------------------------------


class TestBrucoPlots:
    def test_plot_projection_returns_figure(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_projection()
        assert fig is not None
        plt.close("all")

    def test_plot_projection_with_channels(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_projection(channels=["A"])
        assert fig is not None
        plt.close("all")

    def test_plot_projection_with_ranks(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_projection(ranks=[0, 1])
        assert fig is not None
        plt.close("all")

    def test_plot_ranked_returns_figure(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_ranked(top_k=2)
        assert fig is not None
        plt.close("all")

    def test_plot_coherence_default(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_coherence()
        assert fig is not None
        plt.close("all")

    def test_plot_coherence_by_rank(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_coherence(ranks=[0, 1])
        assert fig is not None
        plt.close("all")

    def test_plot_coherence_by_channel(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_coherence(channels=["A"])
        assert fig is not None
        plt.close("all")

    def test_plot_coherence_with_threshold(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_coherence(coherence_threshold=0.5)
        assert fig is not None
        plt.close("all")

    def test_plot_coherence_psd_mode(self):
        import matplotlib.pyplot as plt
        res = _make_result_with_data()
        fig = res.plot_coherence(asd=False)
        assert fig is not None
        plt.close("all")

    def test_coherence_color(self):
        """Test _coherence_color internal method."""
        res = _make_result()
        color = res._coherence_color(0.5)
        assert isinstance(color, str)
        assert color.startswith("rgb(")

    def test_coherence_color_clamps(self):
        """Values outside [0,1] should be clamped."""
        res = _make_result()
        c0 = res._coherence_color(0.0)
        c1 = res._coherence_color(1.0)
        c_neg = res._coherence_color(-0.5)
        c_over = res._coherence_color(1.5)
        assert c_neg == c0
        assert c_over == c1
