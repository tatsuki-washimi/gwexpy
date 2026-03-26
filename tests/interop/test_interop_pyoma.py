"""Tests for pyOMA interoperability.

Uses mock result dicts. Does NOT require pyOMA to be installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gwexpy.frequencyseries import FrequencySeriesMatrix
from gwexpy.interop.pyoma_ import from_pyoma_results


# ---------------------------------------------------------------------------
# Mock result dicts
# ---------------------------------------------------------------------------


def _make_pyoma_results(n_modes=4, n_dof=10):
    rng = np.random.default_rng(42)
    return {
        "Fn": rng.uniform(5, 100, n_modes),
        "Zeta": rng.uniform(0.001, 0.05, n_modes),
        "Phi": rng.random((n_dof, n_modes)) + 1j * rng.random((n_dof, n_modes)),
    }


def _make_pyoma_results_no_phi(n_modes=3):
    rng = np.random.default_rng(0)
    return {
        "Fn": rng.uniform(10, 200, n_modes),
        "Zeta": rng.uniform(0.01, 0.03, n_modes),
    }


# ---------------------------------------------------------------------------
# from_pyoma_results → DataFrame
# ---------------------------------------------------------------------------


class TestFromPyomaDataFrame:
    def test_returns_dataframe(self):
        res = _make_pyoma_results()
        df = from_pyoma_results(pd.DataFrame, res)
        assert isinstance(df, pd.DataFrame)

    def test_summary_without_phi(self):
        res = _make_pyoma_results_no_phi()
        df = from_pyoma_results(pd.DataFrame, res)
        assert "frequency_Hz" in df.columns
        assert "damping_ratio" in df.columns
        assert len(df) == 3

    def test_with_phi(self):
        res = _make_pyoma_results(n_modes=2, n_dof=6)
        df = from_pyoma_results(pd.DataFrame, res)
        assert "mode_1" in df.columns
        assert "mode_2" in df.columns
        assert len(df) == 6


# ---------------------------------------------------------------------------
# from_pyoma_results → FrequencySeriesMatrix
# ---------------------------------------------------------------------------


class TestFromPyomaFSM:
    def test_returns_fsm(self):
        res = _make_pyoma_results(n_modes=3, n_dof=8)
        result = from_pyoma_results(FrequencySeriesMatrix, res)
        assert isinstance(result, FrequencySeriesMatrix)

    def test_no_phi_raises(self):
        res = _make_pyoma_results_no_phi()
        with pytest.raises(ValueError, match="mode shapes"):
            from_pyoma_results(FrequencySeriesMatrix, res)

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="Fn"):
            from_pyoma_results(pd.DataFrame, {})
