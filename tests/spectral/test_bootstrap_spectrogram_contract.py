"""Edge-case contract tests for bootstrap_spectrogram (issue #460 + G5).

Covers the confirmed findings:
  #460 P2 — explicit block_size >= n_time previously fell through to
            np.random.randint(0, high<=0) with an opaque NumPy error.
            Now: explicit block_size >= n_time -> clear ValueError.
  #460 G5 — auto-derived block_size >= n_time -> RuntimeWarning + iid fallback
            (block_size=None) rather than a crash.
  #460 P3 — an all-NaN frequency column with ignore_nan=True silently produced
            NaN center/CI via np.nanmedian/np.nanmean. Now a function-level
            logger.warning diagnoses it; the output NaN is preserved (not
            imputed or hidden).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from gwexpy.spectral import estimation
from gwexpy.spectral.estimation import bootstrap_spectrogram
from gwexpy.spectrogram import Spectrogram


def _spec(n_time=5, n_freq=4, seed=0):
    """Small spectrogram with dt=1s (so block_size in seconds == samples)."""
    rng = np.random.default_rng(seed)
    data = np.abs(rng.standard_normal((n_time, n_freq))) + 1.0
    times = np.arange(n_time, dtype=float)
    frequencies = np.arange(n_freq, dtype=float)
    return Spectrogram(data, times=times, frequencies=frequencies, unit="V")


# --- explicit block_size >= n_time -> ValueError (#460 P2) -------------------


def test_explicit_block_size_greater_than_n_time_raises():
    spec = _spec(n_time=5)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_spectrogram(spec, n_boot=20, block_size=6.0)


def test_explicit_block_size_equal_n_time_raises():
    spec = _spec(n_time=5)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_spectrogram(spec, n_boot=20, block_size=5.0)


# --- auto-derived oversized block_size -> warn + iid fallback (#460 G5) ------


def test_auto_oversized_block_size_warns_and_falls_back(monkeypatch):
    spec = _spec(n_time=5, n_freq=4)
    # Force the auto estimator to produce a block_size >= n_time.
    monkeypatch.setattr(estimation, "_infer_overlap_ratio", lambda sg: 100.0)
    with pytest.warns(RuntimeWarning, match="iid"):
        result = bootstrap_spectrogram(spec, n_boot=20, block_size="auto")
    # Falls back to a valid iid bootstrap rather than crashing.
    assert result.size == 4
    assert np.all(np.isfinite(result.value))


# --- all-NaN frequency column -> logger.warning, NaN preserved (#460 P3) -----


def test_all_nan_frequency_column_warns_and_keeps_nan(caplog):
    spec = _spec(n_time=6, n_freq=4)
    bad = 2
    spec.value[:, bad] = np.nan  # entire frequency column is NaN

    with caplog.at_level(logging.WARNING, logger="gwexpy.spectral.estimation"):
        result = bootstrap_spectrogram(spec, n_boot=30, ignore_nan=True)

    # A function-level diagnostic must be emitted.
    assert any("NaN" in rec.message for rec in caplog.records)
    # The output at the dead bin stays NaN (not imputed, not hidden).
    assert np.isnan(result.value[bad])
    # Other bins remain finite.
    finite_mask = np.ones(4, dtype=bool)
    finite_mask[bad] = False
    assert np.all(np.isfinite(result.value[finite_mask]))


# --- valid path unaffected ---------------------------------------------------


def test_valid_block_bootstrap_path_unchanged():
    spec = _spec(n_time=20, n_freq=5)
    result = bootstrap_spectrogram(spec, n_boot=50, block_size=4.0)
    assert result.size == 5
    assert np.all(np.isfinite(result.value))


def test_valid_iid_bootstrap_path_unchanged():
    spec = _spec(n_time=20, n_freq=5)
    result = bootstrap_spectrogram(spec, n_boot=50)
    assert result.size == 5
    assert np.all(np.isfinite(result.value))
