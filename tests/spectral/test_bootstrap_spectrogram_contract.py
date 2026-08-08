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
  G5 SP1 — integer-dtype spectrogram truncated per-resample mean/median;
            cast to float64 before resampling.
  G5 SP2 — non-finite VIF (e.g. NaN window) silently propagated NaN CI;
            now falls back to factor=1.0.
  G5 SP5 — rebin_width smaller than df converts to bin_size<=1 (silent no-op);
            now emits RuntimeWarning; non-finite rebin_width raises ValueError.
  G5 SP7 — n_boot=1 produces zero-width CI; now emits RuntimeWarning.
  G5 SP4 — all-NaN column in return_map covariance path now emits
            logger.warning before mean-imputation.
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


# --- SP1: integer-dtype truncation -------------------------------------------


def test_integer_dtype_spectrogram_produces_float_output():
    """Integer-valued spectrogram must not truncate bootstrap mean/median to int."""
    rng = np.random.default_rng(1)
    data = rng.integers(1, 10, size=(15, 4)).astype(np.int32)
    times = np.arange(15, dtype=float)
    frequencies = np.arange(4, dtype=float)
    spec = Spectrogram(data, times=times, frequencies=frequencies, unit="ct")
    result = bootstrap_spectrogram(spec, n_boot=100)
    assert result.value.dtype.kind == "f", "output must be floating-point"
    assert np.all(np.isfinite(result.value))


# --- SP2: non-finite VIF in calculate_correlation_factor ---------------------


def test_calculate_correlation_factor_nan_window_returns_1():
    """NaN window values must not propagate NaN; falls back to factor=1.0."""
    from gwexpy.spectral.estimation import calculate_correlation_factor

    factor = calculate_correlation_factor(np.full(32, np.nan), 32, 16, 10)
    assert factor == 1.0


def test_calculate_correlation_factor_inf_window_returns_1():
    """Inf window values must fall back to factor=1.0."""
    from gwexpy.spectral.estimation import calculate_correlation_factor

    factor = calculate_correlation_factor(np.full(32, np.inf), 32, 16, 10)
    assert factor == 1.0


# --- SP5: rebin no-op warning and non-finite guard ---------------------------


def test_rebin_width_smaller_than_df_warns():
    """rebin_width < df → bin_size=0; must warn rather than silently skip."""
    spec = _spec(n_time=20, n_freq=8)
    # df = 1.0 Hz; rebin_width=0.1 < df → bin_size=0
    with pytest.warns(RuntimeWarning, match="bin_size"):
        result = bootstrap_spectrogram(spec, n_boot=30, rebin_width=0.1)
    assert result.size == 8  # rebinning had no effect


def test_rebin_width_nonfinite_raises():
    """Non-finite rebin_width must raise ValueError immediately."""
    spec = _spec(n_time=20, n_freq=8)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_spectrogram(spec, n_boot=30, rebin_width=float("inf"))


def test_rebin_width_nan_raises():
    """rebin_width=NaN must raise (nan > 0 is False, would otherwise be a no-op)."""
    spec = _spec(n_time=20, n_freq=8)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_spectrogram(spec, n_boot=30, rebin_width=float("nan"))


# --- SP7: n_boot=1 zero-width CI warning -------------------------------------


def test_n_boot_1_warns_zero_width_ci():
    """n_boot=1 makes all CI percentiles equal; must warn."""
    spec = _spec(n_time=10, n_freq=4)
    with pytest.warns(RuntimeWarning, match="n_boot=1"):
        result = bootstrap_spectrogram(spec, n_boot=1)
    assert result.size == 4
    np.testing.assert_array_equal(result.error_low.value, 0.0)
    np.testing.assert_array_equal(result.error_high.value, 0.0)


# --- SP4: covariance all-NaN column warning (return_map=True) ----------------


def test_return_map_all_nan_column_warns(caplog):
    """All-NaN column in return_map=True path must emit logger.warning."""
    spec = _spec(n_time=8, n_freq=4)
    bad = 1
    spec.value[:, bad] = np.nan

    with caplog.at_level(logging.WARNING, logger="gwexpy.spectral.estimation"):
        _, bfm = bootstrap_spectrogram(
            spec, n_boot=30, ignore_nan=True, return_map=True
        )

    assert any("NaN" in rec.message for rec in caplog.records)
    cov = bfm.value if hasattr(bfm, "value") else np.asarray(bfm)
    assert np.all(np.isnan(cov[bad, :])) or np.all(np.isnan(cov[:, bad]))
