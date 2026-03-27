"""Tests for gwexpy/timeseries/matrix_spectral.py"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeriesMatrix


def _make_tsm(n_time: int = 1024, n_rows: int = 2, n_cols: int = 1) -> TimeSeriesMatrix:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_rows, n_cols, n_time))
    return TimeSeriesMatrix(data, dt=0.01 * u.s, t0=0.0 * u.s)


# ---------------------------------------------------------------------------
# psd / fft / asd — via _run_spectral_method
# ---------------------------------------------------------------------------

def test_psd_shape():
    tsm = _make_tsm()
    result = tsm.psd(fftlength=0.1)
    assert result.shape[:2] == (2, 1)
    assert result.shape[2] > 0


def test_fft_shape():
    tsm = _make_tsm(n_time=128)
    result = tsm.fft()
    # rfft of 128 points → 65 frequencies
    assert result.shape == (2, 1, 65)


def test_asd_shape():
    tsm = _make_tsm()
    result = tsm.asd(fftlength=0.1)
    assert result.shape[:2] == (2, 1)
    assert result.shape[2] > 0


def test_psd_multi_element():
    """_run_spectral_method iterates over all (i,j) elements."""
    tsm = _make_tsm(n_rows=3, n_cols=2)
    result = tsm.psd(fftlength=0.1)
    assert result.shape[:2] == (3, 2)


# ---------------------------------------------------------------------------
# _vectorized_fft / _vectorized_psd / _vectorized_asd / _vectorized_csd / _vectorized_coherence
# ---------------------------------------------------------------------------

def test_vectorized_fft():
    tsm = _make_tsm(n_time=128)
    result = tsm._vectorized_fft()
    assert result.shape == (2, 1, 65)


def test_vectorized_psd():
    tsm = _make_tsm()
    result = tsm._vectorized_psd(fftlength=0.1)
    assert result.shape[:2] == (2, 1)


def test_vectorized_asd():
    tsm = _make_tsm()
    result = tsm._vectorized_asd(fftlength=0.1)
    # ASD values should be >= 0
    assert np.all(result.value >= 0)


def test_vectorized_csd():
    tsm = _make_tsm()
    result = tsm._vectorized_csd(tsm, fftlength=0.1)
    assert result.shape[:2] == (2, 1)


def test_vectorized_coherence():
    tsm = _make_tsm()
    result = tsm._vectorized_coherence(tsm, fftlength=0.1)
    # Self-coherence should be 1
    assert result.shape[:2] == (2, 1)
    np.testing.assert_allclose(result.value, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# lock_in
# ---------------------------------------------------------------------------

def _make_sine_tsm(f0: float = 10.0, n_time: int = 10000) -> TimeSeriesMatrix:
    t = np.linspace(0, n_time / 1000, n_time, endpoint=False)
    data = np.sin(2 * np.pi * f0 * t)
    data_3d = np.tile(data, (2, 1, 1))
    return TimeSeriesMatrix(data_3d, dt=0.001 * u.s, t0=0.0 * u.s)


def test_lock_in_amp_phase_tuple():
    """Default output='amp_phase' returns two matrices."""
    tsm = _make_sine_tsm()
    r1, r2 = tsm.lock_in(f0=10.0, bandwidth=2.0)
    assert isinstance(r1, TimeSeriesMatrix)
    assert isinstance(r2, TimeSeriesMatrix)
    assert r1.shape[:2] == (2, 1)
    assert r2.shape[:2] == (2, 1)


def test_lock_in_iq_tuple():
    """output='iq' also returns a tuple."""
    tsm = _make_sine_tsm()
    r1, r2 = tsm.lock_in(f0=10.0, bandwidth=2.0, output="iq")
    assert isinstance(r1, TimeSeriesMatrix)
    assert isinstance(r2, TimeSeriesMatrix)


def test_lock_in_complex_single():
    """output='complex' returns a single matrix."""
    tsm = _make_sine_tsm()
    result = tsm.lock_in(f0=10.0, bandwidth=2.0, output="complex")
    assert isinstance(result, TimeSeriesMatrix)
    assert result.shape[:2] == (2, 1)


def test_lock_in_empty_matrix_tuple():
    """Empty matrix returns (copy, copy) for tuple output."""
    empty = TimeSeriesMatrix(
        np.empty((0, 1, 100), dtype=float), dt=0.001 * u.s, t0=0.0 * u.s
    )
    r1, r2 = empty.lock_in(f0=10.0, bandwidth=2.0)
    assert r1.shape[0] == 0
    assert r2.shape[0] == 0


def test_lock_in_empty_matrix_complex():
    """Empty matrix returns single copy for complex output."""
    empty = TimeSeriesMatrix(
        np.empty((0, 1, 100), dtype=float), dt=0.001 * u.s, t0=0.0 * u.s
    )
    result = empty.lock_in(f0=10.0, bandwidth=2.0, output="complex")
    assert result.shape[0] == 0
