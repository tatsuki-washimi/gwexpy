"""Contract tests for gwexpy.fitting.gls (issue #457 + G6).

Covers 5 confirmed findings:
  #457 P2 — GLS.__init__: ill-conditioned cov inversion → RuntimeWarning
  #457 P2 — GLS.solve: ill-conditioned normal equations → RuntimeWarning
  #457 P3 — GeneralizedLeastSquares: silent Cholesky fallback → RuntimeWarning
  G6  F1  — GLS: underdetermined/rank-deficient design → ValueError
  G6  F2  — GeneralizedLeastSquares: non-PSD cov_inv → RuntimeWarning
"""
from __future__ import annotations

import numpy as np
import pytest

from gwexpy.fitting.gls import GLS, GeneralizedLeastSquares

# ---------------------------------------------------------------------------
# GLS.__init__ validation (#457 P2 / G6 F1)
# ---------------------------------------------------------------------------


def test_gls_init_non_2d_X_raises():
    with pytest.raises(ValueError, match="2-D"):
        GLS(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_gls_init_xy_length_mismatch_raises():
    X = np.column_stack([np.ones(4), np.arange(4, dtype=float)])
    y = np.array([1.0, 2.0, 3.0])  # 3 ≠ 4
    with pytest.raises(ValueError, match="length mismatch"):
        GLS(X, y)


def test_gls_init_underdetermined_raises():
    """n_samples < n_params → underdetermined system."""
    X = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3)
    y = np.array([1.0])              # n_samples=1 < n_params=3
    with pytest.raises(ValueError, match="Underdetermined"):
        GLS(X, y)


def test_gls_init_ill_conditioned_cov_warns():
    """Nearly singular cov emits RuntimeWarning instead of silent garbage."""
    eps = np.finfo(float).eps
    # 2×2 near-singular symmetric positive-definite matrix (cond > 1/eps)
    cov = np.array([[1.0, 1.0 - eps], [1.0 - eps, 1.0]])
    X = np.column_stack([np.ones(2), np.array([0.0, 1.0])])
    y = np.array([1.0, 2.0])
    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        GLS(X, y, cov=cov)


# ---------------------------------------------------------------------------
# GLS.solve (#457 P2 / G6 F1)
# ---------------------------------------------------------------------------


def test_gls_solve_basic():
    """Well-conditioned system solves correctly (no warning)."""
    n = 5
    X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    y = 2.0 + 3.0 * np.arange(n, dtype=float)  # exact: [2, 5, 8, 11, 14]
    gls = GLS(X, y)
    beta = gls.solve()
    np.testing.assert_allclose(beta, [2.0, 3.0], atol=1e-10)


def test_gls_solve_ill_conditioned_normal_equations_warns():
    """Near-collinear columns make normal equations ill-conditioned → warning."""
    n = 10
    # Two almost identical columns → XTX is near-singular
    X = np.column_stack([np.ones(n), np.ones(n) + 1e-10 * np.arange(n)])
    y = np.random.default_rng(0).normal(size=n)
    gls = GLS(X, y)
    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        gls.solve()


# ---------------------------------------------------------------------------
# GeneralizedLeastSquares: cov_inv finite + PSD guard (G6 F2)
# ---------------------------------------------------------------------------


def test_gls_cost_nan_cov_inv_raises():
    """cov_inv containing NaN raises ValueError."""

    def linear(x, a):
        return a * x

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    cov_inv = np.eye(3)
    cov_inv[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        GeneralizedLeastSquares(x, y, cov_inv, linear)


def test_gls_cost_inf_cov_inv_raises():
    """cov_inv containing Inf raises ValueError."""

    def linear(x, a):
        return a * x

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    cov_inv = np.eye(3)
    cov_inv[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        GeneralizedLeastSquares(x, y, cov_inv, linear)


def test_gls_cost_indefinite_cov_inv_warns():
    """Indefinite cov_inv (negative eigenvalue) previously caused silent chi2<0; now warns."""

    def linear(x, a, b):
        return a * x + b

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    # Symmetric indefinite matrix: 2I - 4*outer(v,v) has min eigenvalue -2
    n = 4
    v = np.ones(n) / np.sqrt(n)
    cov_inv = 2 * np.eye(n) - 4 * np.outer(v, v)  # min eigenvalue = -2
    with pytest.warns(RuntimeWarning, match="indefinite"):
        gls = GeneralizedLeastSquares(x, y, cov_inv, linear)
    # chi2 should be negative for this cov_inv (the pre-existing silent bug)
    chi2 = gls(0.0, 0.0)
    assert chi2 < 0.0  # documents the known unsafe behavior


def test_gls_cost_asymmetric_cov_inv_with_psd_symmetric_part_accepted():
    """Asymmetric cov_inv whose symmetric part is PSD is accepted.

    chi2 = r^T W r depends only on the symmetric part of W, so a merely
    asymmetric matrix (e.g. from numerical noise in inversion) should not
    be rejected — only a truly indefinite (negative eigenvalue) one should.
    """

    def linear(x, a):
        return a * x

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    # Upper-triangular form: sym part is 0.5*(A+A^T), which is PD
    cov_inv = np.array([[2.0, 1.0, 0.0],
                        [0.0, 2.0, 1.0],
                        [0.0, 0.0, 2.0]])
    gls = GeneralizedLeastSquares(x, y, cov_inv, linear)
    assert gls.ndata == 3


def test_gls_cost_valid_psd_cov_inv_accepted():
    """PSD cov_inv is accepted without warning."""

    def linear(x, a, b):
        return a * x + b

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    cov_inv = np.eye(3) * 4.0  # diagonal positive definite
    gls = GeneralizedLeastSquares(x, y, cov_inv, linear)
    assert gls.ndata == 3
    assert gls(2.0, 0.0) >= 0.0


# ---------------------------------------------------------------------------
# GeneralizedLeastSquares: Cholesky fallback warning (#457 P3)
# ---------------------------------------------------------------------------


def test_gls_cost_cholesky_fallback_warns():
    """Non-PSD cov (Cholesky fails) emits RuntimeWarning; cov_inv path used."""

    def linear(x, a, b):
        return a * x + b

    n = 4
    x = np.arange(n, dtype=float)
    y = np.ones(n)
    cov_inv = np.eye(n)
    # Rank-1 singular cov: Cholesky will fail
    cov = np.ones((n, n))
    with pytest.warns(RuntimeWarning, match="not positive definite"):
        gls = GeneralizedLeastSquares(x, y, cov_inv, linear, cov=cov)
    # Should fall back to cov_inv path
    assert gls.cov_cho is None
    assert gls(1.0, 0.0) >= 0.0
