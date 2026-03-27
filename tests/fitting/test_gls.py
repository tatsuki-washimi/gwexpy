"""Tests for gwexpy/fitting/gls.py — GLS and GeneralizedLeastSquares."""
from __future__ import annotations

import numpy as np
import pytest

from gwexpy.fitting.gls import GLS, GeneralizedLeastSquares


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear(x, a, b):
    return a * x + b


def _make_data(n=10, a=2.0, b=1.0, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(1, 10, n)
    y = a * x + b + rng.normal(0, 0.1, n)
    return x, y


# ---------------------------------------------------------------------------
# GLS (direct solver)
# ---------------------------------------------------------------------------


class TestGLSInit:
    def test_no_cov_uses_identity(self):
        X = np.ones((5, 2))
        y = np.ones(5)
        gls = GLS(X, y)
        np.testing.assert_array_equal(gls.cov_inv, np.eye(5))

    def test_cov_inv_provided(self):
        X = np.eye(3)
        y = np.ones(3)
        cov_inv = 2 * np.eye(3)
        gls = GLS(X, y, cov_inv=cov_inv)
        np.testing.assert_array_equal(gls.cov_inv, cov_inv)

    def test_cov_provided_inverted(self):
        X = np.eye(3)
        y = np.ones(3)
        cov = 2 * np.eye(3)
        gls = GLS(X, y, cov=cov)
        np.testing.assert_allclose(gls.cov_inv, 0.5 * np.eye(3))

    def test_cov_inv_takes_precedence(self):
        X = np.eye(3)
        y = np.ones(3)
        cov = 4 * np.eye(3)
        cov_inv = 3 * np.eye(3)
        gls = GLS(X, y, cov=cov, cov_inv=cov_inv)
        # cov_inv takes precedence
        np.testing.assert_array_equal(gls.cov_inv, cov_inv)


class TestGLSSolve:
    def test_ordinary_least_squares(self):
        # Simple design matrix: y = a*x + b -> X = [[x1, 1], [x2, 1], ...]
        x = np.array([1.0, 2.0, 3.0, 4.0])
        a_true, b_true = 2.0, 1.0
        y = a_true * x + b_true
        X = np.column_stack([x, np.ones_like(x)])
        gls = GLS(X, y)
        beta = gls.solve()
        np.testing.assert_allclose(beta[0], a_true, atol=1e-10)
        np.testing.assert_allclose(beta[1], b_true, atol=1e-10)

    def test_weighted_least_squares(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        a_true, b_true = 2.0, 1.0
        y = a_true * x + b_true
        X = np.column_stack([x, np.ones_like(x)])
        # uniform weighting → same as OLS
        cov_inv = np.eye(4)
        gls = GLS(X, y, cov_inv=cov_inv)
        beta = gls.solve()
        np.testing.assert_allclose(beta[0], a_true, atol=1e-10)
        np.testing.assert_allclose(beta[1], b_true, atol=1e-10)

    def test_correlated_data(self):
        rng = np.random.default_rng(0)
        n = 6
        x = np.linspace(1, 6, n)
        a_true, b_true = 3.0, 0.5
        y = a_true * x + b_true
        X = np.column_stack([x, np.ones(n)])
        cov = np.eye(n) + 0.1 * np.ones((n, n))  # block correlated
        cov_inv = np.linalg.inv(cov)
        gls = GLS(X, y, cov_inv=cov_inv)
        beta = gls.solve()
        np.testing.assert_allclose(beta[0], a_true, atol=1e-8)
        np.testing.assert_allclose(beta[1], b_true, atol=1e-8)


# ---------------------------------------------------------------------------
# GeneralizedLeastSquares
# ---------------------------------------------------------------------------


class TestGeneralizedLeastSquaresInit:
    def test_basic_init(self):
        x, y = _make_data()
        cov_inv = np.eye(len(y))
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear)
        assert gls.ndata == len(y)
        assert gls.cov_cho is None  # no cov provided

    def test_shape_mismatch_raises(self):
        x, y = _make_data(n=5)
        cov_inv = np.eye(10)  # wrong shape
        with pytest.raises(ValueError, match="cov_inv shape"):
            GeneralizedLeastSquares(x, y, cov_inv, _linear)

    def test_with_positive_definite_cov(self):
        x, y = _make_data()
        n = len(y)
        cov = 0.01 * np.eye(n)
        cov_inv = np.linalg.inv(cov)
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear, cov=cov)
        assert gls.cov_cho is not None  # Cholesky succeeds

    def test_with_non_positive_definite_cov(self):
        x, y = _make_data()
        n = len(y)
        # Rank-deficient matrix - not positive definite
        cov = np.zeros((n, n))  # degenerate
        cov_inv = np.eye(n)
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear, cov=cov)
        # Cholesky should fail → fallback to None
        assert gls.cov_cho is None

    def test_parameters_extracted(self):
        x, y = _make_data()
        cov_inv = np.eye(len(y))
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear)
        assert set(gls._parameters.keys()) == {"a", "b"}

    def test_errordef(self):
        from iminuit import Minuit
        assert GeneralizedLeastSquares.errordef == Minuit.LEAST_SQUARES


class TestGeneralizedLeastSquaresCall:
    def test_chi2_at_true_params(self):
        x = np.linspace(1, 5, 10)
        a_true, b_true = 2.0, 1.0
        y = a_true * x + b_true
        cov_inv = np.eye(len(y))
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear)
        chi2 = gls(a_true, b_true)
        assert chi2 == pytest.approx(0.0, abs=1e-20)

    def test_chi2_increases_with_wrong_params(self):
        x = np.linspace(1, 5, 10)
        a_true, b_true = 2.0, 1.0
        y = a_true * x + b_true
        cov_inv = np.eye(len(y))
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear)
        chi2_true = gls(a_true, b_true)
        chi2_wrong = gls(a_true + 1.0, b_true)
        assert chi2_wrong > chi2_true

    def test_chi2_with_cholesky(self):
        x = np.linspace(1, 5, 10)
        a_true, b_true = 2.0, 1.0
        y = a_true * x + b_true
        n = len(y)
        cov = 0.01 * np.eye(n)
        cov_inv = np.linalg.inv(cov)
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear, cov=cov)
        assert gls.cov_cho is not None
        chi2 = gls(a_true, b_true)
        assert chi2 == pytest.approx(0.0, abs=1e-10)

    def test_chi2_cholesky_matches_inv(self):
        rng = np.random.default_rng(7)
        x = np.linspace(1, 5, 8)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 8)
        n = len(y)
        cov = 0.01 * np.eye(n)
        cov_inv = np.linalg.inv(cov)
        a_test, b_test = 2.1, 0.9

        # With Cholesky
        gls_cho = GeneralizedLeastSquares(x, y, cov_inv, _linear, cov=cov)
        chi2_cho = gls_cho(a_test, b_test)

        # Without Cholesky (no cov argument)
        gls_inv = GeneralizedLeastSquares(x, y, cov_inv, _linear)
        chi2_inv = gls_inv(a_test, b_test)

        np.testing.assert_allclose(chi2_cho, chi2_inv, rtol=1e-6)

    def test_ndata(self):
        x, y = _make_data(n=15)
        cov_inv = np.eye(15)
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear)
        assert gls.ndata == 15

    def test_works_with_iminuit(self):
        iminuit = pytest.importorskip("iminuit")
        x = np.linspace(1, 5, 20)
        a_true, b_true = 3.0, 2.0
        rng = np.random.default_rng(42)
        y = a_true * x + b_true + rng.normal(0, 0.05, 20)
        cov_inv = np.eye(20) / 0.05**2
        gls = GeneralizedLeastSquares(x, y, cov_inv, _linear)
        m = iminuit.Minuit(gls, a=1.0, b=0.0)
        m.migrad()
        assert m.valid
        assert abs(m.values["a"] - a_true) < 0.5
