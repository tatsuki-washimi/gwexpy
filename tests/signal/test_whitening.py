"""Tests for gwexpy/signal/preprocessing/whitening.py."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from gwexpy.signal.preprocessing.whitening import WhiteningModel, whiten, _resolve_eps


# ---------------------------------------------------------------------------
# _resolve_eps
# ---------------------------------------------------------------------------

class TestResolveEps:
    def test_resolve_none(self):
        # eps=None → auto mode using get_safe_epsilon
        data = np.random.default_rng(0).normal(size=(10, 3))
        result = _resolve_eps(None, data)
        assert result > 0

    def test_resolve_auto_string(self):
        # eps='auto' → same as None
        data = np.random.default_rng(0).normal(size=(10, 3))
        result = _resolve_eps("auto", data)
        assert result > 0

    def test_resolve_float(self):
        # eps=float → just return as float
        data = np.ones((5, 2))
        result = _resolve_eps(1e-6, data)
        assert result == pytest.approx(1e-6)

    def test_resolve_int(self):
        data = np.ones((5, 2))
        result = _resolve_eps(0, data)
        assert result == 0.0

    def test_resolve_negative_raises(self):
        # Lines 42-43
        data = np.ones((5, 2))
        with pytest.raises(ValueError, match="non-negative"):
            _resolve_eps(-1.0, data)

    def test_resolve_inf_raises(self):
        data = np.ones((5, 2))
        with pytest.raises(ValueError, match="non-negative"):
            _resolve_eps(float("inf"), data)

    def test_resolve_invalid_type_raises(self):
        # Line 45
        data = np.ones((5, 2))
        with pytest.raises(TypeError, match="eps must be"):
            _resolve_eps([1e-6], data)


# ---------------------------------------------------------------------------
# WhiteningModel
# ---------------------------------------------------------------------------

class TestWhiteningModel:
    def test_inverse_transform_basic(self):
        # Basic roundtrip
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 3))
        X_w, model = whiten(X, method="pca")
        X_rec = model.inverse_transform(X_w)
        np.testing.assert_allclose(X_rec, X, atol=1e-10)

    def test_inverse_transform_with_value_attr(self):
        # Line 79 — input with .value attribute
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 2))
        X_w, model = whiten(X, method="pca")

        class WithValue:
            def __init__(self, v):
                self.value = v

        obj = WithValue(X_w)
        X_rec = model.inverse_transform(obj)
        assert X_rec is not None
        assert X_rec.shape == X.shape


# ---------------------------------------------------------------------------
# whiten
# ---------------------------------------------------------------------------

class TestWhiten:
    def test_pca_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        X_w, model = whiten(X, method="pca")
        assert X_w.shape == (50, 3)
        assert isinstance(model, WhiteningModel)

    def test_zca_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        X_w, model = whiten(X, method="zca")
        assert X_w.shape == (50, 3)

    def test_unknown_method_raises(self):
        # Line 164
        X = np.random.default_rng(0).normal(size=(20, 2))
        with pytest.raises(ValueError, match="method must be"):
            whiten(X, method="bad_method")

    def test_eps_float(self):
        # Lines 144-145
        X = np.random.default_rng(0).normal(size=(20, 2))
        X_w, model = whiten(X, eps=1e-5)
        assert X_w.shape == (20, 2)

    def test_eps_invalid_raises(self):
        # Lines 146-147
        X = np.random.default_rng(0).normal(size=(20, 2))
        with pytest.raises(TypeError, match="eps must be"):
            whiten(X, eps="invalid_string")

    def test_return_model_false(self):
        # Line 176
        X = np.random.default_rng(0).normal(size=(20, 2))
        result = whiten(X, return_model=False)
        assert isinstance(result, np.ndarray)

    def test_n_components_pca(self):
        # Line 169 — PCA with reduced components
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        X_w, model = whiten(X, method="pca", n_components=2)
        assert X_w.shape[1] == 2

    def test_n_components_zca_warns(self):
        # Lines 167-168 — ZCA + n_components → warning
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_w, model = whiten(X, method="zca", n_components=2)
            assert any("n_components" in str(warning.message) for warning in w)

    def test_1d_covariance(self):
        # Lines 152-153 — 1D case (single feature), cov is 0D
        X = np.random.default_rng(0).normal(size=(20, 1))
        X_w, model = whiten(X, method="pca")
        assert X_w.shape[1] == 1
