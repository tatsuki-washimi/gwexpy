"""
Tests for gwexpy.signal.preprocessing module.
"""

import numpy as np
from gwexpy.signal.preprocessing import (
    whiten,
    standardize,
    impute,
    WhiteningModel,
    StandardizationModel,
)


class TestWhitening:
    def test_whiten_pca(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X_w, model = whiten(X, method="pca")
        
        assert X_w.shape == (100, 3)
        assert isinstance(model, WhiteningModel)
        
        # Whitened data should have identity covariance
        cov = np.cov(X_w, rowvar=False)
        np.testing.assert_array_almost_equal(cov, np.eye(3), decimal=1)
        
    def test_whiten_zca(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X_w, model = whiten(X, method="zca")
        
        assert X_w.shape == (100, 3)
        
    def test_whiten_n_components(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X_w, model = whiten(X, method="pca", n_components=2)
        
        assert X_w.shape == (100, 2)
        
    def test_inverse_transform(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X_w, model = whiten(X, method="zca")
        
        X_rec = model.inverse_transform(X_w)
        np.testing.assert_array_almost_equal(X_rec, X, decimal=5)


class TestStandardization:
    def test_standardize_zscore(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X_std, model = standardize(X, method="zscore")
        
        assert np.abs(np.mean(X_std)) < 1e-10
        assert np.abs(np.std(X_std) - 1.0) < 1e-10
        assert isinstance(model, StandardizationModel)
        
    def test_standardize_robust(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Outlier
        X_std, model = standardize(X, method="robust")
        
        # Robust should not be as affected by outlier
        assert np.abs(np.median(X_std)) < 1e-10
        
    def test_inverse_transform(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X_std, model = standardize(X, method="zscore")
        
        X_rec = model.inverse_transform(X_std)
        np.testing.assert_array_almost_equal(X_rec, X, decimal=10)


class TestImputation:
    def test_impute_interpolate(self):
        X = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        X_imp = impute(X, method="interpolate")
        
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(X_imp, expected)
        
    def test_impute_ffill(self):
        X = np.array([1.0, np.nan, np.nan, 4.0])
        X_imp = impute(X, method="ffill")
        
        expected = np.array([1.0, 1.0, 1.0, 4.0])
        np.testing.assert_array_almost_equal(X_imp, expected)
        
    def test_impute_bfill(self):
        X = np.array([1.0, np.nan, np.nan, 4.0])
        X_imp = impute(X, method="bfill")
        
        expected = np.array([1.0, 4.0, 4.0, 4.0])
        np.testing.assert_array_almost_equal(X_imp, expected)
        
    def test_impute_mean(self):
        X = np.array([1.0, np.nan, 3.0])
        X_imp = impute(X, method="mean")
        
        # Mean of 1 and 3 is 2
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(X_imp, expected)
        
    def test_impute_max_gap(self):
        times = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        X = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
        
        # Gap between 2 and 10 is > 5, so those NaNs should remain
        X_imp = impute(X, method="interpolate", times=times, max_gap=5.0)
        
        # NaN at index 1 and 2 should remain NaN (gap is 8.0 > 5.0)
        assert np.isnan(X_imp[1])
        assert np.isnan(X_imp[2])
        # NaN at index 4 should be filled (gap is 1.0 < 5.0)
        assert not np.isnan(X_imp[4])
