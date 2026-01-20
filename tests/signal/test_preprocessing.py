"""
Tests for gwexpy.signal.preprocessing module.

This module contains comprehensive pytest tests for fixing the numerical
invariants of the preprocessing algorithms (imputation, standardization, whitening).
"""

import numpy as np
import pytest

from gwexpy.signal.preprocessing import (
    StandardizationModel,
    WhiteningModel,
    impute,
    standardize,
    whiten,
)


# Use fixed seed for reproducibility
@pytest.fixture
def rng():
    """Reproducible random generator."""
    return np.random.default_rng(0)


# ===========================================================================
# 1) standardize のpytest固定
# ===========================================================================


class TestStandardizeZscore:
    """Tests for zscore standardization."""

    def test_zscore_basic_mean_zero_std_one(self, rng):
        """z-score standardization should produce mean≈0 and std≈1 along axis."""
        x = rng.normal(size=(4096, 3))
        # Use axis=0 to standardize along samples (column-wise)
        y, model = standardize(x, method="zscore", ddof=0, axis=0, return_model=True)

        # Each channel should have mean ≈ 0 and std ≈ 1
        np.testing.assert_allclose(np.mean(y, axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(y, axis=0, ddof=0), 1.0, atol=1e-10)

    def test_zscore_with_nan_using_nanmean_nanstd(self, rng):
        """z-score with NaN should use nanmean/nanstd, producing mean≈0, std≈1."""
        x = rng.normal(size=(4096, 3))
        # Inject NaN at regular intervals in channel 0
        x[::100, 0] = np.nan

        # Use axis=0 to standardize along samples
        y, model = standardize(x, method="zscore", ddof=0, axis=0, return_model=True)

        # Each channel should have nanmean ≈ 0 and nanstd ≈ 1
        np.testing.assert_allclose(np.nanmean(y, axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.nanstd(y, axis=0, ddof=0), 1.0, atol=1e-6)

    def test_zscore_1d_array(self, rng):
        """z-score should work with 1D arrays."""
        x = rng.normal(size=1000)
        y, model = standardize(x, method="zscore", ddof=0, return_model=True)

        np.testing.assert_allclose(np.mean(y), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(y, ddof=0), 1.0, atol=1e-10)

    def test_zscore_returns_standardization_model(self, rng):
        """zscore should return StandardizationModel when return_model=True."""
        x = rng.normal(size=(100, 3))
        y, model = standardize(x, method="zscore", return_model=True)

        assert isinstance(model, StandardizationModel)
        assert hasattr(model, "mean")
        assert hasattr(model, "scale")
        assert hasattr(model, "inverse_transform")

    def test_zscore_return_model_false(self, rng):
        """zscore should return only array when return_model=False."""
        x = rng.normal(size=(100, 3))
        result = standardize(x, method="zscore", return_model=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape

    def test_zscore_constant_array_no_zero_division(self):
        """Constant array (std=0) should not cause zero division."""
        x = np.ones((100, 3))
        y, model = standardize(x, method="zscore", return_model=True)

        # Should be finite (no inf/nan from division by zero)
        assert np.all(np.isfinite(y))

    def test_zscore_ddof_parameter(self, rng):
        """ddof parameter should affect std calculation."""
        x = rng.normal(size=(100, 3))
        y0, _ = standardize(x, method="zscore", ddof=0)
        y1, _ = standardize(x, method="zscore", ddof=1)

        # With ddof=1, std is slightly larger, so normalized results differ
        assert not np.allclose(y0, y1)


class TestStandardizeRobust:
    """Tests for robust (MAD-based) standardization."""

    def test_robust_median_zero(self, rng):
        """Robust standardization should produce median ≈ 0 for each channel."""
        x = rng.normal(size=(4096, 3))
        # Use axis=0 to standardize along samples
        y, model = standardize(x, method="robust", axis=0, return_model=True)

        # Each channel should have median ≈ 0
        np.testing.assert_allclose(np.median(y, axis=0), 0.0, atol=1e-10)

    def test_robust_outlier_resistance(self, rng):
        """Robust should be less affected by outliers compared to zscore."""
        x = rng.normal(size=(1000,))
        # Add extreme outliers
        x_with_outliers = x.copy()
        x_with_outliers[0] = 1000.0
        x_with_outliers[1] = -1000.0

        y_robust, _ = standardize(x_with_outliers, method="robust")
        y_zscore, _ = standardize(x_with_outliers, method="zscore")

        # Robust median should still be near 0
        assert abs(np.median(y_robust)) < 1e-6

        # The max value for robust should be larger than zscore
        # (because MAD-based scale is more robust)
        assert np.max(np.abs(y_robust)) > np.max(np.abs(y_zscore))

    def test_robust_constant_array_no_zero_division(self):
        """Constant array (MAD=0) should not cause zero division."""
        x = np.ones((100, 3))
        y, model = standardize(x, method="robust", return_model=True)

        # Should be finite (no inf/nan from division by zero)
        assert np.all(np.isfinite(y))

    def test_robust_returns_standardization_model(self, rng):
        """robust should return StandardizationModel."""
        x = rng.normal(size=(100, 3))
        y, model = standardize(x, method="robust", return_model=True)

        assert isinstance(model, StandardizationModel)

    def test_robust_mad_method_alias(self, rng):
        """'mad' should be an alias for 'robust' method."""
        x = rng.normal(size=(100, 3))
        y_robust, _ = standardize(x, method="robust")
        y_mad, _ = standardize(x, method="mad")

        np.testing.assert_array_equal(y_robust, y_mad)


class TestStandardizeInverseTransform:
    """Tests for standardization inverse_transform."""

    def test_inverse_transform_roundtrip_zscore_1d(self, rng):
        """inverse_transform should recover original data for zscore with 1D array."""
        X = rng.normal(size=100)
        X_std, model = standardize(X, method="zscore")

        X_rec = model.inverse_transform(X_std)
        np.testing.assert_allclose(X_rec, X, rtol=1e-10)

    def test_inverse_transform_roundtrip_robust_1d(self, rng):
        """inverse_transform should recover original data for robust with 1D array."""
        X = rng.normal(size=100)
        X_std, model = standardize(X, method="robust")

        X_rec = model.inverse_transform(X_std)
        np.testing.assert_allclose(X_rec, X, rtol=1e-10)


class TestStandardizeUnknownMethod:
    """Tests for unknown method error."""

    def test_unknown_method_raises_valueerror(self, rng):
        """Unknown method should raise ValueError."""
        x = rng.normal(size=(100, 3))
        with pytest.raises(ValueError, match="Unknown standardization method"):
            standardize(x, method="unknown_method")


# ===========================================================================
# 2) whiten のpytest固定
# ===========================================================================


class TestWhitenPCA:
    """Tests for PCA whitening."""

    def test_pca_whitening_identity_covariance(self, rng):
        """PCA whitening should produce data with covariance ≈ I."""
        # Create correlated data
        n_samples = 1000
        n_features = 4
        Z = rng.normal(size=(n_samples, n_features))
        A = rng.normal(size=(n_features, n_features))
        b = rng.normal(size=n_features)
        X = Z @ A + b

        Xw, model = whiten(X, method="pca", return_model=True)

        cov = np.cov(Xw, rowvar=False)

        # Diagonal elements should be ≈ 1
        np.testing.assert_allclose(np.diag(cov), 1.0, atol=0.1)

        # Off-diagonal elements should be ≈ 0
        off_diag = cov - np.diag(np.diag(cov))
        np.testing.assert_allclose(off_diag, 0.0, atol=0.1)

    def test_pca_whitening_shape_preserved(self, rng):
        """PCA whitening should preserve shape when n_components is None."""
        X = rng.normal(size=(100, 5))
        Xw, model = whiten(X, method="pca", n_components=None)

        assert Xw.shape == X.shape

    def test_pca_whitening_returns_model(self, rng):
        """PCA whitening should return WhiteningModel."""
        X = rng.normal(size=(100, 3))
        Xw, model = whiten(X, method="pca", return_model=True)

        assert isinstance(model, WhiteningModel)
        assert hasattr(model, "mean")
        assert hasattr(model, "W")
        assert hasattr(model, "inverse_transform")

    def test_pca_whitening_n_components(self, rng):
        """PCA whitening with n_components should reduce dimensionality."""
        X = rng.normal(size=(100, 5))
        Xw, model = whiten(X, method="pca", n_components=3)

        assert Xw.shape == (100, 3)


class TestWhitenZCA:
    """Tests for ZCA whitening."""

    def test_zca_whitening_identity_covariance(self, rng):
        """ZCA whitening should produce data with covariance ≈ I."""
        # Create correlated data
        n_samples = 1000
        n_features = 4
        Z = rng.normal(size=(n_samples, n_features))
        A = rng.normal(size=(n_features, n_features))
        X = Z @ A

        Xw, model = whiten(X, method="zca", return_model=True)

        cov = np.cov(Xw, rowvar=False)

        # Diagonal elements should be ≈ 1
        np.testing.assert_allclose(np.diag(cov), 1.0, atol=0.1)

        # Off-diagonal elements should be ≈ 0
        off_diag = cov - np.diag(np.diag(cov))
        np.testing.assert_allclose(off_diag, 0.0, atol=0.1)

    def test_zca_whitening_shape_preserved(self, rng):
        """ZCA whitening should preserve shape."""
        X = rng.normal(size=(100, 5))
        Xw, model = whiten(X, method="zca", n_components=None)

        assert Xw.shape == X.shape


class TestWhitenInverseTransform:
    """Tests for whitening inverse_transform."""

    def test_inverse_transform_pca_full_components(self, rng):
        """inverse_transform should recover original X for PCA with all components."""
        X = rng.normal(size=(50, 3))
        Xw, model = whiten(X, method="pca", n_components=None)

        X_rec = model.inverse_transform(Xw)
        np.testing.assert_allclose(X_rec, X, rtol=1e-6)

    def test_inverse_transform_zca_full_components(self, rng):
        """inverse_transform should recover original X for ZCA with all components."""
        X = rng.normal(size=(50, 3))
        Xw, model = whiten(X, method="zca", n_components=None)

        X_rec = model.inverse_transform(Xw)
        np.testing.assert_allclose(X_rec, X, rtol=1e-6)

    def test_inverse_transform_reduced_components_shape(self, rng):
        """inverse_transform with reduced n_components should produce valid shape."""
        X = rng.normal(size=(50, 5))
        Xw, model = whiten(X, method="pca", n_components=3)

        X_rec = model.inverse_transform(Xw)

        # Shape should match original feature dimension
        assert X_rec.shape == (50, 5)
        # Should be finite
        assert np.all(np.isfinite(X_rec))


class TestWhitenEdgeCases:
    """Tests for whitening edge cases."""

    def test_unknown_method_raises_valueerror(self, rng):
        """Unknown method should raise ValueError."""
        X = rng.normal(size=(100, 3))
        with pytest.raises(ValueError, match="method must be 'pca' or 'zca'"):
            whiten(X, method="unknown")

    def test_1d_feature_case(self, rng):
        """Whitening should work for 1D feature case."""
        X = rng.normal(size=(100, 1))
        Xw, model = whiten(X, method="pca")

        assert Xw.shape == X.shape

    def test_return_model_false(self, rng):
        """whiten should return only array when return_model=False."""
        X = rng.normal(size=(100, 3))
        result = whiten(X, method="pca", return_model=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == X.shape


# ===========================================================================
# 3) impute のpytest固定
# ===========================================================================


class TestImputeTimesValidation:
    """Tests for times validation in impute."""

    def test_times_different_length_raises_valueerror(self):
        """times with different length should raise ValueError."""
        values = np.array([1.0, np.nan, 3.0, 4.0])
        times = np.array([0.0, 1.0, 2.0])  # Length 3, should be 4

        with pytest.raises(ValueError, match="same length"):
            impute(values, times=times, method="interpolate")

    def test_times_not_strictly_increasing_raises_valueerror(self):
        """times not strictly increasing should raise ValueError."""
        values = np.array([1.0, np.nan, 3.0, 4.0])
        times = np.array([0.0, 1.0, 1.0, 2.0])  # Not strictly increasing

        with pytest.raises(ValueError, match="strictly increasing"):
            impute(values, times=times, method="interpolate")

    def test_times_not_1d_raises_valueerror(self):
        """times that is not 1D should raise ValueError."""
        values = np.array([1.0, np.nan, 3.0, 4.0])
        times = np.array([[0.0, 1.0], [2.0, 3.0]])  # 2D array

        with pytest.raises(ValueError, match="1D array"):
            impute(values, times=times, method="interpolate")


class TestImputeInterpolate:
    """Tests for linear interpolation imputation."""

    def test_interpolate_fills_nan(self):
        """Interpolation should fill NaN with linearly interpolated values."""
        values = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        result = impute(values, method="interpolate")

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(result, expected)

    def test_interpolate_with_times(self):
        """Interpolation should use times for interpolation."""
        times = np.array([0.0, 1.0, 3.0, 5.0])
        values = np.array([0.0, np.nan, np.nan, 10.0])
        result = impute(values, times=times, method="interpolate")

        # Linear interpolation: at t=1, value = 0 + (10-0)*(1-0)/(5-0) = 2
        # at t=3, value = 0 + (10-0)*(3-0)/(5-0) = 6
        expected = np.array([0.0, 2.0, 6.0, 10.0])
        np.testing.assert_allclose(result, expected)

    def test_interpolate_edge_nans_stay_nan_with_max_gap(self):
        """Edge NaNs should stay as NaN when max_gap is specified."""
        values = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = impute(values, times=times, method="interpolate", max_gap=1.0)

        # Edge NaNs should remain NaN
        assert np.isnan(result[0])
        assert np.isnan(result[4])

    def test_interpolate_complex_values(self, rng):
        """Interpolation should work with complex values."""
        values = np.array([1.0 + 1j, np.nan, 3.0 + 3j])
        result = impute(values, method="interpolate")

        expected = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j])
        np.testing.assert_allclose(result, expected)


class TestImputeMaxGap:
    """Tests for max_gap constraint in impute."""

    def test_max_gap_large_gap_stays_nan(self):
        """NaNs in gaps larger than max_gap should stay as NaN."""
        times = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        values = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])

        # Gap between t=2 and t=10 is 8.0, which exceeds max_gap=5.0
        result = impute(values, times=times, method="interpolate", max_gap=5.0)

        # NaN at index 1 and 2 should remain NaN (within gap > 5.0)
        assert np.isnan(result[1])
        assert np.isnan(result[2])

        # NaN at index 4 should be filled (gap is 1.0 < 5.0)
        assert not np.isnan(result[4])

    def test_max_gap_small_gap_filled(self):
        """NaNs in gaps smaller than max_gap should be filled."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, np.nan, np.nan, 4.0])

        result = impute(values, times=times, method="interpolate", max_gap=5.0)

        # All NaNs within max_gap should be filled
        assert not np.any(np.isnan(result))


class TestImputeLimit:
    """Tests for limit constraint in impute."""

    def test_ffill_limit_constrains_fill(self):
        """ffill with limit should only fill up to limit consecutive NaNs."""
        values = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])

        result = impute(values, method="ffill", limit=2)

        # First 2 NaNs should be filled, remaining should stay NaN
        np.testing.assert_equal(result[1], 1.0)
        np.testing.assert_equal(result[2], 1.0)
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_bfill_limit_constrains_fill(self):
        """bfill with limit should only fill up to limit consecutive NaNs."""
        values = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])

        result = impute(values, method="bfill", limit=2)

        # Last 2 NaNs before 6.0 should be filled
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        np.testing.assert_equal(result[3], 6.0)
        np.testing.assert_equal(result[4], 6.0)

    def test_ffill_no_limit(self):
        """ffill without limit should fill all consecutive NaNs."""
        values = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])

        result = impute(values, method="ffill")

        # All NaNs should be filled with 1.0
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_bfill_no_limit(self):
        """bfill without limit should fill all consecutive NaNs."""
        values = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])

        result = impute(values, method="bfill")

        # All NaNs should be filled with 6.0
        expected = np.array([1.0, 6.0, 6.0, 6.0, 6.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_limit_longer_run_leaves_nans(self):
        """NaN runs longer than limit should leave some NaNs unfilled."""
        values = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0])

        # limit=3, run length=5
        result_ffill = impute(values, method="ffill", limit=3)
        result_bfill = impute(values, method="bfill", limit=3)

        # ffill: first 3 filled, last 2 NaN
        assert not np.isnan(result_ffill[1])
        assert not np.isnan(result_ffill[2])
        assert not np.isnan(result_ffill[3])
        assert np.isnan(result_ffill[4])
        assert np.isnan(result_ffill[5])

        # bfill: last 3 filled, first 2 NaN
        assert np.isnan(result_bfill[1])
        assert np.isnan(result_bfill[2])
        assert not np.isnan(result_bfill[3])
        assert not np.isnan(result_bfill[4])
        assert not np.isnan(result_bfill[5])


class TestImputeFillValue:
    """Tests for fill_value parameter in impute."""

    def test_fill_value_for_edge_nans_with_max_gap(self):
        """fill_value should be used for edge NaNs when max_gap is specified."""
        values = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = impute(values, times=times, method="interpolate", max_gap=1.0, fill_value=0.0)

        # Edge NaNs should be filled with fill_value
        np.testing.assert_equal(result[0], 0.0)
        np.testing.assert_equal(result[4], 0.0)


class TestImputeMethods:
    """Tests for different imputation methods."""

    def test_mean_imputation(self):
        """mean imputation should fill NaNs with mean of non-NaN values."""
        values = np.array([1.0, np.nan, 3.0])
        result = impute(values, method="mean")

        # Mean of 1.0 and 3.0 is 2.0
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_median_imputation(self):
        """median imputation should fill NaNs with median of non-NaN values."""
        values = np.array([1.0, np.nan, 3.0, 5.0])
        result = impute(values, method="median")

        # Median of 1.0, 3.0, 5.0 is 3.0
        expected = np.array([1.0, 3.0, 3.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_unknown_method_raises_valueerror(self):
        """Unknown method should raise ValueError."""
        values = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="Unknown impute method"):
            impute(values, method="unknown")


class TestImputeNoNan:
    """Tests for arrays without NaN."""

    def test_no_nan_returns_copy(self, rng):
        """Arrays without NaN should be returned unchanged."""
        values = rng.normal(size=100)
        result = impute(values, method="interpolate")

        np.testing.assert_array_equal(result, values)


class TestImputeInterpolateLimitInteraction:
    """Tests for interpolate method with limit constraint."""

    def test_interpolate_with_limit(self):
        """Interpolate with limit should limit consecutive NaN fills."""
        values = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0])

        # limit applies to interpolation as well
        result = impute(values, method="interpolate", limit=2)

        # The first 2 should be filled, remaining should be NaN
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])
        assert np.isnan(result[3])


# ===========================================================================
# 4) API整合性テスト
# ===========================================================================


class TestAPIConsistency:
    """Tests for API consistency across preprocessing functions."""

    def test_standardize_returns_standardization_model(self, rng):
        """standardize(..., return_model=True) should return StandardizationModel."""
        X = rng.normal(size=(100, 3))
        result, model = standardize(X, method="zscore", return_model=True)

        assert isinstance(model, StandardizationModel)

    def test_whiten_returns_whitening_model(self, rng):
        """whiten(..., return_model=True) should return WhiteningModel."""
        X = rng.normal(size=(100, 3))
        result, model = whiten(X, method="pca", return_model=True)

        assert isinstance(model, WhiteningModel)

    def test_standardization_model_has_required_attributes(self, rng):
        """StandardizationModel should have mean, scale, axis, and inverse_transform."""
        X = rng.normal(size=(100, 3))
        _, model = standardize(X, method="zscore", return_model=True)

        assert hasattr(model, "mean")
        assert hasattr(model, "scale")
        assert hasattr(model, "axis")
        assert callable(getattr(model, "inverse_transform"))

    def test_whitening_model_has_required_attributes(self, rng):
        """WhiteningModel should have mean, W, and inverse_transform."""
        X = rng.normal(size=(100, 3))
        _, model = whiten(X, method="pca", return_model=True)

        assert hasattr(model, "mean")
        assert hasattr(model, "W")
        assert callable(getattr(model, "inverse_transform"))

    def test_standardization_transform_inverse_roundtrip_1d(self, rng):
        """Standardization transform-inverse roundtrip should recover original for 1D."""
        X = rng.normal(size=100)
        X_std, model = standardize(X, method="zscore", return_model=True)

        X_rec = model.inverse_transform(X_std)
        np.testing.assert_allclose(X_rec, X, rtol=1e-10)

    def test_whitening_transform_inverse_roundtrip(self, rng):
        """Whitening transform-inverse roundtrip should recover original."""
        X = rng.normal(size=(50, 3))
        X_w, model = whiten(X, method="pca", n_components=None, return_model=True)

        X_rec = model.inverse_transform(X_w)
        np.testing.assert_allclose(X_rec, X, rtol=1e-6)
