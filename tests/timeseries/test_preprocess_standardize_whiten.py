"""
Tests for gwexpy.timeseries.preprocess standardization and whitening functions.

This module tests the specifications defined in the API documentation:
- standardize_timeseries: output unit is always dimensionless_unscaled
- standardize_matrix: axis must be 'time' or 'channel', output dimensionless
- whiten_matrix: PCA outputs flat, ZCA preserves shape, output dimensionless
"""

import numpy as np
import pytest
import warnings
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.timeseries.preprocess import (
    standardize_timeseries,
    standardize_matrix,
    whiten_matrix,
)


class TestStandardizeTimeSeries:
    """Tests for standardize_timeseries function."""

    def test_zscore_basic(self):
        """Test zscore standardization produces mean~0, std~1."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s)
        ts_std, model = standardize_timeseries(ts, method="zscore", ddof=0)

        assert np.abs(np.nanmean(ts_std.value)) < 1e-10
        assert np.abs(np.nanstd(ts_std.value, ddof=0) - 1.0) < 1e-10

    def test_zscore_ddof(self):
        """Test zscore with ddof=1."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s)
        ts_std, model = standardize_timeseries(ts, method="zscore", ddof=1)

        # With ddof=1, std(standardized, ddof=1) should be 1
        assert np.abs(np.nanmean(ts_std.value)) < 1e-10
        assert np.abs(np.nanstd(ts_std.value, ddof=1) - 1.0) < 1e-10

    def test_robust_basic(self):
        """Test robust standardization produces median~0, 1.4826*MAD~1."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s)
        ts_std, model = standardize_timeseries(ts, method="robust")

        # Median should be ~0
        assert np.abs(np.nanmedian(ts_std.value)) < 1e-10
        # 1.4826 * MAD should be ~1
        mad = np.nanmedian(np.abs(ts_std.value - np.nanmedian(ts_std.value)))
        assert np.abs(1.4826 * mad - 1.0) < 1e-10

    def test_nan_preserved(self):
        """Test that NaN values are preserved in output."""
        data = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        ts = TimeSeries(data, dt=1.0 * u.s)
        ts_std, model = standardize_timeseries(ts, method="zscore")

        assert np.isnan(ts_std.value[1])
        assert np.isnan(ts_std.value[4])
        assert not np.isnan(ts_std.value[0])
        assert not np.isnan(ts_std.value[2])
        assert not np.isnan(ts_std.value[3])

    def test_zscore_with_nan(self):
        """Test zscore computes correctly ignoring NaN."""
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s)
        ts_std, model = standardize_timeseries(ts, method="zscore", ddof=0)

        # Check mean and std of non-NaN values
        valid = ~np.isnan(ts_std.value)
        assert np.abs(np.nanmean(ts_std.value[valid])) < 1e-10
        assert np.abs(np.nanstd(ts_std.value[valid], ddof=0) - 1.0) < 1e-10

    def test_robust_with_nan(self):
        """Test robust standardization with NaN values."""
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s)
        ts_std, model = standardize_timeseries(ts, method="robust")

        valid = ~np.isnan(ts_std.value)
        assert np.abs(np.nanmedian(ts_std.value[valid])) < 1e-10

    def test_constant_series_fallback(self):
        """Test that constant series (scale=0) falls back to scale=1."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ts_std, model = standardize_timeseries(ts, method="zscore")
            # Should warn about zero std
            assert any("zero" in str(warning.message).lower() for warning in w)

        # Output should be all zeros (since (5-5)/1 = 0)
        np.testing.assert_array_almost_equal(ts_std.value, np.zeros(5))

    def test_output_unit_dimensionless(self):
        """Test that output unit is always dimensionless_unscaled."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s, unit=u.m)  # Input has meters
        ts_std, model = standardize_timeseries(ts, method="zscore")

        assert ts_std.unit == u.dimensionless_unscaled

    def test_output_unit_dimensionless_robust(self):
        """Test that output unit is dimensionless for robust method."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s, unit=u.kg)
        ts_std, model = standardize_timeseries(ts, method="robust")

        assert ts_std.unit == u.dimensionless_unscaled

    def test_integer_dtype_promotion(self):
        """Test that integer dtype is promoted to float."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ts = TimeSeries(data, dt=1.0 * u.s)
        ts_std, model = standardize_timeseries(ts, method="zscore")

        assert ts_std.value.dtype.kind == 'f'  # float

    def test_metadata_preserved(self):
        """Test that metadata is preserved."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s, t0=100.0 * u.s, name="test_channel")
        ts_std, model = standardize_timeseries(ts, method="zscore")

        assert ts_std.dt == ts.dt
        assert ts_std.t0 == ts.t0
        assert ts_std.name == ts.name

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        data = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries(data, dt=1.0 * u.s)

        with pytest.raises(ValueError, match="Unknown standardization method"):
            standardize_timeseries(ts, method="invalid")


class TestStandardizeMatrix:
    """Tests for standardize_matrix function."""

    def test_axis_time_zscore(self):
        """Test axis='time' standardizes each channel over time."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100) * 10 + 5  # 2x3 channels, 100 time
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)
        mat_std = standardize_matrix(mat, axis="time", method="zscore")

        # Each (row, col) should have mean~0 and std~1 over time
        for i in range(2):
            for j in range(3):
                channel = mat_std.value[i, j, :]
                assert np.abs(np.nanmean(channel)) < 1e-10
                assert np.abs(np.nanstd(channel, ddof=0) - 1.0) < 1e-10

    def test_axis_channel_zscore(self):
        """Test axis='channel' standardizes each time sample across channels."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100) * 10 + 5
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)
        mat_std = standardize_matrix(mat, axis="channel", method="zscore")

        # Each time sample should have mean~0 and std~1 across channels
        for t in range(100):
            sample = mat_std.value[:, :, t].flatten()
            assert np.abs(np.nanmean(sample)) < 1e-10
            assert np.abs(np.nanstd(sample, ddof=0) - 1.0) < 1e-10

    def test_axis_time_robust(self):
        """Test axis='time' with robust method."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100) * 10 + 5
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)
        mat_std = standardize_matrix(mat, axis="time", method="robust")

        # Each channel should have median~0 and 1.4826*MAD~1
        for i in range(2):
            for j in range(3):
                channel = mat_std.value[i, j, :]
                assert np.abs(np.nanmedian(channel)) < 1e-10
                mad = np.nanmedian(np.abs(channel - np.nanmedian(channel)))
                assert np.abs(1.4826 * mad - 1.0) < 0.1  # Looser tolerance for small samples

    def test_axis_channel_robust(self):
        """Test axis='channel' with robust method."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100) * 10 + 5
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)
        mat_std = standardize_matrix(mat, axis="channel", method="robust")

        # Each time sample should have median~0
        for t in range(100):
            sample = mat_std.value[:, :, t].flatten()
            assert np.abs(np.nanmedian(sample)) < 1e-10

    def test_invalid_axis_raises(self):
        """Test that invalid axis values raise ValueError."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 10)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        with pytest.raises(ValueError, match="axis must be 'time' or 'channel'"):
            standardize_matrix(mat, axis="freq")

        with pytest.raises(ValueError, match="axis must be 'time' or 'channel'"):
            standardize_matrix(mat, axis=None)

        with pytest.raises(ValueError, match="axis must be 'time' or 'channel'"):
            standardize_matrix(mat, axis=0)

        with pytest.raises(ValueError, match="axis must be 'time' or 'channel'"):
            standardize_matrix(mat, axis=-1)

    def test_output_unit_dimensionless(self):
        """Test that output unit is dimensionless."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 10)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_std = standardize_matrix(mat, axis="time")

        if hasattr(mat_std, 'unit') and mat_std.unit is not None:
            assert mat_std.unit == u.dimensionless_unscaled

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 10)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        with pytest.raises(ValueError, match="Unknown standardization method"):
            standardize_matrix(mat, axis="time", method="invalid")


class TestWhitenMatrix:
    """Tests for whiten_matrix function."""

    def test_pca_covariance_identity(self):
        """Test PCA whitening produces covariance approximately identity."""
        np.random.seed(42)
        # Create correlated data
        n_time = 500
        data = np.random.randn(2, 3, n_time)
        # Add correlation
        data[0, 1, :] = data[0, 0, :] + 0.5 * np.random.randn(n_time)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="pca")

        # Flatten to (time, features)
        X_w = mat_w.value.reshape(-1, mat_w.shape[-1]).T
        cov = np.cov(X_w, rowvar=False)

        np.testing.assert_array_almost_equal(cov, np.eye(cov.shape[0]), decimal=1)

    def test_zca_covariance_identity(self):
        """Test ZCA whitening produces covariance approximately identity."""
        np.random.seed(42)
        n_time = 500
        data = np.random.randn(2, 3, n_time)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="zca")

        X_w = mat_w.value.reshape(-1, mat_w.shape[-1]).T
        cov = np.cov(X_w, rowvar=False)

        np.testing.assert_array_almost_equal(cov, np.eye(cov.shape[0]), decimal=1)

    def test_pca_n_components_covariance(self):
        """Test PCA with n_components produces correct dimensionality."""
        np.random.seed(42)
        n_time = 500
        k = 3
        data = np.random.randn(2, 3, n_time)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="pca", n_components=k)

        X_w = mat_w.value.reshape(-1, mat_w.shape[-1]).T
        assert X_w.shape[1] == k

        cov = np.cov(X_w, rowvar=False)
        np.testing.assert_array_almost_equal(cov, np.eye(k), decimal=1)

    def test_pca_output_shape_flat(self):
        """Test PCA always outputs flattened shape (features, 1, time)."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="pca")

        # Should be (6, 1, 100) - flattened features
        assert mat_w.shape == (6, 1, 100)

    def test_pca_n_components_output_shape(self):
        """Test PCA with n_components outputs (n_components, 1, time)."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="pca", n_components=4)

        assert mat_w.shape == (4, 1, 100)

    def test_zca_output_shape_original(self):
        """Test ZCA without n_components preserves original shape."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="zca")

        assert mat_w.shape == (2, 3, 100)

    def test_zca_n_components_output_shape(self):
        """Test ZCA with n_components outputs flattened."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mat_w, model = whiten_matrix(mat, method="zca", n_components=4)
            # Should warn about flattened output
            assert any("flattened" in str(warning.message).lower() for warning in w)

        assert mat_w.shape == (4, 1, 100)

    def test_rank_deficient_no_nan(self):
        """Test that rank-deficient data doesn't produce NaN due to eps."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100)
        # Make one channel identical to another (rank deficient)
        data[1, 0, :] = data[0, 0, :]
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="pca", eps=1e-6)

        assert not np.any(np.isnan(mat_w.value))
        assert not np.any(np.isinf(mat_w.value))

    def test_output_unit_dimensionless_pca(self):
        """Test PCA output unit is dimensionless."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="pca")

        if hasattr(mat_w, 'unit') and mat_w.unit is not None:
            assert mat_w.unit == u.dimensionless_unscaled

    def test_output_unit_dimensionless_zca(self):
        """Test ZCA output unit is dimensionless."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 100)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="zca")

        if hasattr(mat_w, 'unit') and mat_w.unit is not None:
            assert mat_w.unit == u.dimensionless_unscaled

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        np.random.seed(42)
        data = np.random.randn(2, 3, 10)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        with pytest.raises(ValueError, match="method must be 'pca' or 'zca'"):
            whiten_matrix(mat, method="invalid")

    def test_2d_matrix(self):
        """Test with 2D input matrix (channels, time) - note TimeSeriesMatrix may reshape."""
        np.random.seed(42)
        data = np.random.randn(5, 100)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        # Store original shape as seen by the matrix
        original_shape = mat.shape

        # PCA
        mat_w_pca, _ = whiten_matrix(mat, method="pca")
        # PCA always flattens to (features, 1, time)
        n_features = int(np.prod(original_shape[:-1]))
        assert mat_w_pca.shape == (n_features, 1, original_shape[-1])

        # ZCA preserves original matrix shape
        mat_w_zca, _ = whiten_matrix(mat, method="zca")
        assert mat_w_zca.shape == original_shape

    def test_single_feature(self):
        """Test with single feature (1D data)."""
        np.random.seed(42)
        data = np.random.randn(1, 1, 100)
        mat = TimeSeriesMatrix(data, dt=0.01 * u.s)

        mat_w, model = whiten_matrix(mat, method="pca")
        assert not np.any(np.isnan(mat_w.value))

        # Variance should be ~1
        var = np.var(mat_w.value.flatten())
        assert np.abs(var - 1.0) < 0.1


class TestModelInverseTransform:
    """Tests for inverse transform capability."""

    def test_standardization_inverse(self):
        """Test standardization inverse transform."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = TimeSeries(data, dt=1.0 * u.s)
        ts_std, model = standardize_timeseries(ts, method="zscore")

        # Inverse transform
        X_rec = model.inverse_transform(ts_std.value)
        np.testing.assert_array_almost_equal(X_rec.flatten(), data, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
