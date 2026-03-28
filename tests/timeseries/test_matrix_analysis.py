"""Tests for gwexpy/timeseries/matrix_analysis.py"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

# --- Mocks for optional dependencies ---
import sys
from unittest.mock import MagicMock

# Mock minepy
try:
    import minepy # noqa: F401
except ImportError:
    mock_minepy = MagicMock()
    mock_minepy.MINE.return_value.mic.return_value = 0.5
    sys.modules["minepy"] = mock_minepy

# Mock dcor
try:
    import dcor # noqa: F401
except ImportError:
    mock_dcor = MagicMock()
    mock_dcor.distance_correlation.return_value = 0.5
    sys.modules["dcor"] = mock_dcor

# Mock sklearn for decomposition
try:
    import sklearn.decomposition # noqa: F401
    from gwexpy.timeseries.decomposition import SKLEARN_AVAILABLE
except ImportError:
    mock_sklearn = MagicMock()
    mock_pca = MagicMock()
    mock_ica = MagicMock()
    
    mock_pca_inst = mock_pca.return_value
    def _mock_pca_fit(X, y=None):
        n_comp = mock_pca.call_args[1].get('n_components', X.shape[1])
        if n_comp is None: n_comp = X.shape[1]
        mock_pca_inst.n_components_ = n_comp
        mock_pca_inst.components_ = np.zeros((n_comp, X.shape[1]))
        mock_pca_inst.explained_variance_ratio_ = np.ones(n_comp) / n_comp
        return mock_pca_inst
    mock_pca_inst.fit.side_effect = _mock_pca_fit
    mock_pca_inst.transform.side_effect = lambda X: np.zeros((X.shape[0], getattr(mock_pca_inst, 'n_components_', 1)))
    mock_pca_inst.inverse_transform.side_effect = lambda X: np.zeros((X.shape[0], getattr(mock_pca_inst, 'components_', np.zeros((1, X.shape[1]))).shape[1]))

    mock_ica_inst = mock_ica.return_value
    def _mock_ica_fit(X, y=None):
        n_comp = mock_ica.call_args[1].get('n_components', X.shape[1])
        if n_comp is None: n_comp = X.shape[1]
        mock_ica_inst.n_components_ = n_comp
        return mock_ica_inst
    mock_ica_inst.fit.side_effect = _mock_ica_fit
    mock_ica_inst.transform.side_effect = lambda X: np.zeros((X.shape[0], getattr(mock_ica_inst, 'n_components_', 1)))
    mock_ica_inst.inverse_transform.side_effect = lambda X: np.zeros((X.shape[0], getattr(mock_ica_inst, 'n_components_', 1)))

    mock_sklearn.decomposition.PCA = mock_pca
    mock_sklearn.decomposition.FastICA = mock_ica
    sys.modules["sklearn"] = mock_sklearn
    sys.modules["sklearn.decomposition"] = mock_sklearn.decomposition
    
    # Force decomposition availability
    import gwexpy.timeseries.decomposition as decomp
    decomp.SKLEARN_AVAILABLE = True
    decomp.PCA = mock_pca
    decomp.FastICA = mock_ica

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix


def _make_tsm(n_rows=2, n_cols=2, n_time=100, seed=0) -> TimeSeriesMatrix:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_rows, n_cols, n_time))
    return TimeSeriesMatrix(data, dt=0.01 * u.s, t0=0.0 * u.s)


# ---------------------------------------------------------------------------
# _resolve_axis
# ---------------------------------------------------------------------------

def test_resolve_axis_time():
    tsm = _make_tsm()
    assert tsm._resolve_axis("time") == tsm._x_axis_norm


def test_resolve_axis_channel():
    tsm = _make_tsm()
    assert tsm._resolve_axis("channel") == 0


def test_resolve_axis_int():
    tsm = _make_tsm()
    assert tsm._resolve_axis(1) == 1


# ---------------------------------------------------------------------------
# Statistical methods — ignore_nan paths
# ---------------------------------------------------------------------------

def test_mean_ignore_nan():
    tsm = _make_tsm()
    result = tsm.mean(ignore_nan=True)
    assert result.shape == (2, 2)


def test_std_ignore_nan():
    tsm = _make_tsm()
    result = tsm.std(ignore_nan=True)
    assert result.shape == (2, 2)


def test_rms_ignore_nan():
    tsm = _make_tsm()
    result = tsm.rms(ignore_nan=True)
    assert result.shape == (2, 2)
    assert np.all(result >= 0)


def test_min_ignore_nan():
    tsm = _make_tsm()
    result = tsm.min(ignore_nan=True)
    assert result.shape == (2, 2)


def test_max_ignore_nan():
    tsm = _make_tsm()
    result = tsm.max(ignore_nan=True)
    assert result.shape == (2, 2)


def test_skewness():
    tsm = _make_tsm()
    result = tsm.skewness()
    assert result.shape == (2, 2)


def test_kurtosis():
    tsm = _make_tsm()
    result = tsm.kurtosis()
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Signal transforms
# ---------------------------------------------------------------------------

def test_hilbert():
    tsm = _make_tsm()
    result = tsm.hilbert()
    assert result.shape == tsm.shape
    assert np.iscomplexobj(result.value)


def test_radian():
    tsm = _make_tsm()
    result = tsm.radian()
    assert result.shape == tsm.shape
    assert result.unit == u.rad


def test_radian_unwrap():
    tsm = _make_tsm()
    result = tsm.radian(unwrap=True)
    assert result.shape == tsm.shape


def test_degree():
    tsm = _make_tsm()
    result = tsm.degree()
    assert result.shape == tsm.shape
    assert result.unit == u.deg


def test_degree_unwrap():
    tsm = _make_tsm()
    result = tsm.degree(unwrap=True)
    assert result.shape == tsm.shape


def test_vectorized_taper():
    tsm = _make_tsm()
    result = tsm._vectorized_taper()
    assert result.shape == tsm.shape


def test_vectorized_detrend_inplace():
    tsm = _make_tsm()
    result = tsm._vectorized_detrend(inplace=True)
    assert result is tsm


# ---------------------------------------------------------------------------
# Resampling (time-bin path)
# ---------------------------------------------------------------------------

def test_resample_time_bin_string():
    tsm = _make_tsm(n_time=200)
    result = tsm.resample("0.1s")
    assert isinstance(result, TimeSeriesMatrix)
    assert result.shape[:2] == (2, 2)
    assert result.shape[2] <= 20  # 200 samples @ 0.01s / 0.1s bins


def test_resample_time_quantity():
    tsm = _make_tsm(n_time=200)
    result = tsm.resample(0.1 * u.s)
    assert isinstance(result, TimeSeriesMatrix)


# ---------------------------------------------------------------------------
# Impute / standardize / whiten
# ---------------------------------------------------------------------------

def test_impute():
    data = np.ones((2, 1, 50))
    data[0, 0, 10] = np.nan
    tsm = TimeSeriesMatrix(data, dt=0.01 * u.s, t0=0.0 * u.s)
    result = tsm.impute()
    assert result.shape == tsm.shape
    assert not np.any(np.isnan(result.value))


def test_standardize():
    tsm = _make_tsm()
    result = tsm.standardize()
    assert result.shape == tsm.shape


def test_whiten_channels_with_model():
    tsm = _make_tsm(n_time=200)
    w, model = tsm.whiten_channels()
    assert w is not None
    assert model is not None


def test_whiten_channels_no_model():
    tsm = _make_tsm(n_time=200)
    result = tsm.whiten_channels(return_model=False)
    assert isinstance(result, TimeSeriesMatrix)


# ---------------------------------------------------------------------------
# Rolling methods
# ---------------------------------------------------------------------------

def test_rolling_mean():
    tsm = _make_tsm()
    result = tsm.rolling_mean(5)
    assert result.shape == tsm.shape


def test_rolling_std():
    tsm = _make_tsm()
    result = tsm.rolling_std(5)
    assert result.shape == tsm.shape


def test_rolling_median():
    tsm = _make_tsm()
    result = tsm.rolling_median(5)
    assert result.shape == tsm.shape


def test_rolling_min():
    tsm = _make_tsm()
    result = tsm.rolling_min(5)
    assert result.shape == tsm.shape


def test_rolling_max():
    tsm = _make_tsm()
    result = tsm.rolling_max(5)
    assert result.shape == tsm.shape


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

def test_crop_float():
    tsm = _make_tsm(n_time=200)
    result = tsm.crop(0.5, 1.0)
    assert result.shape[:2] == (2, 2)
    assert result.shape[2] == 50


def test_crop_quantity():
    tsm = _make_tsm(n_time=200)
    result = tsm.crop(0.5 * u.s, 1.0 * u.s)
    assert result.shape[:2] == (2, 2)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def test_pca_fit_transform():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    pca_res = tsm.pca_fit()
    scores = tsm.pca_transform(pca_res)
    assert scores.shape[2] == 200


def test_pca_inverse_transform():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    pca_res = tsm.pca_fit()
    scores = tsm.pca_transform(pca_res)
    reconstructed = tsm.pca_inverse_transform(pca_res, scores)
    assert reconstructed is not None


def test_pca_convenience():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    scores = tsm.pca()
    assert scores.shape[2] == 200


def test_pca_return_model():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    scores, model = tsm.pca(return_model=True)
    assert model is not None


# ---------------------------------------------------------------------------
# ICA
# ---------------------------------------------------------------------------

def test_ica_fit_transform():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    ica_res = tsm.ica_fit()
    sources = tsm.ica_transform(ica_res)
    assert sources.shape[2] == 200


def test_ica_inverse_transform():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    ica_res = tsm.ica_fit()
    sources = tsm.ica_transform(ica_res)
    reconstructed = tsm.ica_inverse_transform(ica_res, sources)
    assert reconstructed is not None


def test_ica_convenience():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    sources = tsm.ica()
    assert sources.shape[2] == 200


def test_ica_return_model():
    pytest.importorskip("sklearn")
    tsm = _make_tsm(n_time=200)
    sources, model = tsm.ica(return_model=True)
    assert model is not None


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def test_correlation_pairwise():
    tsm = _make_tsm()
    result = tsm.correlation()
    n_ch = 2 * 2
    assert result.shape == (n_ch, n_ch)


def test_correlation_with_1d_target():
    """correlation(target) with ndim==1 → correlation_vector."""
    tsm = _make_tsm(n_time=200)
    target = TimeSeries(
        np.random.default_rng(1).normal(size=200), dt=0.01 * u.s, t0=0.0 * u.s
    )
    df = tsm.correlation(target, method="pearson")
    assert "score" in df.columns
    assert len(df) == 4  # 2x2 channels


def test_mic():
    """mic() delegates to correlation(method='mic')."""
    pytest.importorskip("minepy")
    tsm = _make_tsm(n_time=200)
    target = TimeSeries(
        np.random.default_rng(2).normal(size=200), dt=0.01 * u.s, t0=0.0 * u.s
    )
    df = tsm.mic(target)
    assert "score" in df.columns


def test_distance_correlation():
    pytest.importorskip("dcor")
    tsm = _make_tsm(n_time=200)
    target = TimeSeries(
        np.random.default_rng(3).normal(size=200), dt=0.01 * u.s, t0=0.0 * u.s
    )
    df = tsm.distance_correlation(target)
    assert "score" in df.columns


# ---------------------------------------------------------------------------
# partial_correlation_matrix
# ---------------------------------------------------------------------------

def test_partial_correlation_matrix():
    tsm = _make_tsm(n_time=200)
    result = tsm.partial_correlation_matrix()
    assert result.shape == (4, 4)
    np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)


def test_partial_correlation_matrix_shrinkage_auto():
    tsm = _make_tsm(n_time=200)
    result = tsm.partial_correlation_matrix(shrinkage="auto")
    assert result.shape == (4, 4)


def test_partial_correlation_matrix_shrinkage_float():
    tsm = _make_tsm(n_time=200)
    result = tsm.partial_correlation_matrix(shrinkage=0.1)
    assert result.shape == (4, 4)


def test_partial_correlation_matrix_return_precision():
    tsm = _make_tsm(n_time=200)
    pcorr, precision = tsm.partial_correlation_matrix(return_precision=True)
    assert pcorr.shape == (4, 4)
    assert precision.shape == (4, 4)


def test_partial_correlation_matrix_invalid_estimator():
    tsm = _make_tsm(n_time=200)
    with pytest.raises(ValueError, match="Unknown estimator"):
        tsm.partial_correlation_matrix(estimator="bad")


def test_partial_correlation_matrix_invalid_shrinkage():
    tsm = _make_tsm(n_time=200)
    with pytest.raises(ValueError, match="shrinkage"):
        tsm.partial_correlation_matrix(shrinkage=2.0)


def test_partial_correlation_matrix_too_few_samples():
    data = np.ones((2, 1, 1))
    tsm = TimeSeriesMatrix(data, dt=0.01 * u.s, t0=0.0 * u.s)
    with pytest.raises(ValueError, match="at least 2 samples"):
        tsm.partial_correlation_matrix()
