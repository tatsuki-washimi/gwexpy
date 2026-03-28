import sys
from unittest.mock import MagicMock
import numpy as np
import pytest
from astropy import units as u

# Mock external dependencies before importing gwexpy modules that use them
mock_sklearn = MagicMock()
mock_pca = MagicMock()
mock_ica = MagicMock()
mock_sklearn.decomposition.PCA = mock_pca
mock_sklearn.decomposition.FastICA = mock_ica
sys.modules["sklearn"] = mock_sklearn
sys.modules["sklearn.decomposition"] = mock_sklearn.decomposition

mock_minepy = MagicMock()
sys.modules["minepy"] = mock_minepy

mock_dcor = MagicMock()
sys.modules["dcor"] = mock_dcor

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.timeseries.pipeline import PCATransform, ICATransform
import gwexpy.timeseries.decomposition as decomposition

# Force SKLEARN_AVAILABLE to True for this test session
decomposition.SKLEARN_AVAILABLE = True

def test_pca_transform_mocked():
    # Mock PCA model behavior
    mock_instance = mock_pca.return_value
    mock_instance.n_components_ = 2
    mock_instance.components_ = np.random.randn(2, 3)
    mock_instance.explained_variance_ratio_ = np.array([0.7, 0.3])
    # transform returns (samples, components)
    mock_instance.transform.return_value = np.random.randn(10, 2)
    mock_instance.inverse_transform.return_value = np.random.randn(10, 3)

    tsm = TimeSeriesMatrix(np.random.randn(3, 1, 10), dt=1)
    pca = PCATransform(n_components=2)
    
    # Fit & Transform
    out = pca.fit_transform(tsm)
    
    assert isinstance(out, TimeSeriesMatrix)
    assert out.shape == (2, 1, 10)
    assert mock_pca.called
    assert mock_instance.fit.called
    assert mock_instance.transform.called

    # Inverse Transform
    rec = pca.inverse_transform(out)
    assert isinstance(rec, TimeSeriesMatrix)
    assert rec.shape == (3, 1, 10)
    assert mock_instance.inverse_transform.called

def test_ica_transform_mocked():
    mock_instance = mock_ica.return_value
    mock_instance.transform.return_value = np.random.randn(10, 2)
    mock_instance.inverse_transform.return_value = np.random.randn(10, 3)

    tsm = TimeSeriesMatrix(np.random.randn(3, 1, 10), dt=1)
    ica = ICATransform(n_components=2)
    
    out = ica.fit_transform(tsm)
    assert out.shape == (2, 1, 10)
    assert mock_ica.called

def test_mic_statistics_mocked():
    # Mock minepy.MINE
    mock_mine_instance = mock_minepy.MINE.return_value
    mock_mine_instance.mic.return_value = 0.8
    
    ts1 = TimeSeries([1, 2, 3], dt=1)
    ts2 = TimeSeries([1.1, 2.1, 3.1], dt=1)
    
    # correlation calls MINE internally for method='mic'
    val = ts1.correlation(ts2, method="mic")
    assert val == 0.8
    assert mock_minepy.MINE.called

def test_dcor_statistics_mocked():
    mock_dcor.distance_correlation.return_value = 0.95
    
    ts1 = TimeSeries([1, 2, 3], dt=1)
    ts2 = TimeSeries([1.1, 2.1, 3.1], dt=1)
    
    # correlation calls distance_correlation internally for method='distance'
    val = ts1.correlation(ts2, method="distance")
    assert val == 0.95
    assert mock_dcor.distance_correlation.called
