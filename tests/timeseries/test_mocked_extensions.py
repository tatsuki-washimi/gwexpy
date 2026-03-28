import sys
import numpy as np
import pytest
from astropy import units as u
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix

@pytest.mark.usefixtures("mock_optional_ml_stack")
def test_pca_transform_mocked():
    from gwexpy.timeseries.pipeline import PCATransform
    # Mock PCA model behavior
    mock_pca = sys.modules["sklearn.decomposition"].PCA
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

@pytest.mark.usefixtures("mock_optional_ml_stack")
def test_ica_transform_mocked():
    from gwexpy.timeseries.pipeline import ICATransform
    mock_ica = sys.modules["sklearn.decomposition"].FastICA
    mock_instance = mock_ica.return_value
    mock_instance.transform.return_value = np.random.randn(10, 2)
    mock_instance.inverse_transform.return_value = np.random.randn(10, 3)

    tsm = TimeSeriesMatrix(np.random.randn(3, 1, 10), dt=1)
    ica = ICATransform(n_components=2)
    
    out = ica.fit_transform(tsm)
    assert out.shape == (2, 1, 10)
    assert mock_ica.called

@pytest.mark.usefixtures("mock_optional_ml_stack")
def test_mic_statistics_mocked():
    mock_minepy = sys.modules["minepy"]
    # Mock minepy.MINE
    mock_mine_instance = mock_minepy.MINE.return_value
    mock_mine_instance.mic.return_value = 0.8
    
    ts1 = TimeSeries([1, 2, 3], dt=1)
    ts2 = TimeSeries([1.1, 2.1, 3.1], dt=1)
    
    # correlation calls MINE internally for method='mic'
    val = ts1.correlation(ts2, method="mic")
    assert val == 0.8
    assert mock_minepy.MINE.called

@pytest.mark.usefixtures("mock_optional_ml_stack")
def test_dcor_statistics_mocked():
    mock_dcor = sys.modules["dcor"]
    mock_dcor.distance_correlation.return_value = 0.95
    
    ts1 = TimeSeries([1, 2, 3], dt=1)
    ts2 = TimeSeries([1.1, 2.1, 3.1], dt=1)
    
    # correlation calls distance_correlation internally for method='distance'
    val = ts1.correlation(ts2, method="distance")
    assert val == 0.95
    assert mock_dcor.distance_correlation.called
