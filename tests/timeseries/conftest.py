import sys
import importlib
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

@pytest.fixture
def mock_sklearn_decomposition(monkeypatch):
    """sklearn.decomposition を MagicMock で差し替え、テスト後に自動復元"""
    mock_sklearn = MagicMock()
    mock_pca = MagicMock()
    mock_ica = MagicMock()
    
    # 精巧なモック挙動を定義 (test_pipeline.py のロジックを参考)
    mock_pca_inst = mock_pca.return_value
    def _pca_fit(X, y=None):
        n_comp = mock_pca.call_args[1].get('n_components', X.shape[1])
        if n_comp is None: n_comp = X.shape[1]
        mock_pca_inst.n_components_ = n_comp
        mock_pca_inst.components_ = np.zeros((n_comp, X.shape[1]))
        mock_pca_inst.explained_variance_ratio_ = np.ones(n_comp) / n_comp
        return mock_pca_inst
    mock_pca_inst.fit.side_effect = _pca_fit

    def _pca_transform(X):
        n = getattr(mock_pca_inst, 'n_components_', 1)
        if isinstance(n, MagicMock): n = X.shape[1]
        # If n_components matches X.shape[1], return X for round-trip testing
        if n == X.shape[1]: return X
        return np.zeros((X.shape[0], n))
    mock_pca_inst.transform.side_effect = _pca_transform

    def _pca_inverse_transform(X):
        # We need to return (samples, features)
        components = getattr(mock_pca_inst, 'components_', np.zeros((1, X.shape[1])))
        features = components.shape[1]
        if X.shape[1] == features: return X
        return np.zeros((X.shape[0], features))
    mock_pca_inst.inverse_transform.side_effect = _pca_inverse_transform

    mock_ica_inst = mock_ica.return_value
    def _ica_fit(X, y=None):
        n_comp = mock_ica.call_args[1].get('n_components', X.shape[1])
        if n_comp is None: n_comp = X.shape[1]
        mock_ica_inst.n_components_ = n_comp
        return mock_ica_inst
    mock_ica_inst.fit.side_effect = _ica_fit

    def _ica_transform(X):
        n = getattr(mock_ica_inst, 'n_components_', 1)
        if isinstance(n, MagicMock): n = X.shape[1]
        if n == X.shape[1]: return X
        return np.zeros((X.shape[0], n))
    mock_ica_inst.transform.side_effect = _ica_transform

    def _ica_inverse_transform(X):
        n = getattr(mock_ica_inst, 'n_components_', 1)
        if isinstance(n, MagicMock): n = X.shape[1]
        return np.zeros((X.shape[0], n))
    mock_ica_inst.inverse_transform.side_effect = _ica_inverse_transform

    mock_sklearn.decomposition.PCA = mock_pca
    mock_sklearn.decomposition.FastICA = mock_ica

    with patch.dict(sys.modules, {
        "sklearn": mock_sklearn,
        "sklearn.decomposition": mock_sklearn.decomposition
    }):
        import gwexpy.timeseries.decomposition as decomposition
        importlib.reload(decomposition)
        
        # decomposition モジュール内の参照を強制的にモックに向ける
        monkeypatch.setattr(decomposition, "SKLEARN_AVAILABLE", True)
        monkeypatch.setattr(decomposition, "PCA", mock_pca)
        monkeypatch.setattr(decomposition, "FastICA", mock_ica)
        
        yield (mock_pca, mock_ica)

    # Teardown: sys.modules は patch.dict で戻るが、
    # decomposition モジュールが保持している状態を物理環境に戻すためにリロード
    importlib.reload(decomposition)

@pytest.fixture
def mock_minepy():
    """minepy を MagicMock で差し替え"""
    mock_m = MagicMock()
    # デフォルトの挙動を設定 (test_matrix_analysis.py などを参考)
    mock_m.MINE.return_value.mic.return_value = 0.5
    with patch.dict(sys.modules, {"minepy": mock_m}):
        yield mock_m

@pytest.fixture
def mock_dcor():
    """dcor を MagicMock で差し替え"""
    mock_d = MagicMock()
    # デフォルトの挙動を設定
    mock_d.distance_correlation.return_value = 0.5
    with patch.dict(sys.modules, {"dcor": mock_d}):
        yield mock_d

@pytest.fixture
def mock_optional_ml_stack(mock_sklearn_decomposition, mock_minepy, mock_dcor):
    """sklearn, minepy, dcor 全てをモックするスタック"""
    return {
        "sklearn": mock_sklearn_decomposition,
        "minepy": mock_minepy,
        "dcor": mock_dcor
    }
