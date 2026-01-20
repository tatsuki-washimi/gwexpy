
import numpy as np

from gwexpy.timeseries import TimeSeriesMatrix

try:
    import torch  # noqa: F401 - availability check
except ImportError:
    torch = None
import pytest


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_timeseriesmatrix_to_torch():
    data = np.random.randn(2, 1, 100)
    tsm = TimeSeriesMatrix(data, dt=0.01)

    # 1. to_torch (via InteropMixin on SeriesMatrix)
    tensor = tsm.to_torch()
    assert torch.is_tensor(tensor)
    assert tensor.shape == (2, 1, 100)
    assert np.allclose(tensor.numpy(), data)

    # 2. to_tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        tf = None

    if tf is not None:
        tensor_tf = tsm.to_tensorflow()
        assert tf.is_tensor(tensor_tf)
        assert np.allclose(tensor_tf.numpy(), data)

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_timeseriesmatrix_to_torch_device():
    data = np.random.randn(2, 1, 100)
    tsm = TimeSeriesMatrix(data, dt=0.01)
    # Check that arguments are passed correctly
    tensor = tsm.to_torch(device='cpu', copy=True)
    assert torch.is_tensor(tensor)
