import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries

try:
    import torch  # noqa: F401 - availability check
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_frequencyseries_to_torch():
    data = np.arange(10, dtype=np.float32)
    fs = FrequencySeries(data, df=1.0)

    # 1. to_torch (via InteropMixin)
    tensor = fs.to_torch()
    assert torch.is_tensor(tensor)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.numpy(), data)

    # 2. from_torch
    fs2 = FrequencySeries.from_torch(tensor, frequencies=fs.frequencies)
    assert np.allclose(fs2.value, data)
    assert np.allclose(fs2.frequencies.value, fs.frequencies.value)


try:
    import tensorflow as tf
except ImportError:
    tf = None


@pytest.mark.skipif(tf is None, reason="tensorflow not installed")
def test_frequencyseries_to_tf():
    data = np.arange(10, dtype=np.float32)
    fs = FrequencySeries(data, df=1.0)

    # 1. to_tensorflow (via InteropMixin)
    tensor = fs.to_tensorflow()
    assert tf.is_tensor(tensor)
    assert np.allclose(tensor.numpy(), data)

    # 2. from_tensorflow
    fs2 = FrequencySeries.from_tensorflow(tensor, frequencies=fs.frequencies)
    assert np.allclose(fs2.value, data)
