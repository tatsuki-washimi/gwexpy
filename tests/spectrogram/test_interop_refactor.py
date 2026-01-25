import numpy as np

from gwexpy.spectrogram import Spectrogram, SpectrogramMatrix

try:
    import torch  # noqa: F401 - availability check
except ImportError:
    torch = None


def test_spectrogram_interop():
    data = np.random.randn(10, 20)
    spec = Spectrogram(data, times=np.arange(10), frequencies=np.arange(20))

    if torch:
        tensor = spec.to_torch()
        assert torch.is_tensor(tensor)
        assert tensor.shape == (10, 20)
        assert np.allclose(tensor.numpy(), data)
        print("Spectrogram to_torch passed")


def test_spectrogram_matrix_interop():
    data = np.random.randn(2, 10, 20)
    spec_mat = SpectrogramMatrix(data, times=np.arange(10), frequencies=np.arange(20))

    # Test .value property
    assert spec_mat.value.shape == (2, 10, 20)

    if torch:
        tensor = spec_mat.to_torch()
        assert torch.is_tensor(tensor)
        assert tensor.shape == (2, 10, 20)
        assert np.allclose(tensor.numpy(), data)
        print("SpectrogramMatrix to_torch passed")


if __name__ == "__main__":
    test_spectrogram_interop()
    test_spectrogram_matrix_interop()
