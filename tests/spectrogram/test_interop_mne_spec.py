
import pytest
import numpy as np
from gwexpy.spectrogram import Spectrogram

def test_to_mne_tfr():
    mne = pytest.importorskip("mne")
    if not hasattr(mne.time_frequency, "EpochsTFRArray"):
        pytest.skip("mne.time_frequency.EpochsTFRArray not available (requires MNE >= 1.3)")

    # Create Spectrogram (Frequency, Time) usually in gwexpy?
    # Let's verify shape. Standard Spectrogram is (Time, Frequency).
    # wait.. base gwpy Spectrogram is (Times, Frequencies).
    # Let's assume standard behavior. (n_times, n_freqs)

    data = np.array([[1, 2], [3, 4], [5, 6]]) # 3 times, 2 freqs
    times = np.array([0, 1, 2])
    freqs = np.array([10, 20])

    spec = Spectrogram(data, times=times, frequencies=freqs, name="test_spec")

    tfr = spec.to_mne()

    assert isinstance(tfr, mne.time_frequency.EpochsTFRArray)
    # MNE shape: (n_epochs, n_channels, n_freqs, n_times)
    # Here: (1, 1, 2, 3)

    d = tfr.data
    assert d.shape == (1, 1, 2, 3)

    # Check values. Spectrogram was (3, 2) -> (2, 3) transposed?
    # Spec:
    # t=0: [1, 2] (f=10, 20)
    # t=1: [3, 4]
    # t=2: [5, 6]

    # MNE:
    # ch=0, f=10 -> [1, 3, 5]
    # ch=0, f=20 -> [2, 4, 6]

    assert np.allclose(d[0, 0, 0, :], [1, 3, 5])
    assert np.allclose(d[0, 0, 1, :], [2, 4, 6])

    assert np.allclose(tfr.times, times)
    assert np.allclose(tfr.freqs, freqs)


def test_from_mne_tfr():
    mne = pytest.importorskip("mne")
    if not hasattr(mne.time_frequency, "EpochsTFRArray"):
        pytest.skip("mne.time_frequency.EpochsTFRArray not available")

    # Construct EpochsTFRArray manually
    # shape: (1, 1, 2, 3) -> (1 epoch, 1 ch, 2 freqs, 3 times)
    data = np.array([[[[1, 3, 5], [2, 4, 6]]]])
    times = np.array([0, 1, 2])
    freqs = np.array([10, 20])
    info = mne.create_info(["ch1"], 1.0, ["misc"])

    tfr = mne.time_frequency.EpochsTFRArray(info, data, times, freqs)

    spec = Spectrogram.from_mne(tfr)

    assert isinstance(spec, Spectrogram)
    # GWpy Spectrogram: (times, freqs) -> (3, 2)
    assert spec.shape == (3, 2)

    # Check values
    assert np.allclose(spec.value, [[1, 2], [3, 4], [5, 6]])
    assert spec.name == "ch1"
