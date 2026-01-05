
import pytest
import numpy as np
from gwexpy.frequencyseries import FrequencySeries

def test_to_mne_spectrum():
    mne = pytest.importorskip("mne")

    data = np.array([1, 2, 3])
    freqs = np.array([10, 20, 30])
    fs = FrequencySeries(data, frequencies=freqs, name="test_ch")

    # Check if MNE version supports SpectrumArray
    if not hasattr(mne.time_frequency, "SpectrumArray"):
        pytest.skip("mne.time_frequency.SpectrumArray not available")

    spec = fs.to_mne()

    assert isinstance(spec, mne.time_frequency.SpectrumArray)
    # Check data shape: (n_channels, n_freqs) -> (1, 3) for SpectrumArray (static)
    # Note: installed MNE version enforces 2D for SpectrumArray
    assert spec.get_data().shape == (1, 3)
    assert np.allclose(spec.get_data()[0], data)
    assert np.allclose(spec.freqs, freqs)
    assert spec.ch_names == ["test_ch"]

def test_from_mne_spectrum():
    mne = pytest.importorskip("mne")

    if not hasattr(mne.time_frequency, "SpectrumArray"):
        pytest.skip("mne.time_frequency.SpectrumArray not available")

    data = np.array([[1, 2, 3]]) # (1, 3) -> 1 ch, 3 freqs
    info = mne.create_info(["ch1"], 100.0, ["mag"])
    freqs = np.array([10, 20, 30])

    spec = mne.time_frequency.SpectrumArray(data, info, freqs)

    fs = FrequencySeries.from_mne(spec)

    assert isinstance(fs, FrequencySeries)
    assert np.allclose(fs.value, [1, 2, 3])
    assert np.allclose(fs.frequencies.value, freqs)
    assert fs.name == "ch1"
