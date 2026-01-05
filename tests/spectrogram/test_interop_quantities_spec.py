
import pytest
import numpy as np
from astropy import units as u
from gwexpy.spectrogram import Spectrogram

def test_spectrogram_quantities():
    pq = pytest.importorskip("quantities")
    
    # Create spectrogram
    data = np.array([[1, 2], [3, 4]])
    times = np.array([0, 1])
    freqs = np.array([10, 20])
    spec = Spectrogram(data, times=times, frequencies=freqs, unit="V")
    
    # to_quantities
    q = spec.to_quantities()
    assert isinstance(q, pq.Quantity)
    assert q.units == pq.V
    assert np.allclose(q.magnitude, data)
    
    # from_quantities
    spec_new = Spectrogram.from_quantities(q, times=times, frequencies=freqs)
    assert isinstance(spec_new, Spectrogram)
    assert spec_new.unit == u.V
    assert np.allclose(spec_new.value, data)
