import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries


def test_to_quantities():
    pq = pytest.importorskip("quantities")

    data = np.array([1, 2, 3])
    freqs = np.array([10, 20, 30])
    fs = FrequencySeries(data, frequencies=freqs, unit="V")

    q = fs.to_quantities()
    assert isinstance(q, pq.Quantity)
    assert q.units == pq.V
    assert np.allclose(q.magnitude, data)

    # with explicit units
    q_mv = fs.to_quantities(units="mV")
    assert q_mv.units == pq.mV
    assert np.allclose(q_mv.magnitude, data * 1000)


def test_from_quantities():
    pq = pytest.importorskip("quantities")

    q_data = pq.Quantity([1, 2, 3], "V")
    freqs = np.linspace(0, 10, 3)

    fs = FrequencySeries.from_quantities(q_data, frequencies=freqs)

    assert isinstance(fs, FrequencySeries)
    assert fs.unit == u.V
    assert np.allclose(fs.value, [1, 2, 3])
