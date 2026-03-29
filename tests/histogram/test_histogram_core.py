import numpy as np
import pytest
from astropy import units as u

from gwexpy.histogram import Histogram


def test_histogram_init():
    values = [10, 20, 30]
    edges = [0.0, 1.0, 2.0, 3.0]
    hist = Histogram(values, edges, unit="count", xunit="Hz")

    assert hist.nbins == 3
    assert hist.unit == u.Unit("count")
    assert hist.xunit == u.Unit("Hz")
    assert np.allclose(hist.bin_widths.value, 1.0)
    assert hist.values.unit == u.Unit("count")
    assert hist.edges.unit == u.Unit("Hz")

def test_histogram_validation():
    # Edges length mapping
    with pytest.raises(ValueError):
        Histogram([10, 20], [0, 1])

    # Strictly increasing
    with pytest.raises(ValueError):
        Histogram([10, 20], [0, 2, 1])

def test_histogram_errors():
    values = [10, 20, 30]
    edges = [0, 1, 2, 3]
    sumw2 = [10, 20, 30]
    hist = Histogram(values, edges, sumw2=sumw2)
    assert hist.errors is not None
    assert np.allclose(hist.errors.value, np.sqrt(sumw2))

def test_histogram_fill():
    hist = Histogram([0, 0], [0.0, 1.0, 2.0], unit="count")
    data = [0.1, 0.5, 1.2, 1.9, 0.5]
    h2 = hist.fill(data)
    assert np.allclose(h2.values.value, [3, 2])
    assert h2.unit == u.Unit("count")

def test_histogram_fill_sumw2():
    hist = Histogram([0, 0], [0.0, 1.0, 2.0], unit="count", sumw2=[0, 0])
    data = [0.1, 0.5, 1.2, 1.9, 0.5]
    weights = [2, 1, 3, 1, 1]
    h2 = hist.fill(data, weights=weights)
    # Bin 1 (0-1): values 0.1(w=2), 0.5(w=1), 0.5(w=1). sum=4. sumw2=4+1+1=6
    # Bin 2 (1-2): values 1.2(w=3), 1.9(w=1). sum=4. sumw2=9+1=10
    assert np.allclose(h2.values.value, [4, 4])
    assert np.allclose(h2.sumw2.value, [6, 10])
