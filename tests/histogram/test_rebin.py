import numpy as np
import pytest
from astropy import units as u

from gwexpy.histogram import Histogram


def test_rebin_preserves_integral():
    values = [10, 20, 30, 40]
    edges = [0, 1, 2, 3, 4]
    hist = Histogram(values, edges, unit="count")

    new_edges = [0.5, 1.5, 2.5]
    rebinned = hist.rebin(new_edges)

    # 0.5 to 1.5 covers half of bin 0 (value 5) and half of bin 1 (value 10) -> 15
    # 1.5 to 2.5 covers half of bin 1 (value 10) and half of bin 2 (value 15) -> 25
    assert np.allclose(rebinned.values.value, [15, 25])
    assert rebinned.unit == u.Unit("count")

def test_rebin_handles_units():
    values = [10, 20]
    edges = [0, 1000, 2000]
    hist = Histogram(values, edges, unit="count", xunit="Hz")

    # Rebin using kHz
    new_edges = [0.5, 1.5]
    rebinned = hist.rebin(new_edges, xunit="kHz")

    # internally this maps to 500 Hz to 1500 Hz
    # 500 to 1000 is half of bin 0 (5). 1000 to 1500 is half of bin 1 (10). -> 15.
    assert np.allclose(rebinned.values.value, [15])

def test_integral():
    values = [10, 20, 30, 40]
    edges = [0, 1, 2, 3, 4]
    hist = Histogram(values, edges, unit="count", sumw2=[1, 2, 3, 4])

    val, err = hist.integral(0.5, 2.5)
    # Value should be 5 + 20 + 15 = 40
    assert val.value == 40.0

    # Error variance:
    # A_0 = 0.5, A_1 = 1.0, A_2 = 0.5
    # var = (0.5)^2 * 1 + (1.0)^2 * 2 + (0.5)^2 * 3 = 0.25 + 2 + 0.75 = 3.0
    assert np.allclose(err, np.sqrt(3.0))
