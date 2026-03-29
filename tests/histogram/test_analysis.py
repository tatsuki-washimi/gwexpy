import numpy as np
import pytest
from astropy import units as u

from gwexpy.histogram import Histogram


def test_histogram_stats():
    # Normal distribution centered at 5 with std 1
    np.random.seed(42)
    data = np.random.normal(5, 1, 10000)
    counts, edges = np.histogram(data, bins=100, range=(0, 10))
    h = Histogram(counts, edges, unit=u.dimensionless_unscaled, xunit=u.m)

    # Values should be approximately correct
    assert pytest.approx(h.mean().value, rel=1e-2) == 5.0
    assert pytest.approx(h.std().value, rel=1e-2) == 1.0
    assert pytest.approx(h.median().value, rel=1e-2) == 5.0

    # Quantiles
    assert pytest.approx(h.quantile(0.16).value, rel=1e-2) == 4.0  # 1-sigma low
    assert pytest.approx(h.quantile(0.84).value, rel=1e-2) == 6.0  # 1-sigma high


def test_histogram_integral():
    counts = np.array([1, 2, 3, 2, 1])
    edges = np.array([0, 1, 2, 3, 4, 5])
    h = Histogram(counts, edges, unit=u.ct, xunit=u.s)

    assert h.integral().value == 9
    assert h.integral(start=1.5, end=3.5).value == 5  # Bins: [2,3], [3,4] -> 3+2=5
    assert h.integral(start=2, end=3).value == 3  # Bin: [2,3] -> 3


def test_histogram_min_max():
    counts = np.array([0, 0, 1, 2, 3, 0, 0])
    edges = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    h = Histogram(counts, edges)

    assert h.min().value == 2.0
    assert h.max().value == 5.0  # Upper edge of the last non-empty bin (index 4 is 4-5)


def test_histogram_empty():
    h = Histogram(np.zeros(10), np.arange(11))
    assert np.isnan(h.mean().value)
    assert np.isnan(h.var().value)
    assert np.isnan(h.median().value)
    assert np.isnan(h.min().value)


def test_histogram_fitting():
    # Gaussian data
    np.random.seed(42)
    data = np.random.normal(10, 2, 10000)
    counts, edges = np.histogram(data, bins=50, range=(0, 20))
    h = Histogram(counts, edges, xunit=u.m)

    # Fit to Gaussian
    # We use a simple lambda as model: f(x, mean, std, amplitude)
    def gaussian(x, mu, sig, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    # Needs iminuit
    pytest.importorskip("iminuit")

    result = h.fit(gaussian, p0={"mu": 10, "sig": 2, "amp": 800})

    assert pytest.approx(result.params["mu"].value, rel=1e-1) == 10.0
    assert pytest.approx(result.params["sig"].value, rel=1e-1) == 2.0
