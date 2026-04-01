import numpy as np
import pytest
from astropy import units as u

from gwexpy.histogram import Histogram


def test_histogram_fill_statistical():
    # 1. Test fill with weights as Quantity
    edges = [0, 5, 10] * u.s
    # initial histogram with 10 counts in bin 1
    h = Histogram([10, 0], edges, unit=u.count, cov=np.diag([1.0, 1.0]) * u.count**2)

    # New data with weights in different unit but convertible
    data = [2, 7] * u.s
    weights = [1000, 1000] * u.Unit("mcount") # 1 count

    h_new = h.fill(data, weights=weights)

    assert h_new.values.value[0] == 11.0
    assert h_new.values.value[1] == 1.0

    # 2. Double Management Rule: Covariance diagonal update
    # Initial variance was [1, 1]. New variance from counts is [1^2, 1^2] = [1, 1].
    # Total diagonal should be [2, 2].
    assert np.allclose(np.diag(h_new.cov.value), [2.0, 2.0])

    # 3. Scalar broadcasting
    h_scalar = h.fill(data, weights=10 * u.count)
    assert h_scalar.values.value[0] == 20.0
    assert h_scalar.values.value[1] == 10.0
    # Variance increment: 10^2 = 100. Total: 101.
    assert np.allclose(np.diag(h_scalar.cov.value), [101.0, 101.0])

def test_histogram_integral_boundaries():
    edges = [0, 5, 10] * u.s
    h = Histogram([10, 10], edges, unit=u.count)

    # Normal case
    assert h.integral(2, 7).value == 10.0 # [2, 5] -> 3/5 * 10 = 6, [5, 7] -> 2/5 * 10 = 4. Sum = 10.

    # start == end
    assert h.integral(5.0, 5.0).value == 0.0
    assert h.integral(5.0, 5.0, return_error=True)[0].value == 0.0

    # nearly equal (floating point jitter)
    assert h.integral(5.0, 5.0 + 1e-15).value == 0.0

    # start > end
    with pytest.raises(ValueError, match="integral: start"):
        h.integral(7, 2)

def test_histogram_quantile_plateaus():
    # Bins [0, 1, 2, 3] with counts [10, 0, 10]
    # CDF edges: [0, 0.5, 0.5, 1.0]
    h = Histogram([10, 0, 10], [0, 1, 2, 3], xunit=u.s)

    # q=0.25 (inside first bin)
    assert np.isclose(h.quantile(0.25).value, 0.5)

    # q=0.5 (on a plateau: bin 1 has 0 weight)
    # The plateau covers edges [1, 2]. Midpoint is 1.5.
    assert np.isclose(h.quantile(0.5).value, 1.5)

    # q=0.75 (inside third bin)
    assert np.isclose(h.quantile(0.75).value, 2.5)

    # q=0 and q=1
    assert h.quantile(0).value == 0.0
    assert h.quantile(1).value == 3.0

    # Empty histogram
    h_empty = Histogram([0, 0], [0, 1, 2], xunit=u.s)
    assert np.isnan(h_empty.quantile(0.5).value)

def test_histogram_root_interop_overflow():
    pytest.importorskip("ROOT")
    from gwexpy.interop.root_ import to_th1d

    h = Histogram([10, 10], [0, 5, 10], unit=u.count)
    h.fill([-5, 15], weights=[1, 2]) # 1 underflow, 2 overflow

    th1 = to_th1d(h)
    assert th1.GetBinContent(0) == 1.0  # Underflow
    assert th1.GetBinContent(1) == 10.0 # Bin 1
    assert th1.GetBinContent(2) == 10.0 # Bin 2
    assert th1.GetBinContent(3) == 2.0  # Overflow

    # Check errors
    # Initial: counts=10, weight=1 -> sumw2=10. Error=sqrt(10).
    # New: underflow count=1 -> sumw2=1. Error=1.
    # New: overflow count=2 -> sumw2=4. Error=2.
    assert np.isclose(th1.GetBinError(0), 1.0)
    assert np.isclose(th1.GetBinError(3), 2.0)
