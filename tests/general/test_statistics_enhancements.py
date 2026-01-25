import numpy as np
import pytest
from astropy import units as u

from gwexpy.analysis.bruco import Bruco
from gwexpy.timeseries import TimeSeries

# Mocking Bruco dependencies if necessary, but here we can test the logic directly
# We need TimeSeries for Bruco input


def test_statistics_ndim_check():
    """Test that statistics methods raise ValueError for 2D TimeSeries."""
    # Since TimeSeries constructor might strictly enforce 1D,
    # we verify the Mixin logic using a mock object that simulates a 2D state.

    from gwexpy.timeseries._statistics import StatisticsMixin

    class MockTS(StatisticsMixin):
        def __init__(self):
            self.ndim = 2
            self.shape = (2, 100)
            self.value = np.zeros((2, 100))
            self.sample_rate = 1

        def __len__(self):
            return 100

    ts_mock = MockTS()

    # Try calling a statistical method that uses _prep_stat_data
    # e.g. correlation
    with pytest.raises(
        ValueError, match="Statistical methods are only supported for 1D"
    ):
        ts_mock.correlation(ts_mock)


def test_granger_causality_return_format():
    """Test that granger_causality returns a dictionary with expected keys."""
    # Create valid 1D data
    np.random.seed(42)
    # y causes x with lag
    y_data = np.random.randn(100)
    x_data = np.roll(y_data, 2) + 0.1 * np.random.randn(100)  # x lags y by 2

    ts_x = TimeSeries(x_data, dt=1 * u.s)
    ts_y = TimeSeries(y_data, dt=1 * u.s)

    try:
        import statsmodels  # noqa: F401
    except ImportError:
        pytest.skip("statsmodels not installed")

    # Check x caused by y
    # granger_causality(other) checks if 'other' causes 'self'
    result = ts_x.granger_causality(ts_y, maxlag=5)

    # It should support dictionary-like access
    assert "min_p_value" in result
    assert "best_lag" in result
    assert "p_values" in result

    # Lag 2 should be significant
    assert (
        result["best_lag"] == 2 or result["best_lag"] == 1
    )  # heuristic might vary with noise
    assert result["min_p_value"] < 0.05


def test_bruco_validation():
    """Test Bruco compute parameter validation."""
    bruco = Bruco("TARGET", ["AUX1"], [])

    # fftlength > duration should raise ValueError
    with pytest.raises(ValueError, match="cannot be longer than duration"):
        bruco.compute(start=1000000000, duration=10, fftlength=11)
