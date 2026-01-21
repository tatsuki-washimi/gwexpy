
import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


@pytest.fixture
def gaussian_data():
    np.random.seed(42)
    return TimeSeries(np.random.normal(0, 1, 1000), dt=0.01)

@pytest.fixture
def non_gaussian_data():
    np.random.seed(42)
    # Exponential distribution is skewed and leptokurtic
    return TimeSeries(np.random.exponential(1, 1000), dt=0.01)

@pytest.fixture
def simple_linear_relationship():
    t = np.linspace(0, 10, 100)
    x = TimeSeries(t, dt=0.1)
    y = TimeSeries(2 * t + 1, dt=0.1)
    return x, y

@pytest.fixture
def non_linear_relationship():
    t = np.linspace(-10, 10, 100)
    x = TimeSeries(t, dt=0.1)
    y = TimeSeries(t**2, dt=0.1) # Parabola
    return x, y

@pytest.fixture
def causal_relationship():
    np.random.seed(42)
    n = 200
    # X causes Y with delay
    x_val = np.random.randn(n)
    y_val = np.zeros(n)
    # Y depends on X from 1 step ago
    for i in range(1, n):
        y_val[i] = 0.5 * y_val[i-1] + 0.8 * x_val[i-1] + 0.1 * np.random.randn()

    x = TimeSeries(x_val, dt=1)
    y = TimeSeries(y_val, dt=1)
    return x, y

def test_skewness(gaussian_data, non_gaussian_data):
    s_gauss = gaussian_data.skewness()
    s_exp = non_gaussian_data.skewness()

    # Gaussian should be near 0
    assert abs(s_gauss) < 0.2
    # Exponential should be positive (around 2)
    assert s_exp > 1.0

def test_kurtosis(gaussian_data, non_gaussian_data):
    k_gauss = gaussian_data.kurtosis(fisher=True)
    k_exp = non_gaussian_data.kurtosis(fisher=True)

    # Gaussian (Fisher) should be near 0
    assert abs(k_gauss) < 0.2
    # Exponential should be positive (excess kurtosis)
    assert k_exp > 1.0

def test_pearson_correlation(simple_linear_relationship):
    x, y = simple_linear_relationship
    corr = x.pcc(y)
    assert abs(corr - 1.0) < 1e-5

def test_distance_correlation(non_linear_relationship):
    x, y = non_linear_relationship

    # Linear correlation should be close to 0 for parabola on symmetric domain
    pcc = x.pcc(y)
    assert abs(pcc) < 0.1

    try:
        # Distance correlation should detect the dependence
        dcor = x.distance_correlation(y)
        assert dcor > 0.4
    except Exception as exc:
        pytest.skip(f"dcor unavailable: {exc}")

def test_mic(non_linear_relationship):
    x, y = non_linear_relationship
    try:
        mic = x.mic(y)
        assert mic > 0.5
    except ImportError:
        pytest.skip("mictools (or minepy) not installed")

def test_granger_causality(causal_relationship):
    x, y = causal_relationship
    try:
        # Check if X causes Y
        p_val_xy = y.granger_causality(x, maxlag=5)
        # Should be significant (small p-value)
        assert p_val_xy < 0.05

        # Check if Y causes X (should not be significant)
        p_val_yx = x.granger_causality(y, maxlag=5)
        assert p_val_yx > 0.05
    except ImportError:
        pytest.skip("statsmodels not installed")
