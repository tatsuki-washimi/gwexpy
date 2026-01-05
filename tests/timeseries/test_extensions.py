
import pytest
import numpy as np
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.timeseries.preprocess import align_timeseries_collection

# Optional deps
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import statsmodels
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import hurst
    HAS_HURST = True
except ImportError:
    HAS_HURST = False

# --- Preprocess Tests ---
def test_align_intersection():
    ts1 = TimeSeries([1, 2, 3, 4], t0=0, dt=1, name="ts1") # 0..3
    ts2 = TimeSeries([10, 20, 30], t0=1, dt=1, name="ts2") # 1..3

    # Intersection: [1, 3] overlap (inclusive) -> times [1, 2, 3] -> 3 samples
    mat, times, meta = align_timeseries_collection([ts1, ts2], how="intersection")

    assert mat.shape == (3, 2)
    np.testing.assert_array_equal(times.value, [1, 2, 3])
    # ts1 values at 1, 2, 3 -> 2, 3, 4
    # ts2 values at 1, 2, 3 -> 10, 20, 30
    np.testing.assert_array_equal(mat[:, 0], [2, 3, 4])
    np.testing.assert_array_equal(mat[:, 1], [10, 20, 30])

def test_align_union():
    ts1 = TimeSeries([1, 2], t0=0, dt=1) # 0, 1
    ts2 = TimeSeries([3, 4], t0=2, dt=1) # 2, 3

    mat, times, meta = align_timeseries_collection([ts1, ts2], how="union", fill_value=-1)

    assert mat.shape == (4, 2)
    np.testing.assert_array_equal(times.value, [0, 1, 2, 3])

    # ts1: 1, 2 at 0, 1. Rest filled
    np.testing.assert_array_equal(mat[:, 0], [1, 2, -1, -1])
    # ts2: 3, 4 at 2, 3. Rest filled
    np.testing.assert_array_equal(mat[:, 1], [-1, -1, 3, 4])

def test_align_time_based_with_dimensionless_dt():
    t = np.arange(4) * u.s
    ts_time = TimeSeries([1, 2, 3, 4], times=t, name="ts_time")
    ts_dim = TimeSeries([10, 20, 30, 40], t0=0, dt=1, xunit=u.dimensionless_unscaled, name="ts_dim")

    mat, times, meta = align_timeseries_collection([ts_time, ts_dim], how="intersection")

    assert times.unit == u.s
    assert meta["dt"].unit.physical_type == "time"
    np.testing.assert_array_equal(mat[:, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(mat[:, 1], [10, 20, 30, 40])


def test_align_time_based_rejects_non_time_dt():
    t = np.arange(4) * u.s
    ts_time = TimeSeries([1, 2, 3, 4], times=t, name="ts_time")
    ts_bad = TimeSeries([10, 20, 30, 40], t0=0, dt=1, xunit=u.Hz, name="ts_bad")

    with pytest.raises(ValueError, match="time-like dt"):
        align_timeseries_collection([ts_time, ts_bad], how="intersection")

def test_impute():
    data = [1.0, np.nan, 3.0, np.nan, 5.0]
    ts = TimeSeries(data, dt=1, t0=0)

    res = ts.impute(method="interpolate")
    np.testing.assert_array_almost_equal(res.value, [1, 2, 3, 4, 5])

    # res_bfill = ts.impute(method="bfill")
    # np.testing.assert_array_equal(res_bfill.value, [1, 3, 3, 5, 5])

def test_standardize():
    data = [1., 2., 3., 4., 5.] # explicit float
    ts = TimeSeries(data, dt=1, t0=0)
    ret = ts.standardize(method="zscore")
    if isinstance(ret, tuple):
        res, model = ret
    else:
        res = ret

    assert np.isclose(np.mean(res.value), 0, atol=1e-10)
    assert np.isclose(np.std(res.value), 1, atol=1e-10)


def test_matrix_whiten():
    # Simple correlated data
    # (samples, channels)
    np.random.seed(42)
    t = np.linspace(0, 1, 100) * u.s
    s1 = np.sin(2*np.pi*5*t.value)
    # Full rank for whitening check (otherwise eps dominates)
    s2 = 0.5 * s1 + 0.1 * np.random.randn(len(t))

    # TimeSeriesMatrix expects (rows, cols, samples)
    data = np.stack([s1, s2])[:, None, :]
    mat = TimeSeriesMatrix(data, times=t)

    w_mat, model = mat.whiten_channels(method="pca")

    val = w_mat.value.reshape(-1, 100)

    cov = np.cov(val, rowvar=True)
    np.testing.assert_allclose(cov, np.eye(2), atol=2e-1) # Looser tolerance due to noise/size

# --- PCA/ICA Tests ---

@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn required")
def test_pca():
    t = np.linspace(0, 1, 100) * u.s
    s1 = np.sin(2*np.pi*5*t.value)
    s2 = np.cos(2*np.pi*5*t.value)
    # Mix
    m1 = s1 + s2
    m2 = s1 - s2

    # (2, 1, 100)
    data = np.stack([m1, m2])[:, None, :]
    mat = TimeSeriesMatrix(data, times=t)

    # Fit
    pca_res = mat.pca_fit(n_components=2)
    scores = mat.pca_transform(pca_res)

    # scores should be (2, 1, 100)
    assert scores.shape == (2, 1, 100)

    # Reconstruct
    rec = mat.pca_inverse_transform(pca_res, scores)

    # rec should match mat (2, 1, 100)
    np.testing.assert_allclose(rec.value, mat.value, atol=1e-5)

@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn required")
def test_ica():
    # Synthetic sources
    n_samples = 200
    time = np.linspace(0, 8, n_samples) * u.s
    s1 = np.sin(2 * time.value)
    s2 = np.sign(np.sin(3 * time.value))
    # Mix
    S = np.stack([s1, s2])
    S += 0.2 * np.random.normal(size=S.shape)
    S /= np.std(S, axis=1, keepdims=True)

    A = np.array([[1, 1], [0.5, 2]])
    X = A @ S # (2, 200)

    # (2, 1, 200)
    X_3d = X[:, None, :]

    mat = TimeSeriesMatrix(X_3d, times=time)

    ica_res = mat.ica_fit(n_components=2, random_state=42)
    sources = mat.ica_transform(ica_res)

    assert sources.shape == (2, 1, n_samples)
    # Check correlation with original sources (order/sign undefined)
    # ...

# --- ARIMA Tests ---

@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels required")
def test_arima():
    # AR(1) process: y_t = 0.9 * y_{t-1} + e_t
    np.random.seed(42)
    y = np.zeros(100)
    for i in range(1, 100):
        y[i] = 0.9 * y[i-1] + np.random.normal()

    ts = TimeSeries(y, dt=1, t0=0)

    res = ts.fit_arima(order=(1, 0, 0))
    assert isinstance(res.params_dict()['params'], (list, np.ndarray))

    forecast, conf = res.forecast(steps=5)
    assert len(forecast) == 5
    assert forecast.t0.value == 100.0

# --- Hurst Tests ---
# Mock hurst if not installed?
# Or skip.
@pytest.mark.skipif(not HAS_HURST, reason="hurst package required")
def test_hurst():
    # Random walk H ~ 0.5
    rw = np.cumsum(np.random.randn(1000))
    ts = TimeSeries(rw, dt=1)

    H = ts.hurst(method="rs", simplified=True)
    assert 0.3 < H < 0.7 # Broad check

    # local hurst
    lhs = ts.local_hurst(window=100, step=50)
    assert len(lhs) > 0
    assert lhs.unit == u.dimensionless_unscaled
