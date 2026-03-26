import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import (
    ImputeTransform,
    PCATransform,
    Pipeline,
    StandardizeTransform,
    TimeSeries,
    TimeSeriesMatrix,
)
from gwexpy.timeseries.collections import TimeSeriesDict, TimeSeriesList


def test_pipeline_pca_inverse_roundtrip():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 1, 128))
    data[0, 0, 5] = np.nan
    mat = TimeSeriesMatrix(data, dt=0.1 * u.s, t0=0.0 * u.s)

    pipe = Pipeline(
        [
            ("impute", ImputeTransform(method="mean")),
            ("standardize", StandardizeTransform()),
            ("pca", PCATransform(n_components=3)),
        ]
    )
    transformed = pipe.fit_transform(mat)
    assert isinstance(transformed, TimeSeriesMatrix)

    reconstructed = pipe.inverse_transform(transformed, strict=False)
    assert reconstructed.shape == mat.shape
    assert reconstructed.dt == mat.dt
    assert reconstructed.t0 == mat.t0

    baseline = ImputeTransform(method="mean").transform(mat)
    np.testing.assert_allclose(
        reconstructed.value,
        baseline.value,
        atol=1e-6,
        rtol=1e-3,
    )


def test_rolling_statistics_quantity_window():
    data = np.array([1.0, np.nan, 3.0, 4.0])
    ts = TimeSeries(data, dt=1.0 * u.s)

    mean_ts = ts.rolling_mean(2, nan_policy="omit")
    expected_mean = [1.0, 1.0, 3.0, 3.5]
    np.testing.assert_allclose(mean_ts.value, expected_mean, equal_nan=True)

    mean_q = ts.rolling_mean(2 * u.s, nan_policy="omit")
    np.testing.assert_allclose(mean_q.value, expected_mean, equal_nan=True)

    med_ts = ts.rolling_median(2, nan_policy="omit")
    expected_median = [1.0, 1.0, 3.0, 3.5]
    np.testing.assert_allclose(med_ts.value, expected_median, equal_nan=True)

    std_ts = ts.rolling_std(2, nan_policy="omit", ddof=0)
    expected_std = [0.0, 0.0, 0.0, 0.5]
    np.testing.assert_allclose(std_ts.value, expected_std, equal_nan=True)


# --- ImputeTransform ---

def test_impute_transform_timeseries():
    data = np.array([1.0, np.nan, 3.0, 4.0])
    ts = TimeSeries(data, dt=1.0 * u.s, unit=u.m)
    result = ImputeTransform(method="mean").transform(ts)
    assert isinstance(result, TimeSeries)
    assert not np.any(np.isnan(result.value))


def test_impute_transform_matrix():
    data = np.array([[[1.0, np.nan, 3.0], [2.0, 2.0, 2.0]]])
    mat = TimeSeriesMatrix(data, dt=1.0 * u.s, t0=0.0 * u.s)
    result = ImputeTransform(method="mean").transform(mat)
    assert isinstance(result, TimeSeriesMatrix)
    assert not np.any(np.isnan(result.value))


def test_impute_transform_dict():
    ts1 = TimeSeries([1.0, np.nan, 3.0], dt=1.0 * u.s, unit=u.m)
    ts2 = TimeSeries([np.nan, 2.0, 3.0], dt=1.0 * u.s, unit=u.m)
    td = TimeSeriesDict({"a": ts1, "b": ts2})
    result = ImputeTransform(method="mean").transform(td)
    assert isinstance(result, TimeSeriesDict)
    assert not np.any(np.isnan(result["a"].value))
    assert not np.any(np.isnan(result["b"].value))


def test_impute_transform_list():
    ts1 = TimeSeries([1.0, np.nan, 3.0], dt=1.0 * u.s, unit=u.m)
    tl = TimeSeriesList([ts1])
    result = ImputeTransform(method="mean").transform(tl)
    assert isinstance(result, TimeSeriesList)
    assert not np.any(np.isnan(result[0].value))


def test_impute_transform_unsupported_type():
    with pytest.raises(TypeError):
        ImputeTransform().transform([1, 2, 3])


# --- StandardizeTransform ---

def test_standardize_timeseries_roundtrip():
    ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], dt=1.0 * u.s, unit=u.m)
    st = StandardizeTransform()
    standardized = st.fit_transform(ts)
    assert isinstance(standardized, TimeSeries)
    np.testing.assert_allclose(np.nanmean(standardized.value), 0.0, atol=1e-10)

    restored = st.inverse_transform(standardized)
    np.testing.assert_allclose(restored.value, ts.value, atol=1e-10)


def test_standardize_timeseries_robust():
    ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 100.0], dt=1.0 * u.s, unit=u.m)
    st = StandardizeTransform(robust=True)
    result = st.fit_transform(ts)
    assert isinstance(result, TimeSeries)


def test_standardize_matrix_roundtrip():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(2, 1, 50))
    mat = TimeSeriesMatrix(data, dt=0.1 * u.s, t0=0.0 * u.s)
    st = StandardizeTransform()
    standardized = st.fit_transform(mat)
    restored = st.inverse_transform(standardized)
    np.testing.assert_allclose(restored.value, mat.value, atol=1e-10)


def test_standardize_dict_roundtrip():
    ts1 = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], dt=1.0 * u.s, unit=u.m)
    ts2 = TimeSeries([10.0, 20.0, 30.0, 40.0, 50.0], dt=1.0 * u.s, unit=u.m)
    td = TimeSeriesDict({"a": ts1, "b": ts2})
    st = StandardizeTransform()
    standardized = st.fit_transform(td)
    assert isinstance(standardized, TimeSeriesDict)
    restored = st.inverse_transform(standardized)
    np.testing.assert_allclose(restored["a"].value, ts1.value, atol=1e-10)


def test_standardize_list_roundtrip():
    ts1 = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s, unit=u.m)
    ts2 = TimeSeries([4.0, 5.0, 6.0], dt=1.0 * u.s, unit=u.m)
    tl = TimeSeriesList([ts1, ts2])
    st = StandardizeTransform()
    standardized = st.fit_transform(tl)
    assert isinstance(standardized, TimeSeriesList)
    restored = st.inverse_transform(standardized)
    np.testing.assert_allclose(restored[0].value, ts1.value, atol=1e-10)


def test_standardize_inverse_not_fitted():
    st = StandardizeTransform()
    ts = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s, unit=u.m)
    with pytest.raises(ValueError, match="not been fitted"):
        st.inverse_transform(ts)


def test_standardize_zero_scale():
    """Constant series should not cause division by zero."""
    ts = TimeSeries([5.0, 5.0, 5.0, 5.0], dt=1.0 * u.s, unit=u.m)
    st = StandardizeTransform()
    result = st.fit_transform(ts)
    assert np.all(np.isfinite(result.value))


# --- Pipeline: fit / transform independently ---

def test_pipeline_fit_then_transform():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    data = rng.normal(size=(2, 1, 64))
    mat = TimeSeriesMatrix(data, dt=0.1 * u.s, t0=0.0 * u.s)

    pipe = Pipeline([
        ("standardize", StandardizeTransform()),
        ("pca", PCATransform(n_components=2)),
    ])
    pipe.fit(mat)
    result = pipe.transform(mat)
    assert isinstance(result, TimeSeriesMatrix)
