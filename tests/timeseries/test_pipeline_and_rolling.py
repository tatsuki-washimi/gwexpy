import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import (
    TimeSeries,
    TimeSeriesMatrix,
    Pipeline,
    ImputeTransform,
    StandardizeTransform,
    PCATransform,
)


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
