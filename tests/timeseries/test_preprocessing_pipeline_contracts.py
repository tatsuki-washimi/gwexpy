from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.timeseries.collections import TimeSeriesList
from gwexpy.timeseries.decomposition import PCAResult
from gwexpy.timeseries.pipeline import Pipeline, StandardizeTransform
from gwexpy.timeseries.preprocess import impute_timeseries, standardize_timeseries


def test_standardize_timeseries_preserves_time_metadata_and_array_inverse():
    ts = TimeSeries(
        [2.0, 4.0, 6.0, 8.0],
        t0=100.0 * u.s,
        dt=0.25 * u.s,
        unit=u.m,
        name="calibrated",
        channel="H1:CAL",
    )

    standardized, model = standardize_timeseries(ts, method="zscore")
    restored_values = model.inverse_transform(standardized)

    assert standardized.unit == u.dimensionless_unscaled
    assert standardized.t0 == ts.t0
    assert standardized.dt == ts.dt
    assert standardized.name == ts.name
    assert standardized.channel == ts.channel
    assert isinstance(restored_values, np.ndarray)
    np.testing.assert_allclose(restored_values, ts.value)


def test_impute_timeseries_treats_inf_as_observed_value_not_missing():
    ts = TimeSeries(
        [1.0, np.nan, np.inf, 5.0],
        t0=3.0 * u.s,
        dt=0.25 * u.s,
        unit=u.kg,
        name="inf-policy",
    )

    result = impute_timeseries(ts, method="ffill")

    np.testing.assert_array_equal(result.value, [1.0, 1.0, np.inf, 5.0])
    assert result.t0 == ts.t0
    assert result.dt == ts.dt
    assert result.unit == ts.unit
    assert result.name == ts.name


def test_impute_timeseries_quantity_max_gap_uses_time_axis_units():
    times = np.array([0.0, 100.0, 300.0, 400.0]) * u.ms
    ts = TimeSeries([1.0, np.nan, np.nan, 4.0], times=times, unit=u.V)

    filled = impute_timeseries(ts, method="linear", max_gap=500.0 * u.ms)
    blocked = impute_timeseries(ts, method="linear", max_gap=0.2 * u.s)

    np.testing.assert_allclose(filled.value, [1.0, 1.75, 3.25, 4.0])
    np.testing.assert_allclose(blocked.value, [1.0, np.nan, np.nan, 4.0])
    np.testing.assert_allclose(filled.times.to_value(u.ms), times.to_value(u.ms))
    assert filled.times.unit == times.unit
    assert filled.unit == ts.unit
    assert blocked.unit == ts.unit


def test_pipeline_standardize_inverse_restores_values_and_time_metadata():
    ts = TimeSeries(
        [1.0, 2.0, 4.0, 8.0],
        t0=10.0 * u.s,
        dt=0.5 * u.s,
        unit=u.m,
        name="pipeline-input",
    )
    pipeline = Pipeline([("standardize", StandardizeTransform())])

    transformed = pipeline.fit_transform(ts)
    restored = pipeline.inverse_transform(transformed)

    np.testing.assert_allclose(restored.value, ts.value)
    assert restored.t0 == ts.t0
    assert restored.dt == ts.dt
    assert restored.unit == ts.unit
    assert restored.name == ts.name


def test_multicolumn_matrix_to_list_flattens_in_row_major_order():
    names = ["r0c0", "r0c1", "r1c0", "r1c1"]
    data = np.arange(16.0).reshape(2, 2, 4)
    matrix = TimeSeriesMatrix(
        data,
        t0=1.0 * u.s,
        dt=0.25 * u.s,
        unit=u.m,
        channel_names=names,
    )

    result = matrix.to_list()

    assert isinstance(result, TimeSeriesList)
    assert [series.name for series in result] == names
    assert len(result) == 4
    for index, series in enumerate(result):
        row, col = divmod(index, 2)
        np.testing.assert_array_equal(series.value, data[row, col])
        assert series.t0 == matrix.t0
        assert series.dt == matrix.dt
        assert series.unit == u.m


def test_pca_result_summary_and_metadata_are_passive_contracts():
    class FakePCA:
        components_ = np.array([[1.0, 0.0], [0.0, 1.0]])
        explained_variance_ratio_ = np.array([0.75, 0.25])
        n_components_ = 2

    preprocessing = {"center": True, "scale": None, "nan_policy": "raise"}
    input_meta = {"t0": 4.0 * u.s, "dt": 0.125 * u.s, "original_shape": (2, 1, 8)}
    result = PCAResult(
        sklearn_model=FakePCA(),
        channel_labels=["a", "b"],
        preprocessing=preprocessing,
        input_meta=input_meta,
    )

    np.testing.assert_array_equal(result.components, FakePCA.components_)
    np.testing.assert_array_equal(result.components_, FakePCA.components_)
    np.testing.assert_array_equal(
        result.explained_variance_ratio, FakePCA.explained_variance_ratio_
    )
    assert result.summary_dict() == {
        "explained_variance_ratio": [0.75, 0.25],
        "n_components": 2,
    }
    assert result.channel_labels == ["a", "b"]
    assert result.preprocessing is preprocessing
    assert result.input_meta is input_meta
