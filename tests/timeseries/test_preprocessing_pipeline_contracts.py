from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.timeseries.collections import TimeSeriesList
from gwexpy.timeseries.decomposition import (
    ICAResult,
    PCAResult,
    ica_inverse_transform,
    pca_inverse_transform,
)
from gwexpy.timeseries.pipeline import ImputeTransform, Pipeline, StandardizeTransform
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


def test_pipeline_inverse_strict_error_and_standardize_unsupported_input_contract():
    ts = TimeSeries([1.0, np.nan, 3.0], t0=0.0 * u.s, dt=1.0 * u.s)

    unsupported_inverse = Pipeline([("impute", ImputeTransform(method="ffill"))])
    transformed = unsupported_inverse.fit_transform(ts)

    with pytest.raises(ValueError, match="does not support inverse_transform"):
        unsupported_inverse.inverse_transform(transformed)

    standardize = StandardizeTransform().fit(ts)
    with pytest.raises(TypeError, match="Incompatible input"):
        standardize.inverse_transform(np.asarray(transformed.value))


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


def test_multivariate_standardize_list_inverse_uses_flat_to_list_restoration():
    series_list = TimeSeriesList()
    for index, values in enumerate(([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])):
        series_list.append(
            TimeSeries(
                values,
                t0=5.0 * u.s,
                dt=0.5 * u.s,
                unit=u.m,
                name=f"witness-{index}",
            )
        )

    transform = StandardizeTransform(multivariate=True)
    transformed = transform.fit_transform(series_list)
    restored = transform.inverse_transform(transformed)

    assert isinstance(transformed, TimeSeriesMatrix)
    assert transformed.shape == (2, 1, 3)
    assert transformed.channel_names == ["witness-0", "witness-1"]
    assert isinstance(restored, TimeSeriesList)
    assert [series.name for series in restored] == ["witness-0", "witness-1"]
    for restored_series, original_series in zip(restored, series_list, strict=True):
        np.testing.assert_allclose(restored_series.value, original_series.value)
        assert restored_series.t0 == original_series.t0
        assert restored_series.dt == original_series.dt
        assert restored_series.unit == u.dimensionless_unscaled


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


def test_pca_inverse_reconstructs_shape_from_input_meta_but_time_from_scores(
    monkeypatch,
):
    class FakePCA:
        def inverse_transform(self, scores):
            return np.arange(scores.shape[0] * 4.0).reshape(scores.shape[0], 4)

    import gwexpy.timeseries.decomposition as decomposition

    monkeypatch.setattr(decomposition, "_check_sklearn", lambda name="sklearn": None)
    scores = TimeSeriesMatrix(
        np.ones((2, 1, 3)),
        t0=20.0 * u.s,
        dt=0.25 * u.s,
    )
    result = PCAResult(
        sklearn_model=FakePCA(),
        channel_labels=["r0c0", "r0c1", "r1c0", "r1c1"],
        preprocessing={"center": False, "scale": None},
        input_meta={
            "t0": 5.0 * u.s,
            "dt": 1.0 * u.s,
            "original_shape": (2, 2, 3),
        },
    )

    reconstructed = pca_inverse_transform(result, scores)

    assert isinstance(reconstructed, TimeSeriesMatrix)
    assert reconstructed.shape == (2, 2, 3)
    assert reconstructed.t0 == scores.t0
    assert reconstructed.dt == scores.dt
    assert reconstructed.channel_names == result.channel_labels


def test_ica_result_metadata_and_inverse_reconstruction_contract(monkeypatch):
    class FakeICA:
        def inverse_transform(self, sources):
            return np.arange(sources.shape[0] * 2.0).reshape(sources.shape[0], 2)

    import gwexpy.timeseries.decomposition as decomposition

    monkeypatch.setattr(decomposition, "_check_sklearn", lambda name="sklearn": None)
    sources = TimeSeriesMatrix(
        np.ones((2, 1, 3)),
        t0=30.0 * u.s,
        dt=0.5 * u.s,
    )
    result = ICAResult(
        sklearn_model=FakeICA(),
        channel_labels=["mix-a", "mix-b"],
        preprocessing={"center": False, "scale": None, "prewhiten": False},
        input_meta={
            "t0": 7.0 * u.s,
            "dt": 2.0 * u.s,
            "original_shape": (2, 1, 3),
        },
    )

    reconstructed = ica_inverse_transform(result, sources)

    assert result.channel_labels == ["mix-a", "mix-b"]
    assert result.preprocessing["prewhiten"] is False
    assert isinstance(reconstructed, TimeSeriesMatrix)
    assert reconstructed.shape == (2, 1, 3)
    assert reconstructed.t0 == sources.t0
    assert reconstructed.dt == sources.dt
    assert reconstructed.channel_names == result.channel_labels
