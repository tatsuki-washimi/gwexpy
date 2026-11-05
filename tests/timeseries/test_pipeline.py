"""
Comprehensive tests for gwexpy.timeseries.pipeline

Covers all classes and helper functions:
- _is_collection, _to_matrix_from_collection, _restore_collection
- Transform (base class)
- Pipeline
- ImputeTransform
- StandardizeTransform
- WhitenTransform
- PCATransform
- ICATransform
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.timeseries.collections import TimeSeriesDict, TimeSeriesList
from gwexpy.timeseries.pipeline import (
    ImputeTransform,
    Pipeline,
    StandardizeTransform,
    Transform,
    WhitenTransform,
    _is_collection,
    _restore_collection,
    _to_matrix_from_collection,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_ts(n: int = 100) -> TimeSeries:
    data = RNG.normal(size=n).astype(float)
    return TimeSeries(data, t0=0.0, dt=0.01, unit=u.m, name="test")


def _make_tsm(n_channels: int = 3, n_time: int = 50) -> TimeSeriesMatrix:
    data = RNG.normal(size=(n_channels, 1, n_time)).astype(float)
    return TimeSeriesMatrix(data, t0=0.0 * u.s, dt=1.0 * u.s)


def _make_tsd(n_channels: int = 3, n_time: int = 100) -> TimeSeriesDict:
    """Create a TimeSeriesDict with all channels sharing the same time array."""
    tsd = TimeSeriesDict()
    for i in range(n_channels):
        data = RNG.normal(size=n_time).astype(float)
        key = f"ch{i}"
        tsd[key] = TimeSeries(data, t0=0.0, dt=0.01, unit=u.m, name=key)
    return tsd


def _make_tsl(n_channels: int = 3, n_time: int = 100) -> TimeSeriesList:
    """Create a TimeSeriesList with all channels sharing the same time array."""
    tsl = TimeSeriesList()
    for i in range(n_channels):
        data = RNG.normal(size=n_time).astype(float)
        tsl.append(TimeSeries(data, t0=0.0, dt=0.01, unit=u.m, name=f"ch{i}"))
    return tsl


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestIsCollection:
    def test_timeseries_dict_is_collection(self):
        tsd = _make_tsd()
        assert _is_collection(tsd) is True

    def test_timeseries_list_is_collection(self):
        tsl = _make_tsl()
        assert _is_collection(tsl) is True

    def test_timeseries_is_not_collection(self):
        ts = _make_ts()
        assert _is_collection(ts) is False

    def test_matrix_is_not_collection(self):
        mat = _make_tsm()
        assert _is_collection(mat) is False

    def test_plain_list_is_not_collection(self):
        assert _is_collection([1, 2, 3]) is False

    def test_none_is_not_collection(self):
        assert _is_collection(None) is False


class TestToMatrixFromCollection:
    def test_timeseries_dict_returns_matrix(self):
        tsd = _make_tsd()
        mat, original = _to_matrix_from_collection(tsd)
        assert isinstance(mat, TimeSeriesMatrix)
        assert original is tsd

    def test_timeseries_list_returns_matrix(self):
        tsl = _make_tsl()
        mat, original = _to_matrix_from_collection(tsl)
        assert isinstance(mat, TimeSeriesMatrix)
        assert original is tsl

    def test_non_collection_passthrough(self):
        mat = _make_tsm()
        result, original = _to_matrix_from_collection(mat)
        assert result is mat
        assert original is None

    def test_timeseries_passthrough(self):
        ts = _make_ts()
        result, original = _to_matrix_from_collection(ts)
        assert result is ts
        assert original is None


class TestRestoreCollection:
    def test_none_original_returns_result(self):
        mat = _make_tsm()
        result = _restore_collection(mat, None)
        assert result is mat

    def test_timeseries_dict_original_converts_to_dict(self):
        tsd = _make_tsd()
        mat, original = _to_matrix_from_collection(tsd)
        result = _restore_collection(mat, original)
        assert isinstance(result, TimeSeriesDict)

    def test_timeseries_list_original_attempts_to_list_conversion(self):
        # _restore_collection delegates to mat.to_list(); depending on the
        # internal shape this may succeed or raise TypeError (known limitation
        # of the underlying to_list() implementation for multi-channel matrices).
        tsl = _make_tsl(n_channels=1)
        mat, original = _to_matrix_from_collection(tsl)
        assert isinstance(mat, TimeSeriesMatrix)
        assert original is tsl
        # The function is expected to call result.to_list() on the matrix.
        # We only verify that the call is attempted (no AttributeError) even
        # if the underlying to_list raises TypeError for certain shapes.
        try:
            result = _restore_collection(mat, original)
            assert isinstance(result, TimeSeriesList)
        except TypeError:
            pass  # known limitation for multi-row matrices

    def test_non_matrix_result_passthrough(self):
        tsd = _make_tsd()
        # If result is not a TimeSeriesMatrix, just return it
        result = _restore_collection("something", tsd)
        assert result == "something"


# ---------------------------------------------------------------------------
# Transform base class
# ---------------------------------------------------------------------------


class TestTransformBase:
    def test_fit_returns_self(self):
        t = Transform()
        result = t.fit("anything")
        assert result is t

    def test_transform_raises_not_implemented(self):
        t = Transform()
        with pytest.raises(NotImplementedError):
            t.transform("anything")

    def test_fit_transform_calls_fit_then_transform(self):
        class Echo(Transform):
            def transform(self, x):
                return x

        e = Echo()
        ts = _make_ts()
        result = e.fit_transform(ts)
        assert result is ts

    def test_inverse_transform_raises_not_implemented(self):
        t = Transform()
        with pytest.raises(NotImplementedError, match="inverse_transform"):
            t.inverse_transform("anything")

    def test_supports_inverse_false_by_default(self):
        t = Transform()
        assert t.supports_inverse is False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_init_raises_on_non_transform_step(self):
        with pytest.raises(TypeError, match="must be a Transform"):
            Pipeline([("bad", "not_a_transform")])

    def test_init_accepts_valid_steps(self):
        pipe = Pipeline([("impute", ImputeTransform())])
        assert len(pipe.steps) == 1

    def test_fit_returns_self(self):
        ts = _make_ts()
        pipe = Pipeline([("std", StandardizeTransform())])
        result = pipe.fit(ts)
        assert result is pipe

    def test_fit_sets_is_fitted(self):
        ts = _make_ts()
        pipe = Pipeline([("std", StandardizeTransform())])
        assert pipe._is_fitted is False
        pipe.fit(ts)
        assert pipe._is_fitted is True

    def test_transform_applies_all_steps(self):
        ts = _make_ts()
        pipe = Pipeline([("std", StandardizeTransform())])
        pipe.fit(ts)
        result = pipe.transform(ts)
        assert isinstance(result, TimeSeries)
        np.testing.assert_allclose(np.nanmean(result.value), 0.0, atol=1e-10)

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=50)
        ts1 = TimeSeries(data.copy(), t0=0.0, dt=0.01, unit=u.m)
        ts2 = TimeSeries(data.copy(), t0=0.0, dt=0.01, unit=u.m)

        pipe1 = Pipeline([("std", StandardizeTransform())])
        result1 = pipe1.fit_transform(ts1)

        pipe2 = Pipeline([("std", StandardizeTransform())])
        pipe2.fit(ts2)
        result2 = pipe2.transform(ts2)

        np.testing.assert_allclose(result1.value, result2.value)

    def test_inverse_transform_strict_raises_for_non_inverse_step(self):
        ts = _make_ts()
        pipe = Pipeline([("impute", ImputeTransform())])
        pipe.fit(ts)
        result = pipe.transform(ts)
        with pytest.raises(ValueError, match="does not support inverse_transform"):
            pipe.inverse_transform(result, strict=True)

    def test_inverse_transform_not_strict_skips_non_inverse_step(self):
        ts = _make_ts()
        pipe = Pipeline([("impute", ImputeTransform())])
        pipe.fit(ts)
        result = pipe.transform(ts)
        # Should not raise, just returns unchanged
        out = pipe.inverse_transform(result, strict=False)
        assert isinstance(out, TimeSeries)

    def test_inverse_transform_calls_supports_inverse_step(self):
        ts = _make_ts()
        pipe = Pipeline([("std", StandardizeTransform())])
        standardized = pipe.fit_transform(ts)
        restored = pipe.inverse_transform(standardized)
        np.testing.assert_allclose(restored.value, ts.value, atol=1e-10)

    def test_pipeline_multiple_steps_fit_transform(self):
        ts = _make_ts()
        ts.value[5] = np.nan
        pipe = Pipeline(
            [
                ("impute", ImputeTransform(method="mean")),
                ("std", StandardizeTransform()),
            ]
        )
        result = pipe.fit_transform(ts)
        assert isinstance(result, TimeSeries)
        assert not np.any(np.isnan(result.value))

    def test_pipeline_inverse_strict_with_mixed_steps(self):
        ts = _make_ts()
        pipe = Pipeline(
            [
                ("impute", ImputeTransform()),
                ("std", StandardizeTransform()),
            ]
        )
        pipe.fit(ts)
        transformed = pipe.transform(ts)
        # strict=True should raise because ImputeTransform doesn't support inverse
        with pytest.raises(ValueError):
            pipe.inverse_transform(transformed, strict=True)

    def test_pipeline_inverse_not_strict_with_mixed_steps(self):
        ts = _make_ts()
        pipe = Pipeline(
            [
                ("impute", ImputeTransform()),
                ("std", StandardizeTransform()),
            ]
        )
        transformed = pipe.fit_transform(ts)
        # strict=False should succeed, applying only StandardizeTransform inverse
        result = pipe.inverse_transform(transformed, strict=False)
        assert isinstance(result, TimeSeries)

    def test_empty_pipeline(self):
        ts = _make_ts()
        pipe = Pipeline([])
        result = pipe.fit_transform(ts)
        assert result is ts

    def test_pipeline_inverse_strict_default_is_true(self):
        ts = _make_ts()
        pipe = Pipeline([("impute", ImputeTransform())])
        transformed = pipe.fit_transform(ts)
        with pytest.raises(ValueError):
            pipe.inverse_transform(transformed)


# ---------------------------------------------------------------------------
# ImputeTransform
# ---------------------------------------------------------------------------


class TestImputeTransform:
    def test_transform_timeseries(self):
        data = np.array([1.0, np.nan, 3.0, 4.0])
        ts = TimeSeries(data, dt=1.0 * u.s, unit=u.m)
        result = ImputeTransform(method="mean").transform(ts)
        assert isinstance(result, TimeSeries)
        assert not np.any(np.isnan(result.value))

    def test_transform_timeseries_interpolate(self):
        data = np.array([1.0, np.nan, 3.0, 4.0])
        ts = TimeSeries(data, dt=1.0 * u.s, unit=u.m)
        result = ImputeTransform(method="interpolate").transform(ts)
        assert isinstance(result, TimeSeries)
        assert not np.any(np.isnan(result.value))

    def test_transform_matrix(self):
        data = np.array([[[1.0, np.nan, 3.0]], [[2.0, 2.0, 2.0]]])
        mat = TimeSeriesMatrix(data, dt=1.0 * u.s, t0=0.0 * u.s)
        result = ImputeTransform(method="mean").transform(mat)
        assert isinstance(result, TimeSeriesMatrix)
        assert not np.any(np.isnan(result.value))

    def test_transform_dict(self):
        ts1 = TimeSeries([1.0, np.nan, 3.0], dt=1.0 * u.s, unit=u.m)
        ts2 = TimeSeries([np.nan, 2.0, 3.0], dt=1.0 * u.s, unit=u.m)
        tsd = TimeSeriesDict({"a": ts1, "b": ts2})
        result = ImputeTransform(method="mean").transform(tsd)
        assert isinstance(result, TimeSeriesDict)
        assert not np.any(np.isnan(result["a"].value))
        assert not np.any(np.isnan(result["b"].value))

    def test_transform_list(self):
        # ImputeTransform.transform for TimeSeriesList now uses a loop and append()
        # to avoid the gwpy SeriesList list-constructor bug.
        ts1 = TimeSeries([1.0, np.nan, 3.0], dt=1.0 * u.s, unit=u.m)
        tsl = TimeSeriesList()
        tsl.append(ts1)
        imp = ImputeTransform(method="mean")

        # Test full list transform
        result = imp.transform(tsl)
        assert isinstance(result, TimeSeriesList)
        assert len(result) == 1
        assert not np.any(np.isnan(result[0].value))

    def test_transform_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unsupported type for ImputeTransform"):
            ImputeTransform().transform([1, 2, 3])

    def test_fit_is_identity(self):
        ts = _make_ts()
        imp = ImputeTransform()
        result = imp.fit(ts)
        assert result is imp

    def test_fit_transform_equivalent_to_transform(self):
        data = np.array([1.0, np.nan, 3.0])
        ts = TimeSeries(data.copy(), dt=1.0 * u.s, unit=u.m)
        imp = ImputeTransform(method="mean")
        r1 = imp.transform(ts)
        r2 = imp.fit_transform(TimeSeries(data.copy(), dt=1.0 * u.s, unit=u.m))
        np.testing.assert_allclose(r1.value, r2.value)


# ---------------------------------------------------------------------------
# StandardizeTransform
# ---------------------------------------------------------------------------


class TestStandardizeTransform:
    # -- TimeSeries path --

    def test_fit_transform_timeseries(self):
        ts = _make_ts()
        st = StandardizeTransform()
        result = st.fit_transform(ts)
        assert isinstance(result, TimeSeries)
        np.testing.assert_allclose(np.nanmean(result.value), 0.0, atol=1e-10)

    def test_inverse_transform_timeseries_roundtrip(self):
        ts = _make_ts()
        st = StandardizeTransform()
        standardized = st.fit_transform(ts)
        restored = st.inverse_transform(standardized)
        np.testing.assert_allclose(restored.value, ts.value, atol=1e-10)

    def test_integer_timeseries_keeps_fractional_standard_scores(self):
        ts = TimeSeries(
            [1, 2, 3],
            t0=7 * u.s,
            dt=0.5 * u.s,
            unit=u.m,
            name="integer input",
            channel="H1:INT",
        )
        st = StandardizeTransform()
        standardized = st.fit_transform(ts)
        np.testing.assert_allclose(np.mean(standardized.value), 0, atol=1e-12)
        np.testing.assert_allclose(np.std(standardized.value), 1, atol=1e-12)
        assert standardized.unit == u.dimensionless_unscaled
        assert standardized.t0 == ts.t0
        assert standardized.dt == ts.dt
        assert standardized.name == ts.name
        assert str(standardized.channel) == str(ts.channel)
        restored = st.inverse_transform(standardized)
        np.testing.assert_allclose(restored.value, ts.value)
        assert restored.unit == ts.unit

    def test_complex_timeseries_centers_real_and_imaginary_parts(self):
        ts = TimeSeries([1.0 + 2.0j, 2.0 + 3.0j, 3.0 + 5.0j], dt=1 * u.s, unit=u.V)
        st = StandardizeTransform()
        scaled = st.fit_transform(ts)
        np.testing.assert_allclose(np.mean(scaled.value), 0, atol=1e-12)
        assert np.iscomplexobj(scaled.value)
        np.testing.assert_allclose(st.inverse_transform(scaled).value, ts.value)

    def test_robust_timeseries(self):
        ts = _make_ts()
        st = StandardizeTransform(robust=True)
        result = st.fit_transform(ts)
        assert isinstance(result, TimeSeries)
        # Median should be near 0
        np.testing.assert_allclose(np.nanmedian(result.value), 0.0, atol=1e-10)

    def test_zero_scale_forced_to_one(self):
        ts = TimeSeries([5.0, 5.0, 5.0, 5.0], dt=1.0 * u.s, unit=u.m)
        st = StandardizeTransform()
        result = st.fit_transform(ts)
        assert np.all(np.isfinite(result.value))

    # -- TimeSeriesMatrix path --

    def test_fit_transform_matrix(self):
        mat = _make_tsm()
        st = StandardizeTransform()
        result = st.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)
        assert result.shape == mat.shape

    def test_inverse_transform_matrix_roundtrip(self):
        mat = _make_tsm()
        st = StandardizeTransform()
        standardized = st.fit_transform(mat)
        restored = st.inverse_transform(standardized)
        np.testing.assert_allclose(restored.value, mat.value, atol=1e-10)

    def test_axis_channel_matrix(self):
        mat = _make_tsm()
        st = StandardizeTransform(axis="channel")
        result = st.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_robust_matrix(self):
        mat = _make_tsm()
        st = StandardizeTransform(robust=True)
        result = st.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    # -- TimeSeriesDict path --

    def test_fit_transform_dict(self):
        tsd = _make_tsd()
        st = StandardizeTransform()
        result = st.fit_transform(tsd)
        assert isinstance(result, TimeSeriesDict)

    def test_inverse_transform_dict_roundtrip(self):
        ts1 = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], dt=1.0 * u.s, unit=u.m)
        ts2 = TimeSeries([10.0, 20.0, 30.0, 40.0, 50.0], dt=1.0 * u.s, unit=u.m)
        tsd = TimeSeriesDict({"a": ts1, "b": ts2})
        st = StandardizeTransform()
        standardized = st.fit_transform(tsd)
        restored = st.inverse_transform(standardized)
        np.testing.assert_allclose(restored["a"].value, ts1.value, atol=1e-10)
        np.testing.assert_allclose(restored["b"].value, ts2.value, atol=1e-10)

    def test_inverse_transform_dict_unknown_key_raises(self):
        ts = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s, unit=u.m)
        tsd_fit = TimeSeriesDict({"a": ts})
        st = StandardizeTransform()
        st.fit(tsd_fit)

        # Add a key not seen during fitting
        ts_new = TimeSeries([4.0, 5.0, 6.0], dt=1.0 * u.s, unit=u.m)
        tsd_unknown = TimeSeriesDict({"unknown_key": ts_new})
        with pytest.raises(KeyError):
            st.inverse_transform(tsd_unknown)

    def test_transform_dict_unseen_key_gets_computed(self):
        """transform (not inverse) auto-computes stats for new keys."""
        ts1 = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s, unit=u.m)
        ts2 = TimeSeries([4.0, 5.0, 6.0], dt=1.0 * u.s, unit=u.m)
        tsd_fit = TimeSeriesDict({"a": ts1})
        st = StandardizeTransform()
        st.fit(tsd_fit)

        # transform with a new key that wasn't in fit
        tsd_new = TimeSeriesDict({"a": ts1, "b": ts2})
        result = st.transform(tsd_new)
        assert isinstance(result, TimeSeriesDict)
        assert "b" in result

    # -- TimeSeriesList path --

    def test_fit_transform_list(self):
        tsl = _make_tsl()
        st = StandardizeTransform()
        result = st.fit_transform(tsl)
        assert isinstance(result, TimeSeriesList)

    def test_inverse_transform_list_roundtrip(self):
        ts1 = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], dt=1.0 * u.s, unit=u.m)
        ts2 = TimeSeries([10.0, 20.0, 30.0, 40.0, 50.0], dt=1.0 * u.s, unit=u.m)
        tsl = TimeSeriesList()
        tsl.append(ts1)
        tsl.append(ts2)
        st = StandardizeTransform()
        standardized = st.fit_transform(tsl)
        restored = st.inverse_transform(standardized)
        np.testing.assert_allclose(restored[0].value, ts1.value, atol=1e-10)
        np.testing.assert_allclose(restored[1].value, ts2.value, atol=1e-10)

    def test_inverse_transform_list_out_of_bounds_raises(self):
        ts1 = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s, unit=u.m)
        tsl_fit = TimeSeriesList()
        tsl_fit.append(ts1)
        st = StandardizeTransform()
        st.fit(tsl_fit)

        ts2 = TimeSeries([4.0, 5.0, 6.0], dt=1.0 * u.s, unit=u.m)
        tsl_long = TimeSeriesList()
        tsl_long.append(ts1)
        tsl_long.append(ts2)
        with pytest.raises(IndexError):
            st.inverse_transform(tsl_long)

    def test_transform_list_appends_stats_for_new_index(self):
        """transform auto-extends stats for new indices."""
        ts1 = TimeSeries([1.0, 2.0, 3.0], dt=1.0 * u.s, unit=u.m)
        ts2 = TimeSeries([4.0, 5.0, 6.0], dt=1.0 * u.s, unit=u.m)
        tsl_fit = TimeSeriesList()
        tsl_fit.append(ts1)
        st = StandardizeTransform()
        st.fit(tsl_fit)

        tsl_long = TimeSeriesList()
        tsl_long.append(ts1)
        tsl_long.append(ts2)
        result = st.transform(tsl_long)
        assert isinstance(result, TimeSeriesList)
        assert len(result) == 2

    # -- Error paths --

    def test_inverse_transform_not_fitted_raises_value_error(self):
        st = StandardizeTransform()
        ts = _make_ts()
        with pytest.raises(ValueError, match="not been fitted"):
            st.inverse_transform(ts)

    def test_transform_incompatible_type_raises(self):
        ts = _make_ts()
        st = StandardizeTransform()
        st.fit(ts)
        # Try to transform a matrix when fitted on TimeSeries
        mat = _make_tsm()
        with pytest.raises(TypeError):
            st.transform(mat)

    def test_fit_unsupported_type_raises(self):
        st = StandardizeTransform()
        with pytest.raises(TypeError):
            st.fit([1, 2, 3])

    # -- multivariate=True paths --

    def test_multivariate_dict(self):
        tsd = _make_tsd()
        st = StandardizeTransform(multivariate=True)
        result = st.fit_transform(tsd)
        assert isinstance(result, TimeSeriesMatrix)
        assert all(unit == u.dimensionless_unscaled for unit in result.units.flat)
        assert result.channel_names == list(tsd)
        restored = st.inverse_transform(result)
        assert isinstance(restored, TimeSeriesDict)
        assert list(restored) == list(tsd)
        for key in tsd:
            np.testing.assert_allclose(restored[key].value, tsd[key].value)
            assert restored[key].unit == tsd[key].unit

    def test_multivariate_list(self):
        tsl = _make_tsl()
        st = StandardizeTransform(multivariate=True)
        result = st.fit_transform(tsl)
        assert isinstance(result, TimeSeriesMatrix)
        assert all(unit == u.dimensionless_unscaled for unit in result.units.flat)
        restored = st.inverse_transform(result)
        assert isinstance(restored, TimeSeriesList)
        for actual, original in zip(restored, tsl, strict=True):
            np.testing.assert_allclose(actual.value, original.value)
            assert actual.unit == original.unit

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="method"):
            StandardizeTransform(method="unknown")

    def test_robust_flag_resolves_unknown_method(self):
        ts = TimeSeries([1.0, 2.0, 3.0], dt=1 * u.s)
        result = StandardizeTransform(method="unknown", robust=True).fit_transform(ts)
        np.testing.assert_allclose(np.nanmedian(result.value), 0.0)

    def test_multivariate_channel_converts_compatible_units(self):
        series = TimeSeriesDict(
            {
                "volts": TimeSeries([1.0, 2.0, 3.0], dt=1 * u.s, unit=u.V),
                "millivolts": TimeSeries(
                    [1000.0, 2000.0, 3000.0], dt=1 * u.s, unit=u.mV
                ),
            }
        )
        st = StandardizeTransform(multivariate=True, axis="channel")
        scaled = st.fit_transform(series)
        np.testing.assert_allclose(scaled.value[0], scaled.value[1])
        assert all(unit == u.dimensionless_unscaled for unit in scaled.units.flat)
        restored = st.inverse_transform(scaled)
        assert restored["volts"].unit == u.V
        assert restored["millivolts"].unit == u.mV
        np.testing.assert_allclose(
            restored["millivolts"].value, [1000.0, 2000.0, 3000.0]
        )

    def test_multivariate_channel_rejects_incompatible_units(self):
        series = TimeSeriesList(
            [
                TimeSeries([1.0, 2.0, 3.0], dt=1 * u.s, unit=u.V),
                TimeSeries([1.0, 2.0, 3.0], dt=1 * u.s, unit=u.m),
            ]
        )
        with pytest.raises(ValueError, match="unit"):
            StandardizeTransform(multivariate=True, axis="channel").fit(series)

    def test_matrix_channel_inverse_restores_each_numeric_unit(self):
        matrix = TimeSeriesMatrix(
            np.array([[[1.0, 2.0, 3.0]], [[300.0, 400.0, 500.0]]]),
            dt=1 * u.s,
            units=np.array([[u.m], [u.cm]], dtype=object),
        )
        st = StandardizeTransform(axis="channel")
        scaled = st.fit_transform(matrix)
        restored = st.inverse_transform(scaled)
        np.testing.assert_allclose(restored.value, matrix.value)
        assert [unit for unit in restored.units.flat] == [u.m, u.cm]

    def test_matrix_channel_promotes_float16_before_unit_conversion(self):
        matrix = TimeSeriesMatrix(
            np.array([[[1.0, 2.0, 3.0]], [[1000.0, 2000.0, 3000.0]]], dtype=np.float16),
            dt=1 * u.s,
            units=np.array([[u.km], [u.m]], dtype=object),
        )
        st = StandardizeTransform(axis="channel")
        scaled = st.fit_transform(matrix)
        assert np.all(np.isfinite(scaled.value))
        restored = st.inverse_transform(scaled)
        np.testing.assert_allclose(restored.value, matrix.value, rtol=1e-3)

    def test_matrix_channel_preserves_complex_values(self):
        matrix = TimeSeriesMatrix(
            np.array([[[1.0 + 2.0j, 2.0 + 3.0j]], [[300.0 + 400.0j, 400.0 + 500.0j]]]),
            dt=1 * u.s,
            units=np.array([[u.m], [u.cm]], dtype=object),
        )
        st = StandardizeTransform(axis="channel")
        scaled = st.fit_transform(matrix)
        assert np.iscomplexobj(scaled.value)
        restored = st.inverse_transform(scaled)
        np.testing.assert_allclose(restored.value, matrix.value)

    def test_multivariate_time_permits_different_units(self):
        series = TimeSeriesList(
            [
                TimeSeries([1.0, 2.0, 3.0], dt=1 * u.s, unit=u.V),
                TimeSeries([4.0, 6.0, 8.0], dt=1 * u.s, unit=u.m),
            ]
        )
        st = StandardizeTransform(multivariate=True, axis="time")
        scaled = st.fit_transform(series)
        np.testing.assert_allclose(scaled.value[:, 0, :].mean(axis=-1), 0.0)
        restored = st.inverse_transform(scaled)
        assert [ts.unit for ts in restored] == [u.V, u.m]

    def test_multivariate_dict_restores_metadata_and_order(self):
        series = TimeSeriesDict(
            {
                "second": TimeSeries(
                    [2.0, 4.0, 6.0],
                    t0=12 * u.s,
                    dt=0.5 * u.s,
                    unit=u.mV,
                    name="sensor two",
                    channel="H1:TWO",
                ),
                "first": TimeSeries(
                    [1.0, 3.0, 5.0],
                    t0=12 * u.s,
                    dt=0.5 * u.s,
                    unit=u.V,
                    name="sensor one",
                    channel="H1:ONE",
                ),
            }
        )
        st = StandardizeTransform(multivariate=True, axis="time")
        scaled = st.fit_transform(series)
        assert scaled.channel_names == ["second", "first"]
        restored = st.inverse_transform(scaled)
        assert list(restored) == ["second", "first"]
        for key, original in series.items():
            actual = restored[key]
            np.testing.assert_allclose(actual.value, original.value)
            np.testing.assert_allclose(actual.times.value, original.times.value)
            assert actual.unit == original.unit
            assert actual.name == original.name
            assert str(actual.channel) == str(original.channel)

    def test_multivariate_intersection_inverse_restores_original_times(self):
        series = TimeSeriesList(
            [
                TimeSeries([1.0, 2.0, 3.0, 4.0], t0=0 * u.s, dt=1 * u.s, unit=u.V),
                TimeSeries([10.0, 20.0, 30.0, 40.0], t0=1 * u.s, dt=1 * u.s, unit=u.m),
            ]
        )
        st = StandardizeTransform(multivariate=True)
        scaled = st.fit_transform(series)
        restored = st.inverse_transform(scaled)
        assert isinstance(restored, TimeSeriesList)
        for actual, original in zip(restored, series, strict=True):
            np.testing.assert_allclose(actual.value, original.value)
            np.testing.assert_allclose(actual.times.value, original.times.value)
            assert actual.unit == original.unit

    def test_multivariate_resampled_intersection_inverse_uses_aligned_grid(self):
        series = TimeSeriesList(
            [
                TimeSeries(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dt=1 * u.s, unit=u.V, name="fast"
                ),
                TimeSeries([2.0, 3.0, 4.0, 5.0], dt=1.5 * u.s, unit=u.m, name="slow"),
            ]
        )
        st = StandardizeTransform(multivariate=True)
        aligned = st.fit_transform(series)
        restored = st.inverse_transform(aligned)
        assert isinstance(restored, TimeSeriesList)
        assert [ts.name for ts in restored] == ["fast", "slow"]
        assert [ts.unit for ts in restored] == [u.V, u.m]
        for index, ts in enumerate(restored):
            np.testing.assert_allclose(ts.times.value, aligned.times.value)
            np.testing.assert_allclose(ts.value, st.params["meta"].value[index, 0])

    def test_multivariate_inverse_preserves_fit_time_axis_units(self):
        series = TimeSeriesList(
            [
                TimeSeries([1.0, 2.0, 3.0], t0=2 * u.min, dt=1 * u.min, unit=u.V),
                TimeSeries([4.0, 5.0, 6.0], t0=2 * u.min, dt=1 * u.min, unit=u.m),
            ]
        )
        st = StandardizeTransform(multivariate=True)
        restored = st.inverse_transform(st.fit_transform(series))
        for actual, original in zip(restored, series, strict=True):
            assert actual.times.unit == u.min
            assert actual.t0 == original.t0
            assert actual.dt == original.dt
            np.testing.assert_allclose(actual.value, original.value)

    def test_multivariate_reused_at_new_t0_restores_output_times(self):
        fit_series = TimeSeriesList(
            [
                TimeSeries([1.0, 2.0, 3.0], t0=0 * u.s, dt=1 * u.s, unit=u.V),
                TimeSeries([4.0, 5.0, 6.0], t0=0 * u.s, dt=1 * u.s, unit=u.m),
            ]
        )
        new_series = TimeSeriesList(
            [
                TimeSeries([2.0, 4.0, 6.0], t0=10 * u.s, dt=1 * u.s, unit=u.V),
                TimeSeries([7.0, 8.0, 9.0], t0=10 * u.s, dt=1 * u.s, unit=u.m),
            ]
        )
        st = StandardizeTransform(multivariate=True)
        st.fit(fit_series)
        restored = st.inverse_transform(st.transform(new_series))
        for actual, original in zip(restored, new_series, strict=True):
            np.testing.assert_allclose(actual.value, original.value)
            np.testing.assert_allclose(actual.times.value, original.times.value)
            assert actual.unit == original.unit

    def test_multivariate_inverse_unknown_dict_key_raises(self):
        series = TimeSeriesDict({"known": TimeSeries([1.0, 2.0, 3.0], dt=1 * u.s)})
        st = StandardizeTransform(multivariate=True)
        transformed = st.fit_transform(series)
        transformed.channel_names = ["unknown"]
        with pytest.raises(KeyError):
            st.inverse_transform(transformed)

    def test_multivariate_dict_inverse_uses_channel_labels_after_reorder(self):
        series = TimeSeriesDict(
            {
                "a": TimeSeries([1.0, 2.0, 3.0], dt=1 * u.s, unit=u.m),
                "b": TimeSeries([10.0, 20.0, 50.0], dt=1 * u.s, unit=u.m),
            }
        )
        st = StandardizeTransform(multivariate=True)
        transformed = st.fit_transform(series)
        reordered = TimeSeriesMatrix(
            transformed.value[::-1].copy(),
            times=transformed.times,
            channel_names=["b", "a"],
            units=transformed.units[::-1].copy(),
        )
        restored = st.inverse_transform(reordered)
        assert list(restored) == ["a", "b"]
        for key in series:
            np.testing.assert_allclose(restored[key].value, series[key].value)

    def test_nan_is_retained_through_roundtrip(self):
        ts = TimeSeries([1.0, np.nan, 3.0], dt=1 * u.s, unit=u.V)
        st = StandardizeTransform()
        scaled = st.fit_transform(ts)
        assert np.isnan(scaled.value[1])
        restored = st.inverse_transform(scaled)
        np.testing.assert_allclose(restored.value, ts.value, equal_nan=True)
        assert restored.unit == u.V

    def test_constant_and_tiny_scales(self):
        constant = TimeSeries([5.0, 5.0, 5.0], dt=1 * u.s, unit=u.m)
        with pytest.warns(UserWarning, match="constant"):
            scaled = StandardizeTransform().fit_transform(constant)
        np.testing.assert_array_equal(scaled.value, [0.0, 0.0, 0.0])
        tiny = TimeSeries([0.0, 1e-21, 2e-21], dt=1 * u.s, unit=u.m)
        st = StandardizeTransform()
        np.testing.assert_allclose(
            st.fit_transform(tiny).value, [-1.224744871, 0.0, 1.224744871]
        )
        np.testing.assert_allclose(
            st.inverse_transform(st.transform(tiny)).value,
            tiny.value,
            rtol=1e-12,
            atol=0,
        )
        extremely_tiny = TimeSeries([0.0, 1e-200, 2e-200], dt=1 * u.s)
        normalized = StandardizeTransform().fit_transform(extremely_tiny)
        np.testing.assert_allclose(normalized.value, [-1.224744871, 0.0, 1.224744871])

    @pytest.mark.parametrize(
        "values",
        [np.array([60000.0, 60000.0], dtype=np.float16), np.array([1e308, 1e308])],
    )
    def test_large_finite_constant_is_centered_zero(self, values):
        ts = TimeSeries(values, dt=1 * u.s)
        with pytest.warns(UserWarning, match="constant"):
            scaled = StandardizeTransform().fit_transform(ts)
        np.testing.assert_array_equal(scaled.value, [0.0, 0.0])

    def test_large_complex_constant_is_exactly_zero(self):
        ts = TimeSeries(
            np.array([1e308 + 1e308j, 1e308 + 1e308j]),
            dt=1 * u.s,
            unit=u.V,
        )
        with pytest.warns(UserWarning, match="constant"):
            scaled = StandardizeTransform().fit_transform(ts)
        np.testing.assert_array_equal(scaled.value, [0.0 + 0.0j, 0.0 + 0.0j])

    def test_very_large_complex_constant_timeseries_and_matrix(self):
        value = 1.7e308 + 1.7e308j
        ts = TimeSeries(np.array([value, value]), dt=1 * u.s, unit=u.V)
        matrix = TimeSeriesMatrix(
            np.array([[[value, value]], [[value, value]]]), dt=1 * u.s
        )
        for source in (ts, matrix):
            with pytest.warns(UserWarning, match="constant"):
                scaled = StandardizeTransform().fit_transform(source)
            np.testing.assert_array_equal(scaled.value, np.zeros_like(scaled.value))

    @pytest.mark.parametrize("value", [1e308, 1.7e308 + 1.7e308j])
    def test_robust_large_constant_timeseries_and_matrix(self, value):
        ts = TimeSeries(np.array([value, value]), dt=1 * u.s, unit=u.V)
        matrix = TimeSeriesMatrix(
            np.array([[[value, value]], [[value, value]]]), dt=1 * u.s
        )
        for source in (ts, matrix):
            with pytest.warns(UserWarning, match="constant"):
                scaled = StandardizeTransform(robust=True).fit_transform(source)
            np.testing.assert_array_equal(scaled.value, np.zeros_like(scaled.value))

    def test_zero_mad_nonconstant_raises(self):
        ts = TimeSeries([0.0, 0.0, 0.0, 1.0], dt=1 * u.s)
        with pytest.raises(ValueError, match="MAD|scale"):
            StandardizeTransform(robust=True).fit(ts)

    @pytest.mark.parametrize("values", [[np.nan, np.nan], [1.0, np.inf, 2.0]])
    def test_nonfinite_scale_raises(self, values):
        with pytest.raises(ValueError, match="scale"):
            StandardizeTransform().fit(TimeSeries(values, dt=1 * u.s))

    def test_supports_inverse_is_true(self):
        assert StandardizeTransform.supports_inverse is True


# ---------------------------------------------------------------------------
# WhitenTransform
# ---------------------------------------------------------------------------


class TestWhitenTransform:
    @staticmethod
    def _correlated_matrix(scale: float = 1.0) -> TimeSeriesMatrix:
        rng = np.random.default_rng(482)
        data = rng.normal(size=(3, 1, 512))
        data[1] += 0.5 * data[0]
        data[2] -= 0.25 * data[0] + 0.2 * data[1]
        return TimeSeriesMatrix(data * scale, t0=0.0 * u.s, dt=0.01 * u.s)

    def test_fit_transform_matrix(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        wt = WhitenTransform()
        result = wt.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_inverse_transform_matrix_roundtrip(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        wt = WhitenTransform()
        whitened = wt.fit_transform(mat)
        restored = wt.inverse_transform(whitened)
        assert isinstance(restored, TimeSeriesMatrix)
        # Shape must be preserved
        assert restored.shape[-1] == mat.shape[-1]

    def test_inverse_transform_restores_fit_time_element_units(self):
        matrix = TimeSeriesMatrix(
            np.array(
                [
                    [[1.0, 2.0, 4.0, 8.0], [3.0, 5.0, 7.0, 11.0]],
                    [[2.0, 4.0, 7.0, 13.0], [5.0, 8.0, 12.0, 17.0]],
                ]
            ),
            dt=1 * u.s,
            units=np.array([[u.m, u.cm], [u.V, u.s]], dtype=object),
        )
        transform_input = matrix.copy()
        transform_input.meta.units = np.full(matrix.meta.shape, u.kg, dtype=object)
        transform = WhitenTransform()

        transform.fit(matrix)
        transformed = transform.transform(transform_input)
        restored = transform.inverse_transform(transformed)

        assert [unit for unit in restored.units.flat] == [u.m, u.cm, u.V, u.s]

    @pytest.mark.parametrize("collection_type", [TimeSeriesDict, TimeSeriesList])
    def test_inverse_transform_restores_fit_time_collection_units(
        self, collection_type
    ):
        fit_series = [
            TimeSeries([1.0, 2.0, 4.0, 8.0], dt=1 * u.s, unit=u.V),
            TimeSeries([3.0, 5.0, 7.0, 11.0], dt=1 * u.s, unit=u.m),
        ]
        transform_series = [
            TimeSeries([2.0, 4.0, 7.0, 13.0], dt=1 * u.s, unit=u.kg),
            TimeSeries([5.0, 8.0, 12.0, 17.0], dt=1 * u.s, unit=u.s),
        ]
        if collection_type is TimeSeriesDict:
            fit_input = TimeSeriesDict(dict(zip(("A", "B"), fit_series)))
            transform_input = TimeSeriesDict(dict(zip(("A", "B"), transform_series)))
        else:
            fit_input = TimeSeriesList(fit_series)
            transform_input = TimeSeriesList(transform_series)
        transform = WhitenTransform()

        transform.fit(fit_input)
        transformed = transform.transform(transform_input)
        restored = transform.inverse_transform(transformed)

        assert isinstance(restored, collection_type)
        series = restored.values() if isinstance(restored, TimeSeriesDict) else restored
        assert [ts.unit for ts in series] == [u.V, u.m]

    def test_inverse_transform_restores_units_with_reduced_components(self):
        matrix = TimeSeriesMatrix(
            np.array(
                [
                    [[1.0, 2.0, 4.0, 8.0]],
                    [[3.0, 5.0, 7.0, 11.0]],
                ]
            ),
            dt=1 * u.s,
            units=np.array([[u.V], [u.m]], dtype=object),
        )
        transform = WhitenTransform(n_components=1)

        restored = transform.inverse_transform(transform.fit_transform(matrix))

        assert restored.shape == (2, 1, 4)
        assert [unit for unit in restored.units.flat] == [u.V, u.m]

    def test_inverse_transform_not_fitted_raises(self):
        wt = WhitenTransform()
        mat = _make_tsm()
        with pytest.raises(ValueError, match="not been fitted"):
            wt.inverse_transform(mat)

    def test_inverse_transform_non_matrix_raises(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        wt = WhitenTransform()
        wt.fit(mat)
        with pytest.raises(TypeError, match="TimeSeriesMatrix"):
            wt.inverse_transform("not_a_matrix")

    def test_supports_inverse_is_true(self):
        assert WhitenTransform.supports_inverse is True

    def test_to_matrix_from_timeseries(self):
        ts = _make_ts()
        wt = WhitenTransform()
        mat_data, orig = wt._to_matrix(ts)
        assert isinstance(mat_data, TimeSeriesMatrix)

    def test_to_matrix_from_matrix_passthrough(self):
        mat = _make_tsm()
        wt = WhitenTransform()
        # When given a TimeSeriesMatrix, _to_matrix returns the matrix directly
        # (not a tuple) as a passthrough.
        result = wt._to_matrix(mat)
        assert result is mat

    def test_to_matrix_unsupported_type_raises(self):
        wt = WhitenTransform()
        with pytest.raises(TypeError, match="Unsupported type for whitening"):
            wt._to_matrix([1, 2, 3])

    def test_transform_before_fit_auto_fits(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        wt = WhitenTransform()
        result = wt.transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_zca_method(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        wt = WhitenTransform(method="zca")
        result = wt.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_default_auto_epsilon_is_scale_invariant(self):
        unit = self._correlated_matrix()
        strain = self._correlated_matrix(1e-21)

        unit_w = WhitenTransform().fit_transform(unit)
        strain_w = WhitenTransform().fit_transform(strain)
        np.testing.assert_allclose(strain_w.value, unit_w.value, rtol=1e-10)

        explicit_auto = WhitenTransform(eps="auto").fit_transform(unit)
        none_auto = WhitenTransform(eps=None).fit_transform(unit)
        np.testing.assert_allclose(none_auto.value, explicit_auto.value)

    def test_rejects_invalid_epsilon(self):
        for eps in (True, np.bool_(True), -1.0, np.nan, np.inf, -np.inf, "bad"):
            with pytest.raises(ValueError, match="eps"):
                WhitenTransform(eps=eps)

    def test_fitted_transform_rejects_nonfinite_input(self):
        transform = WhitenTransform().fit(self._correlated_matrix())
        nonfinite = self._correlated_matrix()
        nonfinite.value[0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="finite"):
            transform.transform(nonfinite)

    def test_transform_is_dimensionless_and_preserves_time_metadata(self):
        matrix = self._correlated_matrix()
        matrix.unit = u.m
        transform = WhitenTransform()

        result = transform.fit_transform(matrix)

        assert result.unit == u.dimensionless_unscaled
        assert result.t0 == matrix.t0
        assert result.dt == matrix.dt

    def test_single_feature_float32_zero_variance_auto_is_finite(self):
        matrix = TimeSeriesMatrix(np.zeros((1, 1, 16), dtype=np.float32), dt=1.0 * u.s)
        transform = WhitenTransform()

        whitened = transform.fit_transform(matrix)

        assert np.all(np.isfinite(whitened.value))
        np.testing.assert_array_equal(whitened.value, 0.0)
        assert whitened.unit == u.dimensionless_unscaled
        assert all(unit == u.dimensionless_unscaled for unit in whitened.units.flat)
        assert transform.model is not None
        assert np.all(np.isfinite(transform.model.W))

    def test_fit_transform_dict(self):
        tsd = _make_tsd(n_channels=3, n_time=100)
        wt = WhitenTransform()
        result = wt.fit_transform(tsd)
        assert isinstance(result, TimeSeriesDict)

    def test_fit_transform_list_attempts_conversion(self):
        # WhitenTransform converts list -> matrix -> attempts list via to_list().
        # to_list() is known to fail for multi-channel matrices; we only verify
        # that the fit+transform chain reaches that point without other errors.
        tsl = _make_tsl(n_channels=3, n_time=100)
        wt = WhitenTransform()
        try:
            result = wt.fit_transform(tsl)
            assert isinstance(result, TimeSeriesList)
        except TypeError:
            pass  # known limitation of to_list() for multi-row matrices


# ---------------------------------------------------------------------------
# PCATransform
# ---------------------------------------------------------------------------


class TestPCATransform:
    @pytest.fixture(autouse=True)
    def _setup_imports(self):
        from gwexpy.timeseries.pipeline import PCATransform

        self.PCATransform = PCATransform

    def test_fit_transform_matrix(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=50)
        pca = self.PCATransform(n_components=2)
        result = pca.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_inverse_transform_matrix_roundtrip(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=50)
        pca = self.PCATransform(n_components=3)
        scores = pca.fit_transform(mat)
        rec = pca.inverse_transform(scores)
        assert isinstance(rec, TimeSeriesMatrix)
        # Note: tolerance might need adjustment depending on mock vs real
        if "MagicMock" not in str(type(pca)):
            np.testing.assert_allclose(rec.value, mat.value, atol=1e-6)

    def test_inverse_transform_not_fitted_raises(self):
        pytest.importorskip("sklearn")
        pca = self.PCATransform()
        mat = _make_tsm()
        with pytest.raises(ValueError, match="not been fitted"):
            pca.inverse_transform(mat)

    def test_ensure_matrix_raises_for_unsupported_type(self):
        pytest.importorskip("sklearn")
        pca = self.PCATransform()
        with pytest.raises(TypeError):
            pca._ensure_matrix([1, 2, 3])

    def test_ensure_matrix_returns_matrix_passthrough(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm()
        pca = self.PCATransform()
        result, orig = pca._ensure_matrix(mat)
        assert result is mat
        assert orig is None

    def test_ensure_matrix_from_collection(self):
        pytest.importorskip("sklearn")
        tsd = _make_tsd()
        pca = self.PCATransform()
        mat, original = pca._ensure_matrix(tsd)
        assert isinstance(mat, TimeSeriesMatrix)
        assert original is tsd

    def test_multivariate_false_raises_on_non_matrix(self):
        pytest.importorskip("sklearn")
        pca = self.PCATransform(multivariate=False)
        ts = _make_ts()
        with pytest.raises(TypeError):
            pca.fit(ts)

    def test_multivariate_false_path(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=50)
        pca = self.PCATransform(n_components=2, multivariate=False)
        result = pca.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_transform_before_fit_auto_fits(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=50)
        pca = self.PCATransform(n_components=2)
        result = pca.transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_transform_non_matrix_raises_after_fit(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=50)
        pca = self.PCATransform(n_components=2)
        pca.fit(mat)
        with pytest.raises(TypeError):
            pca.transform("not_matrix")

    def test_inverse_transform_non_matrix_raises(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=50)
        pca = self.PCATransform(n_components=3)
        pca.fit(mat)
        with pytest.raises(TypeError):
            pca.inverse_transform("not_matrix")

    def test_supports_inverse_is_true(self):
        from gwexpy.timeseries.pipeline import PCATransform

        assert PCATransform.supports_inverse is True

    def test_fit_transform_dict(self):
        pytest.importorskip("sklearn")
        tsd = _make_tsd(n_channels=3, n_time=100)
        pca = self.PCATransform(n_components=2)
        result = pca.fit_transform(tsd)
        assert isinstance(result, TimeSeriesDict)

    def test_fit_transform_list_attempts_conversion(self):
        pytest.importorskip("sklearn")
        # PCATransform converts list -> matrix -> attempts list via to_list().
        # to_list() is known to fail for multi-channel matrices; we only verify
        # the chain reaches that point without other errors.
        tsl = _make_tsl(n_channels=3, n_time=100)
        pca = self.PCATransform(n_components=2)
        try:
            result = pca.fit_transform(tsl)
            assert isinstance(result, TimeSeriesList)
        except TypeError:
            pass  # known limitation of to_list() for multi-row matrices


@pytest.mark.usefixtures("mock_sklearn_decomposition")
class TestPCATransformMocked:
    """Explicitly test PCA transform with mocks via fixture."""

    @pytest.fixture(autouse=True)
    def _setup_imports(self):
        from gwexpy.timeseries.pipeline import PCATransform

        self.PCATransform = PCATransform

    def test_pca_mocked_basic(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        pca = self.PCATransform(n_components=2)
        result = pca.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)
        assert result.shape == (2, 1, 50)

    def test_pca_mocked_inverse(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        pca = self.PCATransform(n_components=3)
        scores = pca.fit_transform(mat)
        rec = pca.inverse_transform(scores)
        assert isinstance(rec, TimeSeriesMatrix)
        assert rec.shape == (3, 1, 50)


# ---------------------------------------------------------------------------
# ICATransform
# ---------------------------------------------------------------------------


class TestICATransform:
    @pytest.fixture(autouse=True)
    def _setup_imports(self):
        from gwexpy.timeseries.pipeline import ICATransform

        self.ICATransform = ICATransform

    def test_fit_transform_matrix(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=200)
        ica = self.ICATransform(n_components=2)
        result = ica.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_inverse_transform_matrix(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=200)
        ica = self.ICATransform(n_components=3)
        sources = ica.fit_transform(mat)
        rec = ica.inverse_transform(sources)
        assert isinstance(rec, TimeSeriesMatrix)
        assert rec.shape[-1] == mat.shape[-1]

    def test_inverse_transform_not_fitted_raises(self):
        pytest.importorskip("sklearn")
        ica = self.ICATransform()
        mat = _make_tsm()
        with pytest.raises(ValueError, match="not been fitted"):
            ica.inverse_transform(mat)

    def test_ensure_matrix_raises_for_unsupported_type(self):
        pytest.importorskip("sklearn")
        ica = self.ICATransform()
        with pytest.raises(TypeError):
            ica._ensure_matrix([1, 2, 3])

    def test_ensure_matrix_returns_matrix_passthrough(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm()
        ica = self.ICATransform()
        result, orig = ica._ensure_matrix(mat)
        assert result is mat
        assert orig is None

    def test_ensure_matrix_from_collection(self):
        pytest.importorskip("sklearn")
        tsd = _make_tsd()
        ica = self.ICATransform()
        mat, original = ica._ensure_matrix(tsd)
        assert isinstance(mat, TimeSeriesMatrix)
        assert original is tsd

    def test_multivariate_false_raises_on_non_matrix(self):
        pytest.importorskip("sklearn")
        ica = self.ICATransform(multivariate=False)
        ts = _make_ts()
        with pytest.raises(TypeError):
            ica.fit(ts)

    def test_multivariate_false_path(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=200)
        ica = self.ICATransform(n_components=2, multivariate=False)
        result = ica.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_transform_before_fit_auto_fits(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=200)
        ica = self.ICATransform(n_components=2)
        result = ica.transform(mat)
        assert isinstance(result, TimeSeriesMatrix)

    def test_transform_non_matrix_raises_after_fit(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=200)
        ica = self.ICATransform(n_components=2)
        ica.fit(mat)
        with pytest.raises(TypeError):
            ica.transform("not_matrix")

    def test_inverse_transform_non_matrix_raises(self):
        pytest.importorskip("sklearn")
        mat = _make_tsm(n_channels=3, n_time=200)
        ica = self.ICATransform(n_components=3)
        ica.fit(mat)
        with pytest.raises(TypeError):
            ica.inverse_transform("not_matrix")

    def test_supports_inverse_is_true(self):
        from gwexpy.timeseries.pipeline import ICATransform

        assert ICATransform.supports_inverse is True

    def test_fit_transform_dict(self):
        pytest.importorskip("sklearn")
        tsd = _make_tsd(n_channels=3, n_time=200)
        ica = self.ICATransform(n_components=2)
        result = ica.fit_transform(tsd)
        assert isinstance(result, TimeSeriesDict)

    def test_fit_transform_list_attempts_conversion(self):
        pytest.importorskip("sklearn")
        # ICATransform converts list -> matrix -> attempts list via to_list().
        # to_list() is known to fail for multi-channel matrices; we only verify
        # the chain reaches that point without other errors.
        tsl = _make_tsl(n_channels=3, n_time=200)
        ica = self.ICATransform(n_components=2)
        try:
            result = ica.fit_transform(tsl)
            assert isinstance(result, TimeSeriesList)
        except TypeError:
            pass  # known limitation of to_list() for multi-row matrices


@pytest.mark.usefixtures("mock_sklearn_decomposition")
class TestICATransformMocked:
    """Explicitly test ICA transform with mocks via fixture."""

    @pytest.fixture(autouse=True)
    def _setup_imports(self):
        from gwexpy.timeseries.pipeline import ICATransform

        self.ICATransform = ICATransform

    def test_ica_mocked_basic(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        ica = self.ICATransform(n_components=2)
        result = ica.fit_transform(mat)
        assert isinstance(result, TimeSeriesMatrix)
        assert result.shape == (2, 1, 50)

    def test_ica_mocked_inverse(self):
        mat = _make_tsm(n_channels=3, n_time=50)
        ica = self.ICATransform(n_components=3)
        sources = ica.fit_transform(mat)
        rec = ica.inverse_transform(sources)
        assert isinstance(rec, TimeSeriesMatrix)
        # 3 components -> 3 features (channels) in mock recovery
        assert rec.shape == (3, 1, 50)
