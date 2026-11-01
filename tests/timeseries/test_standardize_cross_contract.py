"""Cross-contract validation between standardize_matrix and StandardizeTransform (Lane X).

Verifies shared unit, axis, and metadata contracts (R-016, R-017, AC-003, AC-006, AC-016, AC-017):
- R-016: Compatible channel unit conversion and incompatible unit rejection across APIs.
- R-016: Independent channel processing along axis='time' across APIs.
- R-017: Degeneracy handling in StandardizeTransform while preserving standardize_matrix behavior.
- AC-003 / AC-006: Dimensionless output units, metadata preservation, and exact roundtrip restoration.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.timeseries.pipeline import StandardizeTransform
from gwexpy.timeseries.preprocess import standardize_matrix


def _make_compatible_matrix(*, dtype=float) -> TimeSeriesMatrix:
    """Create a 2-channel matrix with compatible units (m and cm)."""
    data = np.array([[[1.0, 4.0, 7.0]], [[200.0, 300.0, 500.0]]], dtype=dtype)
    matrix = TimeSeriesMatrix(
        data,
        units=[[u.m], [u.cm]],
        names=[["chan_m"], ["chan_cm"]],
        channels=[["A"], ["B"]],
        x0=0.0 * u.s,
        dt=1.0 * u.s,
        xunit=u.s,
        name="cross_contract_test",
        attrs={"origin": "lane_x"},
    )
    matrix.meta[0, 0]["custom_key"] = "meta_A"
    matrix.meta[1, 0]["custom_key"] = "meta_B"
    return matrix


def _make_incompatible_matrix() -> TimeSeriesMatrix:
    """Create a 2-channel matrix with incompatible units (m and s)."""
    data = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=float)
    return TimeSeriesMatrix(
        data,
        units=[[u.m], [u.s]],
        names=[["chan_m"], ["chan_s"]],
        channels=[["A"], ["B"]],
        x0=0.0 * u.s,
        dt=1.0 * u.s,
        xunit=u.s,
    )


class TestCrossContractR016:
    """R-016 cross-verification: channel conversion, incompatible rejection, time independence."""

    @pytest.mark.parametrize("method", ["zscore", "robust"])
    def test_channel_axis_compatible_units_numerical_agreement(
        self, method: str
    ) -> None:
        """Both standardize_matrix and StandardizeTransform convert compatible units identically."""
        source = _make_compatible_matrix()

        # Lane B: standardize_matrix
        res_matrix = standardize_matrix(source, axis="channel", method=method)

        # Lane C: StandardizeTransform(multivariate=True)
        st = StandardizeTransform(multivariate=True, axis="channel", method=method)
        res_transform = st.fit_transform(source)

        # 1. Numerical values must agree closely
        np.testing.assert_allclose(res_matrix.value, res_transform.value, rtol=1e-12)

        # 2. Both produce dimensionless output units
        assert all(unit == u.dimensionless_unscaled for unit in res_matrix.units.flat)
        assert all(
            unit == u.dimensionless_unscaled for unit in res_transform.units.flat
        )

        # 3. Both preserve source units unchanged
        assert list(source.units.flat) == [u.m, u.cm]

        # 4. StandardizeTransform inverse_transform restores original values and units
        restored = st.inverse_transform(res_transform)
        assert isinstance(restored, TimeSeriesMatrix)
        assert list(restored.units.flat) == [u.m, u.cm]
        np.testing.assert_allclose(restored.value, source.value, rtol=1e-12)

    def test_channel_axis_incompatible_units_both_reject(self) -> None:
        """Both standardize_matrix and StandardizeTransform raise on incompatible units."""
        source = _make_incompatible_matrix()

        # Lane B raises ValueError (wrapping UnitConversionError)
        with pytest.raises(ValueError):
            standardize_matrix(source, axis="channel")

        # Lane C raises UnitConversionError (which is a subclass of ValueError)
        st = StandardizeTransform(multivariate=True, axis="channel")
        with pytest.raises(ValueError):
            st.fit_transform(source)

    @pytest.mark.parametrize("method", ["zscore", "robust"])
    def test_time_axis_independent_unit_processing_agreement(self, method: str) -> None:
        """With axis='time', channels are processed independently regardless of differing units."""
        source = _make_incompatible_matrix()

        # Both succeed despite incompatible channel units because pooling is along time axis
        res_matrix = standardize_matrix(source, axis="time", method=method)

        st = StandardizeTransform(multivariate=True, axis="time", method=method)
        res_transform = st.fit_transform(source)

        # Numerical results agree
        np.testing.assert_allclose(res_matrix.value, res_transform.value, rtol=1e-12)

        # Both dimensionless
        assert all(unit == u.dimensionless_unscaled for unit in res_matrix.units.flat)
        assert all(
            unit == u.dimensionless_unscaled for unit in res_transform.units.flat
        )

        # Restoration recovers original incompatible units
        restored = st.inverse_transform(res_transform)
        assert list(restored.units.flat) == [u.m, u.s]
        np.testing.assert_allclose(restored.value, source.value, rtol=1e-12)


class TestCrossContractR017:
    """R-017 cross-verification: Pipeline degeneracy handling vs standardize_matrix legacy behavior."""

    def test_pipeline_constant_channel_warns_and_forces_unit_scale(self) -> None:
        """StandardizeTransform warns and standardizes constant channel to 0, forcing scale=1.0."""
        data = np.array([[[5.0, 5.0, 5.0, 5.0]], [[1.0, 2.0, 3.0, 4.0]]])
        mat = TimeSeriesMatrix(data, units=[[u.m], [u.m]], dt=1.0 * u.s)

        st = StandardizeTransform(multivariate=True, axis="time")
        with pytest.warns(
            UserWarning, match="Finite constant channel standardized to zero"
        ):
            res = st.fit_transform(mat)

        # Constant channel centered to 0.0 with scale forced to 1.0
        np.testing.assert_allclose(res.value[0, 0], [0.0, 0.0, 0.0, 0.0])

        # Inverse transform restores exact constant values
        restored = st.inverse_transform(res)
        np.testing.assert_allclose(restored.value[0, 0], [5.0, 5.0, 5.0, 5.0])

    def test_standardize_matrix_preserves_legacy_constant_behavior(self) -> None:
        """standardize_matrix does not adopt the pipeline warning or force scale=1.0 for constants."""
        import warnings

        data = np.array([[[5.0, 5.0, 5.0, 5.0]], [[1.0, 2.0, 3.0, 4.0]]])
        mat = TimeSeriesMatrix(data, units=[[u.m], [u.m]], dt=1.0 * u.s)

        # standardize_matrix handles it per legacy path without R-017 UserWarning
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _ = standardize_matrix(mat, axis="time")

        r017_warnings = [
            w
            for w in record
            if "Finite constant channel standardized to zero" in str(w.message)
        ]
        assert len(r017_warnings) == 0

    def test_nan_preservation_agreement(self) -> None:
        """Both APIs preserve NaN locations."""
        data = np.array([[[1.0, np.nan, 3.0, 4.0]], [[2.0, 4.0, np.nan, 8.0]]])
        mat = TimeSeriesMatrix(data, units=[[u.m], [u.m]], dt=1.0 * u.s)

        res_matrix = standardize_matrix(mat, axis="time")
        st = StandardizeTransform(multivariate=True, axis="time")
        res_transform = st.fit_transform(mat)

        assert np.isnan(res_matrix.value[0, 0, 1])
        assert np.isnan(res_transform.value[0, 0, 1])
        assert np.isnan(res_matrix.value[1, 0, 2])
        assert np.isnan(res_transform.value[1, 0, 2])
