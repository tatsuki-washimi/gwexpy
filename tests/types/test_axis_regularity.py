"""Tests for #544 AxisDescriptor.is_regular, mean_delta, and AxisApiMixin helpers."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.types import axis_api
from gwexpy.types.axis import AxisDescriptor


class TestAxisRegularityDefaultInvariance:
    def test_strict_default_unchanged(self):
        exact = AxisDescriptor("x", np.arange(100, dtype=np.float64) * 0.5 * u.m)
        assert exact.regular is True
        assert exact.is_regular() is True
        assert exact.delta.unit == u.m
        assert exact.delta.to_value(u.m) == pytest.approx(0.5)

        jitter = AxisDescriptor("t", np.linspace(0.0, 1.0, 1000, endpoint=False) * u.s)
        # np.linspace has ~1e-13 relative diff (> strict 2.5e-14), so default stays strict False
        assert jitter.regular is False
        assert jitter.is_regular() is False

    def test_descending_and_short_axes(self):
        desc = AxisDescriptor("y", np.array([10.0, 8.0, 6.0, 4.0]) * u.m)
        assert desc.regular is True
        assert desc.delta.to_value(u.m) == pytest.approx(-2.0)
        assert desc.mean_delta().to_value(u.m) == pytest.approx(-2.0)

        short = AxisDescriptor("z", np.array([1.0]) * u.m)
        assert short.regular is True
        assert short.is_regular(max_deviation=1e-3) is True
        assert short.delta is None
        assert short.mean_delta() is None


class TestAxisIsRegularTolerances:
    def test_rtol_and_atol_ulps(self):
        coords = np.linspace(0.0, 1.0, 1000, endpoint=False) * u.s
        ax = AxisDescriptor("t", coords)
        assert ax.is_regular(rtol=1e-9) is True
        assert ax.is_regular(rtol=1e-14) is False

        # Genuine step jump fails even generous small rtol
        nonuniform = AxisDescriptor("t", np.array([0.0, 0.1, 0.3, 0.6]) * u.s)
        assert nonuniform.is_regular(rtol=1e-4) is False

        # atol_ulps covers tiny ULP differences
        arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        arr[2] = np.nextafter(arr[2], np.inf)
        arr[3] = np.nextafter(arr[3], np.inf)
        ax_ulp = AxisDescriptor("x", arr * u.m)
        assert ax_ulp.is_regular(atol_ulps=4) is True

    @pytest.mark.parametrize(
        "bad_kw", [{"rtol": -1e-6}, {"rtol": np.nan}, {"max_deviation": -0.1}]
    )
    def test_invalid_tolerance_rejected(self, bad_kw):
        ax = AxisDescriptor("x", np.arange(5, dtype=float) * u.m)
        with pytest.raises(ValueError):
            ax.is_regular(**bad_kw)

    def test_bool_tolerance_rejected(self):
        ax = AxisDescriptor("x", np.arange(5, dtype=float) * u.m)
        with pytest.raises(TypeError):
            ax.is_regular(max_deviation=True)  # type: ignore[arg-type]


class TestAxisLeastSquaresMeanDelta:
    def test_max_deviation_gps_and_large_n_refinement(self):
        # GPS offset + small step in linspace-like float64
        n = 100_000
        dt = 0.001
        coords = (1_400_000_000.0 + np.linspace(0.0, n * dt, n, endpoint=False)) * u.s
        ax = AxisDescriptor("t", coords)
        assert ax.regular is False
        assert ax.is_regular(max_deviation=1e-6) is True
        md = ax.mean_delta().to_value(u.s)
        assert abs(md - dt) / dt < 1e-11

    def test_floor_rejection_on_loss_of_precision(self):
        # float32 with huge offset makes 1 ULP >= 0.25 * dx -> rejected unless strictly regular
        coords = np.array([1e8 + i for i in range(8)], dtype=np.float32) * u.s
        ax = AxisDescriptor("t", coords)
        if not ax.regular:
            assert ax.is_regular(max_deviation=1e-3) is False

    def test_log_axis_rejected(self):
        ax = AxisDescriptor("f", np.geomspace(1.0, 1000.0, 32) * u.Hz)
        assert ax.regular is False
        assert ax.is_regular(max_deviation=1e-2) is False
        assert ax.is_regular(rtol=1e-2) is False

    def test_scale_invariance(self):
        base = np.linspace(0.0, 10.0, 128)
        ax1 = AxisDescriptor("x", base * u.m)
        ax2 = AxisDescriptor("x", (base * 1e-20) * u.m)
        ax3 = AxisDescriptor("x", (base * 1e20) * u.m)
        for tol in (1e-12, 1e-8):
            assert ax1.is_regular(max_deviation=tol) == ax2.is_regular(
                max_deviation=tol
            )
            assert ax1.is_regular(max_deviation=tol) == ax3.is_regular(
                max_deviation=tol
            )
        assert ax2.mean_delta().to_value(u.m) == pytest.approx(
            ax1.mean_delta().to_value(u.m) * 1e-20, rel=1e-12
        )

    def test_iloc_slice_rejects_descending_coord_slice(self):
        desc = AxisDescriptor("lat", np.array([36.0, 35.5, 35.0, 34.5]) * u.deg)
        with pytest.raises(ValueError, match="ascending"):
            desc.iloc_slice(slice(35.0 * u.deg, 36.0 * u.deg))
        # Single value lookup still works cleanly on descending axes
        assert desc.iloc_nearest(35.5 * u.deg) == 1


class TestAxisApiExposure:
    def test_axis_api_exports_and_field_methods(self):
        assert hasattr(axis_api, "AxisDescriptor")
        assert "AxisDescriptor" in axis_api.__all__

        t = np.linspace(0.0, 1.0, 1000, endpoint=False) * u.s
        sf = ScalarField(
            np.zeros((1000, 2, 2, 1)),
            axis0=t,
            axis1=np.array([0.0, 1.0]) * u.m,
            axis2=np.array([0.0, 2.0]) * u.m,
            axis3=np.array([0.0]) * u.m,
        )
        assert sf.is_axis_regular(0) is False
        assert sf.is_axis_regular(0, max_deviation=1e-9) is True
        assert sf.axis_mean_delta(0).to_value(u.s) == pytest.approx(
            1.0 / 1000.0, rel=1e-12
        )
        assert sf.axis_mean_delta("y").to_value(u.m) == pytest.approx(2.0)
