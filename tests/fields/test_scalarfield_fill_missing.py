"""Tests for #549 ScalarField.fill_missing."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import GeoMetadata, ScalarField
from gwexpy.time._reference import TIME_REFERENCE_ABSOLUTE


def _make_hole_field() -> ScalarField:
    # Planar terrain z = 10*x + 2*y with isolated & distant missing cells
    x = np.arange(5, dtype=float) * 1.0 * u.m
    y = np.arange(5, dtype=float) * 1.0 * u.m
    xx, yy = np.meshgrid(x.value, y.value, indexing="ij")
    data = (10.0 * xx + 2.0 * yy)[None, :, :, None]
    data[0, 2, 2, 0] = np.nan
    data[0, 4, 4, 0] = np.nan
    sf = ScalarField(
        data,
        unit=u.m,
        epoch=1_400_000_000.0,
        axis0=np.array([0.0]) * u.s,
        axis1=x,
        axis2=y,
        axis3=np.array([0.0]) * u.m,
    )
    sf.geo = GeoMetadata(crs="EPSG:32654", active=True)
    return sf


def test_fill_missing_nearest_linear_constant():
    sf = _make_hole_field()
    orig_snapshot = sf.value.copy()

    filled_const = sf.fill_missing(method="constant", value=-1.5 * u.m)
    assert np.isnan(sf.value[0, 2, 2, 0])  # non-destructive
    np.testing.assert_array_equal(np.isnan(sf.value), np.isnan(orig_snapshot))
    assert filled_const.value[0, 2, 2, 0] == pytest.approx(-1.5)
    assert filled_const.value[0, 4, 4, 0] == pytest.approx(-1.5)
    assert filled_const.geo is not None and filled_const.geo.active is True
    assert "constant" in filled_const.geo.history[-1]
    assert filled_const.time_reference == TIME_REFERENCE_ABSOLUTE

    # Linear exact recovery of interior planar hole (2, 2) = 10*2 + 2*2 = 24
    filled_lin = sf.fill_missing(method="linear")
    assert filled_lin.value[0, 2, 2, 0] == pytest.approx(24.0, abs=1e-12)
    # Corner (4, 4) lies outside convex hull for linear interpolation
    assert np.isnan(filled_lin.value[0, 4, 4, 0])
    # Valid samples remain bit-identical
    valid = ~np.isnan(orig_snapshot)
    np.testing.assert_array_equal(filled_lin.value[valid], orig_snapshot[valid])

    filled_near = sf.fill_missing(method="nearest")
    assert not np.any(np.isnan(filled_near.value))


def test_fill_missing_max_distance_and_axes():
    x = np.arange(6, dtype=float) * u.m
    data = np.zeros((1, 6, 1, 1), dtype=float)
    data[0, 1:5, 0, 0] = np.nan
    # Valid samples at x=0 (value 0) and x=5 (value 10)
    data[0, 5, 0, 0] = 10.0
    sf = ScalarField(
        data,
        unit=u.m,
        axis0=np.array([0.0]) * u.s,
        axis1=x,
        axis2=np.array([0.0]) * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    filled = sf.fill_missing(method="linear", axes=("x",), max_distance=1.5 * u.m)
    # x=1 is distance 1 from x=0 -> filled (2.0); x=2,3 are dist 2 -> remain NaN; x=4 is dist 1 -> filled (8.0)
    assert filled.value[0, 1, 0, 0] == pytest.approx(2.0)
    assert np.isnan(filled.value[0, 2, 0, 0])
    assert np.isnan(filled.value[0, 3, 0, 0])
    assert filled.value[0, 4, 0, 0] == pytest.approx(8.0)


def test_fill_missing_errors_and_no_nan():
    all_nan = ScalarField(np.full((1, 3, 3, 1), np.nan), unit=u.m)
    with pytest.raises(ValueError, match="all field values are NaN"):
        all_nan.fill_missing()

    clean = ScalarField(np.ones((1, 3, 3, 1)), unit=u.m)
    with pytest.raises(ValueError, match="requires 'value='"):
        clean.fill_missing(method="constant")
    with pytest.raises(ValueError, match="only valid for method='constant'"):
        clean.fill_missing(method="nearest", value=0.0)
    with pytest.raises(ValueError, match="Invalid fill_missing method"):
        clean.fill_missing(method="cubic")  # type: ignore[arg-type]

    no_nan_res = clean.fill_missing(method="nearest")
    np.testing.assert_array_equal(no_nan_res.value, clean.value)
