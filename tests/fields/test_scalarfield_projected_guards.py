"""Guards for projection operations that run before optional CRS imports."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import GeoMetadata, ScalarField
from gwexpy.fields.scalar import _resample_2d_grid


def test_to_projected_rejects_complex_values_before_resampling() -> None:
    field = ScalarField(
        np.full((1, 3, 3, 1), 2.0 + 3.0j),
        unit=u.m,
        axis0=[0.0] * u.s,
        axis1=np.arange(3) * u.m,
        axis2=np.arange(3) * u.m,
        axis3=[0.0] * u.m,
    )
    field.geo = GeoMetadata(crs="EPSG:3857", active=True)

    with pytest.raises(TypeError, match="complex field values"):
        field.to_projected("EPSG:3857", method="nearest")

    np.testing.assert_array_equal(field.value, np.full((1, 3, 3, 1), 2.0 + 3.0j))


@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
def test_resampling_keeps_outer_half_cell_footprint(method: str) -> None:
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    values = np.full((4, 4), 7.0)
    target_x = np.array([[-0.25, 3.25, -0.75]])
    target_y = np.array([[1.5, 1.5, 1.5]])

    result = _resample_2d_grid(x, y, values, target_x, target_y, method)

    np.testing.assert_allclose(result[0, :2], 7.0)
    assert np.isnan(result[0, 2])


def test_field_arithmetic_rejects_crs_mismatch_with_identical_axes() -> None:
    first = ScalarField(np.ones((1, 3, 3, 1)), unit=u.m)
    first.geo = GeoMetadata(crs="EPSG:3857", active=True)
    second = first.copy()
    second.geo = GeoMetadata(crs="EPSG:32654", active=True)

    with pytest.raises(ValueError, match="CRS mismatch"):
        _ = first + second
    with pytest.raises(ValueError, match="CRS mismatch"):
        _ = second + first

    second.geo = None
    with pytest.raises(ValueError, match="CRS mismatch"):
        _ = first + second

    second.geo = GeoMetadata(crs="epsg:3857", active=True)
    combined = first + second
    assert combined.geo is not None
    assert combined.geo.crs == "EPSG:3857"
