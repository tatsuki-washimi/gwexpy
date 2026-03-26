"""Tests for MetPy interop.

Uses mock xarray DataArrays with _metpy_axis attributes. Does NOT require
MetPy to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from gwexpy.fields import ScalarField
from gwexpy.interop.metpy_ import from_metpy_dataarray


NT, NX, NY, NZ = 3, 4, 5, 6


def _make_metpy_da(
    nt=NT, nx=NX, ny=NY, nz=NZ, units="kelvin",
) -> xr.DataArray:
    """Simulate a MetPy-dequantified DataArray with _metpy_axis attrs."""
    data = np.random.default_rng(42).random((nt, nx, ny, nz))
    da = xr.DataArray(
        data,
        dims=("time", "x", "y", "z"),
        coords={
            "time": np.arange(nt, dtype=float),
            "x": np.linspace(0, 100, nx),
            "y": np.linspace(0, 200, ny),
            "z": np.linspace(0, 10, nz),
        },
        attrs={"units": units},
    )
    da["time"].attrs["_metpy_axis"] = "time"
    da["x"].attrs["_metpy_axis"] = "x"
    da["y"].attrs["_metpy_axis"] = "y"
    da["z"].attrs["_metpy_axis"] = "vertical"
    return da


def _make_2d_metpy_da(nx=NX, ny=NY) -> xr.DataArray:
    data = np.random.default_rng(0).random((nx, ny))
    da = xr.DataArray(
        data,
        dims=("x", "y"),
        coords={
            "x": np.linspace(0, 100, nx),
            "y": np.linspace(0, 200, ny),
        },
        attrs={"units": "m/s"},
    )
    da["x"].attrs["_metpy_axis"] = "x"
    da["y"].attrs["_metpy_axis"] = "y"
    return da


class TestFromMetpyDataarray:
    def test_returns_scalarfield(self):
        da = _make_metpy_da()
        sf = from_metpy_dataarray(ScalarField, da, dequantify=False)
        assert isinstance(sf, ScalarField)

    def test_shape_4d(self):
        da = _make_metpy_da()
        sf = from_metpy_dataarray(ScalarField, da, dequantify=False)
        assert sf.shape == (NT, NX, NY, NZ)

    def test_unit_preserved(self):
        da = _make_metpy_da(units="m/s")
        sf = from_metpy_dataarray(ScalarField, da, dequantify=False)
        assert sf.unit is not None
        assert "m" in str(sf.unit)

    def test_axis0_domain(self):
        da = _make_metpy_da()
        sf = from_metpy_dataarray(ScalarField, da, dequantify=False, axis0_domain="time")
        assert sf.axis0_domain == "time"

    def test_2d_padded(self):
        da = _make_2d_metpy_da()
        sf = from_metpy_dataarray(ScalarField, da, dequantify=False)
        assert sf.ndim == 4
        assert 1 in sf.shape  # at least one singleton

    def test_pint_unit_conversion(self):
        """If units look like Pint, they are converted to astropy."""
        da = _make_metpy_da(units="meter / second")
        sf = from_metpy_dataarray(ScalarField, da, dequantify=False)
        assert sf.unit is not None

    def test_spatial_coords(self):
        da = _make_metpy_da()
        sf = from_metpy_dataarray(ScalarField, da, dequantify=False)
        ax1 = np.asarray(
            sf._axis1_index.value
            if hasattr(sf._axis1_index, "value")
            else sf._axis1_index
        )
        np.testing.assert_allclose(ax1, np.linspace(0, 100, NX))
