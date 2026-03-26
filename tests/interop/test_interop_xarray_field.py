"""Tests for xarray ↔ ScalarField / VectorField bridge.

These tests do NOT require MetPy or wrf-python; only xarray is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from gwexpy.fields import ScalarField, VectorField
from gwexpy.interop.xarray_ import from_xarray_field, to_xarray_field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NT, NX, NY, NZ = 3, 4, 5, 6


def _make_4d_da(nt=NT, nx=NX, ny=NY, nz=NZ, units="m/s") -> xr.DataArray:
    data = np.random.default_rng(0).random((nt, nx, ny, nz))
    return xr.DataArray(
        data,
        dims=("time", "x", "y", "z"),
        coords={
            "time": np.linspace(0, 1, nt),
            "x": np.linspace(0, 1, nx),
            "y": np.linspace(0, 2, ny),
            "z": np.linspace(0, 3, nz),
        },
        attrs={"units": units},
    )


def _make_2d_da(nx=NX, ny=NY) -> xr.DataArray:
    data = np.random.default_rng(1).random((nx, ny))
    return xr.DataArray(
        data,
        dims=("easting", "northing"),
        coords={
            "easting": np.linspace(0, 1, nx),
            "northing": np.linspace(0, 2, ny),
        },
        attrs={"units": "mGal"},
    )


def _make_3d_time_lat_lon(nt=NT, nlat=NY, nlon=NX) -> xr.DataArray:
    data = np.random.default_rng(2).random((nt, nlat, nlon))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(nt, dtype=float),
            "lat": np.linspace(-10, 10, nlat),
            "lon": np.linspace(100, 120, nlon),
        },
        attrs={"units": "K"},
    )


def _make_cf_convention_da(nt=NT, nx=NX) -> xr.DataArray:
    """DataArray with CF Convention axis attributes."""
    data = np.random.default_rng(3).random((nt, nx))
    da = xr.DataArray(
        data,
        dims=("tau", "xi"),
        coords={
            "tau": np.linspace(0, 1, nt),
            "xi": np.linspace(0, 1, nx),
        },
    )
    da["tau"].attrs["axis"] = "T"
    da["xi"].attrs["axis"] = "X"
    return da


def _make_metpy_da(nt=NT, nx=NX) -> xr.DataArray:
    """DataArray with MetPy _metpy_axis attributes."""
    data = np.random.default_rng(4).random((nt, nx))
    da = xr.DataArray(
        data,
        dims=("myT", "myX"),
        coords={
            "myT": np.linspace(0, 1, nt),
            "myX": np.linspace(0, 1, nx),
        },
    )
    da["myT"].attrs["_metpy_axis"] = "time"
    da["myX"].attrs["_metpy_axis"] = "x"
    return da


# ---------------------------------------------------------------------------
# from_xarray_field — 4D DataArray
# ---------------------------------------------------------------------------


class TestFromXarrayField4D:
    def test_returns_scalarfield(self):
        da = _make_4d_da()
        sf = from_xarray_field(ScalarField, da)
        assert isinstance(sf, ScalarField)

    def test_shape(self):
        da = _make_4d_da()
        sf = from_xarray_field(ScalarField, da)
        assert sf.shape == (NT, NX, NY, NZ)

    def test_axis0_domain(self):
        da = _make_4d_da()
        sf = from_xarray_field(ScalarField, da, axis0_domain="time")
        assert sf.axis0_domain == "time"

    def test_unit_preserved(self):
        da = _make_4d_da(units="m/s")
        sf = from_xarray_field(ScalarField, da)
        assert sf.unit is not None
        assert "m" in str(sf.unit)

    def test_spatial_coords_preserved(self):
        da = _make_4d_da()
        sf = from_xarray_field(ScalarField, da)
        ax1 = np.asarray(sf._axis1_index.value if hasattr(sf._axis1_index, "value") else sf._axis1_index)
        np.testing.assert_allclose(ax1, da.coords["x"].values)

    def test_roundtrip(self):
        da = _make_4d_da()
        sf = from_xarray_field(ScalarField, da)
        da2 = to_xarray_field(sf)
        np.testing.assert_allclose(sf.value, da2.values)


# ---------------------------------------------------------------------------
# from_xarray_field — 2D DataArray (easting × northing)
# ---------------------------------------------------------------------------


class TestFromXarrayField2D:
    def test_shape_padded_to_4d(self):
        da = _make_2d_da()
        sf = from_xarray_field(ScalarField, da)
        # Should be (1, NX, NY, 1)
        assert sf.shape[1] == NX
        assert sf.shape[2] == NY
        assert sf.shape[0] == 1 or sf.shape[3] == 1  # at least one singleton

    def test_unit_mGal(self):
        da = _make_2d_da()
        sf = from_xarray_field(ScalarField, da)
        # mGal is known to astropy
        assert sf.unit is not None


# ---------------------------------------------------------------------------
# from_xarray_field — 3D time × lat × lon
# ---------------------------------------------------------------------------


class TestFromXarrayField3D:
    def test_shape(self):
        da = _make_3d_time_lat_lon()
        sf = from_xarray_field(ScalarField, da)
        # Should be (NT, NY, NX, 1) or similar with singleton z
        assert sf.shape[0] == NT
        assert 1 in sf.shape  # at least one singleton

    def test_4d(self):
        da = _make_3d_time_lat_lon()
        sf = from_xarray_field(ScalarField, da)
        assert sf.ndim == 4


# ---------------------------------------------------------------------------
# CF Convention axis attribute detection
# ---------------------------------------------------------------------------


class TestCFConventionDetection:
    def test_cf_axis_t_detected(self):
        da = _make_cf_convention_da()
        sf = from_xarray_field(ScalarField, da)
        assert sf.shape[0] == NT

    def test_cf_axis_x_detected(self):
        da = _make_cf_convention_da()
        sf = from_xarray_field(ScalarField, da)
        ax1 = np.asarray(sf._axis1_index.value if hasattr(sf._axis1_index, "value") else sf._axis1_index)
        np.testing.assert_allclose(ax1, da.coords["xi"].values)


# ---------------------------------------------------------------------------
# MetPy _metpy_axis detection
# ---------------------------------------------------------------------------


class TestMetpyAxisDetection:
    def test_metpy_time_detected(self):
        da = _make_metpy_da()
        sf = from_xarray_field(ScalarField, da)
        assert sf.shape[0] == NT

    def test_metpy_x_detected(self):
        da = _make_metpy_da()
        sf = from_xarray_field(ScalarField, da)
        ax1 = np.asarray(sf._axis1_index.value if hasattr(sf._axis1_index, "value") else sf._axis1_index)
        np.testing.assert_allclose(ax1, da.coords["myX"].values)


# ---------------------------------------------------------------------------
# Dataset → VectorField
# ---------------------------------------------------------------------------


class TestFromXarrayDataset:
    def test_dataset_to_vectorfield(self):
        data = np.random.default_rng(5).random((NT, NX, NY, NZ))
        ds = xr.Dataset(
            {
                "u": xr.DataArray(data, dims=("time", "x", "y", "z")),
                "v": xr.DataArray(data * 0.5, dims=("time", "x", "y", "z")),
                "w": xr.DataArray(data * 0.25, dims=("time", "x", "y", "z")),
            },
            coords={
                "time": np.arange(NT, dtype=float),
                "x": np.linspace(0, 1, NX),
                "y": np.linspace(0, 2, NY),
                "z": np.linspace(0, 3, NZ),
            },
        )
        vf = from_xarray_field(VectorField, ds)
        assert isinstance(vf, VectorField)
        assert set(vf.keys()) == {"u", "v", "w"}

    def test_dataset_components_shape(self):
        data = np.ones((NT, NX, NY, NZ))
        ds = xr.Dataset(
            {"u": xr.DataArray(data, dims=("time", "x", "y", "z"))},
            coords={"time": np.arange(NT, dtype=float)},
        )
        vf = from_xarray_field(VectorField, ds)
        assert vf["u"].shape[0] == NT


# ---------------------------------------------------------------------------
# to_xarray_field
# ---------------------------------------------------------------------------


class TestToXarrayField:
    def test_scalarfield_to_datarray(self):
        da = _make_4d_da()
        sf = from_xarray_field(ScalarField, da)
        da2 = to_xarray_field(sf)
        assert isinstance(da2, xr.DataArray)
        assert da2.ndim == 4

    def test_unit_in_attrs(self):
        da = _make_4d_da(units="m/s")
        sf = from_xarray_field(ScalarField, da)
        da2 = to_xarray_field(sf)
        assert "units" in da2.attrs

    def test_custom_dim_names(self):
        da = _make_4d_da()
        sf = from_xarray_field(ScalarField, da)
        da2 = to_xarray_field(sf, dim_names=("freq", "x1", "x2", "x3"))
        assert da2.dims[0] == "freq"
        assert da2.dims[1] == "x1"

    def test_vectorfield_to_dataset(self):
        data = np.ones((NT, NX, NY, NZ))
        ds = xr.Dataset(
            {
                "u": xr.DataArray(data, dims=("time", "x", "y", "z")),
                "v": xr.DataArray(data, dims=("time", "x", "y", "z")),
            },
        )
        vf = from_xarray_field(VectorField, ds)
        ds2 = to_xarray_field(vf)
        assert isinstance(ds2, xr.Dataset)
        assert set(ds2.data_vars) == {"u", "v"}

    def test_roundtrip_vectorfield(self):
        data = np.random.default_rng(9).random((NT, NX, NY, NZ))
        ds = xr.Dataset(
            {
                "u": xr.DataArray(data, dims=("time", "x", "y", "z")),
                "v": xr.DataArray(data * 2, dims=("time", "x", "y", "z")),
            },
        )
        vf = from_xarray_field(VectorField, ds)
        ds2 = to_xarray_field(vf)
        np.testing.assert_allclose(ds2["u"].values, vf["u"].value)


# ---------------------------------------------------------------------------
# Unit parsing edge cases
# ---------------------------------------------------------------------------


class TestUnitParsing:
    def test_unknown_unit_warns(self):
        da = xr.DataArray(np.ones((3, 4)), dims=("time", "x"), attrs={"units": "gizmo_unit_xyz"})
        with pytest.warns(UserWarning):
            sf = from_xarray_field(ScalarField, da)
        # unit=None passed to ScalarField → defaults to dimensionless
        from astropy import units as u
        assert sf.unit is None or sf.unit == u.dimensionless_unscaled

    def test_no_units_attr(self):
        da = xr.DataArray(np.ones((3, 4)), dims=("time", "x"))
        sf = from_xarray_field(ScalarField, da)
        # ScalarField defaults to dimensionless when unit=None
        from astropy import units as u
        assert sf.unit is None or sf.unit == u.dimensionless_unscaled
