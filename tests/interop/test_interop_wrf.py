"""Tests for wrf-python interop.

Uses mock xarray DataArrays with WRF dimension names.
Does NOT require wrf-python to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from gwexpy.fields import ScalarField
from gwexpy.interop.wrf_ import _extract_1d_from_2d, from_wrf_variable

NT, NZ, NY, NX = 2, 3, 5, 6


def _make_wrf_da(
    nt=NT, nz=NZ, ny=NY, nx=NX,
) -> xr.DataArray:
    """Simulate a wrf.getvar() output with WRF dimension names."""
    data = np.random.default_rng(7).random((nt, nz, ny, nx))

    # Regular grid: XLAT varies along south_north only, XLONG along west_east
    lat_vals = np.linspace(30, 35, ny)
    lon_vals = np.linspace(130, 135, nx)
    XLAT = np.tile(lat_vals[:, np.newaxis], (1, nx))  # (ny, nx)
    XLONG = np.tile(lon_vals[np.newaxis, :], (ny, 1))  # (ny, nx)

    da = xr.DataArray(
        data,
        dims=("Time", "bottom_top", "south_north", "west_east"),
        coords={
            "Time": np.arange(nt, dtype=float),
            "bottom_top": np.arange(nz, dtype=float),
            "south_north": np.arange(ny, dtype=float),
            "west_east": np.arange(nx, dtype=float),
            "XLAT": (("south_north", "west_east"), XLAT),
            "XLONG": (("south_north", "west_east"), XLONG),
        },
        attrs={"units": "K", "description": "Temperature"},
    )
    return da


def _make_wrf_2d_da(ny=NY, nx=NX) -> xr.DataArray:
    """2D WRF variable (e.g. surface pressure)."""
    data = np.random.default_rng(3).random((ny, nx))
    return xr.DataArray(
        data,
        dims=("south_north", "west_east"),
        coords={
            "south_north": np.arange(ny, dtype=float),
            "west_east": np.arange(nx, dtype=float),
        },
        attrs={"units": "Pa"},
    )


# ---------------------------------------------------------------------------
# _extract_1d_from_2d
# ---------------------------------------------------------------------------


class TestExtract1DFrom2D:
    def test_regular_lat(self):
        ny, nx = 5, 6
        lat_vals = np.linspace(30, 35, ny)
        XLAT = np.tile(lat_vals[:, np.newaxis], (1, nx))
        da = xr.DataArray(
            np.zeros((ny, nx)),
            dims=("south_north", "west_east"),
            coords={"XLAT": (("south_north", "west_east"), XLAT)},
        )
        result = _extract_1d_from_2d(da, "XLAT", axis=1)
        assert result is not None
        assert result.shape == (ny,)

    def test_regular_lon(self):
        ny, nx = 5, 6
        lon_vals = np.linspace(130, 135, nx)
        XLONG = np.tile(lon_vals[np.newaxis, :], (ny, 1))
        da = xr.DataArray(
            np.zeros((ny, nx)),
            dims=("south_north", "west_east"),
            coords={"XLONG": (("south_north", "west_east"), XLONG)},
        )
        result = _extract_1d_from_2d(da, "XLONG", axis=0)
        assert result is not None
        assert result.shape == (nx,)

    def test_missing_coord_returns_none(self):
        da = xr.DataArray(np.zeros((3, 4)), dims=("a", "b"))
        assert _extract_1d_from_2d(da, "XLAT", axis=0) is None


# ---------------------------------------------------------------------------
# from_wrf_variable
# ---------------------------------------------------------------------------


class TestFromWrfVariable:
    def test_returns_scalarfield(self):
        da = _make_wrf_da()
        sf = from_wrf_variable(ScalarField, da)
        assert isinstance(sf, ScalarField)

    def test_4d_shape(self):
        da = _make_wrf_da()
        sf = from_wrf_variable(ScalarField, da)
        assert sf.ndim == 4
        assert sf.shape[0] == NT

    def test_axis0_domain(self):
        da = _make_wrf_da()
        sf = from_wrf_variable(ScalarField, da)
        assert sf.axis0_domain == "time"

    def test_unit_preserved(self):
        da = _make_wrf_da()
        sf = from_wrf_variable(ScalarField, da)
        assert sf.unit is not None

    def test_2d_variable(self):
        da = _make_wrf_2d_da()
        sf = from_wrf_variable(ScalarField, da)
        assert sf.ndim == 4
        assert 1 in sf.shape  # singleton dims
