"""Tests for Harmonica interop.

Uses mock xarray DataArrays/Datasets with Harmonica-style coordinates.
Does NOT require Harmonica to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from gwexpy.fields import ScalarField, VectorField
from gwexpy.interop.harmonica_ import _guess_harmonica_unit, from_harmonica_grid

NE, NN = 10, 12  # easting, northing


def _make_gravity_grid(ne=NE, nn=NN) -> xr.DataArray:
    """2D Bouguer gravity grid with easting/northing coords."""
    data = np.random.default_rng(0).random((ne, nn))
    return xr.DataArray(
        data,
        dims=("easting", "northing"),
        coords={
            "easting": np.linspace(0, 1000, ne),
            "northing": np.linspace(0, 2000, nn),
        },
        name="gravity",
    )


def _make_3d_grid(ne=NE, nn=NN, nz=5) -> xr.DataArray:
    """3D grid with easting/northing/upward."""
    data = np.random.default_rng(1).random((ne, nn, nz))
    return xr.DataArray(
        data,
        dims=("easting", "northing", "upward"),
        coords={
            "easting": np.linspace(0, 1000, ne),
            "northing": np.linspace(0, 2000, nn),
            "upward": np.linspace(0, 500, nz),
        },
        name="magnetic",
    )


def _make_dataset(ne=NE, nn=NN) -> xr.Dataset:
    """Dataset with multiple grid variables."""
    data1 = np.random.default_rng(2).random((ne, nn))
    data2 = np.random.default_rng(3).random((ne, nn))
    return xr.Dataset(
        {
            "gravity_disturbance": xr.DataArray(
                data1,
                dims=("easting", "northing"),
                coords={
                    "easting": np.linspace(0, 1000, ne),
                    "northing": np.linspace(0, 2000, nn),
                },
            ),
            "topography": xr.DataArray(
                data2,
                dims=("easting", "northing"),
                coords={
                    "easting": np.linspace(0, 1000, ne),
                    "northing": np.linspace(0, 2000, nn),
                },
            ),
        }
    )


# ---------------------------------------------------------------------------
# _guess_harmonica_unit
# ---------------------------------------------------------------------------


class TestGuessUnit:
    def test_gravity(self):
        assert _guess_harmonica_unit("gravity") == "mGal"

    def test_magnetic(self):
        assert _guess_harmonica_unit("magnetic_anomaly") == "nT"

    def test_topography(self):
        assert _guess_harmonica_unit("topography") == "m"

    def test_unknown(self):
        assert _guess_harmonica_unit("something") == ""


# ---------------------------------------------------------------------------
# from_harmonica_grid — DataArray
# ---------------------------------------------------------------------------


class TestFromHarmonicaDataArray:
    def test_returns_scalarfield(self):
        da = _make_gravity_grid()
        sf = from_harmonica_grid(ScalarField, da)
        assert isinstance(sf, ScalarField)

    def test_4d_shape(self):
        da = _make_gravity_grid()
        sf = from_harmonica_grid(ScalarField, da)
        assert sf.ndim == 4
        # (1, NE, NN, 1) — singleton axis0 and z
        assert sf.shape[0] == 1
        assert sf.shape[1] == NE
        assert sf.shape[2] == NN

    def test_3d_grid(self):
        da = _make_3d_grid()
        sf = from_harmonica_grid(ScalarField, da)
        assert sf.ndim == 4
        assert sf.shape[0] == 1  # no time dim → singleton

    def test_unit_guessed(self):
        da = _make_gravity_grid()
        sf = from_harmonica_grid(ScalarField, da)
        assert sf.unit is not None

    def test_spatial_coords(self):
        da = _make_gravity_grid()
        sf = from_harmonica_grid(ScalarField, da)
        ax1 = np.asarray(
            sf._axis1_index.value
            if hasattr(sf._axis1_index, "value")
            else sf._axis1_index
        )
        np.testing.assert_allclose(ax1, np.linspace(0, 1000, NE))


# ---------------------------------------------------------------------------
# from_harmonica_grid — Dataset
# ---------------------------------------------------------------------------


class TestFromHarmonicaDataset:
    def test_dataset_to_vectorfield(self):
        ds = _make_dataset()
        vf = from_harmonica_grid(VectorField, ds)
        assert isinstance(vf, VectorField)
        assert set(vf.keys()) == {"gravity_disturbance", "topography"}

    def test_dataset_with_data_name(self):
        ds = _make_dataset()
        sf = from_harmonica_grid(ScalarField, ds, data_name="topography")
        assert isinstance(sf, ScalarField)

    def test_invalid_data_name_raises(self):
        ds = _make_dataset()
        with pytest.raises(ValueError, match="not found"):
            from_harmonica_grid(ScalarField, ds, data_name="missing")
