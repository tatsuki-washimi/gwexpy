"""Tests for ScalarField slicing behavior - always maintains 4D."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField


class TestScalarFieldSlicingMaintains4D:
    """Test that ScalarField __getitem__ always returns 4D."""

    @pytest.fixture
    def field(self):
        """Create a test ScalarField."""
        data = np.arange(10 * 8 * 6 * 4).reshape((10, 8, 6, 4))
        t = np.arange(10) * 0.1 * u.s
        x = np.arange(8) * 1.0 * u.m
        y = np.arange(6) * 1.0 * u.m
        z = np.arange(4) * 1.0 * u.m
        return ScalarField(
            data,
            unit=u.V,
            axis0=t,
            axis1=x,
            axis2=y,
            axis3=z,
            axis_names=["t", "x", "y", "z"],
        )

    def test_int_index_axis0_maintains_4d(self, field):
        """Test field[0, :, :, :] returns ScalarField with axis0 length=1."""
        sliced = field[0, :, :, :]

        assert isinstance(sliced, ScalarField)
        assert sliced.ndim == 4
        assert sliced.shape == (1, 8, 6, 4)
        assert sliced.shape[0] == 1

    def test_int_index_axis3_maintains_4d(self, field):
        """Test field[:, :, :, 0] returns ScalarField with axis3 length=1."""
        sliced = field[:, :, :, 0]

        assert isinstance(sliced, ScalarField)
        assert sliced.ndim == 4
        assert sliced.shape == (10, 8, 6, 1)
        assert sliced.shape[3] == 1

    def test_int_index_middle_axis_maintains_4d(self, field):
        """Test int index on middle axis maintains 4D."""
        sliced = field[:, 3, :, :]

        assert isinstance(sliced, ScalarField)
        assert sliced.ndim == 4
        assert sliced.shape == (10, 1, 6, 4)

    def test_multiple_int_indices_maintain_4d(self, field):
        """Test multiple int indices still maintain 4D."""
        sliced = field[0, 1, 2, 3]

        assert isinstance(sliced, ScalarField)
        assert sliced.ndim == 4
        assert sliced.shape == (1, 1, 1, 1)

    def test_mixed_int_slice_maintains_4d(self, field):
        """Test mixed int and slice indexing maintains 4D."""
        sliced = field[2, :, 1:4, 0]

        assert isinstance(sliced, ScalarField)
        assert sliced.ndim == 4
        assert sliced.shape == (1, 8, 3, 1)

    def test_slice_index_preserved_as_quantity(self, field):
        """Test that sliced axis index is preserved as Quantity."""
        sliced = field[5, :, :, :]

        assert isinstance(sliced, ScalarField)
        # axis0 index should be the single value at index 5
        assert sliced._axis0_index.shape == (1,)
        np.testing.assert_allclose(sliced._axis0_index.value, [0.5])
        assert sliced._axis0_index.unit == u.s

    def test_negative_int_index(self, field):
        """Test negative int index works correctly."""
        sliced = field[-1, :, :, :]

        assert isinstance(sliced, ScalarField)
        assert sliced.shape == (1, 8, 6, 4)
        # Should be last time value
        np.testing.assert_allclose(sliced._axis0_index.value, [0.9])

    def test_ellipsis_with_int(self, field):
        """Test ellipsis combined with int index."""
        sliced = field[..., 2]

        assert isinstance(sliced, ScalarField)
        assert sliced.shape == (10, 8, 6, 1)

    def test_unit_preserved(self, field):
        """Test that unit is preserved after slicing."""
        sliced = field[0, 1, 2, 3]

        assert sliced.unit == u.V


class TestScalarFieldSlicingDomain:
    """Test that domain is preserved after slicing."""

    def test_axis0_domain_preserved_time(self):
        """Test axis0_domain='time' preserved after slicing."""
        field = ScalarField(np.zeros((10, 4, 4, 4)), axis0_domain="time")
        sliced = field[0, :, :, :]

        assert sliced.axis0_domain == "time"

    def test_axis0_domain_preserved_frequency(self):
        """Test axis0_domain='frequency' preserved after slicing."""
        field = ScalarField(np.zeros((10, 4, 4, 4)), axis0_domain="frequency")
        sliced = field[0, :, :, :]

        assert sliced.axis0_domain == "frequency"

    def test_space_domains_preserved(self):
        """Test space_domains preserved after slicing."""
        field = ScalarField(
            np.zeros((10, 4, 4, 4)),
            axis_names=["t", "x", "y", "z"],
            space_domain={"x": "real", "y": "k", "z": "real"},
        )
        sliced = field[:, 0, :, :]

        assert sliced.space_domains["x"] == "real"
        assert sliced.space_domains["y"] == "k"
        assert sliced.space_domains["z"] == "real"


class TestScalarFieldSlicingAxisNames:
    """Test that axis names are preserved after slicing."""

    def test_axis_names_preserved(self):
        """Test axis names are preserved after slicing."""
        field = ScalarField(
            np.zeros((10, 4, 4, 4)), axis_names=["t", "x", "y", "z"]
        )
        sliced = field[0, 1, 2, 3]

        assert sliced.axis_names == ("t", "x", "y", "z")

    def test_custom_axis_names_preserved(self):
        """Test custom axis names are preserved."""
        field = ScalarField(
            np.zeros((5, 3, 3, 3)), axis_names=["time", "lon", "lat", "alt"]
        )
        sliced = field[2, :, :, :]

        assert sliced.axis_names == ("time", "lon", "lat", "alt")


class TestScalarFieldIselMaintains4D:
    """Test isel with ScalarField maintains 4D."""

    @pytest.fixture
    def field(self):
        """Create a test ScalarField."""
        return ScalarField(
            np.arange(10 * 8 * 6 * 4).reshape((10, 8, 6, 4)),
            axis_names=["t", "x", "y", "z"],
        )

    def test_isel_int_maintains_4d(self, field):
        """Test isel with int index maintains 4D."""
        sliced = field.isel(t=0)

        assert isinstance(sliced, ScalarField)
        assert sliced.ndim == 4
        assert sliced.shape == (1, 8, 6, 4)

    def test_isel_slice_maintains_4d(self, field):
        """Test isel with slice maintains 4D."""
        sliced = field.isel(x=slice(1, 5))

        assert isinstance(sliced, ScalarField)
        assert sliced.shape == (10, 4, 6, 4)

    def test_isel_multiple_axes(self, field):
        """Test isel on multiple axes."""
        sliced = field.isel(t=2, z=1)

        assert isinstance(sliced, ScalarField)
        assert sliced.shape == (1, 8, 6, 1)


class TestScalarFieldIndexErrors:
    """Test proper error handling for indexing."""

    def test_index_out_of_bounds(self):
        """Test index out of bounds raises IndexError."""
        field = ScalarField(np.zeros((5, 4, 3, 2)))

        with pytest.raises(IndexError):
            _ = field[10, :, :, :]

    def test_too_many_indices(self):
        """Test too many indices raises IndexError."""
        field = ScalarField(np.zeros((5, 4, 3, 2)))

        with pytest.raises(IndexError):
            _ = field[0, 0, 0, 0, 0]
