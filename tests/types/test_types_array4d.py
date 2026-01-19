"""Tests for Array4D basic functionality."""

import numpy as np
import pytest
from astropy import units as u
from astropy.units import dimensionless_unscaled

from gwexpy.types import Array4D


class TestArray4DConstruction:
    """Test Array4D construction and validation."""

    def test_basic_construction(self):
        """Test basic 4D array construction."""
        data = np.random.randn(10, 8, 6, 4)
        arr = Array4D(data)
        assert arr.shape == (10, 8, 6, 4)
        assert arr.ndim == 4

    def test_construction_with_unit(self):
        """Test construction with unit."""
        data = np.ones((2, 3, 4, 5))
        arr = Array4D(data, unit=u.m)
        assert arr.unit == u.m

    def test_construction_non_4d_raises(self):
        """Test that non-4D data raises ValueError."""
        with pytest.raises(ValueError, match="must be 4-dimensional"):
            Array4D(np.zeros((3, 3, 3)))

        with pytest.raises(ValueError, match="must be 4-dimensional"):
            Array4D(np.zeros((3, 3, 3, 3, 3)))

    def test_construction_with_axis_indices(self):
        """Test construction with explicit axis indices."""
        data = np.zeros((4, 5, 6, 7))
        t = np.arange(4) * u.s
        x = np.arange(5) * u.m
        y = np.arange(6) * u.m
        z = np.arange(7) * u.m

        arr = Array4D(data, axis0=t, axis1=x, axis2=y, axis3=z)

        np.testing.assert_array_equal(arr._axis0_index.value, t.value)
        assert arr._axis0_index.unit == u.s
        np.testing.assert_array_equal(arr._axis1_index.value, x.value)

    def test_construction_with_axis_names(self):
        """Test construction with explicit axis names."""
        data = np.zeros((2, 3, 4, 5))
        arr = Array4D(data, axis_names=["t", "x", "y", "z"])

        assert arr.axis_names == ("t", "x", "y", "z")

    def test_construction_axis_names_wrong_length_raises(self):
        """Test that wrong length axis_names raises."""
        data = np.zeros((2, 3, 4, 5))
        with pytest.raises(ValueError, match="axis_names must be length 4"):
            Array4D(data, axis_names=["a", "b", "c"])

    def test_default_axis_names(self):
        """Test default axis names."""
        arr = Array4D(np.zeros((2, 3, 4, 5)))
        assert arr.axis_names == ("axis0", "axis1", "axis2", "axis3")

    def test_default_axis_indices(self):
        """Test that default axis indices are created."""
        arr = Array4D(np.zeros((3, 4, 5, 6)))
        assert len(arr._axis0_index) == 3
        assert arr._axis0_index.unit == dimensionless_unscaled


class TestArray4DAxes:
    """Test axes property and axis manipulation."""

    def test_axes_property(self):
        """Test axes property returns AxisDescriptor tuple."""
        arr = Array4D(
            np.zeros((2, 3, 4, 5)),
            axis_names=["t", "x", "y", "z"],
            axis0=np.arange(2) * u.s,
        )
        axes = arr.axes

        assert len(axes) == 4
        assert axes[0].name == "t"
        assert axes[0].index.unit == u.s
        assert axes[1].name == "x"

    def test_axis_by_name(self):
        """Test axis() method by name."""
        arr = Array4D(np.zeros((2, 3, 4, 5)), axis_names=["t", "x", "y", "z"])
        ax = arr.axis("y")
        assert ax.name == "y"

    def test_axis_by_index(self):
        """Test axis() method by index."""
        arr = Array4D(np.zeros((2, 3, 4, 5)), axis_names=["t", "x", "y", "z"])
        ax = arr.axis(2)
        assert ax.name == "y"


class TestArray4DGetitem:
    """Test Array4D __getitem__ behavior."""

    def test_slice_maintains_4d(self):
        """Test that pure slicing maintains Array4D."""
        arr = Array4D(np.arange(120).reshape((2, 3, 4, 5)))
        sliced = arr[:, 1:3, :, 2:4]

        assert isinstance(sliced, Array4D)
        assert sliced.shape == (2, 2, 4, 2)

    def test_int_index_drops_dimension(self):
        """Test that int index drops dimension (returns plain array)."""
        arr = Array4D(np.arange(120).reshape((2, 3, 4, 5)))
        sliced = arr[0, :, :, :]

        # Array4D allows dimension dropping, results in non-Array4D
        assert sliced.ndim == 3

    def test_axis_indices_sliced_correctly(self):
        """Test that axis indices are sliced correctly."""
        t = np.arange(10) * u.s
        x = np.arange(8) * u.m
        y = np.arange(6) * u.m
        z = np.arange(4) * u.m

        arr = Array4D(
            np.zeros((10, 8, 6, 4)),
            axis0=t,
            axis1=x,
            axis2=y,
            axis3=z,
        )
        sliced = arr[2:5, :, 1:4, :]

        assert isinstance(sliced, Array4D)
        np.testing.assert_array_equal(sliced._axis0_index.value, [2, 3, 4])
        np.testing.assert_array_equal(sliced._axis2_index.value, [1, 2, 3])

    def test_ellipsis_indexing(self):
        """Test ellipsis indexing."""
        arr = Array4D(np.zeros((2, 3, 4, 5)))
        sliced = arr[..., 1:3]

        assert isinstance(sliced, Array4D)
        assert sliced.shape == (2, 3, 4, 2)


class TestArray4DTranspose:
    """Test Array4D transpose and swapaxes."""

    def test_swapaxes(self):
        """Test swapaxes."""
        arr = Array4D(
            np.arange(120).reshape((2, 3, 4, 5)),
            axis_names=["t", "x", "y", "z"],
        )
        swapped = arr.swapaxes(0, 2)

        assert swapped.shape == (4, 3, 2, 5)
        assert swapped.axis_names == ("y", "x", "t", "z")

    def test_transpose(self):
        """Test transpose."""
        arr = Array4D(
            np.arange(120).reshape((2, 3, 4, 5)),
            axis_names=["t", "x", "y", "z"],
        )
        transposed = arr.transpose(3, 2, 1, 0)

        assert transposed.shape == (5, 4, 3, 2)
        assert transposed.axis_names == ("z", "y", "x", "t")

    def test_transpose_by_name(self):
        """Test transpose with axis names."""
        arr = Array4D(
            np.zeros((2, 3, 4, 5)),
            axis_names=["t", "x", "y", "z"],
        )
        transposed = arr.transpose("z", "y", "x", "t")

        assert transposed.shape == (5, 4, 3, 2)

    def test_T_property(self):
        """Test T property (full reverse)."""
        arr = Array4D(np.zeros((2, 3, 4, 5)), axis_names=["t", "x", "y", "z"])
        transposed = arr.T

        assert transposed.shape == (5, 4, 3, 2)
        assert transposed.axis_names == ("z", "y", "x", "t")


class TestArray4DIselSel:
    """Test isel and sel methods."""

    def test_isel_by_name(self):
        """Test isel with axis names."""
        arr = Array4D(
            np.arange(120).reshape((2, 3, 4, 5)),
            axis_names=["t", "x", "y", "z"],
        )
        selected = arr.isel(x=slice(0, 2), z=slice(1, 3))

        assert isinstance(selected, Array4D)
        assert selected.shape == (2, 2, 4, 2)

    def test_isel_int_drops_dimension(self):
        """Test isel with int index drops dimension."""
        arr = Array4D(np.zeros((2, 3, 4, 5)), axis_names=["t", "x", "y", "z"])
        selected = arr.isel(t=0)

        # Plain getitem via int drops axes
        assert selected.ndim == 3

    def test_sel_nearest(self):
        """Test sel with nearest method."""
        t = np.linspace(0, 1, 10) * u.s
        arr = Array4D(
            np.arange(10 * 3 * 4 * 5).reshape((10, 3, 4, 5)),
            axis0=t,
            axis_names=["t", "x", "y", "z"],
        )
        # Select t nearest to 0.55 s (should be index 5 or 6)
        selected = arr.sel(t=0.55 * u.s)

        # sel returns int index which drops dimension
        assert selected.ndim == 3
