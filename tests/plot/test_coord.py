"""Tests for Field4D visualization utilities (Phase 0.1)."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.plot._coord import (
    nearest_index,
    select_value,
    slice_from_index,
    slice_from_value,
)


class TestNearestIndex:
    """Tests for nearest_index function."""

    def test_basic_match(self):
        """Test finding exact match."""
        axis = np.array([0, 1, 2, 3, 4]) * u.m
        assert nearest_index(axis, 2.0 * u.m) == 2

    def test_nearest_rounding(self):
        """Test nearest neighbor selection."""
        axis = np.array([0, 1, 2, 3, 4]) * u.m
        assert nearest_index(axis, 2.3 * u.m) == 2
        assert nearest_index(axis, 2.7 * u.m) == 3

    def test_tie_break_smaller_index(self):
        """Test tie-break rule: smaller index wins."""
        axis = np.array([0, 1, 2, 3, 4]) * u.m
        # Exactly between 2 and 3 -> should return 2
        assert nearest_index(axis, 2.5 * u.m) == 2

    def test_unit_mismatch_raises_valueerror(self):
        """Test that unit mismatch raises ValueError."""
        axis = np.array([0, 1, 2, 3, 4]) * u.m
        with pytest.raises(ValueError, match="Unit mismatch"):
            nearest_index(axis, 2.0 * u.s)

    def test_out_of_range_below_raises_indexerror(self):
        """Test that value below range raises IndexError."""
        axis = np.array([1, 2, 3, 4, 5]) * u.m
        with pytest.raises(IndexError, match="outside axis range"):
            nearest_index(axis, 0.5 * u.m)

    def test_out_of_range_above_raises_indexerror(self):
        """Test that value above range raises IndexError."""
        axis = np.array([1, 2, 3, 4, 5]) * u.m
        with pytest.raises(IndexError, match="outside axis range"):
            nearest_index(axis, 5.5 * u.m)

    def test_edge_values(self):
        """Test values at exact edges."""
        axis = np.array([1, 2, 3, 4, 5]) * u.m
        assert nearest_index(axis, 1.0 * u.m) == 0
        assert nearest_index(axis, 5.0 * u.m) == 4

    def test_unit_conversion(self):
        """Test that compatible units are converted."""
        axis = np.array([0, 1000, 2000, 3000]) * u.mm
        # 2 m = 2000 mm
        assert nearest_index(axis, 2.0 * u.m) == 2


class TestSliceFromIndex:
    """Tests for slice_from_index function."""

    def test_basic_slice(self):
        """Test basic slice creation."""
        s = slice_from_index(5)
        assert s == slice(5, 6)

    def test_zero_index(self):
        """Test slice from index 0."""
        s = slice_from_index(0)
        assert s == slice(0, 1)


class TestSliceFromValue:
    """Tests for slice_from_value function."""

    def test_basic_slice(self):
        """Test basic slice from value."""
        axis = np.array([0, 1, 2, 3, 4]) * u.m
        s = slice_from_value(axis, 2.3 * u.m)
        assert s == slice(2, 3)

    def test_unsupported_method_raises(self):
        """Test that unsupported interpolation method raises ValueError."""
        axis = np.array([0, 1, 2, 3, 4]) * u.m
        with pytest.raises(ValueError, match="Unsupported method"):
            slice_from_value(axis, 2.0 * u.m, method="linear")


class TestSelectValue:
    """Tests for select_value function."""

    def test_real_mode(self):
        """Test extracting real part."""
        data = np.array([1 + 2j, 3 + 4j])
        result = select_value(data, "real")
        np.testing.assert_array_equal(result, [1, 3])

    def test_imag_mode(self):
        """Test extracting imaginary part."""
        data = np.array([1 + 2j, 3 + 4j])
        result = select_value(data, "imag")
        np.testing.assert_array_equal(result, [2, 4])

    def test_abs_mode(self):
        """Test extracting absolute value."""
        data = np.array([3 + 4j])
        result = select_value(data, "abs")
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_angle_mode(self):
        """Test extracting phase angle."""
        data = np.array([1 + 1j])
        result = select_value(data, "angle")
        np.testing.assert_array_almost_equal(result.value, [np.pi / 4])
        assert result.unit == u.rad

    def test_power_mode(self):
        """Test extracting power (squared magnitude)."""
        data = np.array([3 + 4j])
        result = select_value(data, "power")
        np.testing.assert_array_almost_equal(result, [25.0])

    def test_power_with_units(self):
        """Test power mode preserves unit^2."""
        data = np.array([3 + 4j]) * u.V
        result = select_value(data, "power")
        np.testing.assert_array_almost_equal(result.value, [25.0])
        assert result.unit == u.V**2

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Invalid mode"):
            select_value(data, "invalid")

    def test_real_data_passthrough(self):
        """Test that real data passes through modes correctly."""
        data = np.array([1.0, 2.0, 3.0])
        result = select_value(data, "real")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_with_quantity(self):
        """Test with Quantity input."""
        data = np.array([1 + 2j, 3 + 4j]) * u.m
        result = select_value(data, "real")
        np.testing.assert_array_equal(result.value, [1, 3])
        assert result.unit == u.m
