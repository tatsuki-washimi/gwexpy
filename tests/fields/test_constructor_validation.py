"""Regression tests for ScalarField and TensorField constructor shape/axis validation.

Covers Issue #242: silent acceptance of mismatched axis lengths must be rejected
with a clear ValueError, matching the fix applied to VectorField in PR #240.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField, TensorField

DATA_4D = np.ones((4, 3, 5, 6))


# ---------------------------------------------------------------------------
# ScalarField — axis length mismatch
# ---------------------------------------------------------------------------


class TestScalarFieldAxisLengthValidation:
    def test_axis0_wrong_length_raises(self):
        axis0 = np.arange(7) * u.s  # 7 != 4
        with pytest.raises(
            ValueError, match="axis0 length 7 does not match data shape\\[0\\]=4"
        ):
            ScalarField(DATA_4D, axis0=axis0)

    def test_axis1_wrong_length_raises(self):
        axis1 = np.arange(10) * u.m  # 10 != 3
        with pytest.raises(
            ValueError, match="axis1 length 10 does not match data shape\\[1\\]=3"
        ):
            ScalarField(DATA_4D, axis1=axis1)

    def test_axis2_wrong_length_raises(self):
        axis2 = np.arange(2) * u.m  # 2 != 5
        with pytest.raises(
            ValueError, match="axis2 length 2 does not match data shape\\[2\\]=5"
        ):
            ScalarField(DATA_4D, axis2=axis2)

    def test_axis3_wrong_length_raises(self):
        axis3 = np.arange(99) * u.m  # 99 != 6
        with pytest.raises(
            ValueError, match="axis3 length 99 does not match data shape\\[3\\]=6"
        ):
            ScalarField(DATA_4D, axis3=axis3)

    def test_all_axes_correct_length_succeeds(self):
        axis0 = np.linspace(0, 1, 4) * u.s
        axis1 = np.linspace(0, 1, 3) * u.m
        axis2 = np.linspace(0, 1, 5) * u.m
        axis3 = np.linspace(0, 1, 6) * u.m
        sf = ScalarField(DATA_4D, axis0=axis0, axis1=axis1, axis2=axis2, axis3=axis3)
        assert sf.shape == (4, 3, 5, 6)
        assert len(sf._axis0_index) == 4
        assert len(sf._axis1_index) == 3
        assert len(sf._axis2_index) == 5
        assert len(sf._axis3_index) == 6

    def test_default_axes_match_shape(self):
        """When no axis is passed the defaults must match the data dimensions."""
        sf = ScalarField(DATA_4D)
        assert len(sf._axis0_index) == DATA_4D.shape[0]
        assert len(sf._axis1_index) == DATA_4D.shape[1]
        assert len(sf._axis2_index) == DATA_4D.shape[2]
        assert len(sf._axis3_index) == DATA_4D.shape[3]

    def test_axis_length_one_mismatch_raises(self):
        """A single-element axis for a dimension > 1 must be rejected."""
        axis0 = np.array([0.0]) * u.s  # length 1 != 4
        with pytest.raises(ValueError, match="axis0 length"):
            ScalarField(DATA_4D, axis0=axis0)


# ---------------------------------------------------------------------------
# TensorField — ndarray construction path
# ---------------------------------------------------------------------------


class TestTensorFieldNdarrayConstruction:
    def test_non_6d_array_raises(self):
        with pytest.raises(ValueError, match="TensorField rank-2 expects 6D array"):
            TensorField(np.ones((4, 3, 5, 6, 3)))  # 5D — must fail

    def test_non_6d_4d_raises(self):
        with pytest.raises(ValueError, match="TensorField rank-2 expects 6D array"):
            TensorField(np.ones((4, 3, 5, 6)))  # 4D — must fail

    def test_6d_array_creates_components(self):
        arr = np.ones((4, 3, 5, 6, 2, 2))
        tf = TensorField(arr)
        assert tf.rank == 2
        assert set(tf.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_component_axes_are_default_arange(self):
        """Components created from ndarray get default arange() axes (documented behaviour)."""
        arr = np.ones((4, 3, 5, 6, 2, 2))
        tf = TensorField(arr)
        comp = tf[(0, 0)]
        expected_axis0 = np.arange(4)
        np.testing.assert_array_equal(comp._axis0_index.value, expected_axis0)

    def test_component_shapes_match_field_dimensions(self):
        arr = np.ones((4, 3, 5, 6, 3, 3))
        tf = TensorField(arr)
        for key, comp in tf.items():
            assert comp.shape == (4, 3, 5, 6), (
                f"Component {key} has shape {comp.shape}, expected (4, 3, 5, 6)"
            )


# ---------------------------------------------------------------------------
# TensorField — dict construction path
# ---------------------------------------------------------------------------


class TestTensorFieldDictConstruction:
    def _make_sf(self, shape=(4, 3, 5, 6)):
        return ScalarField(np.ones(shape))

    def test_consistent_components_validate_true(self):
        comps = {
            (0, 0): self._make_sf(),
            (0, 1): self._make_sf(),
            (1, 0): self._make_sf(),
            (1, 1): self._make_sf(),
        }
        tf = TensorField(comps, validate=True)
        assert tf.rank == 2

    def test_inconsistent_shapes_validate_true_raises(self):
        comps = {
            (0, 0): self._make_sf(shape=(4, 3, 5, 6)),
            (0, 1): self._make_sf(shape=(4, 3, 5, 7)),  # shape mismatch on axis3
        }
        with pytest.raises((ValueError, Exception)):
            TensorField(comps, validate=True)
