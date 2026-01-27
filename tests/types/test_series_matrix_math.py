#!/usr/bin/env python
"""
Unit tests for SeriesMatrixMathMixin operations.

Tests linear algebra operations:
- trace
- diagonal (list, vector, matrix output modes)
- matrix multiplication (__matmul__)
- determinant (det)
- inverse (inv)
- eigenvalues (eigvals)
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix
from gwexpy.types.seriesmatrix import SeriesMatrix


@pytest.fixture
def square_matrix_2x2():
    """Create a 2x2 SeriesMatrix with 10 samples."""
    data = np.array([
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
         [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]
    ])
    xindex = np.linspace(0, 9, 10)
    return SeriesMatrix(data, xindex=xindex)


@pytest.fixture
def square_matrix_3x3():
    """Create a 3x3 identity-like SeriesMatrix with 5 samples."""
    data = np.zeros((3, 3, 5))
    for i in range(3):
        data[i, i, :] = float(i + 1)  # diagonal: 1, 2, 3
    xindex = np.linspace(0, 4, 5)
    return SeriesMatrix(data, xindex=xindex)


@pytest.fixture
def nonsquare_matrix():
    """Create a 2x3 non-square SeriesMatrix."""
    data = np.random.randn(2, 3, 10)
    xindex = np.linspace(0, 9, 10)
    return SeriesMatrix(data, xindex=xindex)


class TestTrace:
    """Tests for trace() method."""

    def test_trace_basic(self, square_matrix_2x2):
        """Test trace computation on a 2x2 matrix."""
        tr = square_matrix_2x2.trace()
        # trace = (1+2+...+10) + (2+2+...+2) = 55 + 20 = 75 at each sample? No.
        # Actually trace = sum of diagonal elements: [0,0] + [1,1]
        # [0,0] = [1,2,3,4,5,6,7,8,9,10]
        # [1,1] = [2,2,2,2,2,2,2,2,2,2]
        # trace = [3,4,5,6,7,8,9,10,11,12]
        expected = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        np.testing.assert_array_almost_equal(tr.value, expected)

    def test_trace_3x3(self, square_matrix_3x3):
        """Test trace computation on a 3x3 diagonal matrix."""
        tr = square_matrix_3x3.trace()
        # trace = 1 + 2 + 3 = 6 at each sample
        expected = np.full(5, 6.0)
        np.testing.assert_array_almost_equal(tr.value, expected)

    def test_trace_nonsquare_raises(self, nonsquare_matrix):
        """Test that trace raises ValueError for non-square matrix."""
        with pytest.raises(ValueError, match="trace requires a square matrix"):
            nonsquare_matrix.trace()


class TestDiagonal:
    """Tests for diagonal() method."""

    def test_diagonal_list(self, square_matrix_2x2):
        """Test diagonal extraction as list."""
        diag = square_matrix_2x2.diagonal(output="list")
        assert isinstance(diag, list)
        assert len(diag) == 2  # min(2, 2)

    def test_diagonal_vector(self, square_matrix_2x2):
        """Test diagonal extraction as vector (column matrix)."""
        diag = square_matrix_2x2.diagonal(output="vector")
        assert diag.shape3D == (2, 1, 10)

    def test_diagonal_matrix(self, square_matrix_2x2):
        """Test diagonal extraction as matrix (zeros off-diagonal)."""
        diag = square_matrix_2x2.diagonal(output="matrix")
        assert diag.shape3D == (2, 2, 10)
        # Off-diagonal should be zero
        np.testing.assert_array_equal(diag._value[0, 1], 0.0)
        np.testing.assert_array_equal(diag._value[1, 0], 0.0)

    def test_diagonal_invalid_output_raises(self, square_matrix_2x2):
        """Test that invalid output mode raises ValueError."""
        with pytest.raises(ValueError, match="output must be one of"):
            square_matrix_2x2.diagonal(output="invalid")

    def test_diagonal_3x3(self, square_matrix_3x3):
        """Test diagonal on 3x3 matrix."""
        diag = square_matrix_3x3.diagonal(output="list")
        assert len(diag) == 3
        # Check values: diagonal elements are 1, 2, 3
        np.testing.assert_array_almost_equal(diag[0].value, np.full(5, 1.0))
        np.testing.assert_array_almost_equal(diag[1].value, np.full(5, 2.0))
        np.testing.assert_array_almost_equal(diag[2].value, np.full(5, 3.0))


class TestMatMul:
    """Tests for __matmul__ (matrix multiplication)."""

    def test_matmul_identity(self, square_matrix_2x2):
        """Test matrix multiplication with identity-like matrix."""
        # Create identity matrix
        eye_data = np.zeros((2, 2, 10))
        eye_data[0, 0, :] = 1.0
        eye_data[1, 1, :] = 1.0
        eye = SeriesMatrix(eye_data, xindex=np.linspace(0, 9, 10))

        result = square_matrix_2x2 @ eye
        np.testing.assert_array_almost_equal(
            result._value, square_matrix_2x2._value
        )

    def test_matmul_dimension_mismatch_raises(self, square_matrix_2x2):
        """Test that mismatched dimensions raise ValueError."""
        other = SeriesMatrix(np.random.randn(3, 2, 10), xindex=np.linspace(0, 9, 10))
        with pytest.raises(ValueError, match="Matrix dimension mismatch"):
            square_matrix_2x2 @ other

    def test_matmul_sample_mismatch_raises(self, square_matrix_2x2):
        """Test that mismatched sample lengths raise ValueError."""
        other = SeriesMatrix(np.random.randn(2, 2, 5), xindex=np.linspace(0, 4, 5))
        with pytest.raises(ValueError, match="Sample axis length mismatch"):
            square_matrix_2x2 @ other


class TestDeterminant:
    """Tests for det() method."""

    def test_det_identity(self):
        """Test determinant of identity matrix."""
        eye_data = np.zeros((2, 2, 5))
        eye_data[0, 0, :] = 1.0
        eye_data[1, 1, :] = 1.0
        eye = SeriesMatrix(eye_data, xindex=np.linspace(0, 4, 5))

        det = eye.det()
        # det(I) = 1
        np.testing.assert_array_almost_equal(det.value, np.ones(5))

    def test_det_scalar_multiple(self):
        """Test determinant of scalar multiple of identity."""
        eye_data = np.zeros((2, 2, 5))
        eye_data[0, 0, :] = 2.0
        eye_data[1, 1, :] = 2.0
        mat = SeriesMatrix(eye_data, xindex=np.linspace(0, 4, 5))

        det = mat.det()
        # det(2*I) = 4 for 2x2
        np.testing.assert_array_almost_equal(det.value, np.full(5, 4.0))

    def test_det_nonsquare_raises(self, nonsquare_matrix):
        """Test that det raises ValueError for non-square matrix."""
        with pytest.raises(ValueError, match="det requires a square matrix"):
            nonsquare_matrix.det()


class TestInverse:
    """Tests for inv() method."""

    def test_inv_identity(self):
        """Test inverse of identity matrix."""
        eye_data = np.zeros((2, 2, 5))
        eye_data[0, 0, :] = 1.0
        eye_data[1, 1, :] = 1.0
        eye = SeriesMatrix(eye_data, xindex=np.linspace(0, 4, 5))

        inv = eye.inv()
        # inv(I) = I
        np.testing.assert_array_almost_equal(inv._value[0, 0], np.ones(5))
        np.testing.assert_array_almost_equal(inv._value[1, 1], np.ones(5))
        np.testing.assert_array_almost_equal(inv._value[0, 1], np.zeros(5))
        np.testing.assert_array_almost_equal(inv._value[1, 0], np.zeros(5))

    def test_inv_diagonal(self):
        """Test inverse of diagonal matrix."""
        diag_data = np.zeros((2, 2, 5))
        diag_data[0, 0, :] = 2.0
        diag_data[1, 1, :] = 4.0
        mat = SeriesMatrix(diag_data, xindex=np.linspace(0, 4, 5))

        inv = mat.inv()
        # inv([2,0;0,4]) = [0.5,0;0,0.25]
        np.testing.assert_array_almost_equal(inv._value[0, 0], np.full(5, 0.5))
        np.testing.assert_array_almost_equal(inv._value[1, 1], np.full(5, 0.25))

    def test_inv_nonsquare_raises(self, nonsquare_matrix):
        """Test that inv raises ValueError for non-square matrix."""
        with pytest.raises(ValueError, match="inv requires a square matrix"):
            nonsquare_matrix.inv()


class TestUnitHandling:
    """Tests for unit handling in math operations."""

    def test_trace_with_units(self):
        """Test trace preserves units."""
        data = np.ones((2, 2, 5))
        xindex = np.linspace(0, 4, 5)
        meta_arr = np.array([
            [MetaData(unit=u.m), MetaData(unit=u.m)],
            [MetaData(unit=u.m), MetaData(unit=u.m)]
        ])
        mat = SeriesMatrix(data, xindex=xindex, meta=MetaDataMatrix(meta_arr))

        tr = mat.trace()
        assert tr.unit == u.m

    def test_det_with_units(self):
        """Test determinant computes correct unit power."""
        data = np.zeros((2, 2, 5))
        data[0, 0, :] = 2.0
        data[1, 1, :] = 3.0
        xindex = np.linspace(0, 4, 5)
        meta_arr = np.array([
            [MetaData(unit=u.m), MetaData(unit=u.m)],
            [MetaData(unit=u.m), MetaData(unit=u.m)]
        ])
        mat = SeriesMatrix(data, xindex=xindex, meta=MetaDataMatrix(meta_arr))

        det = mat.det()
        # det of 2x2 with unit m: result is m^2
        assert det.unit == u.m**2
