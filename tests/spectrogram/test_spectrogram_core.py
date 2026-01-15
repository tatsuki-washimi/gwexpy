"""
Tests for gwexpy/spectrogram core behaviors.

This module provides pytest tests for:
1. Spectrogram.radian() / Spectrogram.degree() metadata propagation
2. SpectrogramMatrixCoreMixin.frequencies validation
3. SpectrogramMatrixAnalysisMixin.radian/degree metadata propagation
4. SpectrogramMatrix.__array_ufunc__ unit behavior
5. SpectrogramList/SpectrogramDict.to_matrix invariants
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u


# =============================================================================
# 1. Tests for Spectrogram.radian() / Spectrogram.degree()
# =============================================================================

class TestSpectrogramPhase:
    """Test Spectrogram.radian() and .degree() methods."""

    @pytest.fixture
    def complex_spectrogram(self):
        """Create a complex-valued Spectrogram for phase testing."""
        from gwexpy.spectrogram import Spectrogram

        # Create complex data (N=4 time, F=3 freq)
        np.random.seed(42)
        real = np.random.rand(4, 3)
        imag = np.random.rand(4, 3)
        data = real + 1j * imag

        times = np.array([0.0, 0.1, 0.2, 0.3])
        freqs = np.array([10.0, 20.0, 30.0])

        return Spectrogram(
            data,
            times=times,
            frequencies=freqs,
            unit=u.dimensionless_unscaled,
            name="test_complex",
            epoch=0.0,
        )

    def test_radian_returns_spectrogram(self, complex_spectrogram):
        """radian() should return a Spectrogram instance."""
        from gwexpy.spectrogram import Spectrogram

        result = complex_spectrogram.radian()
        assert isinstance(result, Spectrogram)

    def test_radian_shape_preserved(self, complex_spectrogram):
        """radian() should preserve the shape."""
        result = complex_spectrogram.radian()
        assert result.shape == complex_spectrogram.shape

    def test_radian_times_preserved(self, complex_spectrogram):
        """radian() should preserve times (value-equal)."""
        result = complex_spectrogram.radian()
        np.testing.assert_array_almost_equal(
            np.asarray(result.times), np.asarray(complex_spectrogram.times)
        )

    def test_radian_frequencies_preserved(self, complex_spectrogram):
        """radian() should preserve frequencies (value-equal)."""
        result = complex_spectrogram.radian()
        np.testing.assert_array_almost_equal(
            np.asarray(result.frequencies), np.asarray(complex_spectrogram.frequencies)
        )

    def test_radian_unit_is_rad(self, complex_spectrogram):
        """radian() should set unit to rad."""
        result = complex_spectrogram.radian()
        assert result.unit == u.rad

    def test_radian_unwrap_uses_axis0(self, complex_spectrogram):
        """radian(unwrap=True) should unwrap along time axis (axis=0)."""
        # Create a Spectrogram with phase wrapping along time axis
        from gwexpy.spectrogram import Spectrogram

        # Build phase that wraps
        phases = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0],
            [-3.0, -3.0, -3.0],  # wraps from ~3 to -3
            [0.0, 0.0, 0.0],
        ])
        data = np.exp(1j * phases)

        spec = Spectrogram(
            data,
            times=np.array([0.0, 0.1, 0.2, 0.3]),
            frequencies=np.array([10.0, 20.0, 30.0]),
            unit=u.dimensionless_unscaled,
            name="wrap_test",
        )

        result_wrapped = spec.radian(unwrap=False)
        result_unwrapped = spec.radian(unwrap=True)

        # Compare with manual unwrap along axis=0
        expected_unwrapped = np.unwrap(np.angle(data), axis=0)
        np.testing.assert_array_almost_equal(result_unwrapped.value, expected_unwrapped)

    def test_degree_returns_spectrogram(self, complex_spectrogram):
        """degree() should return a Spectrogram instance."""
        from gwexpy.spectrogram import Spectrogram

        result = complex_spectrogram.degree()
        assert isinstance(result, Spectrogram)

    def test_degree_shape_preserved(self, complex_spectrogram):
        """degree() should preserve the shape."""
        result = complex_spectrogram.degree()
        assert result.shape == complex_spectrogram.shape

    def test_degree_times_preserved(self, complex_spectrogram):
        """degree() should preserve times (value-equal)."""
        result = complex_spectrogram.degree()
        np.testing.assert_array_almost_equal(
            np.asarray(result.times), np.asarray(complex_spectrogram.times)
        )

    def test_degree_frequencies_preserved(self, complex_spectrogram):
        """degree() should preserve frequencies (value-equal)."""
        result = complex_spectrogram.degree()
        np.testing.assert_array_almost_equal(
            np.asarray(result.frequencies), np.asarray(complex_spectrogram.frequencies)
        )

    def test_degree_unit_is_deg(self, complex_spectrogram):
        """degree() should set unit to deg."""
        result = complex_spectrogram.degree()
        assert result.unit == u.deg

    def test_degree_unwrap_uses_axis0(self, complex_spectrogram):
        """degree(unwrap=True) should unwrap along time axis (axis=0)."""
        from gwexpy.spectrogram import Spectrogram

        # Build phase that wraps
        phases = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0],
            [-3.0, -3.0, -3.0],
            [0.0, 0.0, 0.0],
        ])
        data = np.exp(1j * phases)

        spec = Spectrogram(
            data,
            times=np.array([0.0, 0.1, 0.2, 0.3]),
            frequencies=np.array([10.0, 20.0, 30.0]),
            unit=u.dimensionless_unscaled,
            name="wrap_test",
        )

        result_unwrapped = spec.degree(unwrap=True)
        expected_unwrapped = np.rad2deg(np.unwrap(np.angle(data), axis=0))
        np.testing.assert_array_almost_equal(result_unwrapped.value, expected_unwrapped)


# =============================================================================
# 2. Tests for SpectrogramMatrixCoreMixin.frequencies
# =============================================================================

class TestSpectrogramMatrixFrequencies:
    """Test SpectrogramMatrixCoreMixin frequencies length validation and df/f0."""

    @pytest.fixture
    def small_matrix(self):
        """Create a small SpectrogramMatrix (3D: N=2, T=4, F=3)."""
        from gwexpy.spectrogram import SpectrogramMatrix

        data = np.random.rand(2, 4, 3)
        times = np.array([0.0, 1.0, 2.0, 3.0])
        freqs = np.array([10.0, 20.0, 30.0])

        return SpectrogramMatrix(
            data,
            times=times,
            frequencies=freqs,
            unit=u.m,
            name="test_matrix",
        )

    def test_frequencies_matching_length_passes(self, small_matrix):
        """Setting frequencies with matching length should pass."""
        new_freqs = np.array([5.0, 10.0, 15.0])
        small_matrix.frequencies = new_freqs
        np.testing.assert_array_equal(small_matrix.frequencies, new_freqs)

    def test_frequencies_mismatching_length_raises(self, small_matrix):
        """Setting frequencies with mismatching length should raise ValueError."""
        wrong_freqs = np.array([1.0, 2.0, 3.0, 4.0])  # length 4 != 3
        with pytest.raises(ValueError, match="frequencies length mismatch"):
            small_matrix.frequencies = wrong_freqs

    def test_f0_equals_first_frequency(self, small_matrix):
        """f0 should equal frequencies[0]."""
        expected_f0 = small_matrix.frequencies[0]
        # Accept both scalar and Quantity
        if hasattr(expected_f0, "value"):
            np.testing.assert_almost_equal(small_matrix.f0.value, expected_f0.value)
        else:
            np.testing.assert_almost_equal(small_matrix.f0, expected_f0)

    def test_df_equals_frequency_spacing(self, small_matrix):
        """df should equal frequencies[1] - frequencies[0] for regular arrays."""
        expected_df = small_matrix.frequencies[1] - small_matrix.frequencies[0]
        # Accept both scalar and Quantity
        if hasattr(expected_df, "value"):
            assert np.isclose(small_matrix.df.value, expected_df.value)
        else:
            assert np.isclose(small_matrix.df.value, expected_df)

    def test_frequencies_with_quantity(self):
        """Frequencies can be set as astropy Quantity."""
        from gwexpy.spectrogram import SpectrogramMatrix

        data = np.random.rand(2, 4, 3)
        times = np.array([0.0, 1.0, 2.0, 3.0])
        freqs = np.array([10.0, 20.0, 30.0]) * u.Hz

        mat = SpectrogramMatrix(
            data, times=times, frequencies=freqs, unit=u.m, name="test_q"
        )

        assert mat.frequencies.unit == u.Hz
        np.testing.assert_array_almost_equal(mat.frequencies.value, [10.0, 20.0, 30.0])


# =============================================================================
# 3. Tests for SpectrogramMatrixAnalysisMixin.radian/degree
# =============================================================================

class TestSpectrogramMatrixPhase:
    """Test SpectrogramMatrixAnalysisMixin radian/degree methods."""

    @pytest.fixture
    def complex_matrix(self):
        """Create a complex-valued SpectrogramMatrix for phase testing."""
        from gwexpy.spectrogram import SpectrogramMatrix

        np.random.seed(42)
        real = np.random.rand(2, 4, 3)
        imag = np.random.rand(2, 4, 3)
        data = real + 1j * imag

        times = np.array([0.0, 1.0, 2.0, 3.0])
        freqs = np.array([10.0, 20.0, 30.0])

        return SpectrogramMatrix(
            data,
            times=times,
            frequencies=freqs,
            unit=u.dimensionless_unscaled,
            name="test_complex_matrix",
            rows=["ch1", "ch2"],
        )

    def test_radian_unit_is_rad(self, complex_matrix):
        """radian() should set unit to rad."""
        result = complex_matrix.radian()
        assert result.unit == u.rad

    def test_radian_shape_preserved(self, complex_matrix):
        """radian() should preserve shape."""
        result = complex_matrix.radian()
        assert result.shape == complex_matrix.shape

    def test_radian_times_preserved(self, complex_matrix):
        """radian() should preserve times."""
        result = complex_matrix.radian()
        np.testing.assert_array_almost_equal(
            np.asarray(result.times), np.asarray(complex_matrix.times)
        )

    def test_radian_frequencies_preserved(self, complex_matrix):
        """radian() should preserve frequencies."""
        result = complex_matrix.radian()
        np.testing.assert_array_almost_equal(
            np.asarray(result.frequencies), np.asarray(complex_matrix.frequencies)
        )

    def test_radian_unwrap_uses_axis_minus2(self, complex_matrix):
        """radian(unwrap=True) should unwrap along axis=-2 (time axis)."""
        result = complex_matrix.radian(unwrap=True)

        # Manual calculation - use view to avoid np.angle issues with SpectrogramMatrix
        expected = np.unwrap(np.angle(complex_matrix.view(np.ndarray)), axis=-2)
        np.testing.assert_array_almost_equal(result.view(np.ndarray), expected)

    def test_degree_unit_is_deg(self, complex_matrix):
        """degree() should set unit to deg."""
        result = complex_matrix.degree()
        assert result.unit == u.deg

    def test_degree_shape_preserved(self, complex_matrix):
        """degree() should preserve shape."""
        result = complex_matrix.degree()
        assert result.shape == complex_matrix.shape

    def test_degree_name_contains_phase_suffix(self, complex_matrix):
        """degree() should have name containing phase suffix."""
        result = complex_matrix.degree()
        # Name should contain "_phase" somewhere
        assert "_phase" in result.name

    def test_radian_meta_updates_unit(self, complex_matrix):
        """radian() should update meta elements unit to rad if meta exists."""
        result = complex_matrix.radian()
        if result.meta is not None:
            for m in result.meta.flat:
                assert m.unit == u.rad

    def test_degree_meta_updates_unit(self, complex_matrix):
        """degree() should update meta elements unit to deg if meta exists."""
        result = complex_matrix.degree()
        if result.meta is not None:
            for m in result.meta.flat:
                assert m.unit == u.deg


# =============================================================================
# 4. Tests for SpectrogramMatrix.__array_ufunc__ unit behavior
# =============================================================================

class TestSpectrogramMatrixUfunc:
    """Test SpectrogramMatrix.__array_ufunc__ unit behavior."""

    @pytest.fixture
    def matrix_with_unit(self):
        """Create a SpectrogramMatrix with unit 'm'."""
        from gwexpy.spectrogram import SpectrogramMatrix

        data = np.random.rand(2, 4, 3)
        times = np.array([0.0, 1.0, 2.0, 3.0])
        freqs = np.array([10.0, 20.0, 30.0])

        return SpectrogramMatrix(
            data,
            times=times,
            frequencies=freqs,
            unit=u.m,
            name="test_m",
        )

    def test_multiply_scalar_keeps_unit(self, matrix_with_unit):
        """(matrix with unit 'm') * 2 should keep unit 'm'."""
        result = matrix_with_unit * 2
        assert result.unit == u.m

    def test_add_compatible_units_works(self):
        """Adding matrices with compatible units should work."""
        from gwexpy.spectrogram import SpectrogramMatrix

        data1 = np.random.rand(2, 4, 3)
        data2 = np.random.rand(2, 4, 3)
        times = np.array([0.0, 1.0, 2.0, 3.0])
        freqs = np.array([10.0, 20.0, 30.0])

        mat1 = SpectrogramMatrix(data1, times=times, frequencies=freqs, unit=u.m)
        mat2 = SpectrogramMatrix(data2, times=times, frequencies=freqs, unit=u.m)

        result = mat1 + mat2
        assert result.unit == u.m
        np.testing.assert_array_almost_equal(result.view(np.ndarray), data1 + data2)

    def test_add_incompatible_units_raises(self):
        """Adding matrices with incompatible units should raise UnitConversionError."""
        from gwexpy.spectrogram import SpectrogramMatrix

        data1 = np.random.rand(2, 4, 3)
        data2 = np.random.rand(2, 4, 3)
        times = np.array([0.0, 1.0, 2.0, 3.0])
        freqs = np.array([10.0, 20.0, 30.0])

        mat1 = SpectrogramMatrix(data1, times=times, frequencies=freqs, unit=u.m)
        mat2 = SpectrogramMatrix(data2, times=times, frequencies=freqs, unit=u.s)

        with pytest.raises(u.UnitConversionError):
            _ = mat1 + mat2

    def test_multiply_units_combine(self):
        """Multiplying matrices with units should combine units correctly."""
        from gwexpy.spectrogram import SpectrogramMatrix

        data1 = np.random.rand(2, 4, 3)
        data2 = np.random.rand(2, 4, 3)
        times = np.array([0.0, 1.0, 2.0, 3.0])
        freqs = np.array([10.0, 20.0, 30.0])

        mat1 = SpectrogramMatrix(data1, times=times, frequencies=freqs, unit=u.m)
        mat2 = SpectrogramMatrix(data2, times=times, frequencies=freqs, unit=u.s)

        result = mat1 * mat2
        # m * s should produce m * s
        expected_unit = u.m * u.s
        assert result.unit.is_equivalent(expected_unit)


# =============================================================================
# 5. Tests for SpectrogramList/SpectrogramDict.to_matrix invariants
# =============================================================================

class TestCollectionsToMatrix:
    """Test SpectrogramList and SpectrogramDict to_matrix invariants."""

    @pytest.fixture
    def spectrogram_factory(self):
        """Factory to create Spectrogram objects."""
        from gwexpy.spectrogram import Spectrogram

        def _create(t_size=4, f_size=3, seed=42):
            np.random.seed(seed)
            data = np.random.rand(t_size, f_size)
            times = np.linspace(0, 1, t_size)
            freqs = np.linspace(10, 30, f_size)
            return Spectrogram(
                data,
                times=times,
                frequencies=freqs,
                unit=u.m,
                name="test_spec",
            )

        return _create

    def test_list_to_matrix_shape_mismatch_raises(self, spectrogram_factory):
        """SpectrogramList.to_matrix should raise ValueError for shape mismatch."""
        from gwexpy.spectrogram import SpectrogramList

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=5, f_size=3, seed=2)  # different t_size

        lst = SpectrogramList([s1, s2])

        with pytest.raises(ValueError, match="[Ss]hape mismatch"):
            lst.to_matrix()

    def test_list_to_matrix_success(self, spectrogram_factory):
        """SpectrogramList.to_matrix should return shape (N, T, F)."""
        from gwexpy.spectrogram import SpectrogramList, SpectrogramMatrix

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=4, f_size=3, seed=2)

        lst = SpectrogramList([s1, s2])
        mat = lst.to_matrix()

        assert isinstance(mat, SpectrogramMatrix)
        assert mat.shape == (2, 4, 3)  # N=2, T=4, F=3

    def test_list_to_matrix_inherits_times(self, spectrogram_factory):
        """to_matrix should inherit times from first element."""
        from gwexpy.spectrogram import SpectrogramList

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=4, f_size=3, seed=2)

        lst = SpectrogramList([s1, s2])
        mat = lst.to_matrix()

        np.testing.assert_array_almost_equal(
            np.asarray(mat.times), np.asarray(s1.times)
        )

    def test_list_to_matrix_inherits_frequencies(self, spectrogram_factory):
        """to_matrix should inherit frequencies from first element."""
        from gwexpy.spectrogram import SpectrogramList

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=4, f_size=3, seed=2)

        lst = SpectrogramList([s1, s2])
        mat = lst.to_matrix()

        np.testing.assert_array_almost_equal(
            np.asarray(mat.frequencies), np.asarray(s1.frequencies)
        )

    def test_list_to_matrix_inherits_unit(self, spectrogram_factory):
        """to_matrix should inherit unit from first element."""
        from gwexpy.spectrogram import SpectrogramList

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=4, f_size=3, seed=2)

        lst = SpectrogramList([s1, s2])
        mat = lst.to_matrix()

        assert mat.unit == s1.unit

    def test_dict_to_matrix_shape_mismatch_raises(self, spectrogram_factory):
        """SpectrogramDict.to_matrix should raise ValueError for shape mismatch."""
        from gwexpy.spectrogram import SpectrogramDict

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=4, f_size=4, seed=2)  # different f_size

        d = SpectrogramDict({"a": s1, "b": s2})

        with pytest.raises(ValueError, match="[Mm]ismatch"):
            d.to_matrix()

    def test_dict_to_matrix_success(self, spectrogram_factory):
        """SpectrogramDict.to_matrix should return shape (N, T, F)."""
        from gwexpy.spectrogram import SpectrogramDict, SpectrogramMatrix

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=4, f_size=3, seed=2)

        d = SpectrogramDict({"a": s1, "b": s2})
        mat = d.to_matrix()

        assert isinstance(mat, SpectrogramMatrix)
        assert mat.shape == (2, 4, 3)

    def test_dict_to_matrix_inherits_times(self, spectrogram_factory):
        """to_matrix should inherit times from first element."""
        from gwexpy.spectrogram import SpectrogramDict

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=4, f_size=3, seed=2)

        d = SpectrogramDict({"a": s1, "b": s2})
        mat = d.to_matrix()

        # Dict ordering may vary, so check against first value
        first_val = list(d.values())[0]
        np.testing.assert_array_almost_equal(
            np.asarray(mat.times), np.asarray(first_val.times)
        )

    def test_dict_to_matrix_inherits_unit(self, spectrogram_factory):
        """to_matrix should inherit unit from first element."""
        from gwexpy.spectrogram import SpectrogramDict

        s1 = spectrogram_factory(t_size=4, f_size=3, seed=1)
        s2 = spectrogram_factory(t_size=4, f_size=3, seed=2)

        d = SpectrogramDict({"a": s1, "b": s2})
        mat = d.to_matrix()

        first_val = list(d.values())[0]
        assert mat.unit == first_val.unit


# =============================================================================
# Import sanity checks
# =============================================================================

class TestImportSanity:
    """Test that re-exports can be imported without error."""

    def test_import_coherence(self):
        """coherence.py should be importable."""
        from gwexpy.spectrogram import coherence
        assert coherence is not None

    def test_import_io_hdf5(self):
        """io/hdf5.py should be importable."""
        from gwexpy.spectrogram.io import hdf5
        assert hdf5 is not None
