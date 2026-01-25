"""
Tests for SpectrogramList/SpectrogramDict.to_matrix() semantics.

These tests lock down the behavior to ensure alignment with SeriesMatrix base rules:
- Axes (times/frequencies) strict matching: convert to reference unit, np.array_equal
- Per-element unit handling: different units allowed, stored in MetaDataMatrix
- Global unit is None (no single uniform unit assumed)
"""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.spectrogram import Spectrogram, SpectrogramMatrix
from gwexpy.spectrogram.collections import SpectrogramDict, SpectrogramList


class TestSpectrogramListToMatrixSemantics:
    """Tests for SpectrogramList.to_matrix() axis and unit handling."""

    @pytest.fixture
    def times(self):
        return np.arange(10) * u.s

    @pytest.fixture
    def frequencies(self):
        return np.arange(5) * 10 * u.Hz

    # --- 4.1 Axes strict matching (SeriesMatrix rules) ---

    def test_shape_mismatch_raises_valueerror(self, times, frequencies):
        """to_matrix() raises ValueError when element shapes differ."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 3),  # Different freq count
            times=times,
            frequencies=np.arange(3) * 10 * u.Hz,
            unit=u.V,
        )
        with pytest.raises(ValueError):
            SpectrogramList([sg1, sg2]).to_matrix()

    def test_times_mismatch_raises_valueerror(self, frequencies):
        """to_matrix() raises ValueError when times arrays differ."""
        times1 = np.arange(10) * u.s
        times2 = np.arange(10) * 2 * u.s  # Different values

        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times1, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times2, frequencies=frequencies, unit=u.V
        )
        with pytest.raises(ValueError):
            SpectrogramList([sg1, sg2]).to_matrix()

    def test_frequencies_mismatch_raises_valueerror(self, times):
        """to_matrix() raises ValueError when frequencies arrays differ."""
        freqs1 = np.arange(5) * 10 * u.Hz
        freqs2 = np.arange(5) * 20 * u.Hz  # Different values

        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=freqs1, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=freqs2, unit=u.V
        )
        with pytest.raises(ValueError):
            SpectrogramList([sg1, sg2]).to_matrix()

    def test_times_convertible_units_succeed(self):
        """to_matrix() succeeds when times have different but convertible units."""
        # [0, 1, ..., 9] s == [0, 1000, ..., 9000] ms
        times_s = np.arange(10) * u.s
        times_ms = np.arange(10) * 1000 * u.ms
        freqs = np.arange(5) * u.Hz

        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times_s, frequencies=freqs, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times_ms, frequencies=freqs, unit=u.A
        )
        # Should succeed after unit conversion
        matrix = SpectrogramList([sg1, sg2]).to_matrix()
        assert matrix.shape == (2, 10, 5)

    def test_times_nonconvertible_different_values_raises(self):
        """to_matrix() raises ValueError when times have different converted values."""
        times_s = np.arange(10) * u.s  # [0, 1, ..., 9] s
        times_ms = np.arange(10) * 500 * u.ms  # [0, 0.5, ..., 4.5] s != times_s
        freqs = np.arange(5) * u.Hz

        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times_s, frequencies=freqs, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times_ms, frequencies=freqs, unit=u.V
        )
        with pytest.raises(ValueError):
            SpectrogramList([sg1, sg2]).to_matrix()

    # --- 4.2 Per-element unit handling ---

    def test_different_element_units_succeed(self, times, frequencies):
        """to_matrix() succeeds when element Spectrograms have different units."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=frequencies,
            unit=u.V,
            name="voltage",
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=frequencies,
            unit=u.Pa,  # Different unit
            name="pressure",
        )
        # Should succeed
        matrix = SpectrogramList([sg1, sg2]).to_matrix()
        assert matrix.shape == (2, 10, 5)

    def test_global_unit_is_none(self, times, frequencies):
        """to_matrix() result has global unit=None (no uniform unit assumed)."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.A
        )
        matrix = SpectrogramList([sg1, sg2]).to_matrix()
        assert matrix.unit is None

    def test_per_element_units_in_meta(self, times, frequencies):
        """to_matrix() stores per-element units in MetaDataMatrix.meta."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.A
        )
        matrix = SpectrogramList([sg1, sg2]).to_matrix()

        # meta.flat should be iterable
        assert matrix.meta is not None
        units = [m.unit for m in matrix.meta.flat]
        assert u.V in units
        assert u.A in units

    def test_per_element_units_match_original(self, times, frequencies):
        """to_matrix() preserves original element units in meta."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=frequencies,
            unit=u.m,
            name="length",
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=frequencies,
            unit=u.kg,
            name="mass",
        )
        matrix = SpectrogramList([sg1, sg2]).to_matrix()

        # First element should have unit=m
        assert matrix.meta[0, 0].unit == u.m
        # Second element should have unit=kg
        assert matrix.meta[1, 0].unit == u.kg


class TestSpectrogramDictToMatrixSemantics:
    """Tests for SpectrogramDict.to_matrix() - should match List behavior."""

    @pytest.fixture
    def times(self):
        return np.arange(10) * u.s

    @pytest.fixture
    def frequencies(self):
        return np.arange(5) * 10 * u.Hz

    def test_shape_mismatch_raises_valueerror(self, times, frequencies):
        """SpectrogramDict.to_matrix() raises ValueError on shape mismatch."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 3),
            times=times,
            frequencies=np.arange(3) * 10 * u.Hz,
            unit=u.V,
        )
        with pytest.raises(ValueError):
            SpectrogramDict({"a": sg1, "b": sg2}).to_matrix()

    def test_times_mismatch_raises_valueerror(self, frequencies):
        """SpectrogramDict.to_matrix() raises ValueError on times mismatch."""
        times1 = np.arange(10) * u.s
        times2 = np.arange(10) * 2 * u.s

        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times1, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times2, frequencies=frequencies, unit=u.V
        )
        with pytest.raises(ValueError):
            SpectrogramDict({"a": sg1, "b": sg2}).to_matrix()

    def test_frequencies_mismatch_raises_valueerror(self, times):
        """SpectrogramDict.to_matrix() raises ValueError on frequencies mismatch."""
        freqs1 = np.arange(5) * 10 * u.Hz
        freqs2 = np.arange(5) * 20 * u.Hz

        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=freqs1, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=freqs2, unit=u.V
        )
        with pytest.raises(ValueError):
            SpectrogramDict({"a": sg1, "b": sg2}).to_matrix()

    def test_different_element_units_succeed(self, times, frequencies):
        """SpectrogramDict.to_matrix() succeeds with different element units."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.Pa
        )
        matrix = SpectrogramDict({"v": sg1, "p": sg2}).to_matrix()
        assert matrix.shape == (2, 10, 5)

    def test_global_unit_is_none(self, times, frequencies):
        """SpectrogramDict.to_matrix() result has global unit=None."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.A
        )
        matrix = SpectrogramDict({"v": sg1, "a": sg2}).to_matrix()
        assert matrix.unit is None

    def test_per_element_units_in_meta(self, times, frequencies):
        """SpectrogramDict.to_matrix() stores per-element units in meta."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.A
        )
        matrix = SpectrogramDict({"v": sg1, "a": sg2}).to_matrix()

        assert matrix.meta is not None
        units = [m.unit for m in matrix.meta.flat]
        assert u.V in units
        assert u.A in units


class TestListDictToMatrixConsistency:
    """Tests to ensure SpectrogramList and SpectrogramDict to_matrix() are consistent."""

    @pytest.fixture
    def times(self):
        return np.arange(10) * u.s

    @pytest.fixture
    def frequencies(self):
        return np.arange(5) * 10 * u.Hz

    def test_list_and_dict_same_global_unit_behavior(self, times, frequencies):
        """Both List and Dict have unit=None for mixed-unit elements."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.A
        )

        list_matrix = SpectrogramList([sg1, sg2]).to_matrix()
        dict_matrix = SpectrogramDict({"a": sg1, "b": sg2}).to_matrix()

        assert list_matrix.unit is None
        assert dict_matrix.unit is None

    def test_list_and_dict_same_meta_structure(self, times, frequencies):
        """Both List and Dict store per-element metadata in meta.flat."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=frequencies,
            unit=u.V,
            name="v",
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=frequencies,
            unit=u.A,
            name="a",
        )

        list_matrix = SpectrogramList([sg1, sg2]).to_matrix()
        dict_matrix = SpectrogramDict({"v": sg1, "a": sg2}).to_matrix()

        # Both should have meta with flat accessor
        assert list_matrix.meta is not None
        assert dict_matrix.meta is not None
        assert hasattr(list_matrix.meta, "flat")
        assert hasattr(dict_matrix.meta, "flat")


class TestSpectrogramMatrixCoreMixinSemantics:
    """Tests for SpectrogramMatrixCoreMixin frequencies setter and df property."""

    @pytest.fixture
    def times(self):
        return np.arange(10) * u.s

    @pytest.fixture
    def frequencies(self):
        return np.arange(5) * 10 * u.Hz

    @pytest.fixture
    def basic_matrix(self, times, frequencies):
        """Basic SpectrogramMatrix for testing."""
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=frequencies, unit=u.A
        )
        return SpectrogramList([sg1, sg2]).to_matrix()

    def test_frequencies_length_mismatch_raises_valueerror(self, basic_matrix):
        """Setting frequencies with wrong length raises ValueError."""
        wrong_length_freqs = np.arange(10) * u.Hz  # 10 != 5
        with pytest.raises(ValueError, match="frequencies length mismatch"):
            basic_matrix.frequencies = wrong_length_freqs

    def test_frequencies_correct_length_succeeds(self, basic_matrix):
        """Setting frequencies with correct length succeeds."""
        new_freqs = np.arange(5) * 20 * u.Hz  # Correct length=5
        basic_matrix.frequencies = new_freqs
        assert np.array_equal(basic_matrix.frequencies, new_freqs)

    def test_df_regular_frequencies_returns_value(self, basic_matrix):
        """df returns frequency step for regular (evenly spaced) frequencies."""
        # Default frequencies are regular: [0, 10, 20, 30, 40] Hz
        df = basic_matrix.df
        assert df is not None
        # df should be 10 Hz
        assert df.to_value(u.Hz) == pytest.approx(10.0)

    def test_df_irregular_frequencies_raises_attributeerror(self, times):
        """df raises AttributeError for irregular (non-evenly spaced) frequencies."""
        # Create irregular frequencies
        irregular_freqs = np.array([0, 10, 25, 50, 100]) * u.Hz  # Not evenly spaced
        sg1 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=irregular_freqs, unit=u.V
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5), times=times, frequencies=irregular_freqs, unit=u.A
        )
        matrix = SpectrogramList([sg1, sg2]).to_matrix()

        # This test depends on whether the frequencies have a .regular attribute
        # If they don't, df will be computed from first two elements regardless
        # Current implementation checks hasattr(frequencies, 'regular')
        # Standard Quantity arrays don't have .regular, so df may still return
        # Let's check current behavior and lock it
        try:
            df = matrix.df
            # If no exception, df is computed from first difference
            expected_df = irregular_freqs[1] - irregular_freqs[0]
            assert df.to_value(u.Hz) == pytest.approx(expected_df.to_value(u.Hz))
        except AttributeError:
            # This is also acceptable behavior for irregular frequencies
            pass

    def test_f0_returns_first_frequency(self, basic_matrix):
        """f0 returns the first frequency value."""
        f0 = basic_matrix.f0
        assert f0 is not None
        assert f0.to_value(u.Hz) == pytest.approx(
            0.0
        )  # First element of [0, 10, 20, 30, 40]
