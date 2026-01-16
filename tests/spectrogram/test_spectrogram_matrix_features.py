import numpy as np
import pytest
from astropy import units as u
from astropy.units import UnitConversionError
from gwexpy.spectrogram import SpectrogramMatrix, Spectrogram, SpectrogramList


class TestSpectrogramMatrixFeatures:
    @pytest.fixture
    def sgm_3d_basic(self):
        # Shape: (2, 10, 5) -> (Batch, Time, Freq)
        # Time: 0..9s, Freq: 0..40Hz (5 pts, df=10)
        times = np.arange(10) * u.s
        freqs = np.arange(5) * 10 * u.Hz
        data = np.random.rand(2, 10, 5)
        return SpectrogramMatrix(
            data, 
            times=times, 
            frequencies=freqs, 
            rows=['ch1', 'ch2'], 
            name="TestSGM",
            unit=u.V
        )

    @pytest.fixture
    def sgm_4d_basic(self):
        # Shape: (2, 2, 10, 5) -> (Row, Col, Time, Freq)
        times = np.arange(10)
        freqs = np.arange(5)
        data = np.random.rand(2, 2, 10, 5)
        return SpectrogramMatrix(
            data, 
            times=times, 
            frequencies=freqs, 
            rows=['R1', 'R2'], 
            cols=['C1', 'C2']
        )

    def test_properties_3d(self, sgm_3d_basic):
        """Test basic properties for 3D matrix."""
        assert sgm_3d_basic.ndim == 3
        assert sgm_3d_basic.shape == (2, 10, 5)
        assert np.array_equal(sgm_3d_basic.times.value, np.arange(10))
        assert np.array_equal(sgm_3d_basic.frequencies.value, np.arange(5)*10)
        assert sgm_3d_basic.dt == 1.0 * u.s
        assert sgm_3d_basic.df == 10.0 * u.Hz
        assert sgm_3d_basic.unit == u.V
        assert list(sgm_3d_basic.rows.keys()) == ['ch1', 'ch2']

    def test_slicing_3d(self, sgm_3d_basic):
        """Test slicing on 3D matrix."""
        # 1. Slice batch by index
        sub = sgm_3d_basic[0]
        # Should return a Spectrogram (scalar access in Batch dim usually returns element if 3D SeriesMatrix logic applies?)
        # For SeriesMatrix (Batch, Sample): [0] returns Series.
        # For SpectrogramMatrix (Batch, Time, Freq): [0] returns Spectrogram (Time, Freq).
        assert isinstance(sub, Spectrogram)
        assert sub.shape == (10, 5)
        assert sub.name == "TestSGM" # Inherits name? Or access metadata?
        # Metadata logic in SeriesMatrix creates dummy names 's00' if not provided for elements.
        # But SpectrogramMatrix.__new__ populates rows/cols metadata.
        
        # 2. Slice batch by label
        sub_label = sgm_3d_basic['ch2']
        assert isinstance(sub_label, Spectrogram)
        assert np.all(sub_label.value == sgm_3d_basic[1].value)

        # 3. Slice range
        sub_slice = sgm_3d_basic[0:1] # Returns SpectrogramMatrix (1, 10, 5)
        assert isinstance(sub_slice, SpectrogramMatrix)
        assert sub_slice.shape == (1, 10, 5)
        assert list(sub_slice.row_keys()) == ['ch1']

    def test_slicing_4d(self, sgm_4d_basic):
        """Test slicing on 4D matrix."""
        # 1. Single element Access (Row, Col)
        # Should return Spectrogram
        spec = sgm_4d_basic[0, 1] 
        assert isinstance(spec, Spectrogram)
        assert spec.shape == (10, 5)
        
        # 2. Label access
        spec_lbl = sgm_4d_basic['R2', 'C1']
        assert isinstance(spec_lbl, Spectrogram)
        
        # 3. Partial Slice (Row only) -> Returns SpectrogramMatrix 4D (1, 2, 10, 5) or reduced 3D?
        # Standard numpy slicing: [0] -> (2, 10, 5).
        # SeriesMatrix usually preserves dimensionality if possible or reduces?
        # Let's see what happens.
        
        sub_row = sgm_4d_basic[0] 
        # Numpy behavior: returns subarray of shape (2, 10, 5).
        # For SpectrogramMatrix, this is effectively a 3D matrix (Col, Time, Freq).
        # We assume it downgrades to 3D matrix where "Batch" = "Col".
        assert isinstance(sub_row, SpectrogramMatrix)
        assert sub_row.shape == (2, 10, 5)
        # Verify metadata update
        # If it becomes 3D, 'rows' should become the 'cols' of the original?
        # This behavior depends on how __array_finalize__ and SeriesMatrix handles dim reduction.
        # This is a bit complex. Let's inspect rows.
        # If we sliced row 0, the remaining dims are (Col, T, F). 
        # So new rows should be C1, C2.
        # assert list(sub_row.row_keys()) == ['C1', 'C2'] 
        
    def test_crop_time(self, sgm_3d_basic):
        """Test cropping along time axis (-2)."""
        # Crop 2s to 8s
        cropped = sgm_3d_basic.crop(start=2*u.s, end=8*u.s)
        # Indices: 2..8 (6 points)
        assert cropped.shape == (2, 6, 5)
        assert cropped.times[0] == 2*u.s
        assert cropped.frequencies.shape[0] == 5 # Freqs untouched

    def test_append_time(self, sgm_3d_basic):
        """Test appending along time axis."""
        # Split into two
        part1 = sgm_3d_basic.crop(end=5*u.s)
        part2 = sgm_3d_basic.crop(start=5*u.s)
        
        # Append
        rejoined = part1.append(part2, inplace=False)
        
        assert rejoined.shape == sgm_3d_basic.shape
        assert np.allclose(rejoined.times.value, sgm_3d_basic.times.value)
        assert np.allclose(rejoined.value, sgm_3d_basic.value)
        
    def test_math_stats(self, sgm_3d_basic):
        """Test statistical methods."""
        # Mean over time (axis -2)
        # Note: SeriesMatrix.mean might default to flattening if axis=None.
        # SeriesMatrix inherited mean usually supports axis.
        
        # Axis -2 is time.
        mean_time = sgm_3d_basic.mean(axis=-2) 
        # Result shape: (2, 5) -> (Batch, Freq)
        assert mean_time.shape == (2, 5)
        # Should return FrequencySeriesMatrix-like structure?
        # OR just np.ndarray/SeriesMatrix?
        # Currently stats returns Quantity or array, wrapped in Matrix if possible?
        # StatisticalMethodsMixin.mean returns result of func.
        # func is np.nanmean. returns ndarray/Quantity.
        assert isinstance(mean_time, (u.Quantity, np.ndarray))
        
        # Mean over freq (axis -1)
        mean_freq = sgm_3d_basic.mean(axis=-1)
        assert mean_freq.shape == (2, 10) # (Batch, Time) -> TimeSeriesMatrix-like?
        
    def test_arithmetic(self, sgm_3d_basic):
        """Test arithmetic operations."""
        # Multiply by scalar
        doubled = sgm_3d_basic * 2
        assert np.allclose(doubled.value, sgm_3d_basic.value * 2)
        # Check per-element unit is preserved (new design: units in meta, not global)
        assert doubled.meta[0, 0].unit == u.V
        assert doubled.meta[1, 0].unit == u.V
        
        # Multiply by unit
        volts_sq = sgm_3d_basic * u.V
        # Each element should be V * V = V^2
        assert volts_sq.meta[0, 0].unit.is_equivalent(u.V**2)
        assert volts_sq.meta[1, 0].unit.is_equivalent(u.V**2)
        
        # Add constant
        offset = sgm_3d_basic + 10*u.V
        assert np.allclose(offset.value, sgm_3d_basic.value + 10)


class TestSpectrogramMatrixPerElementUnits:
    """Tests for per-element unit handling in SpectrogramMatrix (3D only)."""

    @pytest.fixture
    def times(self):
        return np.arange(10) * u.s

    @pytest.fixture
    def freqs(self):
        return np.arange(5) * 10 * u.Hz

    def test_to_matrix_succeeds_with_different_units(self, times, freqs):
        """to_matrix() should succeed when units differ across elements."""
        # Create spectrograms with different units
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.V,
            name="voltage",
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.A,  # Different unit!
            name="current",
        )

        # Should succeed without raising
        sgl = SpectrogramList([sg1, sg2])
        matrix = sgl.to_matrix()

        # Verify shape is correct
        assert matrix.ndim == 3
        assert matrix.shape == (2, 10, 5)

    def test_to_matrix_times_frequencies_match(self, times, freqs):
        """matrix.times and matrix.frequencies should equal the input values."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.V,
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.m/u.s,  # Different unit
        )

        matrix = SpectrogramList([sg1, sg2]).to_matrix()

        # Check times
        assert np.array_equal(np.asarray(matrix.times), np.asarray(times))
        # Check frequencies
        assert np.array_equal(np.asarray(matrix.frequencies), np.asarray(freqs))

    def test_to_matrix_per_element_units_accessible(self, times, freqs):
        """Per-element units should be accessible from matrix.meta."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.V,
            name="voltage",
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.Pa,  # Different unit
            name="pressure",
        )
        sg3 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.K,  # Yet another unit
            name="temperature",
        )

        matrix = SpectrogramList([sg1, sg2, sg3]).to_matrix()

        # Access per-element units from meta
        assert matrix.meta is not None
        assert matrix.meta.shape == (3, 1)  # (N, 1) for 3D matrix

        # Verify each element's unit matches the source
        assert matrix.meta[0, 0].unit == u.V
        assert matrix.meta[1, 0].unit == u.Pa
        assert matrix.meta[2, 0].unit == u.K

        # Verify names are preserved
        assert matrix.meta[0, 0].name == "voltage"
        assert matrix.meta[1, 0].name == "pressure"
        assert matrix.meta[2, 0].name == "temperature"

    def test_to_matrix_raises_on_shape_mismatch(self, times, freqs):
        """to_matrix() should still raise on shape mismatch."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
        )
        sg2 = Spectrogram(
            np.random.rand(10, 6),  # Different freq dimension
            times=times,
            frequencies=np.arange(6) * 10 * u.Hz,
        )

        sgl = SpectrogramList([sg1, sg2])
        with pytest.raises(ValueError):
            sgl.to_matrix()

    def test_scalar_op_preserves_per_element_units(self, times, freqs):
        """Scalar operations should preserve per-element units."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.V,
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.m,  # Different unit
        )

        matrix = SpectrogramList([sg1, sg2]).to_matrix()

        # Multiply by scalar
        result = matrix * 2
        assert result.meta[0, 0].unit == u.V
        assert result.meta[1, 0].unit == u.m

        # Multiply by dimensionless
        result2 = matrix * 3.0
        assert result2.meta[0, 0].unit == u.V
        assert result2.meta[1, 0].unit == u.m

    def test_scalar_unit_op_updates_per_element_units(self, times, freqs):
        """Multiplying by a unit should update per-element units appropriately."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.V,
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.m,
        )

        matrix = SpectrogramList([sg1, sg2]).to_matrix()

        # Multiply by unit
        result = matrix * u.s

        # Each element's unit should be multiplied by 's'
        assert result.meta[0, 0].unit.is_equivalent(u.V * u.s)
        assert result.meta[1, 0].unit.is_equivalent(u.m * u.s)

    def test_binary_matrix_op_incompatible_units_raises(self, times, freqs):
        """Binary matrix operations should raise UnitConversionError for incompatible units."""
        sg1_a = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.V,
        )
        sg2_a = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.m,  # Different unit
        )
        matrix_a = SpectrogramList([sg1_a, sg2_a]).to_matrix()

        sg1_b = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.V,  # Compatible with first element of A
        )
        sg2_b = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.K,  # Incompatible with second element of A (m vs K)
        )
        matrix_b = SpectrogramList([sg1_b, sg2_b]).to_matrix()

        # Addition should fail due to incompatible second elements (m vs K)
        with pytest.raises(UnitConversionError):
            _ = matrix_a + matrix_b

    def test_binary_matrix_op_compatible_units_raises_strict_equality(self, times, freqs):
        """Binary add/sub requires strict unit equality - compatible but different units fail.

        Following SeriesMatrix check_add_sub_compatibility: V != mV and m != cm
        should raise UnitConversionError even though they are equivalent.
        """
        sg1_a = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.V,
        )
        sg2_a = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.m,
        )
        matrix_a = SpectrogramList([sg1_a, sg2_a]).to_matrix()

        sg1_b = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.mV,  # Compatible with V but NOT equal
        )
        sg2_b = Spectrogram(
            np.random.rand(10, 5),
            times=times,
            frequencies=freqs,
            unit=u.cm,  # Compatible with m but NOT equal
        )
        matrix_b = SpectrogramList([sg1_b, sg2_b]).to_matrix()

        # Addition should FAIL because V != mV (strict equality required)
        with pytest.raises(UnitConversionError):
            _ = matrix_a + matrix_b

    def test_binary_mul_updates_per_element_units(self, times, freqs):
        """Multiplication between matrices should update per-element units correctly."""
        sg1_a = Spectrogram(
            np.ones((10, 5)),
            times=times,
            frequencies=freqs,
            unit=u.V,
        )
        sg2_a = Spectrogram(
            np.ones((10, 5)),
            times=times,
            frequencies=freqs,
            unit=u.m,
        )
        matrix_a = SpectrogramList([sg1_a, sg2_a]).to_matrix()

        sg1_b = Spectrogram(
            np.ones((10, 5)) * 2,
            times=times,
            frequencies=freqs,
            unit=u.A,
        )
        sg2_b = Spectrogram(
            np.ones((10, 5)) * 2,
            times=times,
            frequencies=freqs,
            unit=u.s,
        )
        matrix_b = SpectrogramList([sg1_b, sg2_b]).to_matrix()

        result = matrix_a * matrix_b

        # Units should be multiplied element-wise
        assert result.meta[0, 0].unit.is_equivalent(u.V * u.A)
        assert result.meta[1, 0].unit.is_equivalent(u.m * u.s)


class TestSpectrogramMatrixSeriesMatrixRules:
    """Tests for SeriesMatrix base rules compliance in SpectrogramMatrix."""

    @pytest.fixture
    def times_s(self):
        """Times in seconds: [0, 1, 2, ..., 9] s"""
        return np.arange(10) * u.s

    @pytest.fixture
    def times_ms(self):
        """Times in milliseconds equivalent to [0, 1, 2, ..., 9] s"""
        return np.arange(10) * 1000 * u.ms

    @pytest.fixture
    def freqs(self):
        return np.arange(5) * 10 * u.Hz

    def test_to_matrix_axes_strict_but_unit_flexible_different_value_units(
        self, times_s, freqs
    ):
        """to_matrix() succeeds with same times/frequencies but different value units."""
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times_s,
            frequencies=freqs,
            unit=u.V,
            name="voltage",
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times_s,
            frequencies=freqs,
            unit=u.Pa,  # Different unit
            name="pressure",
        )

        matrix = SpectrogramList([sg1, sg2]).to_matrix()
        assert matrix.shape == (2, 10, 5)
        assert matrix.meta[0, 0].unit == u.V
        assert matrix.meta[1, 0].unit == u.Pa

    def test_to_matrix_axes_convertible_units_succeed(self, times_s, times_ms, freqs):
        """to_matrix() succeeds when axis units differ but convert to equal values.

        [0, 1, 2, ..., 9] s == [0, 1000, 2000, ..., 9000] ms after conversion to s.
        """
        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times_s,  # [0, 1, ..., 9] s
            frequencies=freqs,
            unit=u.V,
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times_ms,  # [0, 1000, ..., 9000] ms -> [0, 1, ..., 9] s
            frequencies=freqs,
            unit=u.A,
        )

        matrix = SpectrogramList([sg1, sg2]).to_matrix()
        assert matrix.shape == (2, 10, 5)
        # Axes from first element
        assert np.array_equal(matrix.times.to_value(u.s), np.arange(10))

    def test_to_matrix_axes_different_values_raises(self, freqs):
        """to_matrix() raises ValueError when axis values differ after conversion."""
        times1 = np.arange(10) * u.s  # [0, 1, ..., 9] s
        times2 = np.arange(10) * 500 * u.ms  # [0, 500, ..., 4500] ms != times1

        sg1 = Spectrogram(
            np.random.rand(10, 5),
            times=times1,
            frequencies=freqs,
            unit=u.V,
        )
        sg2 = Spectrogram(
            np.random.rand(10, 5),
            times=times2,
            frequencies=freqs,
            unit=u.V,
        )

        with pytest.raises(ValueError):
            SpectrogramList([sg1, sg2]).to_matrix()

    def test_add_sub_requires_exact_unit_match_m_vs_cm_raises(self, times_s, freqs):
        """Addition fails when per-cell units are m vs cm (even though compatible).

        Following SeriesMatrix check_add_sub_compatibility: u0 != uk raises.
        """
        sg1 = Spectrogram(
            np.ones((10, 5)),
            times=times_s,
            frequencies=freqs,
            unit=u.m,
        )
        sg2 = Spectrogram(
            np.ones((10, 5)),
            times=times_s,
            frequencies=freqs,
            unit=u.m,
        )
        matrix_a = SpectrogramList([sg1, sg2]).to_matrix()

        sg3 = Spectrogram(
            np.ones((10, 5)),
            times=times_s,
            frequencies=freqs,
            unit=u.cm,  # Compatible with m, but NOT equal
        )
        sg4 = Spectrogram(
            np.ones((10, 5)),
            times=times_s,
            frequencies=freqs,
            unit=u.m,  # Same as matrix_a[1]
        )
        matrix_b = SpectrogramList([sg3, sg4]).to_matrix()

        # First element: m vs cm -> NOT equal -> must raise
        with pytest.raises(UnitConversionError):
            _ = matrix_a + matrix_b

    def test_add_sub_exact_unit_match_succeeds(self, times_s, freqs):
        """Addition succeeds when per-cell units are exactly equal."""
        sg1 = Spectrogram(
            np.ones((10, 5)),
            times=times_s,
            frequencies=freqs,
            unit=u.V,
        )
        sg2 = Spectrogram(
            np.ones((10, 5)),
            times=times_s,
            frequencies=freqs,
            unit=u.m,
        )
        matrix_a = SpectrogramList([sg1, sg2]).to_matrix()

        sg3 = Spectrogram(
            np.ones((10, 5)) * 2,
            times=times_s,
            frequencies=freqs,
            unit=u.V,  # Exactly equal to matrix_a[0]
        )
        sg4 = Spectrogram(
            np.ones((10, 5)) * 2,
            times=times_s,
            frequencies=freqs,
            unit=u.m,  # Exactly equal to matrix_a[1]
        )
        matrix_b = SpectrogramList([sg3, sg4]).to_matrix()

        result = matrix_a + matrix_b
        assert result.shape == matrix_a.shape
        assert result.meta[0, 0].unit == u.V
        assert result.meta[1, 0].unit == u.m

    def test_scalar_mul_preserves_per_element_units_no_global(self, times_s, freqs):
        """Scalar multiplication preserves per-element units without global unit assumption."""
        sg1 = Spectrogram(
            np.ones((10, 5)),
            times=times_s,
            frequencies=freqs,
            unit=u.V,
        )
        sg2 = Spectrogram(
            np.ones((10, 5)),
            times=times_s,
            frequencies=freqs,
            unit=u.Pa,
        )
        matrix = SpectrogramList([sg1, sg2]).to_matrix()

        result = matrix * 2

        # Per-element units preserved
        assert result.meta[0, 0].unit == u.V
        assert result.meta[1, 0].unit == u.Pa
        # No single global unit (None or not checked)
        assert result.unit is None
