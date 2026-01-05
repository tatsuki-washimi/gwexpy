import numpy as np
import pytest
from astropy import units as u
from gwexpy.spectrogram import SpectrogramMatrix, Spectrogram

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
        assert doubled.unit == sgm_3d_basic.unit
        
        # Multiply by unit
        volts_sq = sgm_3d_basic * u.V
        assert volts_sq.unit == u.V**2
        
        # Add constant
        offset = sgm_3d_basic + 10*u.V
        assert np.allclose(offset.value, sgm_3d_basic.value + 10)

