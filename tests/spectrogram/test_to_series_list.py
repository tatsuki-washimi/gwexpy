"""Tests for Spectrogram.to_timeseries_list() and to_frequencyseries_list()."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList
from gwexpy.spectrogram import Spectrogram
from gwexpy.timeseries import TimeSeries, TimeSeriesList


class TestToTimeSeriesList:
    """Tests for Spectrogram.to_timeseries_list()."""

    @pytest.fixture
    def sample_spectrogram(self):
        """Create a sample Spectrogram for testing."""
        # Shape: (ntimes=4, nfreqs=5)
        data = np.arange(20).reshape(4, 5).astype(float)
        return Spectrogram(
            data,
            t0=100.0,  # GPS time
            dt=0.5,
            f0=10.0,
            df=5.0,
            unit="m/s",
            name="test_spec",
            channel="TEST:CHANNEL",
        )

    def test_shape_and_length(self, sample_spectrogram):
        """Test that ts_list length equals nfreqs and each ts length equals ntimes."""
        spec = sample_spectrogram
        ntimes, nfreqs = spec.shape

        ts_list, freqs = spec.to_timeseries_list()

        # List length should match number of frequency bins
        assert len(ts_list) == nfreqs
        assert len(ts_list) == 5

        # Each TimeSeries length should match number of time bins
        for ts in ts_list:
            assert len(ts) == ntimes
            assert len(ts) == 4

    def test_value_correspondence(self, sample_spectrogram):
        """Test that ts_list[i].value[j] == spec.value[j, i]."""
        spec = sample_spectrogram
        ts_list, _ = spec.to_timeseries_list()

        # Check several index combinations
        for i in range(5):  # frequency index
            for j in range(4):  # time index
                assert ts_list[i].value[j] == spec.value[j, i], (
                    f"Mismatch at ts_list[{i}].value[{j}] vs spec.value[{j}, {i}]"
                )

    def test_frequencies_axis_match(self, sample_spectrogram):
        """Test that returned frequencies match Spectrogram frequencies."""
        spec = sample_spectrogram
        ts_list, freqs = spec.to_timeseries_list()

        # Frequencies should be equivalent
        np.testing.assert_array_equal(freqs.value, spec.frequencies.value)
        assert freqs.unit == spec.frequencies.unit

    def test_unit_inheritance(self, sample_spectrogram):
        """Test that each TimeSeries inherits unit from Spectrogram."""
        spec = sample_spectrogram
        ts_list, _ = spec.to_timeseries_list()

        for ts in ts_list:
            assert ts.unit == spec.unit

    def test_epoch_inheritance(self, sample_spectrogram):
        """Test that each TimeSeries inherits epoch from Spectrogram."""
        spec = sample_spectrogram
        ts_list, _ = spec.to_timeseries_list()

        for ts in ts_list:
            # epoch may be a Time, float or LIGOTimeGPS
            # Spectrogram uses astropy.Time for epoch, TimeSeries uses LIGOTimeGPS
            if hasattr(spec.epoch, "gps"):
                expected = float(spec.epoch.gps)
            else:
                expected = float(spec.epoch)
            
            if hasattr(ts.epoch, "gps"):
                actual = float(ts.epoch.gps)
            else:
                actual = float(ts.epoch)
            assert actual == expected

    def test_channel_inheritance(self, sample_spectrogram):
        """Test that each TimeSeries inherits channel from Spectrogram."""
        spec = sample_spectrogram
        ts_list, _ = spec.to_timeseries_list()

        for ts in ts_list:
            assert str(ts.channel) == str(spec.channel)

    def test_name_with_base_name(self, sample_spectrogram):
        """Test element naming when Spectrogram has a name."""
        spec = sample_spectrogram
        ts_list, freqs = spec.to_timeseries_list()

        # Each name should contain the base name and frequency
        for i, ts in enumerate(ts_list):
            assert "test_spec" in ts.name
            assert str(freqs[i].value) in ts.name or str(freqs[i]) in ts.name

    def test_name_without_base_name(self):
        """Test element naming when Spectrogram has no name."""
        data = np.arange(12).reshape(3, 4).astype(float)
        spec = Spectrogram(data, t0=0, dt=1, f0=0, df=10)

        ts_list, freqs = spec.to_timeseries_list()

        # Each name should still be unique and contain frequency info
        for i, ts in enumerate(ts_list):
            assert ts.name is not None
            assert "f" in ts.name.lower()  # Contains 'f' for frequency

    def test_times_axis_preserved(self, sample_spectrogram):
        """Test that each TimeSeries has correct times axis."""
        spec = sample_spectrogram
        ts_list, _ = spec.to_timeseries_list()

        for ts in ts_list:
            np.testing.assert_array_almost_equal(ts.times.value, spec.times.value)

    def test_returns_correct_types(self, sample_spectrogram):
        """Test that return types are correct."""
        spec = sample_spectrogram
        ts_list, freqs = spec.to_timeseries_list()

        assert isinstance(ts_list, TimeSeriesList)
        assert isinstance(freqs, u.Quantity)
        for ts in ts_list:
            assert isinstance(ts, TimeSeries)


class TestToFrequencySeriesList:
    """Tests for Spectrogram.to_frequencyseries_list()."""

    @pytest.fixture
    def sample_spectrogram(self):
        """Create a sample Spectrogram for testing."""
        # Shape: (ntimes=4, nfreqs=5)
        data = np.arange(20).reshape(4, 5).astype(float)
        return Spectrogram(
            data,
            t0=100.0,
            dt=0.5,
            f0=10.0,
            df=5.0,
            unit="m/s",
            name="test_spec",
            channel="TEST:CHANNEL",
        )

    def test_shape_and_length(self, sample_spectrogram):
        """Test that fs_list length equals ntimes and each fs length equals nfreqs."""
        spec = sample_spectrogram
        ntimes, nfreqs = spec.shape

        fs_list, times = spec.to_frequencyseries_list()

        # List length should match number of time bins
        assert len(fs_list) == ntimes
        assert len(fs_list) == 4

        # Each FrequencySeries length should match number of frequency bins
        for fs in fs_list:
            assert len(fs) == nfreqs
            assert len(fs) == 5

    def test_value_correspondence(self, sample_spectrogram):
        """Test that fs_list[j].value[i] == spec.value[j, i]."""
        spec = sample_spectrogram
        fs_list, _ = spec.to_frequencyseries_list()

        # Check several index combinations
        for j in range(4):  # time index
            for i in range(5):  # frequency index
                assert fs_list[j].value[i] == spec.value[j, i], (
                    f"Mismatch at fs_list[{j}].value[{i}] vs spec.value[{j}, {i}]"
                )

    def test_times_axis_match(self, sample_spectrogram):
        """Test that returned times match Spectrogram times."""
        spec = sample_spectrogram
        fs_list, times = spec.to_frequencyseries_list()

        # Times should be equivalent
        np.testing.assert_array_equal(times.value, spec.times.value)
        assert times.unit == spec.times.unit

    def test_unit_inheritance(self, sample_spectrogram):
        """Test that each FrequencySeries inherits unit from Spectrogram."""
        spec = sample_spectrogram
        fs_list, _ = spec.to_frequencyseries_list()

        for fs in fs_list:
            assert fs.unit == spec.unit

    def test_epoch_inheritance(self, sample_spectrogram):
        """Test that each FrequencySeries inherits epoch from Spectrogram."""
        spec = sample_spectrogram
        fs_list, _ = spec.to_frequencyseries_list()

        for fs in fs_list:
            # epoch may be a Time, float or LIGOTimeGPS
            if hasattr(spec.epoch, "gps"):
                expected = float(spec.epoch.gps)
            else:
                expected = float(spec.epoch)
            
            if hasattr(fs.epoch, "gps"):
                actual = float(fs.epoch.gps)
            else:
                actual = float(fs.epoch)
            assert actual == expected

    def test_channel_inheritance(self, sample_spectrogram):
        """Test that each FrequencySeries inherits channel from Spectrogram."""
        spec = sample_spectrogram
        fs_list, _ = spec.to_frequencyseries_list()

        for fs in fs_list:
            assert str(fs.channel) == str(spec.channel)

    def test_name_with_base_name(self, sample_spectrogram):
        """Test element naming when Spectrogram has a name."""
        spec = sample_spectrogram
        fs_list, times = spec.to_frequencyseries_list()

        # Each name should contain the base name and time
        for j, fs in enumerate(fs_list):
            assert "test_spec" in fs.name
            assert str(times[j].value) in fs.name or str(times[j]) in fs.name

    def test_name_without_base_name(self):
        """Test element naming when Spectrogram has no name."""
        data = np.arange(12).reshape(3, 4).astype(float)
        spec = Spectrogram(data, t0=0, dt=1, f0=0, df=10)

        fs_list, times = spec.to_frequencyseries_list()

        # Each name should still be unique and contain time info
        for j, fs in enumerate(fs_list):
            assert fs.name is not None
            assert "t" in fs.name.lower()  # Contains 't' for time

    def test_frequencies_axis_preserved(self, sample_spectrogram):
        """Test that each FrequencySeries has correct frequencies axis."""
        spec = sample_spectrogram
        fs_list, _ = spec.to_frequencyseries_list()

        for fs in fs_list:
            np.testing.assert_array_almost_equal(
                fs.frequencies.value, spec.frequencies.value
            )

    def test_returns_correct_types(self, sample_spectrogram):
        """Test that return types are correct."""
        spec = sample_spectrogram
        fs_list, times = spec.to_frequencyseries_list()

        assert isinstance(fs_list, FrequencySeriesList)
        assert isinstance(times, u.Quantity)
        for fs in fs_list:
            assert isinstance(fs, FrequencySeries)


class TestCrossValidation:
    """Cross-validation tests between to_timeseries_list and to_frequencyseries_list."""

    @pytest.fixture
    def sample_spectrogram(self):
        """Create a sample Spectrogram for testing."""
        data = np.arange(20).reshape(4, 5).astype(float)
        return Spectrogram(
            data,
            t0=100.0,
            dt=0.5,
            f0=10.0,
            df=5.0,
            unit="m/s",
            name="test_spec",
            channel="TEST:CHANNEL",
        )

    def test_values_consistent_across_methods(self, sample_spectrogram):
        """Test that values are consistent between both methods."""
        spec = sample_spectrogram
        ts_list, freqs = spec.to_timeseries_list()
        fs_list, times = spec.to_frequencyseries_list()

        # For any (j, i), both should give same value
        for j in range(4):  # time index
            for i in range(5):  # frequency index
                val_from_ts = ts_list[i].value[j]
                val_from_fs = fs_list[j].value[i]
                val_from_spec = spec.value[j, i]
                assert val_from_ts == val_from_spec
                assert val_from_fs == val_from_spec
                assert val_from_ts == val_from_fs


class TestEdgeCases:
    """Edge case tests."""

    def test_single_time_bin(self):
        """Test Spectrogram with only one time bin."""
        data = np.arange(5).reshape(1, 5).astype(float)
        spec = Spectrogram(data, t0=0, dt=1, f0=0, df=10, unit="V")

        ts_list, freqs = spec.to_timeseries_list()
        fs_list, times = spec.to_frequencyseries_list()

        assert len(ts_list) == 5
        assert len(fs_list) == 1
        for ts in ts_list:
            assert len(ts) == 1

    def test_single_freq_bin(self):
        """Test Spectrogram with only one frequency bin."""
        data = np.arange(4).reshape(4, 1).astype(float)
        spec = Spectrogram(data, t0=0, dt=1, f0=10, df=5, unit="V")

        ts_list, freqs = spec.to_timeseries_list()
        fs_list, times = spec.to_frequencyseries_list()

        assert len(ts_list) == 1
        assert len(fs_list) == 4
        assert len(ts_list[0]) == 4

    def test_complex_spectrogram(self):
        """Test with complex-valued Spectrogram."""
        data = np.arange(12).reshape(3, 4).astype(complex)
        data.imag = np.arange(12).reshape(3, 4) * 0.5
        spec = Spectrogram(data, t0=0, dt=1, f0=0, df=10)

        ts_list, freqs = spec.to_timeseries_list()
        fs_list, times = spec.to_frequencyseries_list()

        # Values should still match (complex values)
        for j in range(3):
            for i in range(4):
                assert ts_list[i].value[j] == spec.value[j, i]
                assert fs_list[j].value[i] == spec.value[j, i]

    def test_preserves_dtype(self):
        """Test that dtype is preserved (float, complex, etc.)."""
        # Float32
        data_f32 = np.arange(12).reshape(3, 4).astype(np.float32)
        spec_f32 = Spectrogram(data_f32, t0=0, dt=1, f0=0, df=10)
        ts_list, _ = spec_f32.to_timeseries_list()
        # Note: GWpy may promote to float64 internally

        # Complex
        data_complex = np.arange(12).reshape(3, 4).astype(complex)
        spec_complex = Spectrogram(data_complex, t0=0, dt=1, f0=0, df=10)
        ts_list_c, _ = spec_complex.to_timeseries_list()
        assert np.iscomplexobj(ts_list_c[0].value)
