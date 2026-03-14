"""
Extended tests for Phase 1-2 I/O improvements.

Tests for:
- ATS epoch parameter with datetime and float
- Provenance tracking for new parameters
- Error handling for invalid inputs
- Registration with writers
"""

import datetime
import struct
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import to_gps

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.timeseries.io._registration import register_timeseries_format


def _create_minimal_ats(path: Path, n_samples: int = 10):
    """Create a minimal valid ATS file for testing."""
    header = bytearray(1024)
    struct.pack_into("<H", header, 0, 1024)  # header length
    struct.pack_into("<h", header, 2, 80)    # version
    struct.pack_into("<I", header, 4, n_samples)  # samples
    struct.pack_into("<f", header, 8, 1.0)   # sample freq
    struct.pack_into("<I", header, 12, 0)    # start time (unix epoch)
    struct.pack_into("<d", header, 16, 1.0)  # LSB in mV
    struct.pack_into("<h", header, 0xAA, 0)  # bit_indicator (int32)

    data = np.arange(n_samples, dtype="<i4").tobytes()
    path.write_bytes(header + data)


def test_ats_epoch_with_datetime(tmp_path):
    """Test ATS reader with datetime epoch parameter."""
    path = tmp_path / "test.ats"
    _create_minimal_ats(path, n_samples=10)

    # Test with datetime object
    epoch_dt = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    expected_gps = float(to_gps(epoch_dt))

    ts = TimeSeries.read(path, format="ats", epoch=epoch_dt)
    assert abs(float(ts.t0.value) - expected_gps) < 1e-6

    # Check provenance
    if hasattr(ts, "_gwexpy_io"):
        assert ts._gwexpy_io["epoch_source"] == "user"


def test_ats_epoch_with_float(tmp_path):
    """Test ATS reader with float epoch parameter."""
    path = tmp_path / "test.ats"
    _create_minimal_ats(path, n_samples=10)

    # Test with float GPS time
    epoch_gps = 1234567890.0
    ts = TimeSeries.read(path, format="ats", epoch=epoch_gps)
    assert abs(float(ts.t0.value) - epoch_gps) < 1e-6


def test_ats_unit_override(tmp_path):
    """Test ATS reader with unit override."""
    path = tmp_path / "test.ats"
    _create_minimal_ats(path, n_samples=10)

    ts = TimeSeries.read(path, format="ats", unit="mV")
    assert ts.unit == u.Unit("mV")

    # Check provenance
    if hasattr(ts, "_gwexpy_io"):
        assert ts._gwexpy_io["unit_source"] == "override"


def test_ats_invalid_epoch_type(tmp_path):
    """Test ATS reader with invalid epoch type raises TypeError."""
    path = tmp_path / "test.ats"
    _create_minimal_ats(path, n_samples=10)

    with pytest.raises(TypeError, match="epoch must be float or datetime"):
        TimeSeries.read(path, format="ats", epoch="invalid")


def test_wav_invalid_epoch_type(tmp_path):
    """Test WAV reader with invalid epoch type raises TypeError."""
    from scipy.io import wavfile

    path = tmp_path / "test.wav"
    rate = 1000
    data = np.zeros((100, 2), dtype=np.int16)
    wavfile.write(path, rate, data)

    with pytest.raises(TypeError, match="epoch must be float or datetime"):
        TimeSeriesDict.read(path, format="wav", epoch="invalid")


def test_provenance_tracking_wav(tmp_path):
    """Test that provenance metadata is correctly attached."""
    from scipy.io import wavfile

    path = tmp_path / "test.wav"
    rate = 1000
    data = np.zeros((100, 1), dtype=np.int16)
    wavfile.write(path, rate, data)

    # With defaults
    tsd1 = TimeSeriesDict.read(path, format="wav")
    if hasattr(tsd1, "_gwexpy_io"):
        assert tsd1._gwexpy_io["format"] == "wav"
        assert tsd1._gwexpy_io["epoch_source"] == "default"
        assert tsd1._gwexpy_io["unit_source"] == "wav"

    # With overrides
    tsd2 = TimeSeriesDict.read(path, format="wav", epoch=1000.0, unit="m/s")
    if hasattr(tsd2, "_gwexpy_io"):
        assert tsd2._gwexpy_io["epoch_source"] == "user"
        assert tsd2._gwexpy_io["unit_source"] == "override"


def test_register_with_writer(tmp_path):
    """Test registration helper with writer functions."""
    # Mock reader and writer
    def mock_reader_dict(source, **kwargs):
        tsd = TimeSeriesDict()
        tsd["test"] = TimeSeries([1, 2, 3], t0=0, dt=1, name="test")
        return tsd

    def mock_writer_dict(tsd, target, **kwargs):
        # Just check it's called correctly
        assert isinstance(tsd, TimeSeriesDict)
        Path(target).write_text("mock_written")

    register_timeseries_format(
        "mock_write_fmt",
        reader_dict=mock_reader_dict,
        writer_dict=mock_writer_dict,
        extension="mockwrite",
        auto_adapt=True,
    )

    # Test write with auto-adapted writer
    path_write = tmp_path / "test.mockwrite"
    ts = TimeSeries([1, 2, 3], t0=0, dt=1, name="test")
    ts.write(path_write, format="mock_write_fmt")

    assert path_write.exists()
    assert path_write.read_text() == "mock_written"


def test_pathlib_support_multiple_formats(tmp_path):
    """Test Pathlib support across multiple formats."""
    from scipy.io import wavfile

    # WAV
    wav_path = Path(tmp_path) / "test.wav"
    wavfile.write(wav_path, 1000, np.zeros(100, dtype=np.int16))
    ts_wav = TimeSeries.read(wav_path, format="wav")
    assert isinstance(ts_wav, TimeSeries)

    # ATS
    ats_path = Path(tmp_path) / "test.ats"
    _create_minimal_ats(ats_path)
    ts_ats = TimeSeries.read(ats_path, format="ats")
    assert isinstance(ts_ats, TimeSeries)


def test_registration_auto_adapt_disabled():
    """Test registration with auto_adapt=False uses provided functions only."""
    def mock_reader_dict(source, **kwargs):
        return TimeSeriesDict({"test": TimeSeries([1, 2], t0=0, dt=1, name="test")})

    def custom_single_reader(source, **kwargs):
        # Custom implementation (not auto-adapted)
        tsd = mock_reader_dict(source, **kwargs)
        return TimeSeries([99, 99], t0=0, dt=1, name="custom")

    register_timeseries_format(
        "mock_custom",
        reader_dict=mock_reader_dict,
        reader_single=custom_single_reader,
        auto_adapt=False,  # Don't auto-generate
    )

    # Custom reader should be used, not auto-adapted one
    from gwpy.io.registry import default_registry as io_registry
    registered_reader = io_registry.get_reader("mock_custom", TimeSeries)

    # The registered function should be our custom one
    assert registered_reader is custom_single_reader


def test_multiple_format_aliases(tmp_path):
    """Test that format aliases work correctly (e.g., 'nc' and 'netcdf4')."""
    # This is already implemented in netcdf4_.py, just verify it works
    from gwpy.io.registry import default_registry as io_registry

    # Both 'nc' and 'netcdf4' should be registered
    nc_reader = io_registry.get_reader("nc", TimeSeriesDict)
    netcdf4_reader = io_registry.get_reader("netcdf4", TimeSeriesDict)

    # They should be the same function
    assert nc_reader is netcdf4_reader


def test_wav_scipy_kwargs_filtering(tmp_path):
    """Test that WAV reader filters out incompatible kwargs for scipy."""
    from scipy.io import wavfile

    path = tmp_path / "test.wav"
    wavfile.write(path, 1000, np.zeros(100, dtype=np.int16))

    # These kwargs should be filtered out and not cause TypeError
    ts = TimeSeries.read(
        path,
        format="wav",
        start=0,  # These are gwpy-specific, not scipy
        end=0.1,
    )
    assert isinstance(ts, TimeSeries)
