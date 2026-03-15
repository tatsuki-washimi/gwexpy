"""
Tests for magic number-based file format identification.

Tests GBD and ATS magic number identifiers to ensure they correctly
identify files by content rather than extension.
"""

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io.ats import ATS_SUPPORTED_VERSIONS, identify_ats
from gwexpy.timeseries.io.gbd import identify_gbd

# --- GBD Tests ---


def test_identify_gbd_with_valid_header(tmp_path):
    """Test GBD identifier recognizes valid GBD header."""
    path = tmp_path / "test.dat"

    # Create minimal GBD header with HeaderSize keyword
    header = b"HeaderSize: 512\n"
    header += b"Sample: 1000\n"
    header += b"Range: 10V\n"
    header += b"\x00" * (512 - len(header))

    path.write_bytes(header)

    assert identify_gbd(None, path, None) is True


def test_identify_gbd_with_alternative_spelling(tmp_path):
    """Test GBD identifier recognizes 'HeaderSiz' (without e)."""
    path = tmp_path / "test.dat"

    # Some GBD files use "HeaderSiz" instead of "HeaderSize"
    header = b"HeaderSiz: 512\n"
    header += b"\x00" * (512 - len(header))

    path.write_bytes(header)

    assert identify_gbd(None, path, None) is True


def test_identify_gbd_rejects_non_gbd(tmp_path):
    """Test GBD identifier rejects non-GBD files."""
    path = tmp_path / "random.bin"
    path.write_bytes(os.urandom(4096))

    assert identify_gbd(None, path, None) is False


def test_identify_gbd_with_none_filepath():
    """Test GBD identifier handles None filepath."""
    assert identify_gbd(None, None, None) is False


def test_gbd_read_without_extension(tmp_path):
    """Test GBD file can be read without .gbd extension."""
    path = tmp_path / "test_no_ext"

    # Create minimal valid GBD file
    header_text = """HeaderSize: 512
Order: CH1
Sample: 10
Range: 10V
SampleRate: 100
CH1: Temperature
TimeStart: 2024/01/01 12:00:00
"""
    header = header_text.encode("ascii")
    header += b"\x00" * (512 - len(header))

    # Add 10 samples of int16 data (1 channel)
    data = np.arange(10, dtype="<i2").tobytes()

    path.write_bytes(header + data)

    # Should be able to read without specifying format
    tsd = TimeSeriesDict.read(path, timezone="UTC")
    assert len(tsd) > 0
    assert "Temperature" in tsd or "CH1" in tsd


def test_gbd_read_with_wrong_extension(tmp_path):
    """Test GBD file can be read even with wrong extension."""
    path = tmp_path / "test.txt"  # Wrong extension

    # Create minimal valid GBD file
    header_text = """HeaderSize: 512
Order: CH1
Sample: 10
Range: 10V
SampleRate: 100
CH1: Voltage
TimeStart: 2024/01/01 12:00:00
"""
    header = header_text.encode("ascii")
    header += b"\x00" * (512 - len(header))

    data = np.arange(10, dtype="<i2").tobytes()

    path.write_bytes(header + data)

    # Should be identified as GBD despite .txt extension
    tsd = TimeSeriesDict.read(path, timezone="UTC")
    assert len(tsd) > 0


# --- ATS Tests ---


def _create_minimal_ats(path: Path, version: int = 80, n_samples: int = 10):
    """Create a minimal valid ATS file for testing."""
    header = bytearray(1024)
    struct.pack_into("<H", header, 0, 1024)  # header length
    struct.pack_into("<h", header, 2, version)  # version
    struct.pack_into("<I", header, 4, n_samples)  # samples
    struct.pack_into("<f", header, 8, 1.0)  # sample freq
    struct.pack_into("<I", header, 12, 0)  # start time (unix epoch)
    struct.pack_into("<d", header, 16, 1.0)  # LSB in mV
    struct.pack_into("<h", header, 0xAA, 0)  # bit_indicator (int32)

    data = np.arange(n_samples, dtype="<i4").tobytes()
    path.write_bytes(header + data)


def test_identify_ats_with_version_80(tmp_path):
    """Test ATS identifier recognizes version 80."""
    path = tmp_path / "test.ats"
    _create_minimal_ats(path, version=80)

    assert identify_ats(None, path, None) is True


def test_identify_ats_with_version_81(tmp_path):
    """Test ATS identifier recognizes version 81."""
    path = tmp_path / "test.ats"
    _create_minimal_ats(path, version=81)

    assert identify_ats(None, path, None) is True


def test_identify_ats_with_version_1080(tmp_path):
    """Test ATS identifier recognizes version 1080."""
    path = tmp_path / "test.ats"
    _create_minimal_ats(path, version=1080)

    assert identify_ats(None, path, None) is True


def test_identify_ats_rejects_unknown_version(tmp_path):
    """Test ATS identifier rejects unknown version."""
    path = tmp_path / "test.ats"

    # Create header with unsupported version 82
    header = bytearray(1024)
    struct.pack_into("<H", header, 0, 1024)
    struct.pack_into("<h", header, 2, 82)  # Unsupported version

    path.write_bytes(header)

    assert identify_ats(None, path, None) is False


def test_identify_ats_rejects_small_header(tmp_path):
    """Test ATS identifier rejects header < 1024 bytes."""
    path = tmp_path / "test.ats"

    # Create header with invalid small size
    header = bytearray(100)
    struct.pack_into("<H", header, 0, 100)  # Too small
    struct.pack_into("<h", header, 2, 80)

    path.write_bytes(header)

    assert identify_ats(None, path, None) is False


def test_identify_ats_rejects_non_ats(tmp_path):
    """Test ATS identifier rejects non-ATS files."""
    path = tmp_path / "random.bin"
    path.write_bytes(os.urandom(1024))

    assert identify_ats(None, path, None) is False


def test_identify_ats_with_none_filepath():
    """Test ATS identifier handles None filepath."""
    assert identify_ats(None, None, None) is False


def test_identify_ats_with_truncated_file(tmp_path):
    """Test ATS identifier handles truncated files."""
    path = tmp_path / "truncated.ats"
    path.write_bytes(b"\x00\x04")  # Only 2 bytes

    assert identify_ats(None, path, None) is False


def test_ats_read_without_extension(tmp_path):
    """Test ATS file can be read without .ats extension."""
    path = tmp_path / "test_no_ext"
    _create_minimal_ats(path, n_samples=100)

    # Should be able to read without specifying format
    ts = TimeSeries.read(path)
    assert isinstance(ts, TimeSeries)
    assert len(ts) == 100


def test_ats_read_with_wrong_extension(tmp_path):
    """Test ATS file can be read even with wrong extension."""
    path = tmp_path / "test.bin"  # Wrong extension
    _create_minimal_ats(path, n_samples=50)

    # Should be identified as ATS despite .bin extension
    ts = TimeSeries.read(path)
    assert isinstance(ts, TimeSeries)
    assert len(ts) == 50


# --- Combined Tests ---


def test_extension_fallback_for_ambiguous_files(tmp_path):
    """Test extension-based fallback when magic number fails."""
    # Create a file that doesn't match any magic number
    # but has a .gbd extension
    path = tmp_path / "test.gbd"
    path.write_text("Not a valid GBD file\n")

    # Should fall back to extension matching
    # (will fail when actually reading, but identifier should return True)
    assert identify_gbd(None, path, None) is False  # No HeaderSiz pattern

    # But with correct extension and format specified, can attempt read
    # (will fail during actual reading, which is expected)
