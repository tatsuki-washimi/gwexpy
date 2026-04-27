"""
Tests for audio metadata extraction functionality.

Tests the extract_metadata parameter in audio.py and wav.py readers,
including tinytag integration and error handling.
"""

import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from gwexpy.io.utils import extract_audio_metadata
from gwexpy.timeseries import TimeSeries, TimeSeriesDict
from gwexpy.timeseries.io.audio import read_timeseriesdict_audio
from gwexpy.timeseries.io.wav import read_timeseriesdict_wav

# Check if tinytag is available
try:
    import tinytag

    TINYTAG_AVAILABLE = True
except ImportError:
    TINYTAG_AVAILABLE = False


# --- Helper functions ---


def _create_minimal_wav_with_metadata(path: Path) -> dict:
    """Create a minimal WAV file with embedded metadata for testing."""
    # Create simple WAV data
    sample_rate = 1000
    duration = 0.1
    samples = int(sample_rate * duration)
    data = (np.sin(2 * np.pi * 10 * np.linspace(0, duration, samples)) * 32767).astype(
        np.int16
    )

    wavfile.write(str(path), sample_rate, data)

    # Expected metadata (WAV files typically don't have rich metadata,
    # but tinytag can extract basic info)
    return {
        "duration": pytest.approx(duration, abs=0.01),
        "bitrate": pytest.approx(sample_rate * 16 / 1000, abs=1),  # kbps
    }


# --- Unit tests for extract_audio_metadata ---


@pytest.mark.skipif(not TINYTAG_AVAILABLE, reason="tinytag not installed")
def test_extract_audio_metadata_from_wav(tmp_path):
    """Test basic metadata extraction from WAV file."""
    wav_path = tmp_path / "test.wav"
    expected = _create_minimal_wav_with_metadata(wav_path)

    metadata = extract_audio_metadata(wav_path)

    # WAV files should at least have duration
    assert "duration" in metadata
    assert metadata["duration"] == expected["duration"]


def test_extract_audio_metadata_no_tinytag(tmp_path):
    """Test that missing tinytag produces warning and empty dict."""
    wav_path = tmp_path / "test.wav"
    _create_minimal_wav_with_metadata(wav_path)

    # Mock tinytag import to simulate it not being installed
    with patch.dict("sys.modules", {"tinytag": None}):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metadata = extract_audio_metadata(wav_path)

            # Should return empty dict
            assert metadata == {}

            # Should produce warning
            assert len(w) >= 1
            message = str(w[0].message)
            assert "tinytag is required" in message
            assert "pip install 'gwexpy[audio]'" in message
            assert 'pip install "gwexpy[all]"' in message
            assert "pip install tinytag" not in message
            assert w[0].category is UserWarning


def test_extract_audio_metadata_docstring_points_to_audio_extra():
    """Document the published install paths for tinytag metadata support."""
    doc = extract_audio_metadata.__doc__

    assert doc is not None
    assert "pip install 'gwexpy[audio]'" in doc
    assert 'pip install "gwexpy[all]"' in doc
    assert "coming soon" not in doc


def test_audio_reader_docstrings_point_to_audio_extra():
    """Reader docstrings should match published tinytag install guidance."""
    for func in (read_timeseriesdict_wav, read_timeseriesdict_audio):
        doc = func.__doc__

        assert doc is not None
        assert "pip install 'gwexpy[audio]'" in doc
        assert 'pip install "gwexpy[all]"' in doc
        assert "pip install tinytag" not in doc
        assert "coming soon" not in doc


def test_extract_audio_metadata_invalid_file(tmp_path):
    """Test that invalid files produce warning and empty dict."""
    invalid_path = tmp_path / "invalid.wav"
    invalid_path.write_bytes(b"NOT A WAV FILE")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metadata = extract_audio_metadata(invalid_path)

        # Should return empty dict on error
        assert metadata == {}

        # Should produce warning about extraction failure
        if TINYTAG_AVAILABLE:
            assert len(w) >= 1
            assert "Failed to extract metadata" in str(w[0].message)


def test_extract_audio_metadata_nonexistent_file():
    """Test that nonexistent files produce warning and empty dict."""
    nonexistent = Path("/nonexistent/path/file.wav")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metadata = extract_audio_metadata(nonexistent)

        # Should return empty dict
        assert metadata == {}

        # Should produce warning
        if TINYTAG_AVAILABLE:
            assert len(w) >= 1
            assert "Failed to extract metadata" in str(w[0].message)


# --- Integration tests with wav.py reader ---


def test_wav_metadata_disabled_by_default(tmp_path):
    """Test that extract_metadata defaults to False for WAV reader."""
    wav_path = tmp_path / "test.wav"
    _create_minimal_wav_with_metadata(wav_path)

    tsd = TimeSeriesDict.read(wav_path, format="wav")

    # Check provenance does not contain audio metadata fields
    if hasattr(tsd, "attrs"):
        provenance = tsd.attrs
    elif hasattr(tsd, "_gwexpy_io"):
        provenance = tsd._gwexpy_io
    else:
        pytest.skip("No provenance mechanism available")

    # Should not have metadata fields
    assert "title" not in provenance
    assert "artist" not in provenance
    assert "album" not in provenance


@pytest.mark.skipif(not TINYTAG_AVAILABLE, reason="tinytag not installed")
def test_wav_metadata_extraction(tmp_path):
    """Test WAV reader with extract_metadata=True."""
    wav_path = tmp_path / "test.wav"
    expected = _create_minimal_wav_with_metadata(wav_path)

    tsd = TimeSeriesDict.read(wav_path, format="wav", extract_metadata=True)

    # Check provenance contains metadata
    if hasattr(tsd, "attrs"):
        provenance = tsd.attrs
    elif hasattr(tsd, "_gwexpy_io"):
        provenance = tsd._gwexpy_io
    else:
        pytest.skip("No provenance mechanism available")

    # Should have duration metadata
    assert "duration" in provenance
    assert provenance["duration"] == expected["duration"]


# --- Integration tests with audio.py reader ---


@pytest.mark.skipif(not TINYTAG_AVAILABLE, reason="tinytag not installed")
def test_audio_metadata_disabled_by_default(tmp_path):
    """Test that extract_metadata defaults to False for audio reader."""
    # Skip if pydub not available
    pytest.importorskip("pydub")

    wav_path = tmp_path / "test.wav"
    _create_minimal_wav_with_metadata(wav_path)

    tsd = read_timeseriesdict_audio(wav_path)

    # Check provenance does not contain audio metadata fields
    if hasattr(tsd, "attrs"):
        provenance = tsd.attrs
    elif hasattr(tsd, "_gwexpy_io"):
        provenance = tsd._gwexpy_io
    else:
        pytest.skip("No provenance mechanism available")

    # Should not have metadata fields
    assert "title" not in provenance
    assert "artist" not in provenance
    assert "album" not in provenance


@pytest.mark.skipif(not TINYTAG_AVAILABLE, reason="tinytag not installed")
def test_audio_metadata_extraction(tmp_path):
    """Test audio reader with extract_metadata=True."""
    # Skip if pydub not available
    pytest.importorskip("pydub")

    wav_path = tmp_path / "test.wav"
    expected = _create_minimal_wav_with_metadata(wav_path)

    tsd = read_timeseriesdict_audio(wav_path, extract_metadata=True)

    # Check provenance contains metadata
    if hasattr(tsd, "attrs"):
        provenance = tsd.attrs
    elif hasattr(tsd, "_gwexpy_io"):
        provenance = tsd._gwexpy_io
    else:
        pytest.skip("No provenance mechanism available")

    # Should have duration metadata
    assert "duration" in provenance
    assert provenance["duration"] == expected["duration"]


@pytest.mark.skipif(not TINYTAG_AVAILABLE, reason="tinytag not installed")
def test_extract_audio_metadata_filters_none_values(tmp_path):
    """Test that None metadata values are filtered out."""
    wav_path = tmp_path / "test.wav"
    _create_minimal_wav_with_metadata(wav_path)

    metadata = extract_audio_metadata(wav_path)

    # All values should be non-None
    for key, value in metadata.items():
        assert value is not None, f"Field {key} should not be None"
