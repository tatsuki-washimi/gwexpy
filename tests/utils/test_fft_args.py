"""Tests for gwexpy.utils.fft_args module."""

import pytest
from astropy import units as u

from gwexpy.utils.fft_args import (
    check_deprecated_kwargs,
    get_default_overlap,
    parse_fftlength_or_overlap,
)


class TestParseFftlengthOrOverlap:
    """Tests for parse_fftlength_or_overlap function."""

    def test_none_input(self):
        """Test that None input returns (None, None)."""
        seconds, samples = parse_fftlength_or_overlap(None, sample_rate=256)
        assert seconds is None
        assert samples is None

    def test_float_seconds_no_sample_rate(self):
        """Test float (seconds) without sample_rate."""
        seconds, samples = parse_fftlength_or_overlap(1.5)
        assert seconds == 1.5
        assert samples is None

    def test_float_seconds_with_sample_rate(self):
        """Test float (seconds) with sample_rate conversion."""
        seconds, samples = parse_fftlength_or_overlap(1.5, sample_rate=1024)
        assert abs(seconds - 1.5) < 1e-12
        assert samples == 1536  # 1.5 * 1024

    def test_int_is_seconds_not_samples(self):
        """Test that int is treated as seconds, not samples (GWpy-compatible)."""
        seconds, samples = parse_fftlength_or_overlap(2, sample_rate=100)
        assert seconds == 2.0
        assert samples == 200  # 2 * 100, not 2

    def test_quantity_seconds(self):
        """Test Quantity with time units."""
        seconds, samples = parse_fftlength_or_overlap(2.0 * u.s, sample_rate=512)
        assert seconds == 2.0
        assert samples == 1024  # 2.0 * 512

    def test_quantity_milliseconds(self):
        """Test Quantity with milliseconds."""
        seconds, samples = parse_fftlength_or_overlap(500 * u.ms, sample_rate=1000)
        assert seconds == 0.5
        assert samples == 500  # 0.5 * 1000

    def test_quantity_non_time_unit_raises(self):
        """Test that non-time Quantity raises ValueError."""
        with pytest.raises(ValueError, match="expected a time-like Quantity"):
            parse_fftlength_or_overlap(1.0 * u.Hz, sample_rate=100)

    def test_negative_seconds_raises(self):
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError, match="negative time"):
            parse_fftlength_or_overlap(-1.0, sample_rate=100)

    def test_zero_seconds_allowed(self):
        """Test that zero seconds is allowed (edge case)."""
        seconds, samples = parse_fftlength_or_overlap(0.0, sample_rate=100)
        assert seconds == 0.0
        assert samples == 1  # max(1, round(0.0 * 100))

    def test_invalid_sample_rate_raises(self):
        """Test that invalid sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate is required"):
            parse_fftlength_or_overlap(1.0, sample_rate=-100)

    def test_sample_rate_as_quantity(self):
        """Test sample_rate as Quantity (Hz)."""
        seconds, samples = parse_fftlength_or_overlap(
            1.0, sample_rate=256 * u.Hz, arg_name="fftlength"
        )
        assert seconds == 1.0
        assert samples == 256

    def test_sample_rate_non_frequency_quantity_raises(self):
        """Test that non-frequency sample_rate Quantity raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be a frequency"):
            parse_fftlength_or_overlap(1.0, sample_rate=256 * u.s)

    def test_rounding_to_integer_samples(self):
        """Test that samples are rounded to nearest integer."""
        # 1.49 * 100 = 149 → rounds to 149
        _, samples = parse_fftlength_or_overlap(1.49, sample_rate=100)
        assert samples == 149

        # 1.51 * 100 = 151 → rounds to 151
        _, samples = parse_fftlength_or_overlap(1.51, sample_rate=100)
        assert samples == 151

    def test_arg_name_in_error_message(self):
        """Test that arg_name appears in error messages."""
        with pytest.raises(ValueError, match="overlap:"):
            parse_fftlength_or_overlap(-1.0, sample_rate=100, arg_name="overlap")


class TestCheckDeprecatedKwargs:
    """Tests for check_deprecated_kwargs function."""

    def test_no_deprecated_kwargs(self):
        """Test that valid kwargs pass without error."""
        check_deprecated_kwargs(fftlength=1.0, overlap=0.5, window="hann")

    def test_nperseg_raises_typeerror(self):
        """Test that nperseg raises TypeError."""
        with pytest.raises(TypeError, match="nperseg is removed"):
            check_deprecated_kwargs(nperseg=256)

    def test_noverlap_raises_typeerror(self):
        """Test that noverlap raises TypeError."""
        with pytest.raises(TypeError, match="noverlap is removed"):
            check_deprecated_kwargs(noverlap=128)

    def test_both_deprecated_raises_nperseg_first(self):
        """Test that nperseg error is raised first if both present."""
        with pytest.raises(TypeError, match="nperseg is removed"):
            check_deprecated_kwargs(nperseg=256, noverlap=128)


class TestGetDefaultOverlap:
    """Tests for get_default_overlap function."""

    def test_none_fftlength(self):
        """Test that None fftlength returns None."""
        assert get_default_overlap(None, window="hann") is None

    def test_hann_window_50_percent(self):
        """Test that hann window returns 50% overlap."""
        assert get_default_overlap(1.0, window="hann") == 0.5
        assert get_default_overlap(2.0, window="hann") == 1.0

    def test_hamming_window_50_percent(self):
        """Test that hamming window returns 50% overlap."""
        assert get_default_overlap(1.0, window="hamming") == 0.5

    def test_blackman_window_50_percent(self):
        """Test that blackman window returns 50% overlap."""
        assert get_default_overlap(1.0, window="blackman") == 0.5

    def test_boxcar_window_zero_overlap(self):
        """Test that boxcar window returns 0 overlap."""
        assert get_default_overlap(1.0, window="boxcar") == 0.0
        assert get_default_overlap(2.0, window="rectangular") == 0.0
        assert get_default_overlap(1.5, window="uniform") == 0.0

    def test_unknown_window_50_percent_default(self):
        """Test that unknown window returns 50% overlap (conservative)."""
        assert get_default_overlap(1.0, window="kaiser") == 0.5
        assert get_default_overlap(1.0, window="custom_window") == 0.5

    def test_case_insensitive_window_name(self):
        """Test that window names are case-insensitive."""
        assert get_default_overlap(1.0, window="HANN") == 0.5
        assert get_default_overlap(1.0, window="Boxcar") == 0.0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_typical_workflow_with_defaults(self):
        """Test typical workflow: parse fftlength, get default overlap, parse overlap."""
        # User provides fftlength only
        fftlength_sec, nperseg = parse_fftlength_or_overlap(1.0, sample_rate=256)
        assert fftlength_sec == 1.0
        assert nperseg == 256

        # Get default overlap
        overlap_sec = get_default_overlap(fftlength_sec, window="hann")
        assert overlap_sec == 0.5

        # Parse overlap
        overlap_sec_parsed, noverlap = parse_fftlength_or_overlap(
            overlap_sec, sample_rate=256, arg_name="overlap"
        )
        assert overlap_sec_parsed == 0.5
        assert noverlap == 128

    def test_user_provides_both_fftlength_and_overlap(self):
        """Test when user provides both fftlength and overlap."""
        fftlength_sec, nperseg = parse_fftlength_or_overlap(2.0, sample_rate=512)
        overlap_sec, noverlap = parse_fftlength_or_overlap(
            1.0, sample_rate=512, arg_name="overlap"
        )

        assert nperseg == 1024
        assert noverlap == 512
        assert noverlap == nperseg // 2

    def test_deprecated_kwargs_caught_early(self):
        """Test that deprecated kwargs are caught before parsing."""
        # Simulate a function call with deprecated kwargs
        kwargs = {"nperseg": 256, "noverlap": 128, "window": "hann"}

        with pytest.raises(TypeError, match="nperseg is removed"):
            check_deprecated_kwargs(**kwargs)

        # Verify that parsing never happens
        # (exception raised before parse_fftlength_or_overlap is called)
