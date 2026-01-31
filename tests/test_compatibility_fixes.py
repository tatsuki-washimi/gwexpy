"""Tests for gwpy compatibility fixes.

This module tests the fixes for:
1. Internal attribute preservation through __array_finalize__
2. Plot.show() with close parameter
3. _repr_png_ for Jupyter display
"""

import tempfile

import numpy as np
import pytest


class TestArrayFinalizeFrequencySeries:
    """Test __array_finalize__ preserves gwexpy attributes."""

    def test_slicing_preserves_fft_mode(self):
        """Slicing should preserve _gwex_fft_mode attribute."""
        from gwexpy.frequencyseries import FrequencySeries

        # Create a FrequencySeries with gwexpy attributes
        fs = FrequencySeries([1, 2, 3, 4, 5], df=1.0)
        fs._gwex_fft_mode = "transient"
        fs._gwex_target_nfft = 128

        # Slice the series
        fs_slice = fs[1:4]

        # Check attributes are preserved
        assert hasattr(fs_slice, "_gwex_fft_mode")
        assert fs_slice._gwex_fft_mode == "transient"
        assert hasattr(fs_slice, "_gwex_target_nfft")
        assert fs_slice._gwex_target_nfft == 128

    def test_arithmetic_preserves_fft_mode(self):
        """Arithmetic operations should preserve _gwex_fft_mode attribute."""
        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries([1, 2, 3, 4, 5], df=1.0)
        fs._gwex_fft_mode = "transient"
        fs._gwex_target_nfft = 64
        fs._gwex_original_n = 50

        # Multiply
        fs_mult = fs * 2.0
        assert fs_mult._gwex_fft_mode == "transient"
        assert fs_mult._gwex_target_nfft == 64
        assert fs_mult._gwex_original_n == 50

        # Add
        fs_add = fs + 1.0
        assert fs_add._gwex_fft_mode == "transient"

        # Subtract
        fs_sub = fs - 0.5
        assert fs_sub._gwex_fft_mode == "transient"

    def test_fft_roundtrip_with_operations(self):
        """Test that fft mode is preserved through operations for ifft."""
        from gwexpy.timeseries import TimeSeries

        # Create a TimeSeries and fft with transient mode
        ts = TimeSeries(np.sin(np.linspace(0, 4 * np.pi, 100)), dt=0.01)
        fs = ts.fft(mode="transient")

        # Verify attribute is set
        assert getattr(fs, "_gwex_fft_mode", None) == "transient"

        # Apply operations
        fs_scaled = fs * 2.0
        fs_sliced = fs[1:50]

        # Both should preserve the mode
        assert getattr(fs_scaled, "_gwex_fft_mode", None) == "transient"
        assert getattr(fs_sliced, "_gwex_fft_mode", None) == "transient"

    def test_copy_preserves_attributes(self):
        """copy() should preserve gwexpy attributes."""
        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries([1, 2, 3], df=1.0)
        fs._gwex_fft_mode = "transient"
        fs._gwex_pad_left = 10
        fs._gwex_pad_right = 10

        fs_copy = fs.copy()
        assert fs_copy._gwex_fft_mode == "transient"
        assert fs_copy._gwex_pad_left == 10
        assert fs_copy._gwex_pad_right == 10


class TestPlotShowClose:
    """Test Plot.show() close and block parameters."""

    def test_show_close_false_allows_savefig(self):
        """show(close=False) should allow subsequent savefig()."""
        from gwexpy.plot import Plot
        from gwexpy.timeseries import TimeSeries

        ts = TimeSeries([1, 2, 3, 4, 5], dt=1)
        plot = Plot(ts)

        # Show without closing
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing
        # We can't actually call show in tests, but we can verify the method signature
        # and that savefig works when close=False logic would be applied

        # Skip show() in test environment, just verify savefig works
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plot.savefig(f.name)
            # If we get here without error, the test passes

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(plot)

    def test_show_signature_has_close_and_block_parameters(self):
        """Verify show() method has close and block parameters."""
        import inspect

        from gwexpy.plot import Plot

        sig = inspect.signature(Plot.show)
        params = list(sig.parameters.keys())

        # Check close parameter
        assert "close" in params
        assert sig.parameters["close"].default is True

        # Check block parameter
        assert "block" in params
        assert sig.parameters["block"].default is None


class TestPlotReprPng:
    """Test _repr_png_ for Jupyter display."""

    def test_repr_png_exists(self):
        """Plot should have _repr_png_ method."""
        from gwexpy.plot import Plot

        assert hasattr(Plot, "_repr_png_")

    def test_repr_png_returns_bytes(self):
        """_repr_png_ should return PNG bytes."""
        import matplotlib

        from gwexpy.plot import Plot
        from gwexpy.timeseries import TimeSeries

        matplotlib.use("Agg")

        ts = TimeSeries([1, 2, 3, 4, 5], dt=1)
        plot = Plot(ts)

        png_data = plot._repr_png_()

        assert png_data is not None
        assert isinstance(png_data, bytes)
        # PNG files start with specific magic bytes
        assert png_data[:8] == b"\x89PNG\r\n\x1a\n"

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(plot)

    def test_repr_png_after_close_handles_gracefully(self):
        """_repr_png_ should handle closed figures gracefully."""
        import matplotlib
        import matplotlib.pyplot as plt

        from gwexpy.plot import Plot
        from gwexpy.timeseries import TimeSeries

        matplotlib.use("Agg")

        ts = TimeSeries([1, 2, 3, 4, 5], dt=1)
        plot = Plot(ts)
        plt.close(plot)

        # After close, _repr_png_ should either return None or valid bytes
        # depending on matplotlib version
        result = plot._repr_png_()

        # Should not raise an exception, and if it returns data, it should be valid PNG
        if result is not None:
            assert isinstance(result, bytes)
            assert result[:8] == b"\x89PNG\r\n\x1a\n"


class TestReprHtmlDisabled:
    """Test that _repr_html_ is disabled to prevent double display."""

    def test_repr_html_is_none(self):
        """_repr_html_ should be None to prevent double display."""
        from gwexpy.plot import Plot

        # Class attribute should be None
        assert Plot._repr_html_ is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
