"""Tests for gwexpy/analysis/response.py."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before any plt import

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwexpy.analysis.response import (
    ResponseFunctionResult,
    ResponseFunctionAnalysis,
    detect_step_segments,
)
from gwexpy.spectrogram import Spectrogram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrogram(n_steps=3, n_freqs=50, t0=0.0):
    """Create a minimal Spectrogram for testing."""
    data = np.abs(np.random.default_rng(42).normal(1e-20, 1e-22, (n_steps, n_freqs)))
    freqs = np.linspace(1.0, 100.0, n_freqs)
    times = np.arange(n_steps, dtype=float) + t0
    return Spectrogram(data, times=times, frequencies=freqs, unit="1/Hz**0.5")


def _make_result(n_steps=3, n_freqs=50):
    """Build a minimal ResponseFunctionResult."""
    sg_inj = _make_spectrogram(n_steps, n_freqs, t0=0.0)
    sg_bkg = _make_spectrogram(n_steps, n_freqs, t0=0.0)
    inj_freqs = np.array([10.0, 20.0, 30.0])[:n_steps]
    step_times = np.array([0.0, 10.0, 20.0])[:n_steps]
    cf = np.array([1e-4, 2e-4, 3e-4])[:n_steps]
    return ResponseFunctionResult(
        spectrogram_inj=sg_inj,
        spectrogram_bkg=sg_bkg,
        injected_freqs=inj_freqs,
        step_times=step_times,
        coupling_factors=cf,
        witness_name="W",
        target_name="T",
    )


def _make_sine_timeseries(t_start, t_end, freq_hz, sample_rate=256.0, amplitude=1.0):
    """Create a TimeSeries with a sinusoidal signal."""
    dt = 1.0 / sample_rate
    n = int((t_end - t_start) * sample_rate)
    t = np.arange(n) * dt
    data = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return TimeSeries(data, t0=t_start, dt=dt)


# ---------------------------------------------------------------------------
# ResponseFunctionResult - construction
# ---------------------------------------------------------------------------


class TestResponseFunctionResult:
    def test_construction(self):
        result = _make_result()
        assert result.witness_name == "W"
        assert result.target_name == "T"
        assert len(result.injected_freqs) == 3
        assert len(result.coupling_factors) == 3

    def test_attributes_accessible(self):
        result = _make_result()
        assert isinstance(result.spectrogram_inj, Spectrogram)
        assert isinstance(result.spectrogram_bkg, Spectrogram)
        assert result.step_times.shape == (3,)

    # --- plot() ---

    def test_plot_returns_ax(self):
        import matplotlib.pyplot as plt
        result = _make_result()
        ax = result.plot()
        assert ax is not None
        plt.close("all")

    def test_plot_with_provided_ax(self):
        import matplotlib.pyplot as plt
        result = _make_result()
        fig, ax = plt.subplots()
        returned_ax = result.plot(ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_plot_single_step(self):
        import matplotlib.pyplot as plt
        result = _make_result(n_steps=1)
        ax = result.plot()
        assert ax is not None
        plt.close("all")

    # --- plot_snapshot() ---

    def test_plot_snapshot_by_step_index(self):
        import matplotlib.pyplot as plt
        result = _make_result()
        ax = result.plot_snapshot(step_index=0)
        assert ax is not None
        plt.close("all")

    def test_plot_snapshot_by_freq(self):
        import matplotlib.pyplot as plt
        result = _make_result()
        ax = result.plot_snapshot(freq=10.0)
        assert ax is not None
        plt.close("all")

    def test_plot_snapshot_no_args_raises(self):
        result = _make_result()
        with pytest.raises(ValueError, match="step_index or freq"):
            result.plot_snapshot()

    def test_plot_snapshot_with_ax(self):
        import matplotlib.pyplot as plt
        result = _make_result()
        fig, ax = plt.subplots()
        returned = result.plot_snapshot(step_index=1, ax=ax)
        assert returned is ax
        plt.close("all")

    def test_plot_snapshot_upper_limit_branch(self):
        """Test the 'No Excess' annotation branch (bkg > inj)."""
        import matplotlib.pyplot as plt
        n_freqs = 50
        sg_inj = _make_spectrogram(3, n_freqs)
        # Make bkg > inj to trigger the upper limit annotation
        sg_bkg_data = sg_inj.value * 10.0  # much bigger than inj
        freqs = np.linspace(1.0, 100.0, n_freqs)
        times = np.arange(3, dtype=float)
        sg_bkg = Spectrogram(sg_bkg_data, times=times, frequencies=freqs, unit="1/Hz**0.5")
        result = ResponseFunctionResult(
            spectrogram_inj=sg_inj,
            spectrogram_bkg=sg_bkg,
            injected_freqs=np.array([10.0, 20.0, 30.0]),
            step_times=np.array([0.0, 10.0, 20.0]),
            coupling_factors=np.array([1e-4, 2e-4, 3e-4]),
            witness_name="W",
            target_name="T",
        )
        ax = result.plot_snapshot(step_index=0)
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# detect_step_segments
# ---------------------------------------------------------------------------


class TestDetectStepSegments:
    def test_empty_result_for_quiet_signal(self):
        """Signal with no loud steps returns empty list."""
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1e-23, 512)
        ts = TimeSeries(data, t0=0.0, dt=1.0 / 256)
        segments = detect_step_segments(ts, fftlength=0.5, snr_threshold=1e10)
        assert segments == []

    def test_detects_single_step(self):
        """A clear single-frequency injection is detected as one segment."""
        sample_rate = 512.0
        duration = 30.0
        freq = 50.0
        amplitude = 1.0

        ts = _make_sine_timeseries(0.0, duration, freq, sample_rate, amplitude)
        segments = detect_step_segments(
            ts,
            fftlength=1.0,
            snr_threshold=5.0,
            min_duration=5.0,
            trim_edge=1.0,
            freq_tolerance=2.0,
        )
        # Should detect at least one segment near 50 Hz
        assert len(segments) >= 1
        # Check frequency is close to 50 Hz
        freqs = [seg[2] for seg in segments]
        assert any(abs(f - freq) < 5.0 for f in freqs)

    def test_returns_list_of_tuples(self):
        """Return type is list of (t_start, t_end, freq) tuples."""
        ts = _make_sine_timeseries(0.0, 30.0, 50.0, 256.0, 1.0)
        segments = detect_step_segments(ts, fftlength=1.0, snr_threshold=2.0, min_duration=3.0)
        for seg in segments:
            assert len(seg) == 3
            t_start, t_end, freq = seg
            assert t_end > t_start
            assert freq > 0

    def test_zero_median_fallback(self):
        """Handles the edge case where median level is zero (all-zero data)."""
        data = np.zeros(512)
        ts = TimeSeries(data, t0=0.0, dt=1.0 / 256)
        segments = detect_step_segments(ts, fftlength=0.5, snr_threshold=10.0)
        assert segments == []

    def test_min_duration_filters_short_steps(self):
        """Segments shorter than min_duration are excluded."""
        # Create a 30s signal
        ts = _make_sine_timeseries(0.0, 30.0, 50.0, 256.0, 1.0)
        # Very high min_duration → nothing should be returned
        segments = detect_step_segments(ts, fftlength=1.0, min_duration=100.0)
        assert segments == []


# ---------------------------------------------------------------------------
# ResponseFunctionAnalysis - error paths
# ---------------------------------------------------------------------------


class TestResponseFunctionAnalysisErrors:
    def test_no_segments_auto_detect_false_raises(self):
        """Without auto_detect and no segments provided, raises ValueError."""
        ts = TimeSeries(np.ones(512), t0=0.0, dt=1.0 / 256)
        with pytest.raises(ValueError, match="segments or auto_detect"):
            ResponseFunctionAnalysis().compute(
                witness=ts,
                target=ts,
                segments=None,
                auto_detect=False,
            )

    def test_empty_segments_raises(self):
        """Empty segments list raises ValueError."""
        ts = TimeSeries(np.ones(512), t0=0.0, dt=1.0 / 256)
        with pytest.raises(ValueError, match="No injection steps"):
            ResponseFunctionAnalysis().compute(
                witness=ts,
                target=ts,
                segments=[],
                auto_detect=False,
            )
