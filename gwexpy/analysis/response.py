"""
Response Function Analysis Module for gwexpy.

This module implements the Response Function Model (RFM) based on
Stepped Sine (Discrete) Injections. It prioritizes statistical significance
by calculating averaged ASDs for each stable frequency step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ..spectrogram import Spectrogram
from ..timeseries import TimeSeries

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass
class ResponseFunctionResult:
    """
    Result object for Stepped Sine Response Function Analysis.

    Stores the full spectral data for all injection steps efficiently.
    """

    # 2D Data: (Steps x Frequencies)
    spectrogram_inj: Spectrogram
    spectrogram_bkg: Spectrogram

    # Metadata per step
    injected_freqs: np.ndarray  # [Hz]
    step_times: np.ndarray  # [GPS Start Time]
    coupling_factors: np.ndarray  # Representative CF at injection freq

    witness_name: str
    target_name: str

    def plot(self, ax: Axes | None = None, **kwargs: object) -> Axes:
        """
        Plot the Coupling Factor vs Injected Frequency (The Transfer Function).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        # Sort by frequency for a clean line plot
        sort_idx = np.argsort(self.injected_freqs)

        ax.loglog(
            self.injected_freqs[sort_idx],
            self.coupling_factors[sort_idx],
            "o-",
            label="Measured CF",
            **kwargs,
        )

        ax.set_xlabel("Injected Frequency [Hz]")
        ax.set_ylabel("Coupling Factor")
        ax.set_title(f"Response Function: {self.witness_name} -> {self.target_name}")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
        return ax

    def plot_map(self, ax: Axes | None = None, **kwargs: object) -> Axes:
        """
        Plot the 2D Response Map (Injected Freq vs Target Spectrum).
        Useful to check for non-linear couplings.
        """
        # Create a 2D array: X=Injected Freq, Y=Target Freq, Z=Amplitude
        # Re-sort spectrogram rows by injected frequency
        sort_idx = np.argsort(self.injected_freqs)
        sorted_map = self.spectrogram_inj.value[
            sort_idx, :
        ].T  # Transpose to (Freq, Step)

        from matplotlib.colors import LogNorm

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Use pcolormesh
        X = self.injected_freqs[sort_idx]
        Y = self.spectrogram_inj.yindex.value
        Z = sorted_map

        # Simple auto-level
        vmin = np.nanpercentile(Z[Z > 0], 5)
        vmax = np.nanpercentile(Z[Z > 0], 95)

        c = ax.pcolormesh(
            X, Y, Z, norm=LogNorm(vmin=vmin, vmax=vmax), shading="auto", **kwargs
        )
        plt.colorbar(c, ax=ax, label=f"Target Amplitude [{self.spectrogram_inj.unit}]")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Injected Frequency [Hz]")
        ax.set_ylabel("Response Frequency [Hz]")
        ax.set_title("Response Matrix")
        return ax

    def plot_snapshot(
        self,
        freq: float | None = None,
        step_index: int | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """
        Plot ASDs and Upper Limits for a SPECIFIC injection step.
        """
        # Find the step
        if step_index is None and freq is not None:
            step_index = np.argmin(np.abs(self.injected_freqs - freq))
        if step_index is None:
            raise ValueError("Must specify either step_index or freq.")

        # Extract Data (Lightweight slicing)
        asd_inj = self.spectrogram_inj[step_index]
        asd_bkg = self.spectrogram_bkg[step_index]
        target_freq = self.injected_freqs[step_index]
        cf = self.coupling_factors[step_index]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Spectra
        f_axis = self.spectrogram_inj.frequencies.value
        ax.loglog(
            f_axis,
            asd_inj.value,
            label="Target (Injection)",
            color="tab:red",
            alpha=0.8,
        )
        ax.loglog(
            f_axis,
            asd_bkg.value,
            label="Target (Background)",
            color="tab:gray",
            alpha=0.6,
            linestyle="--",
        )

        # 2. Highlight Injection Point
        # Find closest bin
        idx = np.argmin(np.abs(f_axis - target_freq))
        peak_val = asd_inj.value[idx]
        bkg_val = asd_bkg.value[idx]

        ax.plot(
            target_freq, peak_val, "r*", markersize=12, label="Measured Peak", zorder=10
        )

        # 3. Annotate Excess / Upper Limit
        if peak_val > bkg_val * 1.1:  # Significant excess
            ax.annotate(
                f"Excess\n(CF={cf:.2e})",
                xy=(target_freq, peak_val),
                xytext=(target_freq * 1.15, peak_val * 2),
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            )
        else:
            # Upper Limit Case
            ax.annotate(
                "No Excess\n(Upper Limit)",
                xy=(target_freq, peak_val),
                xytext=(target_freq * 1.15, peak_val * 3),
                arrowprops=dict(arrowstyle="-|>", color="gray", linestyle=":"),
                color="gray",
            )

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"ASD [{asd_inj.unit}]")
        ax.set_title(
            f"Step Snapshot @ {target_freq:.1f} Hz (t0={self.step_times[step_index]:.1f})"
        )
        ax.legend(loc="lower left")
        ax.grid(True, which="both", linestyle=":")

        # Auto-zoom X-axis around injection
        ax.set_xlim(target_freq * 0.1, target_freq * 10.0)

        return ax


def detect_step_segments(
    witness: TimeSeries,
    fftlength: float = 1.0,
    snr_threshold: float = 10.0,
    min_duration: float = 5.0,
    trim_edge: float = 1.0,
    freq_tolerance: float = 1.0,
) -> list[tuple[float, float, float]]:
    """
    Automatically detect time segments where the injection frequency is constant.
    """
    # High-res tracking spectrogram
    # Note: gwpy requires stride >= fftlength. We use stride=fftlength for tracking.
    spec = witness.spectrogram(fftlength=fftlength, overlap=0, stride=fftlength)

    times = spec.times.value
    freqs = spec.frequencies.value

    peak_indices = np.argmax(spec.value, axis=1)
    peak_vals = np.max(spec.value, axis=1)
    peak_freqs_t = freqs[peak_indices]

    # SNR check
    median_level = np.median(spec.value)
    if median_level == 0:
        median_level = 1e-30
    is_loud = peak_vals > (median_level * snr_threshold)

    segments: list[tuple[float, float, float]] = []
    if len(times) == 0:
        return segments

    current_start_idx = 0
    current_freq = peak_freqs_t[0]
    in_segment = is_loud[0]

    for i in range(1, len(times)):
        f = peak_freqs_t[i]
        loud = is_loud[i]
        freq_changed = abs(f - current_freq) > freq_tolerance

        if in_segment:
            if not loud or freq_changed:
                # Close segment
                duration = times[i - 1] - times[current_start_idx]
                if duration >= min_duration:
                    t_start = times[current_start_idx] + trim_edge
                    t_end = times[i - 1] - trim_edge
                    if t_end > t_start:
                        segments.append((t_start, t_end, current_freq))

                # Reset
                in_segment = False
                if loud:  # Immediate new step
                    in_segment = True
                    current_start_idx = i
                    current_freq = f
        else:
            if loud:
                in_segment = True
                current_start_idx = i
                current_freq = f

    # Last segment
    if in_segment:
        duration = times[-1] - times[current_start_idx]
        if duration >= min_duration:
            t_start = times[current_start_idx] + trim_edge
            t_end = times[-1] - trim_edge
            if t_end > t_start:
                segments.append((t_start, t_end, current_freq))

    return segments


class ResponseFunctionAnalysis:
    """
    Analysis engine for Stepped Sine Injections.
    """

    def compute(
        self,
        witness: TimeSeries,
        target: TimeSeries,
        segments: list[tuple[float, float, float]] | None = None,
        fftlength: float = 4.0,
        overlap: float = 0.0,
        # Auto-detect params
        auto_detect: bool = True,
        snr_threshold: float = 10.0,
        min_duration: float = 5.0,
        trim_edge: float = 1.0,
        # Background
        witness_bkg: TimeSeries | None = None,
        target_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> ResponseFunctionResult:
        """
        Perform the analysis.
        """
        # --- 1. Step Detection ---
        if segments is None:
            if not auto_detect:
                raise ValueError("segments or auto_detect=True required.")

            # Use smaller fft for detection tracking
            segments = detect_step_segments(
                witness,
                fftlength=1.0,
                snr_threshold=snr_threshold,
                min_duration=min_duration,
                trim_edge=trim_edge,
            )

        if not segments:
            raise ValueError("No injection steps detected.")

        # --- 2. Computation Loop ---
        # Pre-calculate background ASDs if provided (assuming stationary background)
        # If not, we might crop background dynamically.
        # For efficiency, let's calculate one master background if provided.
        master_asd_tgt_bkg = None
        master_asd_wit_bkg = None

        if target_bkg is not None:
            master_asd_tgt_bkg = target_bkg.asd(
                fftlength=fftlength, overlap=overlap, **kwargs
            )
        if witness_bkg is not None:
            master_asd_wit_bkg = witness_bkg.asd(
                fftlength=fftlength, overlap=overlap, **kwargs
            )

        # Pre-allocate containers
        # Find first segment long enough for FFT to determine frequency axis size
        n_freqs = None
        freq_axis = None
        for t_s, t_e, _ in segments:
            if (t_e - t_s) >= fftlength:
                dummy_res = target.crop(t_s, t_s + fftlength).asd(
                    fftlength=fftlength, **kwargs
                )
                freq_axis = dummy_res.frequencies
                n_freqs = len(freq_axis)
                break

        if n_freqs is None:
            raise ValueError(
                f"No segments found with duration >= fftlength ({fftlength}s)."
            )

        n_steps = len(segments)

        spec_inj = np.zeros((n_steps, n_freqs))
        spec_bkg = np.zeros((n_steps, n_freqs))
        inj_freqs = np.zeros(n_steps)
        step_times = np.zeros(n_steps)
        cf_vals = np.zeros(n_steps)

        for i, (t_s, t_e, f_inj) in enumerate(segments):
            # Crop Injection Data
            if (t_e - t_s) < fftlength:
                continue  # Safe check

            ts_wit_seg = witness.crop(t_s, t_e)
            ts_tgt_seg = target.crop(t_s, t_e)

            # Calculate Injection ASDs
            # Dynamic overlap adjustment to maximize averages
            seg_len = t_e - t_s
            seg_ovlp = overlap if overlap > 0 else (fftlength / 2)
            if seg_len < fftlength:
                seg_ovlp = 0  # Should be skipped anyway

            asd_tgt_inj = ts_tgt_seg.asd(
                fftlength=fftlength, overlap=seg_ovlp, **kwargs
            )
            asd_wit_inj = ts_wit_seg.asd(
                fftlength=fftlength, overlap=seg_ovlp, **kwargs
            )

            # Get Background ASDs
            if master_asd_tgt_bkg is not None:
                val_tgt_bkg = master_asd_tgt_bkg.value
            else:
                # Fallback: estimate from nearby quiet data
                # Try before injection step
                t_b_s_eff = t_s - seg_len - 0.5
                if t_b_s_eff < target.span[0]:
                    # Try after injection step
                    t_b_s_eff = t_e + 0.5

                # Constrain to available data and ensure minimum duration
                t_b_s_eff = max(t_b_s_eff, target.span[0])
                t_b_e_eff = min(t_b_s_eff + max(seg_len, fftlength), target.span[1])

                # Check if we need to shift left if we hit the right boundary
                if (t_b_e_eff - t_b_s_eff) < fftlength:
                    t_b_s_eff = max(target.span[0], t_b_e_eff - fftlength)
                    t_b_e_eff = min(target.span[1], t_b_s_eff + fftlength)

                val_tgt_bkg = (
                    target.crop(t_b_s_eff, t_b_e_eff)
                    .asd(fftlength=fftlength, overlap=seg_ovlp, **kwargs)
                    .value
                )

            if master_asd_wit_bkg is not None:
                val_wit_bkg = master_asd_wit_bkg.value
            else:
                # Same for witness
                t_b_s_eff = t_s - seg_len - 0.5
                if t_b_s_eff < witness.span[0]:
                    t_b_s_eff = t_e + 0.5

                t_b_s_eff = max(t_b_s_eff, witness.span[0])
                t_b_e_eff = min(t_b_s_eff + max(seg_len, fftlength), witness.span[1])

                if (t_b_e_eff - t_b_s_eff) < fftlength:
                    t_b_s_eff = max(witness.span[0], t_b_e_eff - fftlength)
                    t_b_e_eff = min(witness.span[1], t_b_s_eff + fftlength)

                val_wit_bkg = (
                    witness.crop(t_b_s_eff, t_b_e_eff)
                    .asd(fftlength=fftlength, overlap=seg_ovlp, **kwargs)
                    .value
                )

            # Store in arrays
            spec_inj[i, :] = asd_tgt_inj.value
            spec_bkg[i, :] = val_tgt_bkg
            inj_freqs[i] = f_inj
            step_times[i] = t_s

            # Calculate Coupling Factor at f_inj
            # CF = sqrt( (P_tgt_inj - P_tgt_bkg) / (P_wit_inj - P_wit_bkg) )
            assert freq_axis is not None
            f_idx = np.argmin(np.abs(freq_axis.value - f_inj))

            p_tgt_net = asd_tgt_inj.value[f_idx] ** 2 - val_tgt_bkg[f_idx] ** 2
            p_wit_net = asd_wit_inj.value[f_idx] ** 2 - val_wit_bkg[f_idx] ** 2

            if p_tgt_net > 0 and p_wit_net > 0:
                cf_vals[i] = np.sqrt(p_tgt_net / p_wit_net)
            else:
                cf_vals[i] = np.nan  # Upper limit condition

        # Wrap in Containers
        unit = getattr(target, "unit", None) or "dimensionless"

        sg_inj = Spectrogram(
            spec_inj, times=step_times, frequencies=freq_axis, unit=unit, name="Inj"
        )
        sg_bkg = Spectrogram(
            spec_bkg, times=step_times, frequencies=freq_axis, unit=unit, name="Bkg"
        )

        return ResponseFunctionResult(
            spectrogram_inj=sg_inj,
            spectrogram_bkg=sg_bkg,
            injected_freqs=inj_freqs,
            step_times=step_times,
            coupling_factors=cf_vals,
            witness_name=str(witness.name) if witness.name else "Witness",
            target_name=str(target.name) if target.name else "Target",
        )


def estimate_response_function(
    witness: TimeSeries,
    target: TimeSeries,
    segments: list[tuple[float, float, float]] | None = None,
    fftlength: float = 4.0,
    overlap: float = 0.0,
    auto_detect: bool = True,
    snr_threshold: float = 10.0,
    min_duration: float = 5.0,
    trim_edge: float = 1.0,
    witness_bkg: TimeSeries | None = None,
    target_bkg: TimeSeries | None = None,
    **kwargs: object,
) -> ResponseFunctionResult:
    """Helper function."""
    analysis = ResponseFunctionAnalysis()
    return analysis.compute(
        witness,
        target,
        segments=segments,
        fftlength=fftlength,
        overlap=overlap,
        auto_detect=auto_detect,
        snr_threshold=snr_threshold,
        min_duration=min_duration,
        trim_edge=trim_edge,
        witness_bkg=witness_bkg,
        target_bkg=target_bkg,
        **kwargs,
    )
