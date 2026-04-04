"""
Response Function Analysis Module for gwexpy.

This module implements the Response Function Model (RFM) based on
Stepped Sine (Discrete) Injections. It prioritizes statistical significance
by calculating averaged ASDs for each stable frequency step.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from gwpy.segments import Segment

from ..table.segment_table import SegmentTable, RowProxy

logger = logging.getLogger(__name__)

from ..spectrogram import Spectrogram
from ..timeseries import TimeSeries

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _freq_values(series: Any) -> np.ndarray:
    return np.asarray(getattr(series.xindex, "value", series.xindex), dtype=float)


def _estimate_response_row_bytes(
    witness: TimeSeries,
    target: TimeSeries,
    fftlength: float,
    segment_duration: float,
) -> int:
    sample_rate = max(witness.sample_rate.value, target.sample_rate.value)
    n_samples = max(1, int(np.ceil(sample_rate * segment_duration)))
    n_freqs = int(fftlength * sample_rate / 2) + 1
    bytes_est = (2 * n_samples + 4 * n_freqs) * 8
    return int(bytes_est * 1.5)


def _compute_response_row(
    witness: TimeSeries,
    target: TimeSeries,
    segment: Segment,
    injected_freq: float,
    fftlength: float,
    overlap: float,
    kwargs: dict[str, Any],
    master_asd_wit_bkg: Any = None,
    master_asd_tgt_bkg: Any = None,
) -> dict[str, Any]:
    wit_inj = witness.crop(segment[0], segment[1])
    tgt_inj = target.crop(segment[0], segment[1])
    wit_asd_inj = wit_inj.asd(fftlength=fftlength, overlap=overlap, **kwargs)
    tgt_asd_inj = tgt_inj.asd(fftlength=fftlength, overlap=overlap, **kwargs)

    if master_asd_wit_bkg is not None:
        wit_asd_bkg = master_asd_wit_bkg
    else:
        seg_len = segment[1] - segment[0]
        t_b_s = max(segment[0] - seg_len - 0.5, witness.span[0])
        t_b_e = min(t_b_s + max(seg_len, fftlength), witness.span[1])
        if (t_b_e - t_b_s) < fftlength:
            t_b_s = max(witness.span[0], t_b_e - fftlength)
            t_b_e = min(witness.span[1], t_b_s + fftlength)
        wit_asd_bkg = witness.crop(t_b_s, t_b_e).asd(
            fftlength=fftlength, overlap=overlap, **kwargs
        )

    if master_asd_tgt_bkg is not None:
        tgt_asd_bkg = master_asd_tgt_bkg
    else:
        seg_len = segment[1] - segment[0]
        t_b_s = max(segment[0] - seg_len - 0.5, target.span[0])
        t_b_e = min(t_b_s + max(seg_len, fftlength), target.span[1])
        if (t_b_e - t_b_s) < fftlength:
            t_b_s = max(target.span[0], t_b_e - fftlength)
            t_b_e = min(target.span[1], t_b_s + fftlength)
        tgt_asd_bkg = target.crop(t_b_s, t_b_e).asd(
            fftlength=fftlength, overlap=overlap, **kwargs
        )

    f_idx = np.argmin(np.abs(wit_asd_inj.xindex.value - injected_freq))
    p_wit_net = wit_asd_inj.value[f_idx] ** 2 - wit_asd_bkg.value[f_idx] ** 2
    p_tgt_net = tgt_asd_inj.value[f_idx] ** 2 - tgt_asd_bkg.value[f_idx] ** 2
    cf = np.sqrt(p_tgt_net / p_wit_net) if p_wit_net > 0 and p_tgt_net > 0 else np.nan

    return {
        "span": segment,
        "injected_freq": injected_freq,
        "wit_inj": wit_inj,
        "tgt_inj": tgt_inj,
        "wit_asd_inj": wit_asd_inj,
        "tgt_asd_inj": tgt_asd_inj,
        "wit_asd_bkg": wit_asd_bkg,
        "tgt_asd_bkg": tgt_asd_bkg,
        "cf": cf,
    }


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
    table: SegmentTable | None = None

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
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

    def plot_map(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
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
            step_index = int(np.argmin(np.abs(self.injected_freqs - freq)))
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
    freq_tolerance: float = 2.0,
) -> list[tuple[float, float, float]]:
    """
    Automatically detect time segments where the injection frequency is constant.

    Parameters
    ----------
    witness : TimeSeries
        Time series data containing the injection signal.
    fftlength : float, optional
        FFT length for tracking spectrogram (default: 1.0s, giving Δf=1 Hz).
    snr_threshold : float, optional
        Minimum SNR for injection detection (default: 10.0).
    min_duration : float, optional
        Minimum duration for a valid step (default: 5.0s).
    trim_edge : float, optional
        Time to trim from step edges (default: 1.0s).
    freq_tolerance : float, optional
        Maximum frequency change within a step [Hz] (default: 2.0).

        **Note**: For reliable detection, ``freq_tolerance`` should be at least
        ``2 * Δf`` where ``Δf = 1/fftlength``. With the default fftlength=1.0s
        (Δf=1 Hz), a strict tolerance of Δf risks false step boundaries due to
        spectral bin fluctuations. The default 2.0 Hz provides adequate margin.

        Reference: Harris, F.J. "On the use of windows for harmonic analysis
        with the discrete Fourier transform", Proc. IEEE 66(1), 1978.

    Returns
    -------
    list of tuple
        List of (start_time, end_time, frequency) tuples for each detected step.
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
    median_level = float(np.median(spec.value))
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
        # Execution
        n_jobs: int | None = None,
        memory_limit: int = 2 * 1024**3,
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

        # --- 2. SegmentTable Flow ---
        # Filter segments with duration >= fftlength
        valid_segments = [s for s in segments if (s[1] - s[0]) >= fftlength]
        if not valid_segments:
            raise ValueError(f"No segments found with duration >= fftlength ({fftlength}s).")

        seg_objs = [Segment(s, e) for s, e, f in valid_segments]
        injected_freqs_list = [f for s, e, f in valid_segments]

        # 2.3 Pre-calculate master backgrounds if possible
        master_asd_tgt_bkg = None
        master_asd_wit_bkg = None
        if target_bkg is not None:
            master_asd_tgt_bkg = target_bkg.asd(fftlength=fftlength, overlap=overlap, **kwargs)
        if witness_bkg is not None:
            master_asd_wit_bkg = witness_bkg.asd(fftlength=fftlength, overlap=overlap, **kwargs)

        row_bytes = _estimate_response_row_bytes(
            witness,
            target,
            fftlength=fftlength,
            segment_duration=max(s[1] - s[0] for s in valid_segments),
        )
        if row_bytes > memory_limit:
            raise ValueError(
                f"memory_limit ({memory_limit} bytes) is too small for one response row "
                f"(estimated {row_bytes} bytes)."
            )
        batch_size = max(1, memory_limit // row_bytes)
        batches = [
            list(zip(seg_objs[i : i + batch_size], injected_freqs_list[i : i + batch_size]))
            for i in range(0, len(seg_objs), batch_size)
        ]

        Parallel = None
        delayed = None
        n_jobs_eff = 1 if n_jobs is None else n_jobs
        if n_jobs_eff != 1:
            from gwexpy.interop._optional import require_optional

            joblib = require_optional("joblib")
            Parallel, delayed = joblib.Parallel, joblib.delayed

        t_start = time.perf_counter()
        row_results: list[dict[str, Any]] = []
        compute_kwargs = dict(kwargs)

        for batch in batches:
            if Parallel is None:
                batch_results = [
                    _compute_response_row(
                        witness=witness,
                        target=target,
                        segment=segment,
                        injected_freq=injected_freq,
                        fftlength=fftlength,
                        overlap=overlap,
                        kwargs=compute_kwargs,
                        master_asd_wit_bkg=master_asd_wit_bkg,
                        master_asd_tgt_bkg=master_asd_tgt_bkg,
                    )
                    for segment, injected_freq in batch
                ]
            else:
                batch_results = Parallel(n_jobs=n_jobs_eff)(
                    delayed(_compute_response_row)(
                        witness=witness,
                        target=target,
                        segment=segment,
                        injected_freq=injected_freq,
                        fftlength=fftlength,
                        overlap=overlap,
                        kwargs=compute_kwargs,
                        master_asd_wit_bkg=master_asd_wit_bkg,
                        master_asd_tgt_bkg=master_asd_tgt_bkg,
                    )
                    for segment, injected_freq in batch
                )
            row_results.extend(batch_results)

        st = SegmentTable.from_segments(
            [row["span"] for row in row_results],
            injected_freq=[row["injected_freq"] for row in row_results],
        )
        st.add_series_column("wit_inj", data=[row["wit_inj"] for row in row_results], kind="timeseries")
        st.add_series_column("tgt_inj", data=[row["tgt_inj"] for row in row_results], kind="timeseries")
        st.add_series_column("wit_asd_inj", data=[row["wit_asd_inj"] for row in row_results], kind="frequencyseries")
        st.add_series_column("tgt_asd_inj", data=[row["tgt_asd_inj"] for row in row_results], kind="frequencyseries")
        st.add_series_column("wit_asd_bkg", data=[row["wit_asd_bkg"] for row in row_results], kind="frequencyseries")
        st.add_series_column("tgt_asd_bkg", data=[row["tgt_asd_bkg"] for row in row_results], kind="frequencyseries")
        st.add_column("cf", [row["cf"] for row in row_results], kind="meta")

        # --- 3. Wrap in Containers ---
        st_data = st.to_pandas(meta_only=False)
        valid_indices: list[int] = []
        ref_freqs: np.ndarray | None = None

        for idx, row in st_data.iterrows():
            row_series = [
                row["wit_asd_inj"],
                row["tgt_asd_inj"],
                row["wit_asd_bkg"],
                row["tgt_asd_bkg"],
            ]
            base_freqs = _freq_values(row_series[0])
            compatible_within_row = all(
                len(series.value) == len(row_series[0].value)
                and np.array_equal(_freq_values(series), base_freqs)
                for series in row_series[1:]
            )
            if not compatible_within_row:
                warnings.warn(
                    f"Skipping response row {idx} due to incompatible ASD frequency grids.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if ref_freqs is None:
                ref_freqs = base_freqs
            elif not np.array_equal(base_freqs, ref_freqs):
                warnings.warn(
                    f"Skipping response row {idx} due to incompatible ASD frequency grids.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            valid_indices.append(idx)

        if not valid_indices:
            raise ValueError("No compatible response rows remain after frequency alignment.")

        row_mask = [i in set(valid_indices) for i in range(len(st))]
        st = st.select(mask=row_mask)
        st_data = st_data.iloc[valid_indices].reset_index(drop=True)

        spec_inj_data = np.stack([v.value for v in st_data["tgt_asd_inj"]])
        spec_bkg_data = np.stack([v.value for v in st_data["tgt_asd_bkg"]])
        freq_axis = st_data["tgt_asd_inj"][0].xindex
        step_times = np.array([s[0] for s in st_data["span"]])
        cf_vals = st_data["cf"].values
        inj_freqs = st_data["injected_freq"].values

        unit = getattr(target, "unit", None) or "dimensionless"
        sg_inj = Spectrogram(spec_inj_data, times=step_times, frequencies=freq_axis, unit=unit, name="Inj")
        sg_bkg = Spectrogram(spec_bkg_data, times=step_times, frequencies=freq_axis, unit=unit, name="Bkg")

        res = ResponseFunctionResult(
            spectrogram_inj=sg_inj,
            spectrogram_bkg=sg_bkg,
            injected_freqs=inj_freqs,
            step_times=step_times,
            coupling_factors=cf_vals,
            witness_name=str(witness.name) if witness.name else "Witness",
            target_name=str(target.name) if target.name else "Target",
            table=st,
        )

        t_end = time.perf_counter()
        dur = t_end - t_start
        logger.info(
            "Response Function Analysis Complete: %d steps processed in %.2fs "
            "(batches=%d, n_jobs=%s).",
            len(st), dur, len(batches), n_jobs_eff,
        )
        return res


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
    """Backward-compatible wrapper for :class:`ResponseFunctionAnalysis`.

    This function preserves the historical module-level API used by external
    callers and examples. New code should prefer
    ``ResponseFunctionAnalysis().compute(...)``.
    """
    warnings.warn(
        "estimate_response_function() is deprecated; use "
        "ResponseFunctionAnalysis().compute(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ResponseFunctionAnalysis().compute(
        witness=witness,
        target=target,
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
