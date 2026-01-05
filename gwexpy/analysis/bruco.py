
import html
import logging
import os
from typing import List, Optional, Union, Dict, Any, Tuple, Sequence, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from astropy import units as u

# ロガーの設定
logger = logging.getLogger(__name__)


class FastCoherenceEngine:
    """
    Welch coherence engine that caches target FFT/PSD for reuse.
    """
    def __init__(self, target_data: TimeSeries, fftlength: float, overlap: float) -> None:
        self.fftlength = fftlength
        self.overlap = overlap
        self.sample_rate = float(target_data.sample_rate.value)
        self.nperseg = int(round(self.fftlength * self.sample_rate))
        if self.nperseg <= 0:
            raise ValueError("fftlength too small for sample rate")
        self.noverlap = int(round(self.overlap * self.sample_rate))
        if self.noverlap >= self.nperseg:
            raise ValueError("overlap must be smaller than fftlength")

        self._window = np.hanning(self.nperseg).astype(float)
        target_array = np.asarray(target_data.value, dtype=float)
        self._target_len = len(target_array)
        segments = self._segment_array(target_array, self.nperseg, self.noverlap)
        windowed = segments * self._window[None, :]
        self._target_fft = np.fft.rfft(windowed, axis=1)
        self._target_psd = np.mean(np.abs(self._target_fft) ** 2, axis=0)
        self.frequencies = np.fft.rfftfreq(self.nperseg, d=1.0 / self.sample_rate)

    @staticmethod
    def _segment_array(data: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
        step = nperseg - noverlap
        if step <= 0:
            raise ValueError("noverlap must be smaller than nperseg")
        if len(data) < nperseg:
            raise ValueError("data length shorter than fftlength")
        nseg = 1 + (len(data) - nperseg) // step
        shape = (nseg, nperseg)
        strides = (data.strides[0] * step, data.strides[0])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def compute_coherence(self, aux_data: TimeSeries) -> np.ndarray:
        if not np.isclose(float(aux_data.sample_rate.value), self.sample_rate):
            aux_data = aux_data.resample(self.sample_rate * u.Hz)
        aux_array = np.asarray(aux_data.value, dtype=float)
        if len(aux_array) < self._target_len:
            raise ValueError("aux data shorter than cached target length")
        if len(aux_array) > self._target_len:
            aux_array = aux_array[:self._target_len]

        segments = self._segment_array(aux_array, self.nperseg, self.noverlap)
        windowed = segments * self._window[None, :]
        aux_fft = np.fft.rfft(windowed, axis=1)
        aux_psd = np.mean(np.abs(aux_fft) ** 2, axis=0)

        csd = np.mean(self._target_fft * np.conj(aux_fft), axis=0)
        denom = self._target_psd * aux_psd
        coherence = np.zeros_like(denom, dtype=float)
        valid = denom > 0
        coherence[valid] = (np.abs(csd[valid]) ** 2) / denom[valid]
        return coherence


class BrucoResult:
    """
    Hold and analyze Bruco results with Top-N coherence per frequency bin.
    """
    def __init__(
        self,
        frequencies: np.ndarray,
        target_name: str,
        target_spectrum: np.ndarray,
        spectrum_kind: str = "asd",
        top_n: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            frequencies: Frequency vector (Hz).
            target_name: Name of the target channel.
            target_spectrum: ASD or PSD of the target channel (same length as frequencies).
            spectrum_kind: "asd" or "psd" to control projection scaling.
            top_n: Number of top channels to keep per frequency bin.
            metadata: Optional metadata dict for reporting.
        """
        if spectrum_kind not in ("asd", "psd"):
            raise ValueError("spectrum_kind must be 'asd' or 'psd'")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if len(frequencies) != len(target_spectrum):
            raise ValueError("frequencies and target_spectrum must have the same length")

        self.frequencies = frequencies
        self.target_name = target_name
        self.target_spectrum = target_spectrum
        self.spectrum_kind = spectrum_kind
        self.top_n = top_n
        self.metadata = metadata or {}
        self.n_bins = len(frequencies)

        # Top-N storage: [n_bins, top_n]
        self.top_coherence = np.zeros((self.n_bins, top_n), dtype=float)
        self.top_channels = np.full((self.n_bins, top_n), None, dtype=object)

    def update_batch(self, channel_names: Sequence[str], coherences: np.ndarray) -> None:
        """
        Update the Top-N records with a new batch of results.

        Args:
            channel_names: List of channel names in this batch.
            coherences: Coherence matrix of shape (n_channels, n_bins).
                        Must align to self.frequencies.
        """
        if len(channel_names) == 0:
            return
        if coherences.ndim != 2:
            raise ValueError("coherences must be a 2D array")
        if coherences.shape[0] != len(channel_names):
            raise ValueError("coherences rows must match channel_names length")
        if coherences.shape[1] != self.n_bins:
            raise ValueError("coherences columns must match number of frequency bins")

        coh_clean = np.nan_to_num(coherences, nan=0.0, posinf=0.0, neginf=0.0)
        batch_coh_t = coh_clean.T  # (bins, n_new)
        batch_names = np.asarray(channel_names, dtype=object)
        batch_names_tiled = np.tile(batch_names, (self.n_bins, 1))

        combined_coh = np.concatenate([self.top_coherence, batch_coh_t], axis=1)
        combined_names = np.concatenate([self.top_channels, batch_names_tiled], axis=1)

        row_indices = np.arange(self.n_bins)[:, None]
        if combined_coh.shape[1] <= self.top_n:
            sorted_indices = np.argsort(-combined_coh, axis=1)
            top_indices = sorted_indices[:, :self.top_n]
        else:
            partition_indices = np.argpartition(-combined_coh, self.top_n - 1, axis=1)[:, :self.top_n]
            partition_coh = combined_coh[row_indices, partition_indices]
            order = np.argsort(-partition_coh, axis=1)
            top_indices = partition_indices[row_indices, order]

        self.top_coherence = combined_coh[row_indices, top_indices]
        self.top_channels = combined_names[row_indices, top_indices]

    def _apply_projection(self, spectrum: np.ndarray, coherence: np.ndarray) -> np.ndarray:
        coh_safe = np.clip(coherence, 0.0, None)
        if self.spectrum_kind == "asd":
            return spectrum * np.sqrt(coh_safe)
        return spectrum * coh_safe

    def get_noise_projection(self, rank: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate noise projection for the channel at a specific rank (0 = highest coherence).

        Returns:
            (projection, coherence)
        """
        if rank >= self.top_n or rank < 0:
            raise ValueError(f"Rank {rank} is out of range for top_n={self.top_n}")

        coh = self.top_coherence[:, rank]
        proj = self._apply_projection(self.target_spectrum, coh)
        return proj, coh

    def projection_for_channel(self, channel: str) -> np.ndarray:
        """
        Calculate projection spectrum for a specific channel where it appears in Top-N.
        """
        projection = np.zeros(self.n_bins, dtype=float)
        for rank in range(self.top_n):
            mask = self.top_channels[:, rank] == channel
            if np.any(mask):
                projection[mask] = self._apply_projection(
                    self.target_spectrum[mask],
                    self.top_coherence[mask, rank],
                )
        return projection

    def dominant_channel(self, rank: int = 0) -> Optional[str]:
        """
        Return the most frequent channel name at a given rank.
        """
        if rank >= self.top_n or rank < 0:
            raise ValueError(f"Rank {rank} is out of range for top_n={self.top_n}")
        channels = self.top_channels[:, rank]
        channels = channels[channels != None]  # noqa: E711
        if channels.size == 0:
            return None
        values, counts = np.unique(channels, return_counts=True)
        return values[np.argmax(counts)]

    def to_dataframe(self, ranks: Optional[Sequence[int]] = None, stride: int = 1) -> pd.DataFrame:
        """
        Convert results to a long-form DataFrame.
        """
        if ranks is None:
            ranks = list(range(self.top_n))
        ranks = [r for r in ranks if 0 <= r < self.top_n]
        stride = max(stride, 1)

        frames = []
        freqs = self.frequencies[::stride]
        spectrum = self.target_spectrum[::stride]
        for rank in ranks:
            coh = self.top_coherence[::stride, rank]
            proj = self._apply_projection(spectrum, coh)
            ch = self.top_channels[::stride, rank]
            frames.append(
                pd.DataFrame(
                    {
                        "frequency": freqs,
                        "rank": rank + 1,
                        "channel": ch,
                        "coherence": coh,
                        "projection": proj,
                    }
                )
            )
        if not frames:
            return pd.DataFrame(columns=["frequency", "rank", "channel", "coherence", "projection"])
        return pd.concat(frames, ignore_index=True)

    def plot_projection(
        self,
        ranks: Optional[Sequence[int]] = None,
        channels: Optional[Sequence[str]] = None,
        max_channels: int = 3,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot target spectrum and noise projections for selected ranks or channels.
        """
        if ranks is None and channels is None:
            ranks = list(range(min(self.top_n, max_channels)))
        if ranks is not None and isinstance(ranks, int):
            ranks = [ranks]
        if channels is not None and isinstance(channels, str):
            channels = [channels]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.loglog(self.frequencies, self.target_spectrum, label=f"Target: {self.target_name}", color="black", linewidth=1.5)

        if ranks is not None:
            for rank in ranks:
                proj, _ = self.get_noise_projection(rank)
                dominant = self.dominant_channel(rank)
                label = f"Projection (Rank {rank + 1})"
                if dominant:
                    label = f"{label}: {dominant}"
                ax.loglog(self.frequencies, proj, label=label, alpha=0.8)

        if channels is not None:
            for channel in channels[:max_channels]:
                proj = self.projection_for_channel(channel)
                ax.loglog(self.frequencies, proj, label=f"Projection: {channel}", alpha=0.8)

        ax.set_xlabel("Frequency [Hz]")
        ylabel = "ASD" if self.spectrum_kind == "asd" else "PSD"
        ax.set_ylabel(ylabel)
        ax.set_title(f"Noise Projection: {self.target_name}")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.4)
        if len(self.frequencies) > 1:
            ax.set_xlim(self.frequencies[1], self.frequencies[-1])

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        return fig

    def plot_coherence(
        self,
        ranks: Optional[Sequence[int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot coherence spectrum for selected ranks.
        """
        if ranks is None:
            ranks = [0]
        if isinstance(ranks, int):
            ranks = [ranks]

        fig, ax = plt.subplots(figsize=(12, 6))
        for rank in ranks:
            coh = self.top_coherence[:, rank]
            label = f"Rank {rank + 1}"
            ax.semilogx(self.frequencies, coh, label=label)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Coherence")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.legend()
        if len(self.frequencies) > 1:
            ax.set_xlim(self.frequencies[1], self.frequencies[-1])

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        return fig

    def _coherence_color(self, value: float) -> str:
        value = float(np.clip(value, 0.0, 1.0))
        red = 255
        green = int(255 - 160 * value)
        blue = int(255 - 160 * value)
        return f"rgb({red},{green},{blue})"

    def generate_report(
        self,
        output_dir: str,
        max_rows: int = 2000,
        coherence_threshold: float = 0.5,
        plot_ranks: int = 3,
    ) -> str:
        """
        Generate an HTML report with plots and data summary.

        Returns:
            Path to the generated HTML file.
        """
        os.makedirs(output_dir, exist_ok=True)

        proj_plot_path = os.path.join(output_dir, "projection.png")
        self.plot_projection(ranks=list(range(min(plot_ranks, self.top_n))), save_path=proj_plot_path)

        coh_plot_path = os.path.join(output_dir, "coherence.png")
        self.plot_coherence(ranks=list(range(min(plot_ranks, self.top_n))), save_path=coh_plot_path)

        stride = max(1, int(np.ceil(self.n_bins / max_rows))) if max_rows else 1
        table_freqs = self.frequencies[::stride]
        table_channels = self.top_channels[::stride, 0]
        table_coh = self.top_coherence[::stride, 0]
        table_proj = self._apply_projection(self.target_spectrum[::stride], table_coh)

        rows = []
        for freq, ch, coh, proj in zip(table_freqs, table_channels, table_coh, table_proj):
            color = self._coherence_color(coh)
            rows.append(
                "<tr>"
                f"<td>{freq:.3f}</td>"
                f"<td>{html.escape(str(ch))}</td>"
                f"<td style='background-color:{color}'>{coh:.3f}</td>"
                f"<td>{proj:.3e}</td>"
                "</tr>"
            )
        top_table_html = (
            "<h3>Top Channel per Frequency Bin (Rank 1)</h3>"
            "<table class='table'>"
            "<thead><tr><th>Frequency [Hz]</th><th>Channel</th><th>Coherence</th><th>Projection</th></tr></thead>"
            "<tbody>"
            + "\n".join(rows)
            + "</tbody></table>"
        )

        peak_mask = self.top_coherence[:, 0] >= coherence_threshold
        peak_freqs = self.frequencies[peak_mask]
        peak_ch = self.top_channels[peak_mask, 0]
        peak_coh = self.top_coherence[peak_mask, 0]
        peak_proj = self._apply_projection(self.target_spectrum[peak_mask], peak_coh)
        peaks_df = pd.DataFrame(
            {
                "Frequency [Hz]": peak_freqs,
                "Channel": peak_ch,
                "Coherence": peak_coh,
                "Projection": peak_proj,
            }
        )
        if len(peaks_df) > 200:
            peaks_df = peaks_df.sort_values("Coherence", ascending=False).head(200)
        peaks_html = "<h3>Significant Coherence Peaks</h3>"
        if len(peaks_df) > 0:
            peaks_html += peaks_df.to_html(classes="table", index=False)
        else:
            peaks_html += "<p>No peaks above threshold.</p>"

        meta_rows = []
        for key, value in self.metadata.items():
            meta_rows.append(
                f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            )
        meta_html = ""
        if meta_rows:
            meta_html = (
                "<h3>Run Summary</h3>"
                "<table class='table'>"
                + "".join(meta_rows)
                + "</table>"
            )

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bruco Report: {html.escape(self.target_name)}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        img {{ max-width: 100%; border: 1px solid #ddd; margin-bottom: 20px; }}
        .table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
        .table th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Bruco Analysis Report</h1>
    <h2>Target: {html.escape(self.target_name)}</h2>
    {meta_html}
    <h3>Noise Projection</h3>
    <img src="projection.png" alt="Noise Projection Plot">
    <h3>Coherence Spectra</h3>
    <img src="coherence.png" alt="Coherence Plot">
    {top_table_html}
    {peaks_html}
</body>
</html>
"""

        report_path = os.path.join(output_dir, "index.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Report generated at %s", report_path)
        return report_path


class Bruco:
    """
    Brute force Coherence (Bruco) scanner.

    Attributes:
        target (str): The name of the target channel (e.g., DARM).
        aux_channels (List[str]): List of auxiliary channels to scan.
        excluded (List[str]): List of channels to exclude from analysis.
    """

    def __init__(self, target_channel: str, aux_channels: List[str], excluded_channels: Optional[List[str]] = None):
        """
        Initialize the Bruco scanner.

        Args:
            target_channel (str): The main channel to analyze.
            aux_channels (List[str]): A list of all available auxiliary channels.
            excluded_channels (List[str], optional): Channels to ignore (e.g., calibration lines).
        """
        self.target = target_channel
        self.aux_channels = aux_channels
        self.excluded = set(excluded_channels) if excluded_channels else set()

        # Exclude target and excluded channels
        self.channels_to_scan = sorted(list(
            set(self.aux_channels) - self.excluded - {self.target}
        ))

        logger.info(f"Bruco initialized. Target: {self.target}, "
                    f"Auxiliary channels: {len(self.channels_to_scan)} (after exclusions)")

    def compute(
        self,
        start: Union[int, float],
        duration: int,
        fftlength: float = 2.0,
        overlap: float = 1.0,
        nproc: int = 4,
        batch_size: int = 100,
        top_n: int = 5,
        spectrum: str = "asd",
    ) -> BrucoResult:
        """
        Execute the coherence scan.

        Args:
            start (int or float): GPS start time.
            duration (int): Duration of data in seconds.
            fftlength (float): FFT length in seconds.
            overlap (float): Overlap in seconds.
            nproc (int): Number of parallel processes.
            batch_size (int): Channels per batch.
            top_n (int): Number of top channels to keep per frequency bin.
            spectrum (str): "asd" or "psd" for target spectrum and projection.

        Returns:
            BrucoResult: Object containing frequency-wise analysis results.
        """
        end = start + duration
        if fftlength > duration:
            raise ValueError(f"fftlength ({fftlength}) cannot be longer than duration ({duration}).")
        if spectrum not in ("asd", "psd"):
            raise ValueError("spectrum must be 'asd' or 'psd'")

        logger.info(f"Starting Bruco scan at {start} for {duration}s. Batch size: {batch_size}")

        # 1. Fetch Target Data & Calculate Reference Spectrum
        try:
            logger.info(f"Fetching target data: {self.target}")
            target_ts = TimeSeries.get(self.target, start, end)

            # Calculate Target ASD/PSD and Frequencies for reference
            if spectrum == "asd":
                target_spectrum = target_ts.asd(fftlength=fftlength, overlap=overlap)
            else:
                target_spectrum = target_ts.psd(fftlength=fftlength, overlap=overlap)
            target_frequencies = target_spectrum.frequencies.value
            target_spectrum_values = target_spectrum.value

            # Store Frequency resolution for alignment checks
            df = 1.0 / fftlength
            # Verify df matches
            if len(target_frequencies) > 1:
                actual_df = target_frequencies[1] - target_frequencies[0]
                if not np.isclose(df, actual_df, rtol=1e-3):
                    logger.warning(f"Calculated df {df} does not match spectrum df {actual_df}")

        except Exception as e:
            logger.error(f"Failed to fetch or process target channel {self.target}: {e}")
            raise e

        # Initialize Result container
        metadata = {
            "start": start,
            "end": end,
            "duration": duration,
            "fftlength": fftlength,
            "overlap": overlap,
            "spectrum": spectrum,
            "top_n": top_n,
            "batch_size": batch_size,
            "nproc": nproc,
            "target": self.target,
            "target_sample_rate": target_ts.sample_rate.value,
        }
        result = BrucoResult(
            target_frequencies,
            self.target,
            target_spectrum_values,
            spectrum_kind=spectrum,
            top_n=top_n,
            metadata=metadata,
        )

        # 2. Batch Processing
        total_channels = len(self.channels_to_scan)

        for i in range(0, total_channels, batch_size):
            batch_channels = self.channels_to_scan[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(total_channels // batch_size) + 1} "
                        f"({len(batch_channels)} channels)")

            # Fetch batch data
            try:
                aux_data = TimeSeriesDict.get(batch_channels, start, end, allow_tape=True, nproc=nproc)
            except Exception as e:
                logger.warning(f"Batch fetch failed: {e}")
                continue

            # Calculate Coherence for batch
            # Returns list of (name, coherence_array)
            batch_results_list = self._process_batch(
                target_ts,
                aux_data,
                fftlength,
                overlap,
                nproc,
                target_frequencies,
            )

            # Unpack results
            valid_names = []
            valid_coherences = []

            for res in batch_results_list:
                if res is not None:
                    name, coh = res
                    valid_names.append(name)
                    valid_coherences.append(coh)

            if valid_names:
                # Stack coherences -> (n_channels, n_bins)
                batch_coh_matrix = np.vstack(valid_coherences)
                result.update_batch(valid_names, batch_coh_matrix)

        logger.info("Scan complete.")
        return result

    def _process_batch(
        self,
        target_ts: TimeSeries,
        aux_dict: TimeSeriesDict,
        fftlength: float,
        overlap: float,
        nproc: int,
        target_frequencies: np.ndarray,
    ) -> List[Optional[Tuple[str, np.ndarray]]]:
        """
        Helper method to process a single batch.
        Passes target_frequencies to worker to ensure alignment.
        """
        aux_list = list(aux_dict.values())
        if not aux_list:
            return []
        if nproc <= 1:
            return self._calculate_batch_coherence_fast(
                target_ts,
                aux_list,
                fftlength,
                overlap,
                target_frequencies,
            )

        chunk_size = max(1, int(np.ceil(len(aux_list) / nproc)))
        results: List[Optional[Tuple[str, np.ndarray]]] = []
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = [
                executor.submit(
                    self._calculate_batch_coherence_fast,
                    target_ts,
                    chunk,
                    fftlength,
                    overlap,
                    target_frequencies,
                )
                for chunk in self._chunk_aux_list(aux_list, chunk_size)
            ]
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as exc:
                    logger.debug(f"Batch coherence failed: {exc}")

        return results

    @staticmethod
    def _calculate_aligned_coherence(
        target: TimeSeries,
        aux: TimeSeries,
        fftlength: float,
        overlap: float,
        target_frequencies: np.ndarray,
    ) -> Optional[Tuple[str, np.ndarray]]:
        """
        Worker: Calculate coherence and align to target frequencies.
        Returns (channel_name, coherence_values_aligned).
        """
        try:
            # Resampling logic
            if target.sample_rate != aux.sample_rate:
                sr_target = target.sample_rate.value
                sr_aux = aux.sample_rate.value
                common_rate = min(sr_target, sr_aux)

                if sr_target > common_rate:
                    target = target.resample(common_rate * u.Hz)
                if sr_aux > common_rate:
                    aux = aux.resample(common_rate * u.Hz)

            # Crop length
            if len(target) != len(aux):
                min_len = min(len(target), len(aux))
                target = target[:min_len]
                aux = aux[:min_len]

            # Calculate Coherence
            # gwpy coherence returns a FrequencySeries
            coh = target.coherence(aux, fftlength=fftlength, overlap=overlap)
            coh_val = coh.value
            coh_freqs = coh.frequencies.value

            # Align to target_frequencies
            # If sampling rates were identical, and params identical, they should match.
            # But if aux was downsampled, coh is shorter.

            final_coh = np.zeros_like(target_frequencies, dtype=float)

            # Case 1: Identical match (most common if same SR)
            if len(coh_val) == len(target_frequencies):
                final_coh = coh_val

            # Case 2: Aux range is smaller (lower sampling rate)
            elif len(coh_val) < len(target_frequencies):
                # Assume starting at 0Hz and same df
                # Copy what we have
                n = len(coh_val)
                final_coh[:n] = coh_val
                # Leave rest as 0

            # Case 3: Aux range is larger (higher sampling rate? Should not happen due to resample logic)
            # or frequency bins slightly off.
            else:
                # Interpolate just to be safe, or slice if df matches
                # If df is same, slice.
                if len(coh_freqs) > 1 and len(target_frequencies) > 1 and np.isclose(
                    coh_freqs[1] - coh_freqs[0],
                    target_frequencies[1] - target_frequencies[0],
                ):
                    final_coh = coh_val[:len(target_frequencies)]
                else:
                    # Interpolate
                    f = interp1d(coh_freqs, coh_val, bounds_error=False, fill_value=0.0)
                    final_coh = f(target_frequencies)

            return (aux.name, final_coh)

        except Exception:
            return None

    @staticmethod
    def _chunk_aux_list(
        aux_list: List[TimeSeries],
        chunk_size: int,
    ) -> Iterable[List[TimeSeries]]:
        for idx in range(0, len(aux_list), chunk_size):
            yield aux_list[idx : idx + chunk_size]

    @staticmethod
    def _calculate_batch_coherence_fast(
        target: TimeSeries,
        aux_list: List[TimeSeries],
        fftlength: float,
        overlap: float,
        target_frequencies: np.ndarray,
    ) -> List[Optional[Tuple[str, np.ndarray]]]:
        results: List[Optional[Tuple[str, np.ndarray]]] = []
        engine_cache: Dict[Tuple[float, int], FastCoherenceEngine] = {}
        target_cache: Dict[float, TimeSeries] = {}

        for aux in aux_list:
            try:
                res = Bruco._calculate_fast_coherence(
                    target,
                    aux,
                    fftlength,
                    overlap,
                    target_frequencies,
                    engine_cache,
                    target_cache,
                )
                results.append(res)
            except Exception as exc:
                logger.debug(f"Calculation failed for {aux.name}: {exc}")
                results.append(None)

        return results

    @staticmethod
    def _calculate_fast_coherence(
        target: TimeSeries,
        aux: TimeSeries,
        fftlength: float,
        overlap: float,
        target_frequencies: np.ndarray,
        engine_cache: Dict[Tuple[float, int], FastCoherenceEngine],
        target_cache: Dict[float, TimeSeries],
    ) -> Optional[Tuple[str, np.ndarray]]:
        try:
            sr_target = float(target.sample_rate.value)
            sr_aux = float(aux.sample_rate.value)
            if np.isclose(sr_target, sr_aux):
                common_rate = sr_target
                target_rs = target
                aux_rs = aux
            else:
                common_rate = min(sr_target, sr_aux)
                if sr_target > common_rate:
                    target_rs = target_cache.get(common_rate)
                    if target_rs is None:
                        target_rs = target.resample(common_rate * u.Hz)
                        target_cache[common_rate] = target_rs
                else:
                    target_rs = target

                if sr_aux > common_rate:
                    aux_rs = aux.resample(common_rate * u.Hz)
                else:
                    aux_rs = aux

            min_len = min(len(target_rs), len(aux_rs))
            if min_len <= 0:
                return None
            if len(target_rs) != min_len:
                target_rs = target_rs[:min_len]
            if len(aux_rs) != min_len:
                aux_rs = aux_rs[:min_len]

            cache_key = (float(common_rate), int(min_len))
            engine = engine_cache.get(cache_key)
            if engine is None:
                engine = FastCoherenceEngine(target_rs, fftlength, overlap)
                engine_cache[cache_key] = engine

            coh_val = engine.compute_coherence(aux_rs)
            final_coh = np.zeros_like(target_frequencies, dtype=float)

            if len(coh_val) == len(target_frequencies):
                final_coh = coh_val
            elif len(coh_val) < len(target_frequencies):
                final_coh[:len(coh_val)] = coh_val
            else:
                engine_freqs = engine.frequencies
                if len(engine_freqs) > 1 and len(target_frequencies) > 1 and np.isclose(
                    engine_freqs[1] - engine_freqs[0],
                    target_frequencies[1] - target_frequencies[0],
                ):
                    final_coh = coh_val[:len(target_frequencies)]
                else:
                    interp = interp1d(engine_freqs, coh_val, bounds_error=False, fill_value=0.0)
                    final_coh = interp(target_frequencies)

            return (aux.name, final_coh)
        except Exception:
            return None

    @staticmethod
    def _calculate_pair_coherence(
        target: TimeSeries,
        aux: TimeSeries,
        fftlength: float,
        overlap: float,
    ) -> Dict[str, Any]:
        """
        Convenience helper to summarize coherence for a single pair.
        """
        coh = target.coherence(aux, fftlength=fftlength, overlap=overlap)
        coh_val = coh.value
        coh_freqs = coh.frequencies.value
        if len(coh_val) == 0:
            return {"channel": aux.name, "max_coherence": 0.0, "freq_at_max": 0.0}
        idx = int(np.argmax(coh_val))
        return {
            "channel": aux.name,
            "max_coherence": float(coh_val[idx]),
            "freq_at_max": float(coh_freqs[idx]),
        }
