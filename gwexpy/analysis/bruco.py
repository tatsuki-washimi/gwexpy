from __future__ import annotations

import html
import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypedDict

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from scipy.interpolate import interp1d

# ロガーの設定
logger = logging.getLogger(__name__)
_BRUCO_BLOCK_SIZE_DEFAULT = 256
_BRUCO_BLOCK_BYTES_DEFAULT = 64 * 1024 * 1024
_BRUCO_BLOCK_SIZE_MIN = 16
_BRUCO_BLOCK_SIZE_MAX = 1024

BrucoMetadataValue: TypeAlias = str | int | float | bool
BrucoMetadata: TypeAlias = Mapping[str, BrucoMetadataValue]


class BrucoPairSummary(TypedDict):
    channel: str
    max_coherence: float
    freq_at_max: float


def _auto_block_size(n_bins: int, top_n: int) -> int:
    if n_bins <= 0:
        return _BRUCO_BLOCK_SIZE_DEFAULT
    try:
        budget = int(os.getenv("GWEXPY_BRUCO_BLOCK_BYTES", _BRUCO_BLOCK_BYTES_DEFAULT))
    except ValueError:
        budget = _BRUCO_BLOCK_BYTES_DEFAULT
    if budget <= 0:
        budget = _BRUCO_BLOCK_BYTES_DEFAULT
    max_cols = (budget // (n_bins * 8)) - top_n
    if max_cols < _BRUCO_BLOCK_SIZE_MIN:
        return _BRUCO_BLOCK_SIZE_MIN
    if max_cols > _BRUCO_BLOCK_SIZE_MAX:
        return _BRUCO_BLOCK_SIZE_MAX
    return int(max_cols)


def _resolve_block_size(block_size: int | str | None, n_bins: int, top_n: int) -> int:
    if block_size is None:
        block_size = os.getenv("GWEXPY_BRUCO_BLOCK_SIZE", _BRUCO_BLOCK_SIZE_DEFAULT)

    if isinstance(block_size, str):
        if block_size.lower() == "auto":
            return _auto_block_size(n_bins, top_n)
        try:
            block_size = int(block_size)
        except ValueError as exc:
            raise ValueError("block_size must be an int or 'auto'") from exc

    if not isinstance(block_size, int):
        raise TypeError("block_size must be an int or 'auto'")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    return block_size


class FastCoherenceEngine:
    """
    Welch coherence engine that caches target FFT/PSD for reuse.
    """

    def __init__(
        self, target_data: TimeSeries, fftlength: float, overlap: float
    ) -> None:
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
        # Scaling factor to match scipy.signal.welch (scaling='density')
        # - 1/_window_power: Normalizae for window energy loss
        # - 2.0: Compensate for one-sided RFFT energy (excluding DC/Nyquist ideally,
        #        but 2.0 is the standard approximation for density scaling)
        # - 1/sample_rate: Convert to V^2/Hz (Power Spectral Density)
        self._window_power = np.sum(self._window**2)
        self._scale = (
            2.0 / (self.sample_rate * self._window_power)
            if self.sample_rate > 0
            else 1.0
        )

        target_array = np.asarray(target_data.value, dtype=float)
        self._target_len = len(target_array)
        segments = self._segment_array(target_array, self.nperseg, self.noverlap)
        windowed = segments * self._window[None, :]
        self._target_fft = np.fft.rfft(windowed, axis=1)
        self._target_psd = np.mean(np.abs(self._target_fft) ** 2, axis=0) * self._scale
        self.frequencies = np.fft.rfftfreq(self.nperseg, d=1.0 / self.sample_rate)

    @staticmethod
    def _segment_array(data: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
        step = nperseg - noverlap
        if step <= 0:
            raise ValueError("noverlap must be smaller than nperseg")
        if len(data) < nperseg:
            raise ValueError("data length shorter than fftlength")
        # Ensure contiguous array to prevent undefined behavior with as_strided
        data = np.ascontiguousarray(data)
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
            aux_array = aux_array[: self._target_len]

        segments = self._segment_array(aux_array, self.nperseg, self.noverlap)
        windowed = segments * self._window[None, :]
        aux_fft = np.fft.rfft(windowed, axis=1)
        aux_psd = np.mean(np.abs(aux_fft) ** 2, axis=0) * self._scale

        csd = np.mean(self._target_fft * np.conj(aux_fft), axis=0) * self._scale
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
        top_n: int = 5,
        metadata: BrucoMetadata | None = None,
        block_size: int | str | None = None,
    ) -> None:
        """
        Args:
            frequencies: Frequency vector (Hz).
            target_name: Name of the target channel.
            target_spectrum: PSD of the target channel (same length as frequencies).
            top_n: Number of top channels to keep per frequency bin.
            metadata: Optional metadata dict for reporting.
            block_size: Channels per block in Top-N updates (int or 'auto').
        """

        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if len(frequencies) != len(target_spectrum):
            raise ValueError(
                "frequencies and target_spectrum must have the same length"
            )

        self.frequencies = frequencies
        self.target_name = target_name
        self.target_spectrum = target_spectrum
        # Internal calculations use PSD; ASD is derived only for display.
        self.top_n = top_n
        self.metadata: dict[str, BrucoMetadataValue] = (
            dict(metadata) if metadata is not None else {}
        )
        self.n_bins = len(frequencies)
        self.block_size = _resolve_block_size(block_size, self.n_bins, self.top_n)

        # Top-N storage: [n_bins, top_n]
        self.top_coherence = np.zeros((self.n_bins, top_n), dtype=float)
        self.top_channels = np.full((self.n_bins, top_n), None, dtype=object)

    def update_batch(
        self, channel_names: Sequence[str], coherences: np.ndarray
    ) -> None:
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
        batch_names = np.asarray(channel_names, dtype=object)

        # Blocked top-k update to cap memory while keeping vectorized selection.
        block_size = min(batch_names.size, self.block_size)
        for start in range(0, batch_names.size, block_size):
            end = start + block_size
            block_names = batch_names[start:end]
            block_coh = coh_clean[start:end]
            block_coh_t = block_coh.T  # (bins, n_block)
            if block_coh_t.size == 0:
                continue

            open_slots = np.any(np.equal(self.top_channels, None), axis=1)
            block_max = np.max(block_coh_t, axis=1)
            needs_update = open_slots | (block_max > self.top_coherence[:, -1])
            if not np.any(needs_update):
                continue

            combined_coh = np.concatenate(
                [self.top_coherence[needs_update], block_coh_t[needs_update]],
                axis=1,
            )
            name_block = np.broadcast_to(
                block_names, (combined_coh.shape[0], block_names.size)
            )
            combined_names = np.concatenate(
                [self.top_channels[needs_update], name_block],
                axis=1,
            )

            row_indices = np.arange(combined_coh.shape[0])[:, None]
            if combined_coh.shape[1] <= self.top_n:
                sorted_indices = np.argsort(-combined_coh, axis=1)
                top_indices = sorted_indices[:, : self.top_n]
            else:
                partition_indices = np.argpartition(
                    -combined_coh, self.top_n - 1, axis=1
                )[:, : self.top_n]
                partition_coh = combined_coh[row_indices, partition_indices]
                order = np.argsort(-partition_coh, axis=1)
                top_indices = partition_indices[row_indices, order]

            self.top_coherence[needs_update] = combined_coh[row_indices, top_indices]
            self.top_channels[needs_update] = combined_names[row_indices, top_indices]

    def _apply_projection(
        self,
        spectrum: np.ndarray,
        coherence: np.ndarray,
        asd: bool = True,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Apply coherence to PSD and optionally return ASD for display."""
        coh_safe = np.clip(coherence, 0.0, None)
        if threshold > 0:
            comparison_thresh = threshold**2 if asd else threshold
            coh_safe = np.where(coh_safe < comparison_thresh, np.nan, coh_safe)

        proj_psd = spectrum * coh_safe
        if asd:
            return np.sqrt(proj_psd)
        return proj_psd

    def get_noise_projection(
        self,
        rank: int = 0,
        asd: bool = True,
        coherence_threshold: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate noise projection for the channel at a specific rank (0 = highest coherence).

        Args:
            asd: If True (default), return ASD projection. If False, return PSD projection.
            coherence_threshold: Frequencies with coherence below this value contribute zero noise.

        Returns:
            (projection, coherence)
        """
        if rank >= self.top_n or rank < 0:
            raise ValueError(f"Rank {rank} is out of range for top_n={self.top_n}")

        coh = self.top_coherence[:, rank]
        proj = self._apply_projection(
            self.target_spectrum, coh, asd=asd, threshold=coherence_threshold
        )
        return proj, coh

    def projection_for_channel(
        self,
        channel: str,
        asd: bool = True,
        coherence_threshold: float = 0.0,
    ) -> np.ndarray:
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
                    asd=asd,
                    threshold=coherence_threshold,
                )
        return projection

    def dominant_channel(self, rank: int = 0) -> str | None:
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

    def get_ranked_channels(self, limit: int = 5) -> list[str]:
        """
        Get a list of channels ranked by their total coherence contribution.

        Args:
            limit: Maximum number of channels to return.

        Returns:
            List of channel names sorted by importance.
        """
        channel_scores: dict[str, float] = {}
        for rank in range(self.top_n):
            names = self.top_channels[:, rank]
            cohs = self.top_coherence[:, rank]
            # Vectorized summation for this rank?
            # Channels are mixed in the column.
            # Iteration might be slow if bins are huge, but numpy unique is fast.
            # Let's iterate over unique names in this rank.
            unique_names = np.unique(names[names != None])  # noqa: E711
            for name in unique_names:
                mask = names == name
                score = np.nanmax(cohs[mask])
                channel_scores[name] = max(channel_scores.get(name, 0.0), score)

        sorted_channels = sorted(
            channel_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [name for name, score in sorted_channels[:limit]]

    def coherence_for_channel(
        self,
        channel: str,
        asd: bool = True,
    ) -> np.ndarray:
        """
        Get the coherence spectrum for a specific channel.
        Values are NaN where the channel is not in the Top-N.

        Args:
            channel: Channel name.
            asd: If True, return Amplitude Coherence. If False, Squared Coherence.

        Returns:
            Coherence spectrum (same length as frequencies).
        """
        coherence = np.full(self.n_bins, np.nan, dtype=float)

        for rank in range(self.top_n):
            mask = self.top_channels[:, rank] == channel
            if np.any(mask):
                coherence[mask] = self.top_coherence[mask, rank]

        if asd:
            return np.sqrt(coherence)
        return coherence

    def to_dataframe(
        self,
        ranks: Sequence[int] | None = None,
        stride: int = 1,
        asd: bool = True,
        coherence_threshold: float = 0.0,
    ) -> pd.DataFrame:
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
            proj = self._apply_projection(
                spectrum, coh, asd=asd, threshold=coherence_threshold
            )
            ch = self.top_channels[::stride, rank]
            coh_display = np.sqrt(np.clip(coh, 0.0, None)) if asd else coh
            frames.append(
                pd.DataFrame(
                    {
                        "frequency": freqs,
                        "rank": rank + 1,
                        "channel": ch,
                        "coherence": coh_display,
                        "projection": proj,
                    }
                )
            )
        if not frames:
            return pd.DataFrame(
                columns=["frequency", "rank", "channel", "coherence", "projection"]
            )
        return pd.concat(frames, ignore_index=True)

    def plot_projection(
        self,
        ranks: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
        max_channels: int = 3,
        asd: bool = True,
        coherence_threshold: float = 0.0,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot target spectrum and noise projections for selected ranks or channels.

        Default behavior (ranks=None, channels=None):
            Plots the Top-K contributors (per-channel mode).
        """
        if ranks is None and channels is None:
            channels = self.get_ranked_channels(max_channels)

        if ranks is not None and isinstance(ranks, int):
            ranks = [ranks]
        if channels is not None and isinstance(channels, str):
            channels = [channels]

        fig, ax = plt.subplots(figsize=(12, 8))

        target = self.target_spectrum
        if asd:
            target = np.sqrt(target)

        ax.loglog(
            self.frequencies,
            target,
            label=f"Target: {self.target_name}",
            color="black",
            linewidth=1,
            alpha=0.5,
        )

        if ranks is not None:
            for rank in ranks:
                proj, _ = self.get_noise_projection(
                    rank, asd=asd, coherence_threshold=coherence_threshold
                )
                dominant = self.dominant_channel(rank)
                label = f"Projection (Rank {rank + 1})"
                if dominant:
                    label = f"{label} [Dom: {dominant}]"
                ax.loglog(
                    self.frequencies,
                    proj,
                    label=label,
                    alpha=0.9,
                    linewidth=2,
                    marker=".",
                )

        if channels is not None:
            for channel in channels[:max_channels]:
                proj = self.projection_for_channel(
                    channel, asd=asd, coherence_threshold=coherence_threshold
                )
                ax.loglog(
                    self.frequencies,
                    proj,
                    label=f"Projection: {channel}",
                    alpha=0.9,
                    linewidth=2,
                    marker=".",
                )

        ax.set_xlabel("Frequency [Hz]")
        ylabel = "ASD" if asd else "PSD"
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
        ranks: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
        max_channels: int = 3,
        asd: bool = True,
        coherence_threshold: float = 0.0,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot coherence spectrum for selected ranks or channels.

        Default behavior (ranks=None, channels=None):
            Plots the Top-K contributors (per-channel mode).

        Args:
            asd: If True (default), plot Amplitude Coherence (sqrt(Coh^2)).
                 If False, plot Squared Coherence (Coh^2).
            coherence_threshold: Draw a horizontal line at this value (default 0.0=off).
        """
        if ranks is None and channels is None:
            channels = self.get_ranked_channels(max_channels)

        if ranks is not None and isinstance(ranks, int):
            ranks = [ranks]
        if channels is not None and isinstance(channels, str):
            channels = [channels]

        fig, ax = plt.subplots(figsize=(12, 6))

        if ranks is not None:
            for rank in ranks:
                coh = self.top_coherence[:, rank]
                if asd:
                    coh = np.sqrt(np.clip(coh, 0.0, None))
                label = f"Rank {rank + 1}"
                ax.semilogx(self.frequencies, coh, label=label)

        if channels is not None:
            for channel in channels:
                coh = self.coherence_for_channel(channel, asd=asd)
                ax.semilogx(
                    self.frequencies, coh, label=f"Coherence: {channel}", alpha=0.9
                )
        ax.set_xlabel("Frequency [Hz]")
        ylabel = "Coherence (Amplitude)" if asd else "Coherence (Power)"
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)

        # Plot threshold line
        if coherence_threshold > 0:
            ax.axhline(
                coherence_threshold,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold ({coherence_threshold:.2g})",
            )

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
        asd: bool = True,
    ) -> str:
        """
        Generate an HTML report with plots and data summary.

        Args:
            asd: If True (default), report and plots use ASD units.

        Returns:
            Path to the generated HTML file.
        """
        os.makedirs(output_dir, exist_ok=True)

        proj_plot_path = os.path.join(output_dir, "projection.png")
        self.plot_projection(
            ranks=list(range(min(plot_ranks, self.top_n))),
            asd=asd,
            coherence_threshold=coherence_threshold,
            save_path=proj_plot_path,
        )
        coh_plot_path = os.path.join(output_dir, "coherence.png")
        self.plot_coherence(
            ranks=list(range(min(plot_ranks, self.top_n))),
            asd=asd,
            coherence_threshold=coherence_threshold,
            save_path=coh_plot_path,
        )

        stride = max(1, int(np.ceil(self.n_bins / max_rows))) if max_rows else 1
        table_freqs = self.frequencies[::stride]
        table_channels = self.top_channels[::stride, 0]
        table_coherent_values = self.top_coherence[::stride, 0]

        table_proj = self._apply_projection(
            self.target_spectrum[::stride],
            table_coherent_values,
            asd=asd,
            threshold=coherence_threshold,
        )

        rows = []
        for freq, ch, coh, proj in zip(
            table_freqs, table_channels, table_coherent_values, table_proj
        ):
            # Display coherence value (amplitude or squared)
            display_coh = coh
            if asd:
                display_coh = np.sqrt(np.clip(coh, 0.0, None))

            color = self._coherence_color(coh)  # Color logic uses Squared [0-1]
            rows.append(
                "<tr>"
                f"<td>{freq:.3f}</td>"
                f"<td>{html.escape(str(ch))}</td>"
                f"<td style='background-color:{color}'>{display_coh:.3f}</td>"
                f"<td>{proj:.3e}</td>"
                "</tr>"
            )
        top_table_html = (
            "<h3>Top Channel per Frequency Bin (Rank 1)</h3>"
            "<table class='table'>"
            "<thead><tr><th>Frequency [Hz]</th><th>Channel</th><th>Coherence</th><th>Projection</th></tr></thead>"
            "<tbody>" + "\n".join(rows) + "</tbody></table>"
        )

        thresh_coh = coherence_threshold**2 if asd else coherence_threshold
        peak_mask = self.top_coherence[:, 0] >= thresh_coh
        peak_freqs = self.frequencies[peak_mask]
        peak_ch = self.top_channels[peak_mask, 0]
        peak_coh = self.top_coherence[peak_mask, 0]

        peak_coh = self.top_coherence[peak_mask, 0]

        peak_proj = self._apply_projection(
            self.target_spectrum[peak_mask],
            peak_coh,
            asd=asd,
            threshold=coherence_threshold,
        )

        display_peak_coh = peak_coh
        col_coh_name = "Coherence (Amplitude)" if asd else "Coherence (Squared)"
        if asd:
            display_peak_coh = np.sqrt(np.clip(peak_coh, 0.0, None))

        peaks_df = pd.DataFrame(
            {
                "Frequency [Hz]": peak_freqs,
                "Channel": peak_ch,
                f"{col_coh_name}": display_peak_coh,
                "Projection": peak_proj,
            }
        )
        if len(peaks_df) > 200:
            peaks_df = peaks_df.sort_values(f"{col_coh_name}", ascending=False).head(
                200
            )
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
                "<table class='table'>" + "".join(meta_rows) + "</table>"
            )

        html_content = (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            f"    <title>Bruco Report: {html.escape(self.target_name)}</title>\n"
            "    <style>\n"
            "        body { font-family: sans-serif; margin: 20px; }\n"
            "        img { max-width: 100%; border: 1px solid #ddd; margin-bottom: 20px; }\n"
            "        .table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n"
            "        .table th, .table td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }\n"
            "        .table th { background-color: #f2f2f2; }\n"
            "    </style>\n"
            "</head>\n"
            "<body>\n"
            "    <h1>Bruco Analysis Report</h1>\n"
            f"    <h2>Target: {html.escape(self.target_name)}</h2>\n"
            f"    {meta_html}\n"
            "    <h3>Noise Projection</h3>\n"
            '    <img src="projection.png" alt="Noise Projection Plot">\n'
            "    <h3>Coherence Spectra</h3>\n"
            '    <img src="coherence.png" alt="Coherence Plot">\n'
            f"    {top_table_html}\n"
            f"    {peaks_html}\n"
            "</body>\n"
            "</html>"
        )

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

    def __init__(
        self,
        target_channel: str,
        aux_channels: list[str],
        excluded_channels: list[str] | None = None,
    ) -> None:
        """
        Initialize the Bruco scanner.

        Args:
            target_channel (str): The main channel to analyze.
            aux_channels (List[str]): A list of all available auxiliary channels.
            excluded_channels (List[str], optional): Channels to ignore (e.g., calibration lines).
        """
        self.target = str(target_channel)
        self.aux_channels = [str(c) for c in aux_channels]
        self.excluded = (
            {str(c) for c in excluded_channels} if excluded_channels else set()
        )

        # Exclude target and excluded channels
        self.channels_to_scan = sorted(
            list(set(self.aux_channels) - self.excluded - {self.target})
        )

        logger.info(
            f"Bruco initialized. Target: {self.target}, "
            f"Auxiliary channels: {len(self.channels_to_scan)} (after exclusions)"
        )

    def compute(
        self,
        start: int | float | None = None,
        duration: int | None = None,
        fftlength: float = 2.0,
        overlap: float = 1.0,
        nproc: int = 4,
        batch_size: int = 100,
        top_n: int = 5,
        block_size: int | str | None = None,
        target_data: TimeSeries | None = None,
        aux_data: TimeSeriesDict | Iterable[TimeSeries] | None = None,
        preprocess_batch: Callable[[TimeSeriesDict], TimeSeriesDict] | None = None,
    ) -> BrucoResult:
        """
        Execute the coherence scan.

        Args:
            start (int or float, optional): GPS start time. Required if not inferable from data.
            duration (int, optional): Duration of data in seconds. Required if not inferable.
            fftlength (float): FFT length in seconds.
            overlap (float): Overlap in seconds.
            nproc (int): Number of parallel processes.
            batch_size (int): Channels per batch.
            top_n (int): Number of top channels to keep per frequency bin.
            block_size (int or 'auto', optional): Channels per block in Top-N updates.
            target_data (TimeSeries, optional): Pre-loaded target channel data.
            aux_data (TimeSeriesDict or Iterable[TimeSeries], optional): Pre-loaded auxiliary channels data.
                         Can be a dictionary-like object or an iterable/generator yielding TimeSeries.
            preprocess_batch (Callable, optional): Batch preprocessing callback.

        Returns:
            BrucoResult: Object containing frequency-wise analysis results.
        """
        # --- 0. Infer start/duration if missing ---
        if start is None or duration is None:
            # 1. Infer from target_data
            if target_data is not None:
                if start is None:
                    start = float(target_data.t0.value)
                if duration is None:
                    duration = int(round(target_data.duration.value))
            # 2. Infer from aux_data (only if dict-like)
            elif isinstance(aux_data, Mapping):
                # Check if empty (keys/len)
                if len(aux_data) > 0:
                    try:
                        # Peek at first item
                        # TimeSeriesDict / Dict
                        first_ts = next(iter(aux_data.values()))

                        if start is None:
                            start = float(first_ts.t0.value)
                        if duration is None:
                            duration = int(round(first_ts.duration.value))
                    except Exception as e:
                        logger.debug(
                            f"Failed to infer start/duration from aux_data: {e}"
                        )
                        pass  # infer failed

            # Final check
            if start is None or duration is None:
                raise ValueError(
                    "Arguments 'start' and 'duration' must be provided if they cannot be inferred from 'target_data' or 'aux_data' (Dict)."
                )

        end = start + duration
        if fftlength > duration:
            raise ValueError(
                f"fftlength ({fftlength}) cannot be longer than duration ({duration})."
            )

        logger.info(
            f"Starting Bruco scan at {start} for {duration}s. Batch size: {batch_size}"
        )

        # 1. Prepare Target Data & Calculate Reference Spectrum
        try:
            if target_data is not None:
                logger.info("Using provided target data.")
                target_ts = target_data
            else:
                logger.info(f"Fetching target data: {self.target}")
                target_ts = TimeSeries.get(self.target, start, end)

            # Calculate Target PSD reference (Internal storage is PSD)
            target_spectrum = target_ts.psd(fftlength=fftlength, overlap=overlap)

            target_frequencies = target_spectrum.frequencies.value
            target_spectrum_values = target_spectrum.value

            # Store Frequency resolution for alignment checks
            df = 1.0 / fftlength
            # Verify df matches
            if len(target_frequencies) > 1:
                actual_df = target_frequencies[1] - target_frequencies[0]
                if not np.isclose(df, actual_df, rtol=1e-3):
                    logger.warning(
                        f"Calculated df {df} does not match spectrum df {actual_df}"
                    )

        except Exception as e:
            logger.error(
                f"Failed to fetch or process target channel {self.target}: {e}"
            )
            raise e

        # Initialize Result container
        metadata = {
            "start": start,
            "end": end,
            "duration": duration,
            "fftlength": fftlength,
            "overlap": overlap,
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
            top_n=top_n,
            metadata=metadata,
            block_size=block_size,
        )

        # 2. Processing

        # Case A: Aux data provided
        if aux_data is not None:
            # Case A-1: Dictionary-like (Everything in memory)
            if isinstance(aux_data, Mapping):
                logger.info(
                    f"Using provided auxiliary data dict ({len(aux_data)} channels)."
                )
                all_aux_channels = list(aux_data.keys())
                total_channels = len(all_aux_channels)

                # Validation: check span overlap
                # Only check the first channel for performance? Or checks per batch?
                # User asked to error if span contradicts.
                # Let's check the first one if possible.
                if total_channels > 0:
                    first_ts = aux_data[all_aux_channels[0]]
                    # Simple check: does it cover start ~ end?
                    # allow small tolerance
                    if (
                        first_ts.t0.value > start + 0.1
                        or (first_ts.t0.value + first_ts.duration.value) < end - 0.1
                    ):
                        msg = (
                            f"Aux data TimeSeries (t0={first_ts.t0.value}, dur={first_ts.duration.value}) "
                            f"does not cover requested analysis span (start={start}, end={end})."
                        )
                        logger.error(msg)
                        raise ValueError(msg)

                for i in range(0, total_channels, batch_size):
                    batch_keys = all_aux_channels[i : i + batch_size]
                    logger.info(
                        f"Processing data batch {i // batch_size + 1}/{(total_channels // batch_size) + 1} "
                        f"({len(batch_keys)} channels)"
                    )

                    # Slice the dict
                    batch_dict = TimeSeriesDict()
                    for k in batch_keys:
                        batch_dict[k] = aux_data[k]

                    # Apply preprocessing if callback provided
                    if preprocess_batch is not None:
                        try:
                            batch_dict = preprocess_batch(batch_dict)
                        except Exception as e:
                            logger.error(f"Preprocessing failed for batch: {e}")
                            raise e

                    self._run_and_update_batch(
                        result,
                        target_ts,
                        batch_dict,
                        fftlength,
                        overlap,
                        nproc,
                        target_frequencies,
                    )

            # Case A-2: Iterable/Generator (Memory efficient streaming)
            elif aux_data is not None:
                logger.info("Using provided auxiliary data generator/iterable.")
                batch_dict = TimeSeriesDict()
                batch_count = 0

                for ts in aux_data:
                    # Ensure it has a name, handle potential missing name if not TimeSeries
                    if not hasattr(ts, "name") or not ts.name:
                        logger.warning(
                            "Skipping unnamed TimeSeries in aux_data iterator."
                        )
                        continue

                    # Optional: Check span for streamed data?
                    # Might be computationally expensive to check every single one strictly,
                    # but valuable for safety.
                    if (
                        ts.t0.value > start + 0.1
                        or (ts.t0.value + ts.duration.value) < end - 0.1
                    ):
                        # We can skip or raise. User requested error/strictness.
                        # But for generator, maybe just skipping invalid ones is better?
                        # "spanが矛盾する場合はエラー" -> Raise.
                        raise ValueError(
                            f"Streamed TimeSeries {ts.name} does not cover requested span."
                        )

                    batch_dict[ts.name] = ts

                    if len(batch_dict) >= batch_size:
                        batch_count += 1
                        logger.info(
                            f"Processing generator batch {batch_count} ({len(batch_dict)} channels)"
                        )

                        self._run_and_update_batch(
                            result,
                            target_ts,
                            batch_dict,
                            fftlength,
                            overlap,
                            nproc,
                            target_frequencies,
                        )
                        batch_dict = TimeSeriesDict()  # Clear memory

                # Process remaining
                if len(batch_dict) > 0:
                    batch_count += 1
                    logger.info(
                        f"Processing generator batch {batch_count} ({len(batch_dict)} channels)"
                    )
                    self._run_and_update_batch(
                        result,
                        target_ts,
                        batch_dict,
                        fftlength,
                        overlap,
                        nproc,
                        target_frequencies,
                    )

        # Case B: Fetching Loop (Execute if channels_to_scan is not empty)
        if self.channels_to_scan:
            total_channels = len(self.channels_to_scan)
            logger.info(
                f"Starting auto-fetch for {total_channels} channels configured in Bruco."
            )

            for i in range(0, total_channels, batch_size):
                batch_channels = self.channels_to_scan[i : i + batch_size]
                logger.info(
                    f"Processing auto-fetch batch {i // batch_size + 1}/{(total_channels // batch_size) + 1} "
                    f"({len(batch_channels)} channels)"
                )

                # Fetch batch data
                try:
                    batch_dict = TimeSeriesDict.get(
                        batch_channels, start, end, allow_tape=True, nproc=nproc
                    )
                except Exception as e:
                    logger.warning(
                        f"Batch fetch failed: {e}. Falling back to individual fetch."
                    )
                    batch_dict = TimeSeriesDict()
                    for ch in batch_channels:
                        try:
                            # Fetch individually
                            batch_dict[ch] = TimeSeries.get(
                                ch, start, end, allow_tape=True
                            )
                        except Exception as ch_err:
                            logger.warning(
                                f"Failed to fetch individual channel {ch}: {ch_err}"
                            )

                    if not batch_dict:
                        logger.warning(
                            "No valid channels in this batch after fallback. Skipping."
                        )
                        continue

                # Apply preprocessing if callback provided
                if preprocess_batch is not None:
                    try:
                        batch_dict = preprocess_batch(batch_dict)
                    except Exception as e:
                        logger.error(f"Preprocessing failed for batch: {e}")
                        raise e

                self._run_and_update_batch(
                    result,
                    target_ts,
                    batch_dict,
                    fftlength,
                    overlap,
                    nproc,
                    target_frequencies,
                )

        logger.info("Scan complete.")
        return result

    def _run_and_update_batch(
        self,
        result: BrucoResult,
        target_ts: TimeSeries,
        aux_dict: TimeSeriesDict,
        fftlength: float,
        overlap: float,
        nproc: int,
        target_frequencies: np.ndarray,
    ) -> None:
        # Calculate Coherence for batch
        # Returns list of (name, coherence_array)
        batch_results_list = self._process_batch(
            target_ts,
            aux_dict,
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

    def _process_batch(
        self,
        target_ts: TimeSeries,
        aux_dict: TimeSeriesDict,
        fftlength: float,
        overlap: float,
        nproc: int,
        target_frequencies: np.ndarray,
    ) -> list[tuple[str, np.ndarray] | None]:
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
        results: list[tuple[str, np.ndarray] | None] = []
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
    ) -> tuple[str, np.ndarray] | None:
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
                if (
                    len(coh_freqs) > 1
                    and len(target_frequencies) > 1
                    and np.isclose(
                        coh_freqs[1] - coh_freqs[0],
                        target_frequencies[1] - target_frequencies[0],
                    )
                ):
                    final_coh = coh_val[: len(target_frequencies)]
                else:
                    # Interpolate
                    f = interp1d(coh_freqs, coh_val, bounds_error=False, fill_value=0.0)
                    final_coh = f(target_frequencies)

            return (aux.name, final_coh)

        except Exception as e:
            logger.debug(
                f"Coherence calculation failed for channel {getattr(aux, 'name', 'unknown')}: {e}"
            )
            return None

    @staticmethod
    def _chunk_aux_list(
        aux_list: list[TimeSeries],
        chunk_size: int,
    ) -> Iterable[list[TimeSeries]]:
        for idx in range(0, len(aux_list), chunk_size):
            yield aux_list[idx : idx + chunk_size]

    @staticmethod
    def _calculate_batch_coherence_fast(
        target: TimeSeries,
        aux_list: list[TimeSeries],
        fftlength: float,
        overlap: float,
        target_frequencies: np.ndarray,
    ) -> list[tuple[str, np.ndarray] | None]:
        results: list[tuple[str, np.ndarray] | None] = []
        engine_cache: dict[tuple[float, int], FastCoherenceEngine] = {}
        target_cache: dict[float, TimeSeries] = {}

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
        engine_cache: dict[tuple[float, int], FastCoherenceEngine],
        target_cache: dict[float, TimeSeries],
    ) -> tuple[str, np.ndarray] | None:
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
                final_coh[: len(coh_val)] = coh_val
            else:
                engine_freqs = engine.frequencies
                if (
                    len(engine_freqs) > 1
                    and len(target_frequencies) > 1
                    and np.isclose(
                        engine_freqs[1] - engine_freqs[0],
                        target_frequencies[1] - target_frequencies[0],
                    )
                ):
                    final_coh = coh_val[: len(target_frequencies)]
                else:
                    interp = interp1d(
                        engine_freqs, coh_val, bounds_error=False, fill_value=0.0
                    )
                    final_coh = interp(target_frequencies)

            return (aux.name, final_coh)
        except Exception as e:
            logger.debug(
                f"Coherence calculation (fast engine) failed for channel {getattr(aux, 'name', 'unknown')}: {e}"
            )
            return None

    @staticmethod
    def _calculate_pair_coherence(
        target: TimeSeries,
        aux: TimeSeries,
        fftlength: float,
        overlap: float,
    ) -> BrucoPairSummary:
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
