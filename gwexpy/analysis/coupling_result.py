"""
CouplingResult: result container for a single Witness -> Target pair.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gwexpy.plot import Plot

    from ..frequencyseries import FrequencySeries
    from ..timeseries import TimeSeries
    from ..types.typing import IndexLike


class CouplingResult:
    """
    Result object for a SINGLE Witness -> Target pair.
    """

    def __init__(
        self,
        cf: FrequencySeries,
        psd_witness_inj: FrequencySeries,
        psd_witness_bkg: FrequencySeries,
        psd_target_inj: FrequencySeries,
        psd_target_bkg: FrequencySeries,
        valid_mask: np.ndarray,
        witness_name: str,
        target_name: str,
        cf_ul: FrequencySeries | None = None,
        ts_witness_bkg: TimeSeries | None = None,
        ts_target_bkg: TimeSeries | None = None,
        ts_witness_inj: TimeSeries | None = None,
        ts_target_inj: TimeSeries | None = None,
        fftlength: float | None = None,
        overlap: float | None = None,
    ) -> None:
        self.cf = cf
        self.cf_ul = cf_ul
        self.psd_witness_inj = psd_witness_inj
        self.psd_witness_bkg = psd_witness_bkg
        self.psd_target_inj = psd_target_inj
        self.psd_target_bkg = psd_target_bkg
        self.valid_mask = valid_mask
        self.witness_name = witness_name
        self.target_name = target_name
        self.ts_witness_bkg = ts_witness_bkg
        self.ts_target_bkg = ts_target_bkg
        self.ts_witness_inj = ts_witness_inj
        self.ts_target_inj = ts_target_inj
        self.fftlength = fftlength
        self.overlap = overlap

    @property
    def frequencies(self) -> IndexLike:
        return self.cf.xindex

    def plot_cf(
        self,
        figsize: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        **kwargs: object,
    ) -> Plot:
        """Plot the Coupling Function and its Upper Limit."""

        # Crop data if xlim provided
        cf_plot = self.cf
        cf_ul_plot = self.cf_ul

        if xlim is not None:
            if cf_plot is not None:
                cf_plot = cf_plot.copy().crop(*xlim)
            if cf_ul_plot is not None:
                cf_ul_plot = cf_ul_plot.copy().crop(*xlim)

        cf_plot.name = "Coupling Function"

        # Handle figsize via kwargs if needed, or set explicitly
        if figsize is not None:
            kwargs["figsize"] = figsize

        label = kwargs.pop("label", "Coupling Function")
        plot = cf_plot.plot(
            color="tab:green",
            marker=".",
            linestyle="-",
            markersize=3,
            label=label,
            **kwargs,
        )
        ax = plot.gca()

        if cf_ul_plot is not None:
            cf_ul_plot.name = "Upper Limit"
            ax.plot(
                cf_ul_plot,
                color="lightskyblue",
                marker=".",
                linestyle="-",
                markersize=3,
                label="Upper Limit",
            )

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel(f"CF Magnitude [{cf_plot.unit}]")
        ax.set_title(f"Coupling Function: {self.witness_name} -> {self.target_name}")
        ax.legend()

        if xlim is not None:
            ax.set_xlim(*xlim)

        return plot

    def plot(
        self,
        figsize: tuple[float, float] = (10, 12),
        xlim: tuple[float, float] | None = None,
    ) -> Plot:
        """
        Create a diagnostic plot showing ASDs and the resulting CF.
        """
        from gwexpy.plot import Plot

        # Helper to crop series safely
        def crop_if_needed(series):
            if series is None or xlim is None:
                return series
            return series.copy().crop(*xlim)

        # Helper to compute background stats
        def get_bkg_stats(ts_bkg, psd_bkg):
            # Crop PSD first if needed
            psd_bkg_eff = crop_if_needed(psd_bkg)

            # Median/Mean from PSD
            asd_mean = psd_bkg_eff**0.5
            asd_mean.name = (
                f"Background ({ts_bkg.name if ts_bkg is not None else 'Target'})"
            )

            p10_asd = None
            p90_asd = None

            if ts_bkg is not None and self.fftlength is not None:
                try:
                    spec = ts_bkg.spectrogram(
                        stride=self.fftlength,
                        fftlength=self.fftlength,
                        overlap=self.overlap if self.overlap else 0,
                        method="welch",
                        window="hann",
                    )
                    # For percentiles, we crop the spectrogram itself if possible or crop result
                    # Cropping spectrogram is more efficient
                    if xlim is not None:
                        spec = spec.crop_frequencies(*xlim)

                    p10 = spec.percentile(10)
                    p90 = spec.percentile(90)
                    p10_asd = p10**0.5
                    p90_asd = p90**0.5
                except (RuntimeError, TypeError, ValueError):
                    logger.warning(
                        "Could not compute background percentiles for %s",
                        ts_bkg.name if ts_bkg is not None else "Target",
                        exc_info=True,
                    )

            return asd_mean, p10_asd, p90_asd

        # --- Prepare Data ---

        # Witness
        psd_wit_inj_c = crop_if_needed(self.psd_witness_inj)
        asd_wit_inj = psd_wit_inj_c**0.5
        asd_wit_inj.name = "Injection (Witness)"
        asd_wit_mean, wit_p10, wit_p90 = get_bkg_stats(
            self.ts_witness_bkg, self.psd_witness_bkg
        )
        asd_wit_mean.name = "Background (Witness)"

        # Target
        psd_tgt_inj_c = crop_if_needed(self.psd_target_inj)
        asd_tgt_inj = psd_tgt_inj_c**0.5
        asd_tgt_inj.name = "Injection (Target)"
        asd_tgt_mean, tgt_p10, tgt_p90 = get_bkg_stats(
            self.ts_target_bkg, self.psd_target_bkg
        )
        asd_tgt_mean.name = "Background (Target)"

        # Derived
        cf_c = crop_if_needed(self.cf)
        cf_ul_c = crop_if_needed(self.cf_ul)
        psd_wit_bkg_c = crop_if_needed(self.psd_witness_bkg)

        # Create Plot
        plot = Plot(geometry=(3, 1), figsize=figsize, sharex=True)
        ax0 = plot.axes[0]
        ax1 = plot.axes[1]
        ax2 = plot.axes[2]

        # 1. Witness ASDs
        # Background
        if wit_p10 is not None and wit_p90 is not None:
            plot.plot_mmm(
                asd_wit_mean,
                wit_p10,
                wit_p90,
                ax=ax0,
                color="black",
                linestyle="-",
                zorder=5,
                alpha_fill=0.1,
            )
        else:
            ax0.plot(
                asd_wit_mean,
                color="black",
                linestyle="-",
                zorder=5,
                label=asd_wit_mean.name,
            )

        # Injection
        ax0.plot(
            asd_wit_inj, color="red", linestyle="-", zorder=4, label=asd_wit_inj.name
        )

        ax0.set_ylabel(f"ASD [{asd_wit_inj.unit}]")
        ax0.set_title(f"Witness: {self.witness_name}")
        ax0.legend()
        ax0.grid(True, which="both", linestyle=":")

        # 2. Target ASDs
        # Background
        if tgt_p10 is not None and tgt_p90 is not None:
            plot.plot_mmm(
                asd_tgt_mean,
                tgt_p10,
                tgt_p90,
                ax=ax1,
                color="black",
                linestyle="-",
                zorder=5,
                alpha_fill=0.1,
            )
        else:
            ax1.plot(
                asd_tgt_mean,
                color="black",
                linestyle="-",
                zorder=5,
                label=asd_tgt_mean.name,
            )

        # Injection
        ax1.plot(
            asd_tgt_inj, color="red", linestyle="-", zorder=4, label=asd_tgt_inj.name
        )

        # Projection (Witness Bkg * CF)
        if cf_c is not None:
            asd_wit_bkg = psd_wit_bkg_c**0.5
            projection_asd = asd_wit_bkg * cf_c
            projection_asd.name = "Projection"
            ax1.plot(
                projection_asd,
                color="tab:green",
                marker=".",
                linestyle="-",
                markersize=3,
                zorder=6,
                label=projection_asd.name,
            )

        if cf_ul_c is not None:
            asd_wit_bkg = psd_wit_bkg_c**0.5
            projection_ul = asd_wit_bkg * cf_ul_c
            projection_ul.name = "Projection UL"
            ax1.plot(
                projection_ul,
                color="lightskyblue",
                marker=".",
                linestyle="-",
                markersize=3,
                zorder=6,
                label=projection_ul.name,
            )

        ax1.set_ylabel(f"ASD [{asd_tgt_inj.unit}]")
        ax1.set_title(f"Target: {self.target_name}")
        ax1.legend()
        ax1.grid(True, which="both", linestyle=":")

        # 3. Coupling Function
        cf_c.name = "Coupling Function"
        ax2.plot(
            cf_c,
            color="tab:green",
            marker=".",
            linestyle="-",
            markersize=3,
            label=cf_c.name,
        )

        if cf_ul_c is not None:
            cf_ul_c.name = "Upper Limit"
            ax2.plot(
                cf_ul_c,
                color="lightskyblue",
                marker=".",
                linestyle="-",
                markersize=3,
                label=cf_ul_c.name,
            )

        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel(f"CF [{cf_c.unit}]")
        ax2.set_title(f"Coupling Function ({self.witness_name} -> {self.target_name})")
        ax2.grid(True, which="both", linestyle=":")
        ax2.legend()

        # Use log scale for all axes
        for ax in plot.axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            if xlim is not None:
                ax.set_xlim(*xlim)

        plot.tight_layout()
        return plot

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def _significance_array(self) -> np.ndarray:
        """
        有意度 (ASD_inj - ASD_bkg) / ASD_bkg を計算して返す。

        背景 ASD をゼロ除算から防護するため epsilon ガードを適用する。
        """
        asd_inj = np.sqrt(np.abs(self.psd_witness_inj.value))
        asd_bkg = np.sqrt(np.abs(self.psd_witness_bkg.value))
        eps = np.finfo(float).tiny
        return (asd_inj - asd_bkg) / np.where(asd_bkg > 0, asd_bkg, eps)

    def to_csv(self, filepath: str | os.PathLike) -> None:
        """
        結合係数を CSV 形式で保存する。

        列: frequency, cf, cf_ul, significance, inj_asd, bkg_asd
        """
        freqs = self.cf.xindex.value
        cf_vals = self.cf.value
        cf_ul_vals = self.cf_ul.value if self.cf_ul is not None else np.full_like(cf_vals, np.nan)
        sig_vals = self._significance_array()
        inj_asd = np.sqrt(np.abs(self.psd_witness_inj.value))
        bkg_asd = np.sqrt(np.abs(self.psd_witness_bkg.value))

        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frequency", "cf", "cf_ul", "significance", "inj_asd", "bkg_asd"])
            for row in zip(freqs, cf_vals, cf_ul_vals, sig_vals, inj_asd, bkg_asd):
                writer.writerow(row)

    @classmethod
    def from_csv(cls, filepath: str | os.PathLike) -> CouplingResult:
        """
        CSV ファイルから CouplingResult を復元する（ラウンドトリップ用）。

        to_csv() で書き出したファイルのみ対応。cf / cf_ul / ASD のみ復元し、
        その他フィールドは最小限のダミーで補完する。
        """
        from gwexpy.frequencyseries import FrequencySeries

        rows: list[dict[str, str]] = []
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"CSV file is empty: {filepath}")

        freqs = np.array([float(r["frequency"]) for r in rows])
        cf_vals = np.array([float(r["cf"]) for r in rows])
        cf_ul_raw = np.array([float(r["cf_ul"]) for r in rows])
        inj_asd = np.array([float(r["inj_asd"]) for r in rows])
        bkg_asd = np.array([float(r["bkg_asd"]) for r in rows])

        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        def _fs(vals: np.ndarray) -> FrequencySeries:
            return FrequencySeries(vals, f0=freqs[0], df=df)

        cf_ul = _fs(cf_ul_raw) if not np.all(np.isnan(cf_ul_raw)) else None
        psd_inj = _fs(inj_asd**2)
        psd_bkg = _fs(bkg_asd**2)
        valid_mask = np.isfinite(cf_vals)

        return cls(
            cf=_fs(cf_vals),
            psd_witness_inj=psd_inj,
            psd_witness_bkg=psd_bkg,
            psd_target_inj=psd_inj,
            psd_target_bkg=psd_bkg,
            valid_mask=valid_mask,
            witness_name="",
            target_name="",
            cf_ul=cf_ul,
        )

    def to_txt(self, filepath: str | os.PathLike) -> None:
        """
        結合係数を NInjA.py 互換テキスト形式で保存する。

        フォーマット::

            # frequency(Hz) coupling_factor uncertainty significance
            1.0 0.123 0.045 2.73
            ...

        uncertainty は cf_ul - cf（cf_ul がない場合は NaN）。
        """
        freqs = self.cf.xindex.value
        cf_vals = self.cf.value
        cf_ul_vals = self.cf_ul.value if self.cf_ul is not None else np.full_like(cf_vals, np.nan)
        uncertainty = cf_ul_vals - cf_vals
        sig_vals = self._significance_array()

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("# frequency(Hz) coupling_factor uncertainty significance\n")
            for f, cf, unc, sig in zip(freqs, cf_vals, uncertainty, sig_vals):
                fh.write(f"{f} {cf} {unc} {sig}\n")

    # ------------------------------------------------------------------
    # Visualization (Phase 3)
    # ------------------------------------------------------------------

    def plot_significance(
        self,
        threshold: float = 3.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        figsize: tuple[float, float] = (12, 6),
    ) -> Any:
        """
        有意度スペクトラムプロット（(ASD_inj - ASD_bkg) / ASD_bkg vs 周波数）。

        Parameters
        ----------
        threshold : float
            閾値水平線の値（デフォルト 3.0）。0 以下の場合は描画しない。
        freq_min, freq_max : float, optional
            表示する周波数範囲 [Hz]。
        figsize : tuple
            Figure サイズ。

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        freqs = self.cf.xindex.value
        sig_vals = self._significance_array()

        mask = np.ones(len(freqs), dtype=bool)
        if freq_min is not None:
            mask &= freqs >= freq_min
        if freq_max is not None:
            mask &= freqs <= freq_max

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(freqs[mask], sig_vals[mask], color="tab:blue", linewidth=1.0, label="Significance")

        if threshold > 0:
            ax.axhline(
                threshold,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label=f"threshold = {threshold}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"$(ASD_{inj} - ASD_{bkg}) / ASD_{bkg}$")
        ax.set_title(
            f"Significance: {self.witness_name} -> {self.target_name}"
        )
        ax.legend()
        ax.grid(True, which="both", linestyle=":")

        if freq_min is not None or freq_max is not None:
            lo = freq_min if freq_min is not None else freqs[mask][0]
            hi = freq_max if freq_max is not None else freqs[mask][-1]
            ax.set_xlim(lo, hi)

        fig.tight_layout()
        return fig

    def plot_asdgram(
        self,
        freq_min: float | None = None,
        freq_max: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        figsize: tuple[float, float] = (14, 6),
    ) -> Any:
        """
        ASD スペクトログラム + パーセンタイルオーバーレイ（2 列レイアウト）。

        左パネル: 注入時 ASD spectrogram, 右パネル: 背景時 ASD spectrogram。
        両者に 50%, 90%, 99% パーセンタイルラインを重ねる。

        Parameters
        ----------
        freq_min, freq_max : float, optional
            表示する周波数範囲 [Hz]。
        vmin, vmax : float, optional
            カラーバースケール。None の場合は自動設定。
        figsize : tuple
            Figure サイズ。

        Raises
        ------
        ValueError
            `ts_witness_inj` または `ts_witness_bkg` が None の場合。
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if self.ts_witness_inj is None:
            raise ValueError(
                "ts_witness_inj is required for plot_asdgram(). "
                "Set it when constructing CouplingResult."
            )
        if self.ts_witness_bkg is None:
            raise ValueError(
                "ts_witness_bkg is required for plot_asdgram(). "
                "Set it when constructing CouplingResult."
            )
        if self.fftlength is None:
            raise ValueError("fftlength is required for plot_asdgram().")

        overlap = self.overlap if self.overlap is not None else 0.0

        def _specgram(ts: Any) -> Any:
            spec = ts.spectrogram(
                stride=self.fftlength,
                fftlength=self.fftlength,
                overlap=overlap,
                method="welch",
                window="hann",
            )
            if freq_min is not None or freq_max is not None:
                lo = freq_min if freq_min is not None else 0.0
                hi = freq_max if freq_max is not None else np.inf
                spec = spec.crop_frequencies(lo, hi)
            return spec

        spec_inj = _specgram(self.ts_witness_inj)
        spec_bkg = _specgram(self.ts_witness_bkg)

        asd_inj = spec_inj**0.5
        asd_bkg = spec_bkg**0.5

        freqs_plot = asd_inj.frequencies.value
        times_inj = asd_inj.times.value
        times_bkg = asd_bkg.times.value

        data_inj = asd_inj.value
        data_bkg = asd_bkg.value

        _vmin = vmin if vmin is not None else float(np.nanpercentile(data_inj[data_inj > 0], 1))
        _vmax = vmax if vmax is not None else float(np.nanpercentile(data_inj[data_inj > 0], 99))
        norm = LogNorm(vmin=max(_vmin, 1e-40), vmax=max(_vmax, _vmin * 2))

        fig, (ax_inj, ax_bkg) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        asd_unit = getattr(self.ts_witness_inj, "unit", "")
        pct_colors = {50: "white", 90: "yellow", 99: "red"}
        pct_styles = {50: "-", 90: "--", 99: ":"}

        for ax, data, times, title in [
            (ax_inj, data_inj, times_inj, f"Injection: {self.witness_name}"),
            (ax_bkg, data_bkg, times_bkg, f"Background: {self.witness_name}"),
        ]:
            c = ax.pcolormesh(
                times, freqs_plot, data.T,
                norm=norm, shading="auto", cmap="viridis",
            )
            fig.colorbar(c, ax=ax, label=f"ASD [{asd_unit}]")

            # 各周波数ビンの時刻方向パーセンタイルを周波数軸に沿った輪郭として描画
            # twin x 軸で ASD スケールのパーセンタイルプロファイルを重ねる
            ax_twin = ax.twiny()
            for pct in (50, 90, 99):
                pct_vals = np.nanpercentile(data, pct, axis=0)  # shape: (n_freqs,)
                ax_twin.plot(
                    pct_vals,
                    freqs_plot,
                    color=pct_colors[pct],
                    linestyle=pct_styles[pct],
                    linewidth=1.0,
                    label=f"p{pct}",
                )
            ax_twin.set_xscale("log")
            ax_twin.set_xlabel(f"ASD [{asd_unit}]", fontsize=8)
            ax_twin.legend(fontsize=7, loc="upper right")

            ax.set_yscale("log")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_title(title)

        ax_bkg.set_ylabel("")
        fig.suptitle(
            f"ASDgram: {self.witness_name} -> {self.target_name}", y=1.02
        )
        fig.tight_layout()
        return fig

    def plot_snrgram(
        self,
        freq_min: float | None = None,
        freq_max: float | None = None,
        snrmax: float = 100.0,
        figsize: tuple[float, float] = (12, 6),
    ) -> Any:
        """
        SNR スペクトログラム（中央値正規化: (ASD_inj - median_bkg) / median_bkg）。

        Parameters
        ----------
        freq_min, freq_max : float, optional
            表示する周波数範囲 [Hz]。
        snrmax : float
            カラーバー上限（clamp 値）。デフォルト 100。
        figsize : tuple
            Figure サイズ。

        Raises
        ------
        ValueError
            `ts_witness_inj` または `ts_witness_bkg` が None の場合。
        """
        import matplotlib.pyplot as plt

        if self.ts_witness_inj is None:
            raise ValueError(
                "ts_witness_inj is required for plot_snrgram(). "
                "Set it when constructing CouplingResult."
            )
        if self.ts_witness_bkg is None:
            raise ValueError(
                "ts_witness_bkg is required for plot_snrgram(). "
                "Set it when constructing CouplingResult."
            )
        if self.fftlength is None:
            raise ValueError("fftlength is required for plot_snrgram().")

        overlap = self.overlap if self.overlap is not None else 0.0

        def _specgram(ts: Any) -> Any:
            spec = ts.spectrogram(
                stride=self.fftlength,
                fftlength=self.fftlength,
                overlap=overlap,
                method="welch",
                window="hann",
            )
            if freq_min is not None or freq_max is not None:
                lo = freq_min if freq_min is not None else 0.0
                hi = freq_max if freq_max is not None else np.inf
                spec = spec.crop_frequencies(lo, hi)
            return spec

        spec_inj = _specgram(self.ts_witness_inj)
        spec_bkg = _specgram(self.ts_witness_bkg)

        asd_inj = spec_inj.value**0.5
        asd_bkg = spec_bkg.value**0.5

        freqs_plot = spec_inj.frequencies.value
        times_inj = spec_inj.times.value

        eps = np.finfo(float).tiny
        median_bkg = np.nanmedian(asd_bkg, axis=0)
        median_bkg = np.where(median_bkg > 0, median_bkg, eps)

        snr = (asd_inj - median_bkg[np.newaxis, :]) / median_bkg[np.newaxis, :]
        snr = np.clip(snr, -snrmax, snrmax)

        fig, ax = plt.subplots(figsize=figsize)
        c = ax.pcolormesh(
            times_inj,
            freqs_plot,
            snr.T,
            vmin=0,
            vmax=snrmax,
            shading="auto",
            cmap="inferno",
        )
        fig.colorbar(c, ax=ax, label="SNR")
        ax.set_yscale("log")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(
            f"SNRgram: {self.witness_name} -> {self.target_name}"
        )
        fig.tight_layout()
        return fig

    @classmethod
    def from_txt(cls, filepath: str | os.PathLike) -> CouplingResult:
        """
        TXT ファイルから CouplingResult を復元する（ラウンドトリップ用）。

        to_txt() で書き出したファイルのみ対応。cf と cf_ul のみ復元し、
        その他フィールドは最小限のダミーで補完する。
        """
        from gwexpy.frequencyseries import FrequencySeries

        freqs_list: list[float] = []
        cf_list: list[float] = []
        uncertainty_list: list[float] = []

        with open(filepath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                freqs_list.append(float(parts[0]))
                cf_list.append(float(parts[1]))
                uncertainty_list.append(float(parts[2]))

        if not freqs_list:
            raise ValueError(f"TXT file contains no data rows: {filepath}")

        freqs = np.array(freqs_list)
        cf_vals = np.array(cf_list)
        unc_vals = np.array(uncertainty_list)

        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        def _fs(vals: np.ndarray) -> FrequencySeries:
            return FrequencySeries(vals, f0=freqs[0], df=df)

        cf_ul_vals = cf_vals + unc_vals
        cf_ul = _fs(cf_ul_vals) if not np.all(np.isnan(cf_ul_vals)) else None
        dummy_psd = _fs(np.ones_like(cf_vals))
        valid_mask = np.isfinite(cf_vals)

        return cls(
            cf=_fs(cf_vals),
            psd_witness_inj=dummy_psd,
            psd_witness_bkg=dummy_psd,
            psd_target_inj=dummy_psd,
            psd_target_bkg=dummy_psd,
            valid_mask=valid_mask,
            witness_name="",
            target_name="",
            cf_ul=cf_ul,
        )


# ------------------------------------------------------------------
# CouplingResultCollection
# ------------------------------------------------------------------


class CouplingResultCollection(dict):
    """
    複数の CouplingResult を管理するコンテナ。

    使用例::

        results = CouplingResultCollection()
        results['WIT-TGT'] = coupling_result_1
        results['WIT-TGT2'] = coupling_result_2
        results.to_summary_csv("summary.csv")
    """

    def __init__(self, mapping: dict[str, Any] | None = None) -> None:
        super().__init__(mapping or {})

    def to_summary_csv(self, filepath: str | os.PathLike) -> None:
        """
        全結果を単一 CSV に集約して保存する。

        列: channel_pair, frequency, cf, cf_ul, significance, inj_asd, bkg_asd
        """
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["channel_pair", "frequency", "cf", "cf_ul", "significance", "inj_asd", "bkg_asd"]
            )
            for key, result in self.items():
                if not isinstance(result, CouplingResult):
                    continue
                freqs = result.cf.xindex.value
                cf_vals = result.cf.value
                cf_ul_vals = (
                    result.cf_ul.value
                    if result.cf_ul is not None
                    else np.full_like(cf_vals, np.nan)
                )
                sig_vals = result._significance_array()
                inj_asd = np.sqrt(np.abs(result.psd_witness_inj.value))
                bkg_asd = np.sqrt(np.abs(result.psd_witness_bkg.value))
                for row in zip(freqs, cf_vals, cf_ul_vals, sig_vals, inj_asd, bkg_asd):
                    writer.writerow([key, *row])

    def plot_comparison(
        self,
        freq_min: float | None = None,
        freq_max: float | None = None,
        threshold: float = 3.0,
        figsize: tuple[float, float] = (12, 8),
    ) -> Any:
        """
        複数の結合係数を重ねたプロット。

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        xlim = (freq_min, freq_max) if freq_min is not None and freq_max is not None else None

        for label, result in self.items():
            if not isinstance(result, CouplingResult):
                continue
            cf = result.cf
            if xlim is not None:
                cf = cf.copy().crop(*xlim)
            ax.plot(cf.xindex.value, cf.value, marker=".", linestyle="-", markersize=3, label=label)

        # 閾値ラインを水平線として描画
        if threshold > 0:
            ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8, label=f"threshold={threshold}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Coupling Function")
        ax.set_title("Coupling Function Comparison")
        ax.legend()
        ax.grid(True, which="both", linestyle=":")
        if xlim is not None:
            ax.set_xlim(*xlim)

        return fig
