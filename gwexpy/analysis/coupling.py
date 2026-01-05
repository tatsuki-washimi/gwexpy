"""
Coupling Function Analysis Module for gwexpy.

Estimates the coupling function (CF) with flexible threshold strategies:
- RatioThreshold: Mean power ratio.
- SigmaThreshold: Statistical significance (Gaussian assumption).
- PercentileThreshold: Data-driven percentile (Robust to non-Gaussianity).
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict

from ..timeseries import TimeSeries, TimeSeriesDict
from ..frequencyseries import FrequencySeries


# --- Threshold Strategies ---

class ThresholdStrategy(ABC):
    """Abstract base class for excess detection strategies."""

    @abstractmethod
    def check(self, psd_inj: FrequencySeries, psd_bkg: FrequencySeries,
              raw_bkg: Optional[TimeSeries] = None, **kwargs) -> np.ndarray:
        """
        Return a boolean mask where P_inj is considered 'excess'.

        Parameters
        ----------
        psd_inj : FrequencySeries
            PSD of the injection data.
        psd_bkg : FrequencySeries
            PSD of the background data (usually mean or median).
        raw_bkg : TimeSeries, optional
            Raw background time series data. Required for PercentileThreshold
            to calculate the distribution of PSDs across segments.
        """
        pass

class RatioThreshold(ThresholdStrategy):
    """
    Checks if P_inj > ratio * P_bkg_mean.
    Simple and fast.
    """
    def __init__(self, ratio: float = 2.0):
        self.ratio = ratio

    def check(self, psd_inj, psd_bkg, **kwargs):
        return psd_inj.value > (psd_bkg.value * self.ratio)

class SigmaThreshold(ThresholdStrategy):
    """
    Checks if P_inj > P_bkg + sigma * std_error.
    Assumes Gaussian noise statistics.
    """
    def __init__(self, sigma: float = 3.0):
        self.sigma = sigma

    def check(self, psd_inj, psd_bkg, n_avg=1.0, **kwargs):
        if n_avg <= 0:
            return np.ones_like(psd_inj.value, dtype=bool)

        factor = 1.0 + (self.sigma / np.sqrt(n_avg))
        return psd_inj.value > (psd_bkg.value * factor)

class PercentileThreshold(ThresholdStrategy):
    """
    Checks if P_inj > factor * Percentile(P_bkg_segments).

    This requires re-calculating the PSD spectrogram of the background
    to find the percentile at each frequency bin.

    Parameters
    ----------
    percentile : float, default=95
        The percentile of the background distribution (0-100).
    factor : float, default=2.0
        Multiplier for the percentile value.
        Threshold = factor * P_bkg_percentile
    """
    def __init__(self, percentile: float = 95, factor: float = 1.0):
        self.percentile = percentile
        self.factor = factor

    def check(self, psd_inj, psd_bkg, raw_bkg=None, fftlength=None, overlap=None, **kwargs):
        if raw_bkg is None or fftlength is None:
            raise ValueError("PercentileThreshold requires 'raw_bkg' time series and 'fftlength' to calculate distributions.")

        # Calculate Spectrogram (Time-Frequency map) of background
        # We need the variation over time segments
        spec = raw_bkg.spectrogram(
            stride=fftlength,
            fftlength=fftlength,
            overlap=overlap if overlap else 0,
            method="welch",
            window="hann"
        )

        # Calculate the percentile along the time axis (axis=0)
        # Result is a FrequencySeries-like array of the Xth percentile background
        p_bkg_values = spec.percentile(self.percentile)

        # Apply factor
        threshold = p_bkg_values.value * self.factor

        # Determine excess
        return psd_inj.value > threshold


# --- Result Class ---

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
        cf_ul: Optional[FrequencySeries] = None,
        ts_witness_bkg: Optional[TimeSeries] = None,
        ts_target_bkg: Optional[TimeSeries] = None,
        fftlength: Optional[float] = None,
        overlap: Optional[float] = None
    ):
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
        self.fftlength = fftlength
        self.overlap = overlap

    @property
    def frequencies(self):
        return self.cf.xindex

    def plot_cf(self, figsize=None, xlim=None, **kwargs):
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
            kwargs['figsize'] = figsize

        label = kwargs.pop('label', "Coupling Function")
        plot = cf_plot.plot(color="tab:green", marker=".", linestyle="-", markersize=3, label=label, **kwargs)
        ax = plot.gca()

        if cf_ul_plot is not None:
             cf_ul_plot.name = "Upper Limit"
             ax.plot(cf_ul_plot, color="lightskyblue", marker=".", linestyle="-", markersize=3, label="Upper Limit")

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel(f"CF Magnitude [{cf_plot.unit}]")
        ax.set_title(f"Coupling Function: {self.witness_name} -> {self.target_name}")
        ax.legend()

        if xlim is not None:
            ax.set_xlim(*xlim)

        return plot

    def plot(self, figsize=(10, 12), xlim=None):
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
            asd_mean = psd_bkg_eff ** 0.5
            asd_mean.name = f"Background ({ts_bkg.name if ts_bkg is not None else 'Target'})"

            p10_asd = None
            p90_asd = None

            if ts_bkg is not None and self.fftlength is not None:
                try:
                    spec = ts_bkg.spectrogram(
                        stride=self.fftlength,
                        fftlength=self.fftlength,
                        overlap=self.overlap if self.overlap else 0,
                        method="welch",
                        window="hann"
                    )
                    # For percentiles, we crop the spectrogram itself if possible or crop result
                    # Cropping spectrogram is more efficient
                    if xlim is not None:
                        spec = spec.crop_frequencies(*xlim)

                    p10 = spec.percentile(10)
                    p90 = spec.percentile(90)
                    p10_asd = p10 ** 0.5
                    p90_asd = p90 ** 0.5
                except Exception as e:
                    print(f"Warning: Could not compute percentiles: {e}")

            return asd_mean, p10_asd, p90_asd

        # --- Prepare Data ---

        # Witness
        psd_wit_inj_c = crop_if_needed(self.psd_witness_inj)
        asd_wit_inj = psd_wit_inj_c ** 0.5
        asd_wit_inj.name = "Injection (Witness)"
        asd_wit_mean, wit_p10, wit_p90 = get_bkg_stats(self.ts_witness_bkg, self.psd_witness_bkg)
        asd_wit_mean.name = "Background (Witness)"

        # Target
        psd_tgt_inj_c = crop_if_needed(self.psd_target_inj)
        asd_tgt_inj = psd_tgt_inj_c ** 0.5
        asd_tgt_inj.name = "Injection (Target)"
        asd_tgt_mean, tgt_p10, tgt_p90 = get_bkg_stats(self.ts_target_bkg, self.psd_target_bkg)
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
            plot.plot_mmm(asd_wit_mean, wit_p10, wit_p90, ax=ax0, color="black", linestyle="-", zorder=5, alpha_fill=0.1)
        else:
            ax0.plot(asd_wit_mean, color="black", linestyle="-", zorder=5, label=asd_wit_mean.name)

        # Injection
        ax0.plot(asd_wit_inj, color="red", linestyle="-", zorder=4, label=asd_wit_inj.name)

        ax0.set_ylabel(f"ASD [{asd_wit_inj.unit}]")
        ax0.set_title(f"Witness: {self.witness_name}")
        ax0.legend()
        ax0.grid(True, which="both", linestyle=":")

        # 2. Target ASDs
        # Background
        if tgt_p10 is not None and tgt_p90 is not None:
            plot.plot_mmm(asd_tgt_mean, tgt_p10, tgt_p90, ax=ax1, color="black", linestyle="-", zorder=5, alpha_fill=0.1)
        else:
            ax1.plot(asd_tgt_mean, color="black", linestyle="-", zorder=5, label=asd_tgt_mean.name)

        # Injection
        ax1.plot(asd_tgt_inj, color="red", linestyle="-", zorder=4, label=asd_tgt_inj.name)

        # Projection (Witness Bkg * CF)
        if cf_c is not None:
            asd_wit_bkg = psd_wit_bkg_c ** 0.5
            projection_asd = asd_wit_bkg * cf_c
            projection_asd.name = "Projection"
            ax1.plot(projection_asd, color="tab:green", marker=".", linestyle="-", markersize=3, zorder=6, label=projection_asd.name)

        if cf_ul_c is not None:
            asd_wit_bkg = psd_wit_bkg_c ** 0.5
            projection_ul = asd_wit_bkg * cf_ul_c
            projection_ul.name = "Projection UL"
            ax1.plot(projection_ul, color="lightskyblue", marker=".", linestyle="-", markersize=3, zorder=6, label=projection_ul.name)

        ax1.set_ylabel(f"ASD [{asd_tgt_inj.unit}]")
        ax1.set_title(f"Target: {self.target_name}")
        ax1.legend()
        ax1.grid(True, which="both", linestyle=":")

        # 3. Coupling Function
        cf_c.name = "Coupling Function"
        ax2.plot(cf_c, color="tab:green", marker=".", linestyle="-", markersize=3, label=cf_c.name)

        if cf_ul_c is not None:
             cf_ul_c.name = "Upper Limit"
             ax2.plot(cf_ul_c, color="lightskyblue", marker=".", linestyle="-", markersize=3, label=cf_ul_c.name)

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


# --- Analysis Class ---

class CouplingFunctionAnalysis:
    """
    Analysis class to estimate Coupling Functions (CF).
    """

    def compute(
        self,
        data_inj: TimeSeriesDict,
        data_bkg: TimeSeriesDict,
        fftlength: float,
        witness: Optional[str] = None,
        overlap: float = 0,
        threshold_witness: ThresholdStrategy = RatioThreshold(25.0),
        threshold_target: ThresholdStrategy = RatioThreshold(4.0),
        **kwargs
    ) -> Union[CouplingResult, Dict[str, CouplingResult]]:
        """
        Compute Coupling Function(s) from TimeSeriesDicts.

        Parameters
        ----------
        data_inj : TimeSeriesDict
            Injection data (Witness + Targets).
        data_bkg : TimeSeriesDict
            Background data (Witness + Targets).
        fftlength : float
            FFT length in seconds.
        witness : str, optional
            The name (key) of the witness channel.
            If None, the FIRST channel in `data_inj` is used.
        overlap : float, optional
            Overlap in seconds (default 0).
        threshold_witness : ThresholdStrategy
            Strategy to determine if Witness is excited.
        threshold_target : ThresholdStrategy
            Strategy to determine if Target is excited.
        """
        # --- 1. Identify Witness Channel ---
        all_channels = list(data_inj.keys())

        if witness is None:
            witness_key = all_channels[0]
        else:
            witness_key = witness
            if witness_key not in data_inj:
                raise KeyError(f"Witness channel '{witness_key}' not found in input data.")

        if witness_key not in data_bkg:
             raise KeyError(f"Witness channel '{witness_key}' not found in background data.")

        # --- 2. Separate Data ---
        ts_wit_inj = data_inj[witness_key]
        ts_wit_bkg = data_bkg[witness_key]
        target_keys = [k for k in all_channels if k != witness_key]

        if not target_keys:
            raise ValueError("No target channels found. Data must contain at least 2 channels.")

        # --- 3. Compute PSDs & N_avg ---
        psd_kwargs = {"fftlength": fftlength, "overlap": overlap, "method": "welch", "window": "hann"}
        psd_kwargs.update(kwargs)

        # Helper to pass extra data needed by PercentileThreshold
        check_kwargs = {
            "fftlength": fftlength,
            "overlap": overlap,
            "n_avg": 1.0 # Will be updated
        }

        # Estimate number of averages (N_avg)
        duration_inj = ts_wit_inj.span[1] - ts_wit_inj.span[0]
        duration_bkg = ts_wit_bkg.span[1] - ts_wit_bkg.span[0]
        eff_ovlp = overlap

        n_avg_inj = max(1, (duration_inj - eff_ovlp) / (fftlength - eff_ovlp))
        n_avg_bkg = max(1, (duration_bkg - eff_ovlp) / (fftlength - eff_ovlp))
        check_kwargs["n_avg"] = min(n_avg_inj, n_avg_bkg)

        # Witness PSDs
        psd_wit_inj = ts_wit_inj.psd(**psd_kwargs)
        psd_wit_bkg = ts_wit_bkg.psd(**psd_kwargs)

        # Check Witness Excess
        # Note: We pass raw_bkg in case PercentileThreshold is used
        mask_wit = threshold_witness.check(
            psd_wit_inj, psd_wit_bkg,
            raw_bkg=ts_wit_bkg,
            **check_kwargs
        )

        delta_wit = psd_wit_inj.value - psd_wit_bkg.value

        results = {}

        # --- 4. Loop over Targets ---
        for tgt_key in target_keys:
            if tgt_key not in data_bkg:
                continue

            ts_tgt_inj = data_inj[tgt_key]
            ts_tgt_bkg = data_bkg[tgt_key]

            # Target PSDs
            psd_tgt_inj = ts_tgt_inj.psd(**psd_kwargs)
            psd_tgt_bkg = ts_tgt_bkg.psd(**psd_kwargs)

            if not np.allclose(psd_wit_inj.xindex.value, psd_tgt_inj.xindex.value):
                 warnings.warn(f"Frequency mismatch for {tgt_key}. Skipping.")
                 continue

            # Check Target Excess
            mask_tgt = threshold_target.check(
                psd_tgt_inj, psd_tgt_bkg,
                raw_bkg=ts_tgt_bkg,
                **check_kwargs
            )

            delta_tgt = psd_tgt_inj.value - psd_tgt_bkg.value

            # --- 5. Compute CF ---
            valid_mask = mask_wit & mask_tgt & (delta_wit > 0) & (delta_tgt > 0)

            cf_values = np.full_like(delta_wit, np.nan)

            with np.errstate(divide='ignore', invalid='ignore'):
                sq_cf = delta_tgt[valid_mask] / delta_wit[valid_mask]
                cf_values[valid_mask] = np.sqrt(sq_cf)

            try:
                cf_unit = psd_tgt_inj.unit.is_unity() and "dimensionless" or (ts_tgt_inj.unit / ts_wit_inj.unit)
            except Exception:
                cf_unit = "dimensionless"

            cf = FrequencySeries(
                cf_values,
                xindex=psd_wit_inj.xindex,
                unit=cf_unit,
                name=f"CF: {witness_key} -> {tgt_key}"
            )

            # --- Calculate Upper Limit (UL) ---
            # Condition: Witness is excessively coupled, but Target is NOT.
            # We want to know: What is the MAX coupling that *could* exist given the Target noise floor?
            # UL ~ ASD_tgt_bkg / (ASD_wit_inj - ASD_wit_bkg) ~ sqrt(PSD_tgt_bkg / delta_wit)

            mask_ul = mask_wit & (~mask_tgt) & (delta_wit > 0)
            ul_values = np.full_like(delta_wit, np.nan)

            with np.errstate(divide='ignore', invalid='ignore'):
                sq_ul = psd_tgt_bkg.value / delta_wit
                ul_values[mask_ul] = np.sqrt(sq_ul[mask_ul])

            cf_ul = FrequencySeries(
                ul_values,
                xindex=psd_wit_inj.xindex,
                unit=cf_unit,
                name=f"CF Upper Limit: {witness_key} -> {tgt_key}"
            )

            results[tgt_key] = CouplingResult(
                cf=cf,
                cf_ul=cf_ul,
                psd_witness_inj=psd_wit_inj,
                psd_witness_bkg=psd_wit_bkg,
                psd_target_inj=psd_tgt_inj,
                psd_target_bkg=psd_tgt_bkg,
                valid_mask=valid_mask,
                witness_name=witness_key,
                target_name=tgt_key,
                ts_witness_bkg=ts_wit_bkg,
                ts_target_bkg=ts_tgt_bkg,
                fftlength=fftlength,
                overlap=overlap
            )

        if len(results) == 1:
            return list(results.values())[0]

        return results


# Functional interface
def estimate_coupling(
    data_inj: TimeSeriesDict,
    data_bkg: TimeSeriesDict,
    fftlength: float,
    witness: Optional[str] = None,
    threshold_witness: Union[ThresholdStrategy, float] = 25.0,
    threshold_target: Union[ThresholdStrategy, float] = 4.0,
    **kwargs
) -> Union[CouplingResult, Dict[str, CouplingResult]]:
    """Helper function to estimate CF."""

    def _ensure_strategy(val):
        if isinstance(val, (int, float)):
            return RatioThreshold(val)
        return val

    tw = _ensure_strategy(threshold_witness)
    tt = _ensure_strategy(threshold_target)

    analysis = CouplingFunctionAnalysis()
    return analysis.compute(
        data_inj, data_bkg, fftlength,
        witness=witness,
        threshold_witness=tw,
        threshold_target=tt,
        **kwargs
    )
