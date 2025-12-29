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
from typing import Optional, Union, Dict, List

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
        target_name: str
    ):
        self.cf = cf
        self.psd_witness_inj = psd_witness_inj
        self.psd_witness_bkg = psd_witness_bkg
        self.psd_target_inj = psd_target_inj
        self.psd_target_bkg = psd_target_bkg
        self.valid_mask = valid_mask
        self.witness_name = witness_name
        self.target_name = target_name

    @property
    def frequencies(self):
        return self.cf.xindex

    def plot(self, **kwargs):
        """Plot the Coupling Function."""
        plot = self.cf.plot(**kwargs)
        ax = plot.gca()
        ax.set_yscale("log")
        ax.set_ylabel(f"CF Magnitude [{self.cf.unit}]")
        ax.set_title(f"Coupling Function: {self.witness_name} -> {self.target_name}")
        return plot

    def plot_diagnosis(self):
        """
        Create a diagnostic plot showing PSDs and the resulting CF.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # 1. Witness PSDs
        f_axis = self.psd_witness_inj.xindex.value
        ax[0].loglog(f_axis, self.psd_witness_inj.value, label="Injection (Witness)", color="tab:blue")
        ax[0].loglog(self.psd_witness_bkg.xindex.value, self.psd_witness_bkg.value, label="Background (Witness)", color="tab:cyan", linestyle="--")
        ax[0].set_ylabel(f"PSD [{self.psd_witness_inj.unit}]")
        ax[0].set_title(f"Witness: {self.witness_name}")
        ax[0].legend()
        ax[0].grid(True, which="both", linestyle=":")

        # 2. Target PSDs
        ax[1].loglog(self.psd_target_inj.xindex.value, self.psd_target_inj.value, label="Injection (Target)", color="tab:red")
        ax[1].loglog(self.psd_target_bkg.xindex.value, self.psd_target_bkg.value, label="Background (Target)", color="tab:orange", linestyle="--")
        ax[1].set_ylabel(f"PSD [{self.psd_target_inj.unit}]")
        ax[1].set_title(f"Target: {self.target_name}")
        ax[1].legend()
        ax[1].grid(True, which="both", linestyle=":")

        # 3. Coupling Function
        ax[2].loglog(self.cf.xindex.value, self.cf.value, label="Coupling Function", color="black")
        
        ax[2].set_xlabel("Frequency [Hz]")
        ax[2].set_ylabel(f"CF [{self.cf.unit}]")
        ax[2].set_title(f"Coupling Function ({self.witness_name} -> {self.target_name})")
        ax[2].grid(True, which="both", linestyle=":")
        ax[2].legend()

        plt.tight_layout()
        return fig


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
        threshold_witness: ThresholdStrategy = RatioThreshold(2.0),
        threshold_target: ThresholdStrategy = RatioThreshold(1.1),
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

            results[tgt_key] = CouplingResult(
                cf=cf,
                psd_witness_inj=psd_wit_inj,
                psd_witness_bkg=psd_wit_bkg,
                psd_target_inj=psd_tgt_inj,
                psd_target_bkg=psd_tgt_bkg,
                valid_mask=valid_mask,
                witness_name=witness_key,
                target_name=tgt_key
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
    threshold_witness: Union[ThresholdStrategy, float] = 2.0,
    threshold_target: Union[ThresholdStrategy, float] = 1.1,
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