"""
Coupling Function Analysis Module for gwexpy.

Estimates the coupling function (CF) with flexible threshold strategies:
- RatioThreshold: Mean power ratio.
- SigmaThreshold: Statistical significance (Gaussian assumption).
- PercentileThreshold: Data-driven percentile (Robust to non-Gaussianity).
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

from ..frequencyseries import FrequencySeries
from ..timeseries import TimeSeries, TimeSeriesDict

if TYPE_CHECKING:
    from gwexpy.plot import Plot
    from gwexpy.types.typing import IndexLike


def _index_values(index: object) -> np.ndarray:
    values = getattr(index, "value", index)
    return np.asarray(values, dtype=float)


# --- Threshold Strategies ---


class ThresholdStrategy(ABC):
    """Abstract base class for excess detection strategies."""

    @abstractmethod
    def check(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
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

    @abstractmethod
    def threshold(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        """Return the PSD threshold values used by this strategy."""
        pass


class RatioThreshold(ThresholdStrategy):
    """
    Checks if P_inj > ratio * P_bkg_mean.

    Statistical Assumptions:
        - No specific statistical distribution is assumed.
        - Tests if injection power exceeds the background level by a fixed factor.

    Usage:
        - Best for simple, physical excess screening where precise statistical significance is less critical.
        - Extremely fast as it requires no variance estimation.
    """

    def __init__(self, ratio: float = 2.0) -> None:
        self.ratio = ratio

    def check(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        return psd_inj.value > (psd_bkg.value * self.ratio)

    def threshold(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        return psd_bkg.value * self.ratio


class SigmaThreshold(ThresholdStrategy):
    """
    Checks if P_inj > P_bkg + sigma * std_error.

    Statistical Assumptions
    -----------------------
    - Background Power Spectral Density (PSD) at each bin approximately follows
      a Gaussian distribution (valid when n_avg is sufficiently large).
    - The parameter `n_avg` represents the number of independent averages
      (e.g., in Welch's method).
    - Assumes standard deviation of the noise reduces as `1 / sqrt(n_avg)`.

    Meaning of Threshold
    --------------------
    - ``threshold = mean + sigma * (mean / sqrt(n_avg))``
    - This is a **statistical significance test**, NOT a physical upper limit.
    - It identifies frequencies where the injection is statistically
      distinguishable from background variance.

    Gaussian Approximation Validity
    -------------------------------
    Welch PSD estimates follow a χ² distribution with 2K degrees of freedom
    (K = n_avg). The Gaussian approximation is valid when K ≥ 10 (approximately).

    For K < 10, consider:
    - Using `PercentileThreshold` (empirical distribution, no Gaussian assumption)
    - Increasing FFT averaging by using longer data or shorter fftlength

    References
    ----------
    - Welch, P.D. (1967): PSD estimation via overlapped segment averaging
    - Bendat & Piersol, Random Data (4th ed., 2010), Ch. 11

    Warning
    -------
    This method relies heavily on the Gaussian and stationary assumptions.
    It may be unreliable if:
    - The background contains significant non-Gaussian features (glitches)
    - `n_avg` is small (< ~10), where the central limit theorem has not converged
    - There are strong spectral lines (non-stationary or deterministic signals)

    In such cases, `PercentileThreshold` is recommended as it uses the
    empirical distribution.
    """

    # Minimum n_avg for reliable Gaussian approximation
    _MIN_NAVG_GAUSSIAN = 10

    def __init__(self, sigma: float = 3.0) -> None:
        self.sigma = sigma

    def _check_gaussian_validity(self, n_avg: float) -> None:
        """Warn if Gaussian approximation may be unreliable."""
        if n_avg < self._MIN_NAVG_GAUSSIAN:
            warnings.warn(
                f"SigmaThreshold: n_avg={n_avg:.1f} < {self._MIN_NAVG_GAUSSIAN}. "
                f"Gaussian approximation may be inaccurate for χ²(2K) with K < 10. "
                f"Consider using PercentileThreshold for more robust results.",
                UserWarning,
                stacklevel=3,
            )

    def check(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        n_avg = kwargs.get("n_avg", 1.0)
        if not isinstance(n_avg, (int, float)):
            raise TypeError("SigmaThreshold expects numeric n_avg.")
        if n_avg <= 0:
            return np.ones_like(psd_inj.value, dtype=bool)

        self._check_gaussian_validity(n_avg)
        factor = 1.0 + (self.sigma / np.sqrt(n_avg))
        return psd_inj.value > (psd_bkg.value * factor)

    def threshold(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        n_avg = kwargs.get("n_avg", 1.0)
        if not isinstance(n_avg, (int, float)):
            raise TypeError("SigmaThreshold expects numeric n_avg.")
        if n_avg <= 0:
            return psd_bkg.value
        factor = 1.0 + (self.sigma / np.sqrt(n_avg))
        return psd_bkg.value * factor


class PercentileThreshold(ThresholdStrategy):
    """
    Checks if P_inj > factor * Percentile(P_bkg_segments).

    Statistical Assumptions:
        - Uses the **Empirical Distribution** of the background PSDs.
        - Does not assume Gaussianity; robust against outliers and non-stationary glitches.

    Usage:
        - Requires `raw_bkg` (time series) to compute the distribution across time segments.
        - Higher computational cost than `SigmaThreshold` but more reliable for real-world non-Gaussian data.

    Parameters
    ----------
    percentile : float, default=95
        The percentile of the background distribution (0-100).
    factor : float, default=2.0
        Multiplier for the percentile value.
        Threshold = factor * P_bkg_percentile
    """

    def __init__(self, percentile: float = 95, factor: float = 1.0) -> None:
        self.percentile = percentile
        self.factor = factor

    def check(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        threshold = self.threshold(
            psd_inj,
            psd_bkg,
            raw_bkg=raw_bkg,
            **kwargs,
        )
        return psd_inj.value > threshold

    def threshold(
        self,
        psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        fftlength = kwargs.get("fftlength")
        overlap = kwargs.get("overlap")
        if raw_bkg is None or fftlength is None:
            raise ValueError(
                "PercentileThreshold requires 'raw_bkg' time series and 'fftlength' to calculate distributions."
            )

        if not isinstance(fftlength, (int, float)):
            raise TypeError("PercentileThreshold expects numeric fftlength.")
        if overlap is not None and not isinstance(overlap, (int, float)):
            raise TypeError("PercentileThreshold expects numeric overlap.")

        # Calculate Spectrogram (Time-Frequency map) of background
        # We need the variation over time segments
        spec = raw_bkg.spectrogram(
            stride=fftlength,
            fftlength=fftlength,
            overlap=overlap if overlap else 0,
            method="welch",
            window="hann",
        )

        # Calculate the percentile along the time axis (axis=0)
        # Result is a FrequencySeries-like array of the Xth percentile background
        p_bkg_values = spec.percentile(self.percentile)

        # Apply factor
        threshold = p_bkg_values.value * self.factor

        return threshold


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
        cf_ul: FrequencySeries | None = None,
        ts_witness_bkg: TimeSeries | None = None,
        ts_target_bkg: TimeSeries | None = None,
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


# --- Analysis Class ---

# --- Helper for Parallel Processing ---


def _process_single_target(
    tgt_key: str,
    ts_tgt_inj: TimeSeries,
    ts_tgt_bkg: TimeSeries,
    psd_kwargs: dict[str, object],
    psd_wit_inj: FrequencySeries,
    psd_wit_bkg: FrequencySeries,
    mask_wit: np.ndarray,
    delta_wit: np.ndarray,
    witness_key: str,
    ts_wit_inj: TimeSeries,
    ts_wit_bkg: TimeSeries,
    threshold_target: ThresholdStrategy,
    check_kwargs: Mapping[str, object],
    fftlength: float,
    overlap: float,
    freq_mask: np.ndarray | None,
) -> tuple[str, CouplingResult] | None:
    """
    Process a single target channel.
    This function is defined at module level to ensuring picklability for multiprocessing.
    """
    # Target PSDs
    psd_tgt_inj = ts_tgt_inj.psd(**psd_kwargs)
    psd_tgt_bkg = ts_tgt_bkg.psd(**psd_kwargs)

    # Frequency check
    if not np.allclose(
        _index_values(psd_wit_inj.xindex), _index_values(psd_tgt_inj.xindex)
    ):
        warnings.warn(f"Frequency mismatch for {tgt_key}. Skipping.")
        return None

    # Check Target Excess
    mask_tgt = threshold_target.check(
        psd_tgt_inj, psd_tgt_bkg, raw_bkg=ts_tgt_bkg, **check_kwargs
    )

    delta_tgt = psd_tgt_inj.value - psd_tgt_bkg.value

    # --- Compute CF ---
    valid_mask = mask_wit & mask_tgt & (delta_wit > 0) & (delta_tgt > 0)
    if freq_mask is not None:
        valid_mask = valid_mask & freq_mask

    cf_values = np.full_like(delta_wit, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        sq_cf = delta_tgt[valid_mask] / delta_wit[valid_mask]
        cf_values[valid_mask] = np.sqrt(sq_cf)

    try:
        cf_unit = (
            psd_tgt_inj.unit.is_unity()
            and "dimensionless"
            or (ts_tgt_inj.unit / ts_wit_inj.unit)
        )
    except (AttributeError, TypeError, ValueError):
        logger.debug(
            "Automatic CF unit determination failed, falling back to dimensionless.",
            exc_info=True,
        )
        cf_unit = "dimensionless"

    cf = FrequencySeries(
        cf_values,
        xindex=psd_wit_inj.xindex,
        unit=cf_unit,
        name=f"CF: {witness_key} -> {tgt_key}",
    )

    # --- Calculate Upper Limit (UL) ---
    mask_ul = mask_wit & (~mask_tgt) & (delta_wit > 0)
    if freq_mask is not None:
        mask_ul = mask_ul & freq_mask

    try:
        psd_tgt_threshold = threshold_target.threshold(
            psd_tgt_inj, psd_tgt_bkg, raw_bkg=ts_tgt_bkg, **check_kwargs
        )
    except AttributeError:
        psd_tgt_threshold = psd_tgt_bkg.value

    if hasattr(psd_tgt_threshold, "value"):
        psd_tgt_threshold = psd_tgt_threshold.value

    delta_thr = psd_tgt_threshold - psd_tgt_bkg.value
    mask_ul = mask_ul & (delta_thr > 0)

    ul_values = np.full_like(delta_wit, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        sq_ul = delta_thr / delta_wit
        ul_values[mask_ul] = np.sqrt(sq_ul[mask_ul])

    cf_ul = FrequencySeries(
        ul_values,
        xindex=psd_wit_inj.xindex,
        unit=cf_unit,
        name=f"CF Upper Limit: {witness_key} -> {tgt_key}",
    )

    res = CouplingResult(
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
        overlap=overlap,
    )
    return tgt_key, res


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
        witness: str | None = None,
        frange: tuple[float, float] | None = None,
        overlap: float = 0,
        threshold_witness: ThresholdStrategy = RatioThreshold(25.0),
        threshold_target: ThresholdStrategy = RatioThreshold(4.0),
        n_jobs: int | None = None,
        **kwargs: object,
    ) -> CouplingResult | dict[str, CouplingResult]:
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
        frange : tuple of float, optional
            Frequency range (fmin, fmax) to evaluate CF and CF upper limit.
            Values outside the range are set to NaN.
        overlap : float, optional
            Overlap in seconds (default 0).
        threshold_witness : ThresholdStrategy
            Strategy to determine if Witness is excited.
        threshold_target : ThresholdStrategy
            Strategy to determine if Target is excited.
        n_jobs : int, optional
            Number of jobs for parallel processing. None means 1 unless in a joblib.parallel_config context.
            -1 means using all processors.
        """
        # --- 1. Identify Witness Channel ---
        all_channels = list(data_inj.keys())

        if witness is None:
            witness_key = all_channels[0]
        else:
            witness_key = witness
            if witness_key not in data_inj:
                raise KeyError(
                    f"Witness channel '{witness_key}' not found in input data."
                )

        if witness_key not in data_bkg:
            raise KeyError(
                f"Witness channel '{witness_key}' not found in background data."
            )

        # --- 2. Separate Data ---
        ts_wit_inj = data_inj[witness_key]
        ts_wit_bkg = data_bkg[witness_key]
        target_keys = [k for k in all_channels if k != witness_key]

        if not target_keys:
            raise ValueError(
                "No target channels found. Data must contain at least 2 channels."
            )

        # --- 3. Compute PSDs & N_avg ---
        psd_kwargs = {
            "fftlength": fftlength,
            "overlap": overlap,
            "method": "welch",
            "window": "hann",
        }
        psd_kwargs.update(kwargs)

        # Helper to pass extra data needed by PercentileThreshold
        check_kwargs = {
            "fftlength": fftlength,
            "overlap": overlap,
            "n_avg": 1.0,  # Will be updated
        }

        # Estimate number of averages (N_avg)
        duration_inj = ts_wit_inj.span[1] - ts_wit_inj.span[0]
        duration_bkg = ts_wit_bkg.span[1] - ts_wit_bkg.span[0]
        eff_ovlp = overlap

        # Guard against fftlength == overlap (would cause division by zero)
        if fftlength <= eff_ovlp:
            raise ValueError(
                f"fftlength ({fftlength}) must be greater than overlap ({eff_ovlp}). "
                f"Otherwise, n_avg cannot be computed (division by zero)."
            )

        n_avg_inj = max(1, (duration_inj - eff_ovlp) / (fftlength - eff_ovlp))
        n_avg_bkg = max(1, (duration_bkg - eff_ovlp) / (fftlength - eff_ovlp))
        check_kwargs["n_avg"] = min(n_avg_inj, n_avg_bkg)

        # Witness PSDs
        psd_wit_inj = ts_wit_inj.psd(**psd_kwargs)
        psd_wit_bkg = ts_wit_bkg.psd(**psd_kwargs)

        # Frequency mask for CF evaluation
        freq_mask = None
        if frange is not None:
            if len(frange) != 2:
                raise ValueError("frange must be a tuple of (fmin, fmax)")
            fmin, fmax = frange
            if fmin is None:
                fmin_val = -np.inf
            else:
                fmin_val = (
                    float(getattr(fmin, "to_value", lambda _: fmin)("Hz"))
                    if hasattr(fmin, "to_value")
                    else float(fmin)
                )
            if fmax is None:
                fmax_val = np.inf
            else:
                fmax_val = (
                    float(getattr(fmax, "to_value", lambda _: fmax)("Hz"))
                    if hasattr(fmax, "to_value")
                    else float(fmax)
                )
            if fmin_val > fmax_val:
                raise ValueError("frange must satisfy fmin <= fmax")
            freqs = _index_values(psd_wit_inj.xindex)
            freq_mask = (freqs >= fmin_val) & (freqs <= fmax_val)

        # Check Witness Excess
        # Note: We pass raw_bkg in case PercentileThreshold is used
        mask_wit = threshold_witness.check(
            psd_wit_inj, psd_wit_bkg, raw_bkg=ts_wit_bkg, **check_kwargs
        )

        delta_wit = psd_wit_inj.value - psd_wit_bkg.value

        results = {}

        # --- 4. Parallel Loop over Targets ---

        # Determine joblib usage
        from gwexpy.interop._optional import require_optional

        try:
            joblib = require_optional("joblib")
            Parallel, delayed = joblib.Parallel, joblib.delayed
        except ImportError:
            # Fallback for when joblib is strictly not installed even though we tried
            # Or if user opted out? No, if require_optional fails it raises ImportError.
            # But here we want smooth fallback if user doesn't have it?
            # Actually require_optional raises informative error.
            # If n_jobs is 1 or None, we can just run sequential loop and avoid import error if joblib missing?
            # But the user might want parallel.
            # Let's say: if n_jobs is explicit (not None/1), we require logic.
            # But to keep code clean, let's use joblib if available, else standard loop?
            # We updated _optional.py, so `require_optional` will tell user to install it.
            # But if n_jobs=1 (default-ish), we shouldn't crash if joblib missing.
            n_jobs_eff = n_jobs if n_jobs is not None else 1
            if n_jobs_eff == 1:
                # Sequential Fallback (no joblib needed)
                Parallel = None
            else:
                joblib = require_optional("joblib")
                Parallel, delayed = joblib.Parallel, joblib.delayed

        if Parallel is None:
            # Sequential execution
            for tgt_key in target_keys:
                if tgt_key not in data_bkg:
                    continue

                res = _process_single_target(
                    tgt_key,
                    data_inj[tgt_key],
                    data_bkg[tgt_key],
                    psd_kwargs,
                    psd_wit_inj,
                    psd_wit_bkg,
                    mask_wit,
                    delta_wit,
                    witness_key,
                    ts_wit_inj,
                    ts_wit_bkg,
                    threshold_target,
                    check_kwargs,
                    fftlength,
                    overlap,
                    freq_mask,
                )
                if res:
                    results[res[0]] = res[1]
        else:
            # Parallel execution
            # Prepare generator
            par_results = Parallel(n_jobs=n_jobs)(
                delayed(_process_single_target)(
                    tgt_key,
                    data_inj[tgt_key],
                    data_bkg[tgt_key],
                    psd_kwargs,
                    psd_wit_inj,
                    psd_wit_bkg,
                    mask_wit,
                    delta_wit,
                    witness_key,
                    ts_wit_inj,
                    ts_wit_bkg,
                    threshold_target,
                    check_kwargs,
                    fftlength,
                    overlap,
                    freq_mask,
                )
                for tgt_key in target_keys
                if tgt_key in data_bkg
            )

            for res in par_results:
                if res:
                    results[res[0]] = res[1]

        if len(results) == 1:
            return list(results.values())[0]

        return results


# Functional interface
def estimate_coupling(
    data_inj: TimeSeriesDict,
    data_bkg: TimeSeriesDict,
    fftlength: float,
    witness: str | None = None,
    frange: tuple[float, float] | None = None,
    threshold_witness: ThresholdStrategy | float = 25.0,
    threshold_target: ThresholdStrategy | float = 4.0,
    n_jobs: int | None = None,
    **kwargs: Any,
) -> CouplingResult | dict[str, CouplingResult]:
    """Helper function to estimate CF.

    Parameters
    ----------
    frange : tuple of float, optional
        Frequency range (fmin, fmax) to evaluate CF and CF upper limit.
        Values outside the range are set to NaN.
    """

    def _ensure_strategy(
        val: ThresholdStrategy | float,
    ) -> ThresholdStrategy:
        if isinstance(val, (int, float)):
            return RatioThreshold(val)
        return val

    tw = _ensure_strategy(threshold_witness)
    tt = _ensure_strategy(threshold_target)

    analysis = CouplingFunctionAnalysis()
    return analysis.compute(
        data_inj,
        data_bkg,
        fftlength,
        witness=witness,
        frange=frange,
        threshold_witness=tw,
        threshold_target=tt,
        n_jobs=n_jobs,
        **kwargs,
    )
