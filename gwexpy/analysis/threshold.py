"""
Threshold strategies for Coupling Function Analysis.

This module provides excess-detection strategies used by CouplingFunctionAnalysis:
- ThresholdStrategy: Abstract base class
- RatioThreshold: Simple power-ratio test
- SigmaThreshold: Gaussian significance test
- PercentileThreshold: Empirical percentile (Appendix B)
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..table.segment_table import SegmentTable
    from ..frequencyseries import FrequencySeries
    from ..timeseries import TimeSeries


def _index_values(index: object) -> np.ndarray:
    values = getattr(index, "value", index)
    return np.asarray(values, dtype=float)


def _align_psd_values_to_reference(
    values: np.ndarray,
    freqs: np.ndarray,
    ref_freqs: np.ndarray,
    method: str = "clip",
) -> np.ndarray | None:
    """Align PSD values to a reference frequency grid.

    Parameters
    ----------
    values : np.ndarray
        Source PSD values.
    freqs : np.ndarray
        Source frequency axis.
    ref_freqs : np.ndarray
        Reference frequency axis.
    method : {'clip', 'interpolate'}, default: 'clip'
        Alignment method. 'clip' returns None if frequencies don't match exactly.
        'interpolate' uses linear interpolation if the maximum bin shift is <= 1.

    Returns
    -------
    np.ndarray or None
        Aligned values, or None if alignment is not possible under the given policy.
    """
    if np.array_equal(freqs, ref_freqs):
        return np.asarray(values, dtype=float)

    if freqs.ndim != 1 or ref_freqs.ndim != 1 or freqs.size == 0 or ref_freqs.size == 0:
        return None

    # Common range check
    freq_min = np.min(freqs)
    freq_max = np.max(freqs)
    if ref_freqs[0] < freq_min or ref_freqs[-1] > freq_max:
        # Strictly prohibit extrapolation
        return None

    if method == "clip":
        # In clip mode, we only allow exact matches (handled above)
        return None

    if method == "interpolate":
        # Estimate index-wise bin shift on the overlapping prefix.
        df = np.median(np.diff(freqs)) if len(freqs) > 1 else 0
        if df > 0:
            n_common = min(len(freqs), len(ref_freqs))
            max_shift = (
                np.max(np.abs(freqs[:n_common] - ref_freqs[:n_common])) / df
            )
            if max_shift > 1.0:
                return None

        return np.interp(ref_freqs, freqs, values)

    return None


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
        **kwargs: object,
    ) -> np.ndarray:
        return psd_inj.value > (psd_bkg.value * self.ratio)

    def threshold(
        self,
        _psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
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
        _raw_bkg: TimeSeries | None = None,
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
        _psd_inj: FrequencySeries,
        psd_bkg: FrequencySeries,
        _raw_bkg: TimeSeries | None = None,
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
    Threshold strategy based on empirical percentile of background distribution.

    This strategy follows Appendix B of the PEM injection paper, using the
    99.7th percentile of background segments and a correction factor
    to account for finite-averaging and χ² distribution scaling.

    Parameters
    ----------
    percentile : float, default=99.7
        The percentile of the background distribution (0-100).
        99.7% equivalent to 3-sigma for Gaussian noise.
    factor : float, default=2.6
        Correction factor (multiplier) for the percentile value.
        The value 2.6 is recommended in Appendix B.1 to set reduced χ² ≈ 1.
    freq_align : {'clip', 'interpolate'}, default='clip'
        Frequency alignment strategy for background segments.
    """

    def __init__(
        self,
        percentile: float = 99.7,
        factor: float = 2.6,
        freq_align: str = "clip",
    ) -> None:
        self.percentile = percentile
        self.factor = factor
        self.freq_align = freq_align

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
        _psd_bkg: FrequencySeries,
        raw_bkg: TimeSeries | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        fftlength = kwargs.get("fftlength")
        overlap = kwargs.get("overlap")
        bkg_table = kwargs.get("bkg_table")

        percentile = kwargs.get("percentile", self.percentile)
        factor = kwargs.get("factor", self.factor)
        freq_align = kwargs.get("freq_align", self.freq_align)

        if bkg_table is not None:
            # --- SegmentTable Mode ---
            if "psd" not in bkg_table.columns:
                raise ValueError("SegmentTable provided to PercentileThreshold has no 'psd' column.")

            # Extract PSDs (handles both meta values and lazy payload cells)
            psds = [bkg_table.row(i)["psd"] for i in range(len(bkg_table))]

            target_unit = psd_inj.unit
            ref_freqs = _index_values(psd_inj.xindex)

            psd_values = []
            for row_idx, p in enumerate(psds):
                p_val = p.value
                p_freqs = _index_values(p.xindex)

                # Frequency Alignment Check
                if not np.array_equal(p_freqs, ref_freqs):
                    aligned = _align_psd_values_to_reference(
                        p_val, p_freqs, ref_freqs, method=freq_align
                    )
                    if aligned is None:
                        warnings.warn(
                            "Skipping background PSD row "
                            f"{row_idx} because its frequency grid does not cover the "
                            "injection PSD grid or exceeds bin shift tolerance.",
                            UserWarning,
                            stacklevel=3,
                        )
                        continue
                    p_val = aligned

                if p.unit != target_unit:
                    try:
                        # Convert units if needed (Quantity support)
                        from astropy.units import Quantity
                        p_q = Quantity(p_val, p.unit)
                        p_val = p_q.to(target_unit).value
                    except Exception:
                        warnings.warn(
                            f"Unit conversion failed for bkg_table row: {p.unit} -> {target_unit}. "
                            f"Using numeric values.",
                            UserWarning,
                            stacklevel=3,
                        )

                psd_values.append(p_val)

            if not psd_values:
                raise ValueError(
                    "No compatible background PSD rows remain after frequency alignment."
                )

            data_matrix = np.stack(psd_values)
            p_bkg_values = np.percentile(data_matrix, percentile, axis=0)
        else:
            # --- Legacy / Raw TimeSeries Mode ---
            if raw_bkg is None or fftlength is None:
                raise ValueError(
                    "PercentileThreshold requires 'bkg_table', or 'raw_bkg' + 'fftlength'."
                )

            if not isinstance(fftlength, (int, float)):
                raise TypeError("PercentileThreshold expects numeric fftlength.")
            if overlap is not None and not isinstance(overlap, (int, float)):
                raise TypeError("PercentileThreshold expects numeric overlap.")

            # Calculate Spectrogram (Time-Frequency map) of background
            spec = raw_bkg.spectrogram(
                stride=fftlength,
                fftlength=fftlength,
                overlap=overlap if overlap else 0,
                method="welch",
                window="hann",
            )
            p_bkg_values = spec.percentile(percentile).value

        return p_bkg_values * factor
