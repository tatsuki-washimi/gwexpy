"""Coupling Function Analysis Module for gwexpy.

Estimates the coupling function (CF) with flexible threshold strategies.
Threshold strategies are defined in :mod:`gwexpy.analysis.threshold`.
The result container is defined in :mod:`gwexpy.analysis.coupling_result`.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..table.segment_table import SegmentTable

from ..frequencyseries import FrequencySeries
from ..timeseries import TimeSeries, TimeSeriesDict
from .coupling_result import CouplingResult
from .threshold import (
    PercentileThreshold,
    RatioThreshold,
    SigmaThreshold,
    ThresholdStrategy,
    _align_psd_values_to_reference,  # re-export for test compatibility  # noqa: F401
    _index_values,
)

__all__ = [
    "CouplingFunctionAnalysis",
    "CouplingResult",
    "PercentileThreshold",
    "RatioThreshold",
    "SigmaThreshold",
    "ThresholdStrategy",
    "_align_psd_values_to_reference",
    "estimate_bkg_mem_bytes",
    "estimate_coupling",
]


def estimate_bkg_mem_bytes(
    duration: float, fftlength: float, stride: float, fs: float
) -> tuple[int, int]:
    """Estimate memory required for a background SegmentTable.

    Returns
    -------
    bytes_est : int
        Estimated memory in bytes (including 20% overhead).
    n_rows : int
        Number of rows.

    """
    n_rows = max(1, int(np.floor((duration - fftlength) / stride)) + 1)
    n_freqs = int(fftlength * fs / 2) + 1
    bytes_est = n_rows * n_freqs * 8  # float64
    return int(bytes_est * 1.2), n_rows


def _build_bkg_segment_table(
    ts_bkg: TimeSeries,
    fftlength: float,
    overlap: float = 0,
    stride: float | None = None,
    memory_limit: int = 2 * 1024**3,  # Default 2GB
    **kwargs: Any,
) -> SegmentTable:
    """Helper to convert a background TimeSeries into a SegmentTable of PSDs.
    """
    from gwpy.segments import Segment

    from ..table.segment_table import SegmentTable

    if stride is None:
        stride = fftlength

    duration = ts_bkg.span[1] - ts_bkg.span[0]
    fs = ts_bkg.sample_rate.value

    # --- Step-wise Stride Adjustment for Memory Control ---
    mem_size, n_rows = estimate_bkg_mem_bytes(duration, fftlength, stride, fs)

    if mem_size > memory_limit:
        orig_stride = stride
        # Increase stride automatically up to 10x fftlength to fit memory
        while mem_size > memory_limit and stride < (fftlength * 10):
            stride *= 2
            mem_size, n_rows = estimate_bkg_mem_bytes(duration, fftlength, stride, fs)

        if mem_size > memory_limit:
            raise ValueError(
                f"Background SegmentTable memory estimate ({mem_size/1e6:.2f} MB) "
                f"exceeds limit ({memory_limit/1e6:.2f} MB) even after stride adjustment. "
                "Increase 'memory_limit' or use a larger 'bkg_stride'."
            )
        logger.warning(
            "Increased bkg_stride from %.2fs to %.2fs to fit memory limit (%.2f MB).",
            orig_stride,
            stride,
            memory_limit / 1e6,
        )

    logger.info(
        "Building bkg SegmentTable: %d rows, estimated %.2f MB.",
        n_rows,
        mem_size / 1e6,
    )

    segments = []
    t0 = ts_bkg.span[0]
    for i in range(n_rows):
        t_s = t0 + i * stride
        t_e = t_s + fftlength
        segments.append(Segment(t_s, t_e))

    st = SegmentTable.from_segments(segments)

    # --- Materialization Strategy for Joblib Pickling ---
    # We materialize PSDs directly into the table row by row.
    # Since we already verified it fits in memory_limit, this is safe and pickle-stable.
    kept_segments = []
    psds = []
    for seg in segments:
        cropped = ts_bkg.crop(seg[0], seg[1])
        if (cropped.span[1] - cropped.span[0]) < fftlength:
            warnings.warn(
                "Skipping background segment because its cropped duration is "
                f"shorter than fftlength ({fftlength}s).",
                UserWarning,
                stacklevel=2,
            )
            continue
        p = cropped.psd(
            fftlength=fftlength, overlap=overlap, **kwargs
        )
        kept_segments.append(seg)
        psds.append(p)

    if not psds:
        raise ValueError("No background segments remain after boundary checks.")

    if len(psds) != len(segments):
        st = SegmentTable.from_segments(kept_segments)

    st.add_series_column("psd", data=psds, kind="frequencyseries")
    return st


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
    bkg_table: SegmentTable | None = None,
) -> tuple[str, CouplingResult] | None:
    """Process a single target channel.
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
        psd_tgt_inj, psd_tgt_bkg, raw_bkg=ts_tgt_bkg, bkg_table=bkg_table, **check_kwargs
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
            psd_tgt_inj, psd_tgt_bkg, raw_bkg=ts_tgt_bkg, bkg_table=bkg_table, **check_kwargs
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
        ts_witness_inj=ts_wit_inj,
        ts_witness_bkg=ts_wit_bkg,
        ts_target_inj=ts_tgt_inj,
        ts_target_bkg=ts_tgt_bkg,
        fftlength=fftlength,
        overlap=overlap,
    )
    return tgt_key, res


# --- Analysis Class ---


class CouplingFunctionAnalysis:
    """Analysis class to estimate Coupling Functions (CF).
    """

    @classmethod
    def from_time_windows(
        cls,
        data: TimeSeriesDict,
        bkg_window: tuple[float, float],
        inj_window: tuple[float, float],
        witness: str | None = None,
        fftlength: float = 2.0,
        overlap: float = 0,
        threshold_strategy: ThresholdStrategy | float = 3.0,
        frange: tuple[float, float] | None = None,
        percentile_factor: float = 2.6,
        bkg_stride: float | None = None,
        memory_limit: int = 2 * 1024**3,
        n_jobs: int | None = None,
        **kwargs: Any,
    ) -> CouplingResult | dict[str, CouplingResult]:
        """Compute Coupling Function by specifying time windows explicitly.

        Parameters
        ----------
        data : TimeSeriesDict
            Input data containing the full time range (both background and injection).
        bkg_window : tuple of float
            GPS time window (t_start, t_end) for the background period.
        inj_window : tuple of float
            GPS time window (t_start, t_end) for the injection period.
        witness : str, optional
            The name of the witness channel. If None, the first channel is used.
        fftlength : float
            FFT length in seconds.
        overlap : float
            Overlap in seconds (default 0).
        threshold_strategy : ThresholdStrategy or float
            Threshold strategy used for both Witness and Target.
            If a float is provided, it is interpreted as a :class:`RatioThreshold`.
        frange : tuple of float, optional
            Frequency range (fmin, fmax) to evaluate.
        percentile_factor : float
            Correction factor for :class:`PercentileThreshold` (Appendix B.1).
        bkg_stride : float, optional
            Stride in seconds for building the background SegmentTable.
        memory_limit : int
            Maximum memory in bytes for the background SegmentTable (default 2 GB).
        n_jobs : int, optional
            Number of parallel jobs. None means 1; -1 uses all cores.

        Returns
        -------
        CouplingResult or dict of CouplingResult
            A CouplingResult for a single target, or a dict for multiple targets.

        Examples
        --------
        >>> result = CouplingFunctionAnalysis.from_time_windows(
        ...     data,
        ...     bkg_window=(1000000000, 1000000100),
        ...     inj_window=(1000000200, 1000000300),
        ...     witness="V1:ENV_WIT",
        ...     fftlength=4.0,
        ... )

        """
        bkg_start, bkg_end = bkg_window
        inj_start, inj_end = inj_window

        if bkg_end <= bkg_start:
            raise ValueError(
                f"bkg_window end ({bkg_end}) must be greater than start ({bkg_start})."
            )
        if inj_end <= inj_start:
            raise ValueError(
                f"inj_window end ({inj_end}) must be greater than start ({inj_start})."
            )

        data_bkg = TimeSeriesDict(
            {k: v.crop(bkg_start, bkg_end) for k, v in data.items()}
        )
        data_inj = TimeSeriesDict(
            {k: v.crop(inj_start, inj_end) for k, v in data.items()}
        )

        def _ensure_strategy(val: ThresholdStrategy | float) -> ThresholdStrategy:
            if isinstance(val, (int, float)):
                return RatioThreshold(val)
            return val

        strategy = _ensure_strategy(threshold_strategy)

        instance = cls()
        return instance.compute(
            data_inj=data_inj,
            data_bkg=data_bkg,
            fftlength=fftlength,
            witness=witness,
            frange=frange,
            overlap=overlap,
            threshold_witness=strategy,
            threshold_target=strategy,
            percentile_factor=percentile_factor,
            bkg_stride=bkg_stride,
            memory_limit=memory_limit,
            n_jobs=n_jobs,
            **kwargs,
        )

    @classmethod
    def from_time_windows_batch(
        cls,
        data: TimeSeriesDict,
        bkg_window: tuple[float, float],
        inj_windows: list[tuple[float, float]],
        witness: str | None = None,
        fftlength: float = 2.0,
        overlap: float = 0,
        threshold_strategy: ThresholdStrategy | float = 3.0,
        frange: tuple[float, float] | None = None,
        percentile_factor: float = 2.6,
        bkg_stride: float | None = None,
        memory_limit: int = 2 * 1024**3,
        n_jobs: int | None = None,
        **kwargs: Any,
    ) -> list[CouplingResult | dict[str, CouplingResult]]:
        """Compute Coupling Functions for multiple injection windows in a batch.

        Parameters
        ----------
        data : TimeSeriesDict
            Input data containing the full time range.
        bkg_window : tuple of float
            GPS time window (t_start, t_end) for the background period.
            Commonly used across all injection windows.
        inj_windows : list of tuple of float
            List of injection windows, each as (t_start, t_end).
        witness : str, optional
            The name of the witness channel. If None, the first channel is used.
        fftlength : float
            FFT length in seconds.
        overlap : float
            Overlap in seconds (default 0).
        threshold_strategy : ThresholdStrategy or float
            Threshold strategy used for both Witness and Target.
            If a float is provided, it is interpreted as a :class:`RatioThreshold`.
        frange : tuple of float, optional
            Frequency range (fmin, fmax) to evaluate.
        percentile_factor : float
            Correction factor for :class:`PercentileThreshold` (Appendix B.1).
        bkg_stride : float, optional
            Stride in seconds for building the background SegmentTable.
        memory_limit : int
            Maximum memory in bytes for the background SegmentTable (default 2 GB).
        n_jobs : int, optional
            Number of parallel jobs. None means 1; -1 uses all cores.

        Returns
        -------
        list of CouplingResult or dict of CouplingResult
            List of results corresponding to each injection window.

        """
        if not inj_windows:
            raise ValueError("inj_windows must contain at least one window.")

        results: list[CouplingResult | dict[str, CouplingResult]] = []
        for inj_window in inj_windows:
            res = cls.from_time_windows(
                data=data,
                bkg_window=bkg_window,
                inj_window=inj_window,
                witness=witness,
                fftlength=fftlength,
                overlap=overlap,
                threshold_strategy=threshold_strategy,
                frange=frange,
                percentile_factor=percentile_factor,
                bkg_stride=bkg_stride,
                memory_limit=memory_limit,
                n_jobs=n_jobs,
                **kwargs,
            )
            results.append(res)
        return results

    def auto_calibrate_percentile_factor(
        self,
        raw_bkg: TimeSeries,
        target: TimeSeries,
        fftlength: float,
        overlap: float = 0,
        percentile: float = 99.7,
        factor_range: tuple[float, float] = (0.1, 10.0),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Automatically calibrate the percentile correction factor (Appendix B).

        Finds the factor `c` such that the background noise distribution
        best matches the expected statistical floor, aiming for reduced chi2 ≈ 1.
        """
        from scipy.optimize import minimize_scalar

        # 1. Prepare Background Distribution
        # We use a SegmentTable for the background to get the empirical distribution
        st_bkg = _build_bkg_segment_table(
            raw_bkg, fftlength, overlap=overlap, **kwargs
        )
        # psd_matrix shape: (segments, frequencies)
        psd_matrix = np.stack([row["psd"].value for row in st_bkg])
        p_emp = np.percentile(psd_matrix, percentile, axis=0)

        # 2. Target Baseline (Mean/Median PSD)
        psd_tgt_bkg = target.psd(fftlength=fftlength, overlap=overlap, **kwargs).value

        # 3. Objective Function: |Reduced Chi2 - 1|
        # We want: mean_target ≈ factor * percentile_bkg
        def objective(c: float) -> float:
            threshold = p_emp * c
            # Simple reduced chi2 proxy: average ratio
            # (In a real scenario, this would be more complex weighting)
            r_chi2 = np.mean(psd_tgt_bkg / threshold)
            return (r_chi2 - 1.0) ** 2

        res = minimize_scalar(objective, bounds=factor_range, method="bounded")

        if not res.success:
            warnings.warn("Auto-calibration failed to converge. Using best guess.")

        return {
            "percentile_factor": float(res.x),
            "reduced_chi2": float(np.sqrt(res.fun) + 1.0),
            "success": bool(res.success),
        }

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
        percentile_factor: float = 2.6,
        bkg_stride: float | None = None,
        memory_limit: int = 2 * 1024**3,
        n_jobs: int | None = None,
        **kwargs: object,
    ) -> CouplingResult | dict[str, CouplingResult]:
        """Compute Coupling Function(s) from TimeSeriesDicts.

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
            "factor": percentile_factor,
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

        t_start = time.perf_counter()

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

        # --- 4. Parallel Setup ---
        Parallel = None
        delayed = None
        n_jobs_eff = n_jobs if n_jobs is not None else 1

        if n_jobs_eff != 1:
            from gwexpy.interop._optional import require_optional
            joblib = require_optional("joblib")
            Parallel, delayed = joblib.Parallel, joblib.delayed

        # --- 3.5 Prepare Background SegmentTables for PercentileThreshold ---
        st_wit_bkg = None
        if isinstance(threshold_witness, PercentileThreshold):
            st_wit_bkg = _build_bkg_segment_table(
                ts_wit_bkg,
                fftlength,
                overlap,
                stride=bkg_stride,
                memory_limit=memory_limit,
                **kwargs,
            )

        target_bkg_tables: dict[str, SegmentTable] = {}
        if isinstance(threshold_target, PercentileThreshold):
            for tgt_key in target_keys:
                if tgt_key not in data_bkg:
                    continue
                st_tgt_bkg = _build_bkg_segment_table(
                    data_bkg[tgt_key],
                    fftlength,
                    overlap,
                    stride=bkg_stride,
                    memory_limit=memory_limit,
                    **kwargs,
                )
                target_bkg_tables[tgt_key] = st_tgt_bkg

        # Check Witness Excess
        mask_wit = threshold_witness.check(
            psd_wit_inj, psd_wit_bkg, raw_bkg=ts_wit_bkg, bkg_table=st_wit_bkg, **check_kwargs
        )

        delta_wit = psd_wit_inj.value - psd_wit_bkg.value

        results = {}

        # --- 4. Parallel Loop over Targets ---

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
                    bkg_table=target_bkg_tables.get(tgt_key),
                )
                if res:
                    results[res[0]] = res[1]
        else:
            # Parallel execution
            assert Parallel is not None
            assert delayed is not None
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
                    bkg_table=target_bkg_tables.get(tgt_key),
                )
                for tgt_key in target_keys
                if tgt_key in data_bkg
            )

            for res in par_results:
                if res:
                    results[res[0]] = res[1]

        t_end = time.perf_counter()
        dur = t_end - t_start

        # --- 5. Summary Logging ---
        n_targets = len(results)
        n_total = len(target_keys)
        logger.info(
            "Coupling Analysis Complete: %d/%d channels estimated in %.2fs. "
            "percentile_factor=%.2f",
            n_targets, n_total, dur, percentile_factor
        )

        if len(results) == 1:
            return list(results.values())[0]

        return results


# Functional interface
def estimate_coupling(
    data_inj: TimeSeriesDict,
    data_bkg: TimeSeriesDict | None = None,
    fftlength: float = 2.0,
    witness: str | None = None,
    frange: tuple[float, float] | None = None,
    overlap: float = 0,
    threshold_witness: ThresholdStrategy | float = 25.0,
    threshold_target: ThresholdStrategy | float = 4.0,
    percentile_factor: float = 2.6,
    bkg_stride: float | None = None,
    memory_limit: int = 2 * 1024**3,
    n_jobs: int | None = None,
    bkg_window: tuple[float, float] | None = None,
    inj_window: tuple[float, float] | None = None,
    **kwargs: Any,
) -> CouplingResult | dict[str, CouplingResult]:
    """Helper function to estimate CF.

    Parameters
    ----------
    data_inj : TimeSeriesDict
        Injection data (Witness + Targets).
        If ``bkg_window`` / ``inj_window`` are specified, please pass data covering
        the full time range. In this case, ``data_bkg`` is not required.
    data_bkg : TimeSeriesDict, optional
        Background data (Witness + Targets).
        Can be omitted if ``bkg_window`` is specified.
    fftlength : float
        FFT length in seconds.
    witness : str, optional
        The name (key) of the witness channel.
        If None, the FIRST channel in ``data_inj`` is used.
    frange : tuple of float, optional
        Frequency range (fmin, fmax) to evaluate CF and CF upper limit.
        Values outside the range are set to NaN.
    overlap : float, optional
        Overlap in seconds (default 0).
    threshold_witness : ThresholdStrategy or float
        Strategy to determine if Witness is excited.
        If a float is given, it is interpreted as a ratio for :class:`RatioThreshold`.
    threshold_target : ThresholdStrategy or float
        Strategy to determine if Target is excited.
        If a float is given, it is interpreted as a ratio for :class:`RatioThreshold`.
    percentile_factor : float, optional
        Correction factor for :class:`PercentileThreshold` (Appendix B.1).
        Default is 2.6.
    bkg_stride : float, optional
        Stride in seconds for background SegmentTable construction.
        Defaults to ``fftlength``.
    memory_limit : int, optional
        Maximum memory in bytes for the background SegmentTable.
        Default is 2 GB.
    n_jobs : int, optional
        Number of jobs for parallel processing.
        ``None`` means 1; ``-1`` means all processors.
    bkg_window : tuple of float, optional
        GPS time window (t_start, t_end) for the background period.
        If specified, background data is cropped from ``data_inj``, so ``data_bkg`` is not required.
        Recommended to be used with ``inj_window``.
    inj_window : tuple of float, optional
        GPS time window (t_start, t_end) for the injection period.
        Works as part of the time-window API when used with ``bkg_window``.
    **kwargs
        Additional keyword arguments forwarded to PSD computation.

    """

    def _ensure_strategy(
        val: ThresholdStrategy | float,
    ) -> ThresholdStrategy:
        if isinstance(val, (int, float)):
            return RatioThreshold(val)
        return val

    # --- Time window mode ---
    if bkg_window is not None:
        # Both inj_window and bkg_window must be specified together
        if inj_window is None:
            raise ValueError(
                "inj_window must be specified when bkg_window is given. "
                "Use estimate_coupling(..., bkg_window=(...), inj_window=(...))"
            )
        tw = _ensure_strategy(threshold_witness)
        analysis = CouplingFunctionAnalysis()
        return analysis.from_time_windows(
            data=data_inj,
            bkg_window=bkg_window,
            inj_window=inj_window,
            witness=witness,
            fftlength=fftlength,
            overlap=overlap,
            threshold_strategy=tw,
            frange=frange,
            percentile_factor=percentile_factor,
            bkg_stride=bkg_stride,
            memory_limit=memory_limit,
            n_jobs=n_jobs,
            **kwargs,
        )

    # --- Legacy two-dict mode ---
    if data_bkg is None:
        raise ValueError(
            "data_bkg is required when bkg_window is not specified. "
            "Either pass data_bkg or use bkg_window + inj_window."
        )

    tw = _ensure_strategy(threshold_witness)
    tt = _ensure_strategy(threshold_target)

    analysis = CouplingFunctionAnalysis()
    return analysis.compute(
        data_inj,
        data_bkg,
        fftlength,
        witness=witness,
        frange=frange,
        overlap=overlap,
        threshold_witness=tw,
        threshold_target=tt,
        percentile_factor=percentile_factor,
        bkg_stride=bkg_stride,
        memory_limit=memory_limit,
        n_jobs=n_jobs,
        **kwargs,
    )
