"""
High-level fitting pipeline for spectral analysis.

This module provides integrated workflows for common analysis patterns,
combining bootstrap estimation, GLS fitting, and MCMC in a single API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gwexpy.frequencyseries import BifrequencyMap, FrequencySeries
    from gwexpy.spectrogram import Spectrogram
    from gwexpy.timeseries import TimeSeries

    from .core import FitResult


__all__ = ["fit_bootstrap_spectrum"]


def fit_bootstrap_spectrum(
    data_or_spectrogram: TimeSeries | Spectrogram,
    model_fn: Callable,
    freq_range: tuple[float, float] | None = None,
    method: str = "median",
    rebin_width: float | None = None,
    block_size: int | None = None,
    ci: float = 0.68,
    window: str = "hann",
    nperseg: int = 16,
    noverlap: int | None = None,
    n_boot: int = 1000,
    initial_params: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    fixed: list | None = None,
    run_mcmc: bool = False,
    mcmc_walkers: int = 32,
    mcmc_steps: int = 5000,
    mcmc_burn_in: int = 500,
    plot: bool = True,
    progress: bool = True,
) -> FitResult:
    """
    Integrated spectral analysis pipeline with bootstrap, GLS fitting, and MCMC.

    This function provides a unified workflow for:
    1. Converting TimeSeries to Spectrogram (if needed)
    2. Bootstrap resampling to estimate PSD and covariance
    3. GLS fitting with proper frequency correlation
    4. Optional MCMC for Bayesian parameter inference
    5. Visualization of results

    Parameters
    ----------
    data_or_spectrogram : TimeSeries or Spectrogram
        Input data. If TimeSeries, a spectrogram will be computed automatically.
    model_fn : callable
        Model function with signature `model(f, *params) -> y`.
        First argument must be frequency array.
        Example: `lambda f, A, alpha: A * f**alpha`
    freq_range : tuple of (fmin, fmax), optional
        Frequency range for fitting. If None, use all frequencies.
    method : str, optional
        Bootstrap averaging method: 'median' (default) or 'mean'.
    rebin_width : float, optional
        Frequency rebinning width in Hz. If None, no rebinning.
    block_size : int, optional
        Block size for block bootstrap. If None, standard bootstrap.
    ci : float, optional
        Confidence interval for bootstrap errors. Default is 0.68 (1-sigma).
    window : str, optional
        Window function for spectrogram and correlation correction.
        Default is 'hann'.
    nperseg : int, optional
        Segment length for spectrogram generation. Default is 16.
    noverlap : int, optional
        Overlap for spectrogram. If None, uses nperseg // 2.
    n_boot : int, optional
        Number of bootstrap resamples. Default is 1000.
    initial_params : dict, optional
        Initial parameter values for fitting, e.g., {"A": 10, "alpha": -1.5}.
    bounds : dict, optional
        Parameter bounds, e.g., {"A": (0, 100), "alpha": (-5, 0)}.
    fixed : list, optional
        List of parameter names to fix during fitting.
    run_mcmc : bool, optional
        Whether to run MCMC after fitting. Default is False.
    mcmc_walkers : int, optional
        Number of MCMC walkers. Default is 32.
    mcmc_steps : int, optional
        Number of MCMC steps. Default is 5000.
    mcmc_burn_in : int, optional
        MCMC burn-in steps to discard. Default is 500.
    plot : bool, optional
        Whether to display plots. Default is True.
    progress : bool, optional
        Whether to show progress bars for MCMC. Default is True.

    Returns
    -------
    FitResult
        Fit result object containing:
        - Best-fit parameters and errors
        - Chi-square and reduced chi-square
        - Covariance matrix (in `cov_inv` and accessible `cov` attribute)
        - MCMC samples and intervals (if run_mcmc=True)
        - Plotting methods

    Examples
    --------
    >>> from gwexpy.fitting.highlevel import fit_bootstrap_spectrum
    >>>
    >>> # Define model
    >>> def power_law(f, A, alpha):
    ...     return A * f**alpha
    >>>
    >>> # Run pipeline
    >>> result = fit_bootstrap_spectrum(
    ...     data,
    ...     model_fn=power_law,
    ...     freq_range=(5, 50),
    ...     rebin_width=0.25,
    ...     block_size=4,
    ...     initial_params={"A": 10, "alpha": -1.5},
    ...     run_mcmc=True,
    ... )
    >>>
    >>> # Access results
    >>> print(result.params)
    >>> print(result.parameter_intervals)  # If MCMC was run

    Notes
    -----
    The covariance matrix from bootstrap is stored and used for GLS fitting,
    properly accounting for frequency correlations in the spectral estimate.

    See Also
    --------
    gwexpy.spectral.bootstrap_spectrogram : Bootstrap resampling function
    gwexpy.fitting.fit_series : Lower-level fitting function
    gwexpy.fitting.GeneralizedLeastSquares : GLS cost function
    """
    from gwexpy.fitting import fit_series
    from gwexpy.spectral import bootstrap_spectrogram

    # 1. Input check and Spectrogram generation
    # Check if input is a TimeSeries (has sample_rate attribute)
    if hasattr(data_or_spectrogram, "sample_rate"):
        # TimeSeries -> Spectrogram
        if noverlap is None:
            noverlap = nperseg // 2

        spectrogram = data_or_spectrogram.spectrogram2(
            stride=nperseg - noverlap,
            fftlength=nperseg,
            overlap=noverlap,
            window=window,
        )
    else:
        # Assume it's already a Spectrogram
        spectrogram = data_or_spectrogram

    # 2. Bootstrap PSD & covariance estimation
    bootstrap_result = bootstrap_spectrogram(
        spectrogram,
        n_boot=n_boot,
        method=method,
        ci=ci,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        block_size=block_size,
        rebin_width=rebin_width,
        return_map=True,  # Always get covariance map for GLS
        ignore_nan=True,
    )

    psd: FrequencySeries = bootstrap_result[0]
    cov_map: BifrequencyMap = bootstrap_result[1]

    # 3. Frequency range cropping
    if freq_range is not None:
        fmin, fmax = freq_range
        psd = psd.crop(fmin, fmax)

        # Crop covariance map to match
        frequencies = psd.frequencies.value

        # Get full frequency array from cov_map
        cov_freqs = cov_map.frequency1.value

        # Find indices in covariance matrix
        mask = (cov_freqs >= fmin) & (cov_freqs <= fmax)
        cov_cropped = cov_map.value[np.ix_(mask, mask)]

        # Recreate BifrequencyMap
        from astropy import units as u

        from gwexpy.frequencyseries import BifrequencyMap

        freq_unit = psd.frequencies.unit
        cov_map = BifrequencyMap.from_points(
            cov_cropped,
            f2=u.Quantity(frequencies, unit=freq_unit),
            f1=u.Quantity(frequencies, unit=freq_unit),
            unit=cov_map.unit,
            name=cov_map.name,
        )

    # 4. GLS Fitting
    result = fit_series(
        psd,
        model_fn,
        cov=cov_map,
        p0=initial_params,
        limits=bounds,
        fixed=fixed,
    )

    # Store additional metadata
    result.psd = psd
    result.cov = cov_map
    result.bootstrap_method = method

    # 5. MCMC (optional)
    if run_mcmc:
        result.run_mcmc(
            n_walkers=mcmc_walkers,
            n_steps=mcmc_steps,
            burn_in=mcmc_burn_in,
            progress=progress,
        )

    # 6. Visualization (optional)
    if plot:
        _plot_bootstrap_fit(result, psd, run_mcmc)

    return result


def _plot_bootstrap_fit(result: FitResult, psd: FrequencySeries, show_mcmc: bool):
    """Create visualization for bootstrap fit results."""
    import matplotlib.pyplot as plt

    n_plots = 2 if show_mcmc and result.samples is not None else 1

    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        axes = [ax1]

    ax = axes[0]

    # Plot PSD with error bars
    frequencies = psd.frequencies.value
    y = psd.value

    # Get asymmetric errors if available
    if hasattr(psd, "error_low") and hasattr(psd, "error_high"):
        yerr = [psd.error_low.value, psd.error_high.value]
    else:
        yerr = result.dy

    ax.errorbar(
        frequencies,
        y,
        yerr=yerr,
        fmt=".",
        color="black",
        label="Bootstrap PSD",
        alpha=0.7,
        capsize=2,
    )

    # Plot fit
    x_plot = np.linspace(frequencies.min(), frequencies.max(), 200)
    y_fit = result.model(x_plot, **result.params)
    ax.plot(x_plot, y_fit, "r-", lw=2, label="GLS Fit")

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add parameter annotation
    param_text = "\n".join(
        [f"{k} = {v:.3g} ± {result.errors[k]:.2g}" for k, v in result.params.items()]
    )
    ax.text(
        0.05,
        0.95,
        param_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Corner plot if MCMC was run
    if show_mcmc and result.samples is not None and n_plots > 1:
        # Create corner plot in separate figure
        result.plot_corner()
        plt.figure(fig.number)  # Switch back to main figure

    plt.tight_layout()
    plt.show()
