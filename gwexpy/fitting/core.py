from __future__ import annotations

import inspect
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from iminuit import Minuit
from iminuit.util import describe

from .models import get_model

logger = logging.getLogger(__name__)


class ParameterValue(float):
    """
    Representation of a fit parameter value, supporting .value and .error attributes.
    Inherits from float for backward compatibility with numeric operations.
    """

    def __new__(cls, value, error=None):
        return super().__new__(cls, value)

    def __init__(self, value, error=None):
        self.value = value
        self.error = error


class Fitter:
    """
    High-level fitter class that wraps the fit_series function.
    """

    def __init__(self, model: Any):
        self.model = model

    def fit(self, series: Any, **kwargs: Any) -> FitResult:
        """Fit the provided series to the model."""
        return fit_series(series, self.model, **kwargs)


# Optional imports for MCMC
try:
    import corner
    import emcee
except ImportError:
    emcee = None
    corner = None


class ComplexLeastSquares:
    """
    Least Squares cost function for complex-valued data.
    Minimizes the sum of squared residuals for both Real and Imaginary parts.
    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, x, y, dy, model):
        self.x = x
        self.y = y  # Complex data
        self.dy = (
            dy  # Error (assumed isotropic for real/imag unless specified otherwise)
        )
        self.model = model

        # Determine parameters from model (skipping 'x')
        # describe returns a list of parameter names
        params = describe(model)[1:]
        self._parameters = {name: None for name in params}

    def __call__(self, *args):
        # Calculate model prediction
        ym = self.model(self.x, *args)

        # Calculate residuals for Real and Imag parts
        res_real = (self.y.real - ym.real) / self.dy
        res_imag = (self.y.imag - ym.imag) / self.dy

        # Sum of squared residuals
        chi2 = np.sum(res_real**2 + res_imag**2)
        return chi2

    @property
    def ndata(self):
        # Effectively 2 * len(x) data points
        return 2 * len(self.x)


class RealLeastSquares:
    """
    Least Squares cost function for real-valued data.

    This is a small, dependency-light replacement for `iminuit.cost.LeastSquares`
    to avoid optional JIT/caching side effects in some environments.
    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, x, y, dy, model):
        self.x = x
        self.y = y
        self.dy = dy
        self.model = model

        params = describe(model)[1:]
        self._parameters = {name: None for name in params}

    def __call__(self, *args):
        ym = self.model(self.x, *args)
        res = (self.y - ym) / self.dy
        return np.sum(res**2)

    @property
    def ndata(self):
        return len(self.x)


class FitResult:
    def __init__(
        self,
        minuit_obj,
        model,
        x,
        y,
        dy=None,
        cost_func=None,
        x_label=None,
        y_label=None,
        x_kind=None,
        x_data=None,
        y_data=None,
        dy_data=None,
        x_fit_range=None,
        cov_inv=None,
        unit=None,
        x_unit=None,
    ):
        self.minuit = minuit_obj
        self._model = model
        self.x = x
        self.y = y
        self.unit = unit
        self.x_unit = x_unit
        self.dy = dy
        self.cost_func = cost_func
        self.x_label = x_label
        self.y_label = y_label
        self.x_kind = x_kind
        self.x_data = x_data if x_data is not None else x
        self.y_data = y_data if y_data is not None else y
        self.dy_data = dy_data
        self.x_fit_range = x_fit_range
        self.cov_inv = cov_inv  # Inverse covariance matrix for GLS
        self.sampler = None
        self.samples = None
        self.mcmc_labels = None
        self.has_dy = dy is not None
        if self.dy is None:
            self.dy = np.ones_like(y, dtype=float)

    @property
    def model(self):
        """
        Fitted model function.
        Can be called as model(x) to use best-fit parameters,
        or as ``model(x, **params)`` to use specific parameters.
        Returns a Quantity with units if the original data had units.
        """

        def bound_model(x, *args, **kwargs):
            if not args and not kwargs:
                # Use best fit parameters
                kwargs = {k: float(v) for k, v in self.params.items()}

            # Handle X unit conversion if x is a Quantity/Series
            if hasattr(x, "to") and self.x_unit:
                try:
                    x_val = x.to(self.x_unit).value
                except (AttributeError, u.UnitConversionError):
                    # Fallback if conversion fails
                    x_val = getattr(x, "value", np.asarray(x))
            else:
                x_val = getattr(x, "value", np.asarray(x))

            res = self._model(x_val, *args, **kwargs)

            # Unit propagation: apply the Y-unit stored in FitResult
            if self.unit and not hasattr(res, "unit"):
                return res * u.Unit(self.unit)
            return res

        return bound_model

    @property
    def params(self):
        """Best fit parameters (dict of ParameterValue)."""
        return {
            name: ParameterValue(self.minuit.values[name], self.minuit.errors[name])
            for name in self.minuit.parameters
        }

    @property
    def errors(self):
        """Parameter errors (dict)."""
        return {name: self.minuit.errors[name] for name in self.minuit.parameters}

    @property
    def chi2(self):
        """Chi-square value (valid only for LeastSquares-like costs)."""
        # Both LeastSquares and ComplexLeastSquares return chi2 as fval
        return self.minuit.fval

    @property
    def ndof(self):
        """Number of degrees of freedom."""
        n_data = len(self.x)
        if np.iscomplexobj(self.y):
            # For complex data, we have Real and Imag parts, so effectively 2 * N data points
            n_data *= 2
        n_params = self.minuit.nfit
        return max(0, n_data - n_params)

    @property
    def reduced_chi2(self):
        """Reduced Chi-square value."""
        return self.chi2 / self.ndof if self.ndof > 0 else np.nan

    def __str__(self):
        """Delegate to Minuit's pretty printer."""
        return str(self.minuit)

    def _repr_html_(self):
        """Jupyter notebook integration."""
        return self.minuit._repr_html_()

    def plot(self, ax=None, num_points=1000, **kwargs):
        """
        Plot data and best-fit curve.
        For complex data, delegates to bode_plot().
        """
        is_complex = np.iscomplexobj(self.y)

        if is_complex:
            return self.bode_plot(ax=ax, num_points=num_points, **kwargs)

        show_errorbar = kwargs.pop("show_errorbar", None)
        if show_errorbar is None:
            show_errorbar = kwargs.pop("show_errorbars", self.has_dy)

        xscale = kwargs.pop("xscale", None)
        x_range = kwargs.pop("x_range", None)

        # Real Plot
        if ax is None:
            fig, ax = plt.subplots()

        # Prefer GWpy's time scaling for TimeSeries fits (GPS-aware)
        if xscale is None and self.x_kind == "time" and ax.get_xscale() == "linear":
            try:
                import gwpy.plot.gps  # noqa: F401  (registers GPS scales)

                ax.set_xscale("auto-gps")
            except (ImportError, ValueError):
                pass
        elif xscale is not None:
            ax.set_xscale(xscale)

        fit_zorder = kwargs.setdefault("zorder", 5)
        data_zorder = max(0, fit_zorder - 1)
        err_zorder = max(0, data_zorder - 1)

        # Plot Data
        x_data = np.asarray(self.x_data)
        y_data = np.asarray(self.y_data)

        if ax.get_yscale() == "log":
            mask = y_data > 0
            x_data = x_data[mask]
            y_data = y_data[mask]

        # Error bars: if we have full dy, use it; otherwise, fall back to fit-range dy only
        if show_errorbar:
            dy_full = self.dy_data
            if dy_full is not None:
                dy = np.asarray(dy_full)
                if np.isscalar(dy) or dy.shape == ():
                    dy = np.full_like(y_data, float(dy), dtype=float)
                else:
                    dy = dy.astype(float, copy=False)
                    dy = dy[mask] if ax.get_yscale() == "log" else dy

                yerr = dy
                if ax.get_yscale() == "log":
                    lower = np.minimum(dy, y_data * (1 - 1e-12))
                    yerr = np.vstack([lower, dy])

                ax.errorbar(
                    x_data,
                    y_data,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    label="_nolegend_",
                    zorder=err_zorder,
                )
            else:
                x_fit = np.asarray(self.x)
                y_fit = np.asarray(self.y)
                dy_fit_arr = np.asarray(self.dy)
                if ax.get_yscale() == "log":
                    fit_mask = y_fit > 0
                    x_fit = x_fit[fit_mask]
                    y_fit = y_fit[fit_mask]
                    dy_fit_arr = (
                        dy_fit_arr[fit_mask] if dy_fit_arr.shape != () else dy_fit_arr
                    )

                if dy_fit_arr.shape == ():
                    dy_fit_arr = np.full_like(y_fit, float(dy_fit_arr), dtype=float)

                yerr = dy_fit_arr
                if ax.get_yscale() == "log":
                    lower = np.minimum(dy_fit_arr, y_fit * (1 - 1e-12))
                    yerr = np.vstack([lower, dy_fit_arr])

                ax.errorbar(
                    x_fit,
                    y_fit,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    label="_nolegend_",
                    zorder=err_zorder,
                )

        ax.plot(
            x_data,
            y_data,
            ".",
            label="Data",
            color="black",
            zorder=data_zorder,
        )

        # Plot Model (Fit) in front of points
        kwargs.setdefault("color", "red")
        if x_range is None:
            x_range = self.x_fit_range
        if x_range is not None:
            x0, x1 = x_range
            x_plot = np.linspace(min(x0, x1), max(x0, x1), num_points)
        else:
            x_plot = np.linspace(min(self.x), max(self.x), num_points)
        y_plot = self.model(x_plot, **self.params)
        ax.plot(x_plot, y_plot, label="Fit", **kwargs)

        # If we're using a GWpy GPS scale, let GWpy format the X label
        # during draw() (it uses the epoch + unit). Do not override here.
        uses_gps_x = False
        try:
            from gwpy.plot.gps import GPS_SCALES

            uses_gps_x = ax.get_xscale() in GPS_SCALES
        except ImportError:
            uses_gps_x = False

        if self.x_label and not uses_gps_x:
            ax.set_xlabel(self.x_label)
        if self.y_label:
            ax.set_ylabel(self.y_label)

        ax.legend()
        return ax.figure

    def bode_plot(self, ax=None, num_points=1000, **kwargs):
        """
        Create a Bode plot (Magnitude and Phase) for the fit result.
        """
        show_errorbar = kwargs.pop("show_errorbar", None)
        if show_errorbar is None:
            show_errorbar = kwargs.pop("show_errorbars", self.has_dy)
        x_range = kwargs.pop("x_range", None)

        if ax is None:
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
        elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) == 2:
            ax_mag, ax_phase = ax
        else:
            raise ValueError(
                "For bode_plot, 'ax' must be a list/tuple of 2 axes (mag, phase), or None."
            )

        kwargs.setdefault("color", "red")
        fit_zorder = kwargs.setdefault("zorder", 5)
        data_zorder = max(0, fit_zorder - 1)
        err_zorder = max(0, data_zorder - 1)

        # --- Magnitude ---
        ax_mag.set_xscale("log")
        ax_mag.set_yscale("log")

        # Data: use only positive x for log scale, and set natural x-range
        x_all = np.asarray(self.x_data)
        pos = x_all > 0
        if np.any(pos):
            x_min = float(np.min(x_all[pos]))
            x_max = float(np.max(x_all[pos]))
        else:
            x_min, x_max = 1e-1, 1.0

        if x_min == x_max:
            x_min = x_min / 10 if x_min > 0 else 1e-1
            x_max = x_max * 10 if x_max > 0 else 1.0

        if x_range is None:
            x_range = self.x_fit_range
        if x_range is not None:
            x0, x1 = x_range
            x0, x1 = float(min(x0, x1)), float(max(x0, x1))
            x0 = max(x0, x_min)
            x1 = min(x1, x_max)
            if x0 <= 0:
                x0 = x_min
            x_plot = np.logspace(np.log10(x0), np.log10(x1), num_points)
        else:
            x_fit = np.asarray(self.x)
            pos_fit = x_fit > 0
            if np.any(pos_fit):
                xf0 = float(np.min(x_fit[pos_fit]))
                xf1 = float(np.max(x_fit[pos_fit]))
            else:
                xf0, xf1 = x_min, x_max
            x_plot = np.logspace(np.log10(xf0), np.log10(xf1), num_points)

        ym_plot = self.model(x_plot, **self.params)

        ax_mag.set_xscale("log")
        ax_mag.set_yscale("log")

        # Data (Second)
        mag_data = np.abs(self.y_data)
        x_data = x_all
        mag_data = np.asarray(mag_data)

        if ax_mag.get_yscale() == "log":
            mask = (mag_data > 0) & (x_data > 0)
            x_data = x_data[mask]
            mag_data = mag_data[mask]

        # Error bars behind points/fit (and log-safe)
        if show_errorbar:
            dy_full = self.dy_data
            if dy_full is not None:
                dy = np.asarray(dy_full)
                if np.isscalar(dy) or dy.shape == ():
                    dy = np.full_like(mag_data, float(dy), dtype=float)
                else:
                    dy = dy.astype(float, copy=False)
                    dy = dy[mask] if ax_mag.get_yscale() == "log" else dy

                yerr = dy
                if ax_mag.get_yscale() == "log":
                    lower = np.minimum(dy, mag_data * (1 - 1e-12))
                    yerr = np.vstack([lower, dy])

                ax_mag.errorbar(
                    x_data,
                    mag_data,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    label="_nolegend_",
                    zorder=err_zorder,
                )
            else:
                # Fit-range errorbars only
                x_fit = np.asarray(self.x)
                mag_fit = np.abs(np.asarray(self.y))
                dy_fit = np.asarray(self.dy)
                if ax_mag.get_yscale() == "log":
                    fit_mask = (mag_fit > 0) & (x_fit > 0)
                    x_fit = x_fit[fit_mask]
                    mag_fit = mag_fit[fit_mask]
                    dy_fit = dy_fit[fit_mask] if dy_fit.shape != () else dy_fit

                if dy_fit.shape == ():
                    dy_fit = np.full_like(mag_fit, float(dy_fit), dtype=float)

                yerr = dy_fit
                if ax_mag.get_yscale() == "log":
                    lower = np.minimum(dy_fit, mag_fit * (1 - 1e-12))
                    yerr = np.vstack([lower, dy_fit])

                ax_mag.errorbar(
                    x_fit,
                    mag_fit,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    label="_nolegend_",
                    zorder=err_zorder,
                )

        ax_mag.plot(
            x_data,
            mag_data,
            ".",
            label="Data",
            color="black",
            zorder=data_zorder,
        )

        # Fit curve in front of points
        ax_mag.plot(x_plot, np.abs(ym_plot), label="Fit", **kwargs)

        ax_mag.set_xlim(x_min, x_max)

        ax_mag.set_ylabel("Magnitude")
        ax_mag.legend()
        ax_mag.grid(True, which="both", alpha=0.3)

        # --- Phase ---
        # Model (First)
        ax_phase.set_xscale("log")
        ax_phase.plot(x_plot, np.angle(ym_plot, deg=True), label="Fit", **kwargs)

        # Data (Second)
        phase_data = np.angle(self.y_data, deg=True)  # Degrees
        x_phase = x_all[pos] if np.any(pos) else x_all
        phase_data = np.asarray(phase_data)
        phase_data = phase_data[pos] if np.any(pos) else phase_data
        ax_phase.plot(
            x_phase, phase_data, ".", label="Data", color="black", zorder=data_zorder
        )

        ax_phase.set_ylabel("Phase [deg]")
        if self.x_label:
            ax_phase.set_xlabel(self.x_label)
        else:
            ax_phase.set_xlabel("Frequency / Time")

        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.set_xlim(x_min, x_max)

        return (ax_mag, ax_phase)

    def run_mcmc(self, n_walkers=32, n_steps=3000, burn_in=500, progress=True):
        """
        Run MCMC using emcee starting from the best-fit parameters.

        This method supports both standard least squares and GLS (Generalized
        Least Squares) error structures. If `cov_inv` is available, the log
        probability is computed using the full covariance structure.

        For complex-valued data (e.g., Transfer Functions), the cost function
        is computed using the magnitude of residuals or the Hermitian form
        in the case of GLS, ensuring correct handling of real and imaginary parts.

        Parameters
        ----------
        n_walkers : int, optional
            Number of MCMC walkers. Default is 32.
        n_steps : int, optional
            Number of MCMC steps per walker. Default is 3000.
        burn_in : int, optional
            Number of initial steps to discard. Default is 500.
        progress : bool, optional
            Whether to show progress bar. Default is True.

        Returns
        -------
        sampler : emcee.EnsembleSampler
            The emcee sampler object containing the full chain.
        """
        if emcee is None:
            raise ImportError(
                "Please install 'emcee' and 'corner' to use MCMC features."
            )

        # 1. Parameter Information Extraction
        # Filter out fixed parameters for MCMC
        float_params = [p for p in self.minuit.parameters if not self.minuit.fixed[p]]
        ndim = len(float_params)

        # Dictionary of fixed parameters
        fixed_params = {
            p: self.minuit.values[p]
            for p in self.minuit.parameters
            if self.minuit.fixed[p]
        }

        # Cache values for log_prob closure
        x = self.x
        y = self.y
        model = self.model
        cov_inv = self.cov_inv
        dy = self.dy
        all_param_names = list(self.minuit.parameters)
        limits_dict = {
            p: self.minuit.limits[p]
            for p in self.minuit.parameters
            if self.minuit.limits[p] != (None, None)
        }

        # Log Probability Function
        def log_prob(theta):
            # theta: array of float values for float_params

            # Construct full parameter dictionary
            current_params = fixed_params.copy()
            for name, val in zip(float_params, theta):
                current_params[name] = val

                # Check limits defined in minuit
                if name in limits_dict:
                    vmin, vmax = limits_dict[name]
                    if vmin is not None and val < vmin:
                        return -np.inf
                    if vmax is not None and val > vmax:
                        return -np.inf

            try:
                # Get model prediction
                args = [current_params[p] for p in all_param_names]
                ym = model(x, *args)
                r = y - ym

                # Compute log probability based on error structure
                if cov_inv is not None:
                    # GLS: use full covariance structure
                    # Handle complex residuals by taking real part of Hermitian form
                    val = r.conj() @ cov_inv @ r
                    # Ensure we aren't discarding meaningful imaginary parts (should be ~0 for Hermitian form)
                    chi2 = float(np.real(val))
                else:
                    # Standard: use diagonal errors
                    # Use np.abs() to handle complex residuals (TransferFunction) correctly
                    chi2 = float(np.sum(np.abs(r / dy) ** 2))

                return -0.5 * chi2

            except (ValueError, TypeError, ZeroDivisionError):
                # Expected numerical errors
                return -np.inf
            except (AttributeError, RuntimeError):
                # Unexpected errors - log full context but keep the walker alive
                logger.exception("Unexpected error in MCMC log_prob")
                return -np.inf

        # Initial state: small ball around minuit result
        p0_float = np.array([self.minuit.values[p] for p in float_params])
        # Use hessian errors for initialization spread, or small value if fixed/zero
        stds = np.array(
            [
                self.minuit.errors[p]
                if self.minuit.errors[p] > 0
                else 1e-4 * abs(v) + 1e-8
                for p, v in zip(float_params, p0_float)
            ]
        )

        pos = p0_float + stds * 1e-1 * np.random.randn(n_walkers, ndim)

        # Run emcee
        self.sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        self.sampler.run_mcmc(pos, n_steps, progress=progress)

        # Save flattened samples (discarding burn-in)
        self.samples = self.sampler.get_chain(discard=burn_in, flat=True)
        self.mcmc_labels = float_params
        self._burn_in = burn_in

        return self.sampler

    @property
    def parameter_intervals(self):
        """
        Get parameter confidence intervals from MCMC samples.

        Returns 16th, 50th, and 84th percentiles for each parameter,
        corresponding to median and ±1σ bounds.

        Returns
        -------
        dict
            Dictionary mapping parameter names to (lower, median, upper) tuples.

        Raises
        ------
        RuntimeError
            If run_mcmc() has not been called.
        """
        if self.samples is None:
            raise RuntimeError("Run .run_mcmc() first.")

        intervals = {}
        labels = self.mcmc_labels or []
        for i, name in enumerate(labels):
            samples_i = self.samples[:, i]
            q16, q50, q84 = np.percentile(samples_i, [16, 50, 84])
            intervals[name] = (q16, q50, q84)

        return intervals

    @property
    def mcmc_chain(self):
        """
        Get the full MCMC chain (not flattened, not discarded).

        Returns
        -------
        ndarray
            Shape (n_steps, n_walkers, n_params). Returns None if MCMC not run.
        """
        if self.sampler is None:
            return None
        return self.sampler.get_chain()

    def plot_corner(self, show_titles=True, quantiles=None, **kwargs):
        """
        Plot corner plot of MCMC samples.

        Parameters
        ----------
        show_titles : bool, optional
            Whether to show parameter value titles. Default is True.
        quantiles : list, optional
            Quantiles for title display. Default is [0.16, 0.5, 0.84].
        **kwargs
            Additional arguments passed to corner.corner().

        Returns
        -------
        figure : matplotlib.figure.Figure
            The corner plot figure.
        """
        if corner is None:
            raise ImportError("Please install 'corner' to use plot_corner.")
        if self.samples is None:
            raise RuntimeError("Run .run_mcmc() first.")

        # Set defaults
        if quantiles is None:
            quantiles = [0.16, 0.5, 0.84]

        # Show BestFit truth lines
        if self.mcmc_labels:
            truths = [self.minuit.values[p] for p in self.mcmc_labels]
            kwargs.setdefault("truths", truths)
            kwargs.setdefault("labels", self.mcmc_labels)

        kwargs.setdefault("show_titles", show_titles)
        kwargs.setdefault("quantiles", quantiles)
        kwargs.setdefault("title_kwargs", {"fontsize": 10})

        fig = corner.corner(self.samples, **kwargs)

        # Add annotation if GLS was used
        if self.cov_inv is not None:
            fig.text(
                0.95,
                0.95,
                "GLS fit",
                ha="right",
                va="top",
                fontsize=10,
                style="italic",
                transform=fig.transFigure,
            )

        return fig

    def plot_fit_band(
        self, ax=None, num_points=200, n_samples=100, alpha=0.3, **kwargs
    ):
        """
        Plot the fit with uncertainty band from MCMC samples.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        num_points : int, optional
            Number of points for model curve. Default is 200.
        n_samples : int, optional
            Number of MCMC samples to use for band. Default is 100.
        alpha : float, optional
            Alpha transparency for uncertainty band. Default is 0.3.
        **kwargs
            Additional arguments passed to ax.fill_between().

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        if self.samples is None:
            raise RuntimeError("Run .run_mcmc() first.")

        if ax is None:
            fig, ax = plt.subplots()

        # Generate x values for plotting
        x_plot = np.linspace(np.min(self.x), np.max(self.x), num_points)

        # Get random subset of samples
        n_total = len(self.samples)
        indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)

        # Fixed parameters
        fixed_params = {
            p: self.minuit.values[p]
            for p in self.minuit.parameters
            if self.minuit.fixed[p]
        }
        all_param_names = list(self.minuit.parameters)

        # Compute model curves for samples
        y_samples = []
        labels = self.mcmc_labels or []
        for idx in indices:
            sample = self.samples[idx]
            current_params = fixed_params.copy()
            for name, val in zip(labels, sample):
                current_params[name] = val
            args = [current_params[p] for p in all_param_names]
            y_samples.append(self.model(x_plot, *args))

        y_samples_arr = np.array(y_samples)

        if np.iscomplexobj(y_samples_arr):
            # For complex data, we usually want to plot the magnitude band
            y_samples_arr = np.abs(y_samples_arr)

        # Compute percentiles
        y_lower = np.percentile(y_samples_arr, 16, axis=0)
        y_upper = np.percentile(y_samples_arr, 84, axis=0)

        # Plot data
        ax.errorbar(
            self.x,
            self.y,
            yerr=self.dy if self.has_dy else None,
            fmt=".",
            color="black",
            label="Data",
            zorder=2,
        )

        # Plot best fit
        y_best = self.model(x_plot, **self.params)
        ax.plot(x_plot, y_best, color="red", label="Best fit", zorder=3)

        # Plot uncertainty band
        band_color = kwargs.pop("color", "blue")
        ax.fill_between(
            x_plot,
            y_lower,
            y_upper,
            alpha=alpha,
            color=band_color,
            label="68% credible",
            zorder=1,
            **kwargs,
        )

        if self.x_label:
            ax.set_xlabel(self.x_label)
        if self.y_label:
            ax.set_ylabel(self.y_label)
        ax.legend()

        return ax


def fit_series(
    series,
    model,
    x_range=None,
    sigma=None,
    cov=None,
    cost_function=None,
    p0=None,
    limits=None,
    fixed=None,
    **kwargs,
):
    """
    Fit a Series object using iminuit.
    Supports real and complex valued Series (simultaneous Re/Im fit).

    Parameters
    ----------
    series : Series
        Data series to fit.
    model : callable or str
        Model function or name (e.g., "gaussian", "power_law").
    x_range : tuple, optional
        (x_min, x_max) to crop data before fitting.
    sigma : array-like or scalar, optional
        Per-point error estimates. Ignored if `cov` or `cost_function` is provided.
    cov : BifrequencyMap or 2D ndarray, optional
        Covariance matrix for Generalized Least Squares (GLS) fitting.
        If provided, overrides `sigma` and uses GLS χ² minimization.
        Ignored if `cost_function` is provided.
    cost_function : callable, optional
        User-defined cost function for Minuit. If provided, takes priority over
        `sigma`, `cov`, and automatic cost function selection.
        Must be callable with signature `cost_function(*params) -> float`.
        Should have `errordef` attribute (default: 1.0 for least squares).
        Parameter names are extracted via `iminuit.util.describe()`.
    p0 : dict or list, optional
        Initial parameter values.
    limits : dict, optional
        Parameter limits, e.g., {"A": (0, 100)}.
    fixed : list, optional
        List of parameter names to fix during fit.
    **kwargs
        Additional arguments passed to Minuit.

    Returns
    -------
    FitResult
        Object containing fit results, parameters, and plotting methods.
    """
    # 0. モデルの解決
    if isinstance(model, str):
        model_name = model
        model = get_model(model_name)

    # 1. データの準備 (Crop & Unit Stripping)
    target = series.crop(*x_range) if x_range else series

    # x軸の取得
    x_label = "x"
    y_label = "y"

    # full-range data for plotting
    if hasattr(series, "frequencies"):
        x_full = series.frequencies.value
    elif hasattr(series, "times"):
        x_full = series.times.value
    else:
        x_full = series.xindex.value
    y_full = series.value

    if hasattr(target, "frequencies"):
        x = target.frequencies.value
        x_label = "Frequency"
        x_kind = "frequency"
        if hasattr(target, "xunit") and str(target.xunit) != "dimensionless":
            x_label += f" [{target.xunit}]"
    elif hasattr(target, "times"):
        x = target.times.value
        x_label = "Time"
        x_kind = "time"
        if hasattr(target, "xunit") and str(target.xunit) != "dimensionless":
            x_label += f" [{target.xunit}]"
    else:
        x = target.xindex.value
        if hasattr(target, "xunit") and str(target.xunit) != "dimensionless":
            x_label += f" [{target.xunit}]"
        x_kind = "index"

    y = target.value

    # Determine y-label
    if hasattr(target, "unit") and str(target.unit) != "dimensionless":
        y_label_unit = f"[{target.unit}]"
    else:
        y_label_unit = ""

    if hasattr(target, "name") and target.name:
        y_label = f"{target.name}"
        if y_label_unit:
            y_label += f" {y_label_unit}"
    elif y_label_unit:
        y_label = f"Amplitude {y_label_unit}"

    # 誤差の処理
    original_len = len(series)
    sigma_full_for_plot: np.ndarray | float | None = None
    if sigma is None:
        # 重みなし最小二乗 (Cost function internally uses 1.0)
        dy = np.ones(len(y))
        sigma_for_result = None
    else:
        # If sigma is a scalar, broadcast it
        if np.isscalar(sigma):
            sigma_val = float(sigma)  # type: ignore[arg-type]
            dy = np.full(len(y), sigma_val)
            sigma_full_for_plot = sigma_val
        else:
            sigma_arr = np.asarray(sigma)
            sigma_full_for_plot = sigma_arr if len(sigma_arr) == original_len else None
            if len(sigma_arr) == original_len and x_range is not None:
                # Crop sigma by x range using the full x array
                x0, x1 = x_range
                lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
                idx0 = int(np.searchsorted(x_full, lo, side="left"))
                idx1 = int(np.searchsorted(x_full, hi, side="right"))
                dy = sigma_arr[idx0:idx1]
            else:
                dy = sigma_arr

        # Final length check
        if len(dy) != len(y):
            raise ValueError(f"Sigma length mismatch: got {len(dy)}, expected {len(y)}")
        sigma_for_result = dy

    is_complex = np.iscomplexobj(y)

    # 2. Cost Function
    # Priority: cost_function > cov > sigma > default
    cov_inv_for_result = None  # Will be set if GLS is used

    if cost_function is not None:
        # User-provided cost function takes highest priority
        cost = cost_function
        # Try to extract cov_inv from cost function if it's a GLS
        if hasattr(cost_function, "cov_inv"):
            cov_inv_for_result = cost_function.cov_inv
    elif cov is not None:
        # GLS mode: use covariance matrix
        if is_complex:
            raise NotImplementedError(
                "GLS fitting is not yet supported for complex data. "
                "Please use sigma instead."
            )

        from .gls import GeneralizedLeastSquares

        # Handle BifrequencyMap or 2D ndarray
        cov_arr = None
        cov_inv = None

        # Check if cov is a BifrequencyMap (duck typing to avoid import issues)
        if hasattr(cov, "inverse") and hasattr(cov, "value"):
            # BifrequencyMap: get inverse and extract value
            inv_map = cov.inverse()
            cov_inv = np.asarray(inv_map.value)
            cov_arr = np.asarray(cov.value)
        else:
            # Assume 2D ndarray covariance
            cov_arr = np.asarray(cov)
            if cov_arr.ndim != 2:
                raise ValueError(f"cov must be a 2D array, got {cov_arr.ndim}D")
            cov_inv = np.linalg.pinv(cov_arr)

        # Check dimension matches
        n = len(y)
        if cov_inv.shape != (n, n):
            raise ValueError(
                f"Covariance matrix shape {cov_arr.shape} does not match "
                f"data length {n}. Expected ({n}, {n})."
            )

        # Generate dy from diagonal of covariance for plotting error bars
        diag_cov = np.diag(cov_arr)
        # Handle potential negative values from numerical issues
        dy = np.sqrt(np.maximum(diag_cov, 0))
        sigma_for_result = dy
        sigma_full_for_plot = None  # Full-range sigma not available for GLS yet

        cost = GeneralizedLeastSquares(x, y, cov_inv, model)
        cov_inv_for_result = cov_inv  # Save for MCMC
    elif is_complex:
        cost = ComplexLeastSquares(x, y, dy, model)
    else:
        cost = RealLeastSquares(x, y, dy, model)

    # 3. Minuit 初期化
    # Get parameter names from the cost function (which got them from model)
    param_names = describe(cost)
    sig = inspect.signature(model)

    init_params = {}

    # Handle list/tuple p0 by mapping positional args to parameter names
    if p0 is not None and isinstance(p0, (list, tuple, np.ndarray)):
        # Optionally check lengths
        for i, val in enumerate(p0):
            if i < len(param_names):
                init_params[param_names[i]] = val

    for name in param_names:
        # If already set by list/tuple, skip
        if name in init_params:
            continue

        if p0 and isinstance(p0, dict) and name in p0:
            init_params[name] = p0[name]
        elif (
            name in sig.parameters
            and sig.parameters[name].default is not inspect.Parameter.empty
        ):
            init_params[name] = sig.parameters[name].default
        else:
            # Fallback for parameters without p0 or default
            init_params[name] = 1.0

    m = Minuit(cost, **init_params)

    # 4. Limit / Fix の適用
    if limits:
        for name, (vmin, vmax) in limits.items():
            m.limits[name] = (vmin, vmax)

    if fixed:
        for name in fixed:
            m.fixed[name] = True

    # 5. 実行
    m.migrad()
    m.hesse()

    return FitResult(
        m,
        model,
        x,
        y,
        dy=sigma_for_result,
        cost_func=cost,
        x_label=x_label,
        y_label=y_label,
        x_kind=x_kind,
        x_data=x_full,
        y_data=y_full,
        dy_data=sigma_full_for_plot,
        x_fit_range=x_range,
        cov_inv=cov_inv_for_result,
        unit=getattr(target, "unit", None),
        x_unit=getattr(target, "xunit", None),
    )
