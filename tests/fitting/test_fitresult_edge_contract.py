"""Contract tests for FitResult / bootstrap-spectrum edge-input guards.

Covers the degenerate/edge-input findings from the Phase 1 numerical
robustness sweep grouped under issue #458 (plus the G9 supplement finding).
Each guard converts an opaque NumPy/emcee crash or a silent-NaN fit into a
clear, actionable error.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fitting.core import FitResult
from gwexpy.fitting.core import emcee as _emcee
from gwexpy.fitting.highlevel import (
    _count_free_params,
    _plot_bootstrap_fit,
    fit_bootstrap_spectrum,
)


def _empty_fitresult():
    """Build a FitResult whose x/y are empty (e.g. cropped to zero length)."""
    m = MagicMock()
    m.fval = 0.0
    m.nfit = 1
    m.parameters = ()
    return FitResult(m, lambda x: x, np.array([]), np.array([]))


# --- #458 [P2] core.py: FitResult.plot on empty x -------------------------
def test_plot_empty_x_raises():
    fr = _empty_fitresult()
    with pytest.raises(ValueError, match="FitResult.x is empty"):
        fr.plot()


# --- #458 [P2] core.py: FitResult.plot_fit_band on empty x ----------------
def test_plot_fit_band_empty_x_raises():
    fr = _empty_fitresult()
    fr.samples = np.empty((100, 0))
    fr.mcmc_labels = []
    with pytest.raises(ValueError, match="FitResult.x is empty"):
        fr.plot_fit_band()


# --- #458 [P2] core.py: run_mcmc with all params fixed (ndim == 0) ---------
@pytest.mark.skipif(
    _emcee is None,
    reason="run_mcmc ndim==0 guard is only reachable when emcee is installed",
)
def test_run_mcmc_all_fixed_raises():
    m = MagicMock()
    m.parameters = ("a",)
    m.fixed = {"a": True}
    fr = FitResult(m, lambda x, a: a * x, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="at least one free parameter"):
        fr.run_mcmc()


# --- #458 [P2] highlevel.py: freq_range selects no bins -------------------
def _spectrogram_10_to_59hz():
    data = np.random.rand(20, 50)
    from gwexpy.spectrogram import Spectrogram

    return Spectrogram(data, dt=1.0 * u.s, f0=10 * u.Hz, df=1 * u.Hz)


def test_fit_bootstrap_spectrum_empty_freq_range_raises():
    spec = _spectrogram_10_to_59hz()
    with pytest.raises(ValueError, match="selects no frequency bins"):
        fit_bootstrap_spectrum(
            spec,
            lambda f, A, alpha: A * f**alpha,
            freq_range=(1000, 2000),
            initial_params={"A": 1, "alpha": -1},
            plot=False,
        )


# --- G9 [P3] highlevel.py: freq_range selects too few bins -----------------
def test_fit_bootstrap_spectrum_too_few_bins_raises():
    spec = _spectrogram_10_to_59hz()
    # A single bin (10-10 Hz) with 2 free params => ndof <= 0.
    with pytest.raises(ValueError, match="too few bins"):
        fit_bootstrap_spectrum(
            spec,
            lambda f, A, alpha: A * f**alpha,
            freq_range=(10, 10),
            initial_params={"A": 1, "alpha": -1},
            plot=False,
        )


def test_count_free_params():
    model = lambda f, A, alpha: A * f**alpha  # noqa: E731
    # From initial_params dict.
    assert _count_free_params(model, {"A": 1, "alpha": -1}, None) == 2
    # fixed reduces the count.
    assert _count_free_params(model, {"A": 1, "alpha": -1}, ["alpha"]) == 1
    # Inferred from signature (leading frequency arg dropped).
    assert _count_free_params(model, None, None) == 2


# --- #458 [P3] highlevel.py: _plot_bootstrap_fit single-panel layout -------
def test_plot_bootstrap_fit_single_panel_with_mcmc():
    import matplotlib

    matplotlib.use("Agg")
    from gwexpy.frequencyseries import FrequencySeries

    freqs = u.Quantity(np.arange(10.0, 20.0), unit=u.Hz)
    psd = FrequencySeries(np.linspace(1.0, 2.0, len(freqs)), frequencies=freqs)

    result = MagicMock()
    result.samples = np.random.randn(50, 2)
    result.params = {"A": 1.0, "alpha": -1.0}
    result.errors = {"A": 0.1, "alpha": 0.1}
    result.model = lambda x, **p: np.ones_like(x)
    result.dy = np.ones(len(freqs))
    result.plot_corner = MagicMock(return_value=None)
    del result.error_low  # force the result.dy errorbar branch

    _plot_bootstrap_fit(result, psd, show_mcmc=True)

    import matplotlib.pyplot as plt

    fig = plt.gcf()
    # Exactly one panel, occupying the full figure (1x1 grid, not 1x2 with a
    # blank right half as in the pre-#458 layout).
    assert len(fig.axes) == 1
    geometry = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert geometry == (1, 1)
    plt.close("all")


# --- #458 [P3] highlevel.py: _plot_bootstrap_fit empty PSD ----------------
def test_plot_bootstrap_fit_empty_psd_raises():
    import matplotlib

    matplotlib.use("Agg")
    from gwexpy.frequencyseries import FrequencySeries

    psd = FrequencySeries(np.array([]), frequencies=u.Quantity([], unit=u.Hz))
    with pytest.raises(ValueError, match="no frequency bins"):
        _plot_bootstrap_fit(MagicMock(), psd, False)
