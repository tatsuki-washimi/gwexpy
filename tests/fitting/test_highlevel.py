"""Tests for gwexpy/fitting/highlevel.py - _plot_bootstrap_fit helper."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from unittest.mock import patch

from gwpy.frequencyseries import FrequencySeries
from gwexpy.fitting import fit_series
from gwexpy.fitting.highlevel import _plot_bootstrap_fit


def _linear(x, a, b):
    return a * x + b


def _make_fit_result():
    """Create a FitResult via fit_series for use in plot tests."""
    x = np.linspace(1, 10, 20)
    a_true, b_true = 2.0, 1.0
    rng = np.random.default_rng(0)
    y = a_true * x + b_true + rng.normal(0, 0.1, len(x))
    dy = np.full_like(y, 0.1)
    fs = FrequencySeries(y, frequencies=x, unit="1/Hz")
    result = fit_series(fs, _linear)
    return result


def _make_psd():
    x = np.linspace(1, 10, 20)
    y = 2.0 * x + 1.0
    return FrequencySeries(y, frequencies=x, unit="1/Hz")


class TestPlotBootstrapFit:
    def test_basic_no_mcmc(self):
        """_plot_bootstrap_fit runs without errors when show_mcmc=False."""
        import matplotlib.pyplot as plt
        result = _make_fit_result()
        psd = _make_psd()
        with patch("matplotlib.pyplot.show"):  # suppress GUI
            _plot_bootstrap_fit(result, psd, show_mcmc=False)
        plt.close("all")

    def test_with_mcmc_no_samples(self):
        """When show_mcmc=True but samples is None, uses single-plot layout."""
        import matplotlib.pyplot as plt
        result = _make_fit_result()
        result.samples = None  # explicitly no MCMC samples
        psd = _make_psd()
        with patch("matplotlib.pyplot.show"):
            _plot_bootstrap_fit(result, psd, show_mcmc=True)
        plt.close("all")

    def test_psd_with_error_attrs(self):
        """Uses error_low/error_high if available on psd."""
        import matplotlib.pyplot as plt
        result = _make_fit_result()
        psd = _make_psd()
        # Attach asymmetric error attributes
        x = np.linspace(1, 10, 20)
        err_fs = FrequencySeries(np.full(20, 0.1), frequencies=x, unit="1/Hz")
        psd.error_low = err_fs
        psd.error_high = err_fs
        with patch("matplotlib.pyplot.show"):
            _plot_bootstrap_fit(result, psd, show_mcmc=False)
        plt.close("all")
