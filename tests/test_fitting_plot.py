import numpy as np
import pytest

try:
    from gwexpy.fitting import fit_series  # noqa: F401
except ImportError as exc:
    pytest.skip(
        f"gwexpy.fitting optional dependencies unavailable: {exc}",
        allow_module_level=True,
    )


def test_fit_result_plot_includes_errorbars():
    import matplotlib.container as mc
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    from gwexpy.fitting import fit_series
    from gwexpy.fitting.models import gaussian
    from gwexpy.timeseries import TimeSeries

    np.random.seed(0)
    t = np.linspace(0, 10, 30)
    y = gaussian(t, A=10, mu=5, sigma=1) + np.random.normal(0, 0.1, len(t))
    ts = TimeSeries(y, times=t)

    res = fit_series(ts, "gaus", sigma=0.5, p0={"A": 8, "mu": 4, "sigma": 1.5})

    fig = Figure()
    FigureCanvas(fig)
    ax = fig.subplots()
    res.plot(ax=ax)

    assert any(isinstance(c, mc.ErrorbarContainer) for c in ax.containers)
    lines = {l.get_label(): l for l in ax.lines}
    assert lines["Fit"].get_zorder() > lines["Data"].get_zorder()
    assert ax.get_xscale() == "auto-gps"
    assert ax.get_xlabel() == ""


def test_fit_result_plot_shows_full_data_and_fit_range_only():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    from gwexpy.fitting import fit_series
    from gwexpy.fitting.models import gaussian
    from gwexpy.timeseries import TimeSeries

    np.random.seed(0)
    t = np.linspace(0, 10, 101)
    y = gaussian(t, A=10, mu=5, sigma=1) + np.random.normal(0, 0.1, len(t))
    ts = TimeSeries(y, times=t)

    res = fit_series(
        ts, "gaus", x_range=(4, 6), sigma=0.5, p0={"A": 8, "mu": 4.5, "sigma": 1.5}
    )

    fig = Figure()
    FigureCanvas(fig)
    ax = fig.subplots()
    res.plot(ax=ax)

    lines = {l.get_label(): l for l in ax.lines}
    x_data = lines["Data"].get_xdata()
    x_fit = lines["Fit"].get_xdata()

    assert len(x_data) == len(t)
    assert float(np.min(x_fit)) >= 4
    assert float(np.max(x_fit)) <= 6


def test_fit_result_bode_plot_includes_magnitude_errorbars():
    import matplotlib.container as mc
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    from gwexpy.fitting import fit_series
    from gwexpy.frequencyseries import FrequencySeries

    np.random.seed(0)
    f = np.logspace(1, 3, 40)

    def cmodel(x, A, phi):
        return A * np.exp(1j * phi) * np.ones_like(x)

    y = cmodel(f, 3.0, 0.3) + 0.05 * (
        np.random.normal(size=len(f)) + 1j * np.random.normal(size=len(f))
    )
    fs = FrequencySeries(y, frequencies=f)
    res = fit_series(fs, cmodel, sigma=0.1, p0={"A": 1.0, "phi": 0.0})

    fig = Figure()
    FigureCanvas(fig)
    ax_mag, ax_phase = fig.subplots(2, 1, sharex=True)
    res.bode_plot(ax=(ax_mag, ax_phase))

    assert any(isinstance(c, mc.ErrorbarContainer) for c in ax_mag.containers)
    lines = {l.get_label(): l for l in ax_mag.lines}
    assert lines["Fit"].get_zorder() > lines["Data"].get_zorder()
    xmin, xmax = ax_mag.get_xlim()
    assert xmin >= min(f)
    assert xmax <= max(f)


def test_fit_result_bode_plot_shows_full_data_and_fit_range_only():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    from gwexpy.fitting import fit_series
    from gwexpy.frequencyseries import FrequencySeries

    np.random.seed(0)
    f = np.logspace(0, 2, 80)

    def cmodel(x, A, phi):
        return A * np.exp(1j * phi) * np.ones_like(x)

    y = cmodel(f, 3.0, 0.3) + 0.05 * (
        np.random.normal(size=len(f)) + 1j * np.random.normal(size=len(f))
    )
    fs = FrequencySeries(y, frequencies=f)

    res = fit_series(fs, cmodel, x_range=(10, 20), sigma=0.1, p0={"A": 1.0, "phi": 0.0})

    fig = Figure()
    FigureCanvas(fig)
    ax_mag, ax_phase = fig.subplots(2, 1, sharex=True)
    res.bode_plot(ax=(ax_mag, ax_phase))

    lines = {l.get_label(): l for l in ax_mag.lines}
    x_data = lines["Data"].get_xdata()
    x_fit = lines["Fit"].get_xdata()

    assert float(np.min(x_data)) <= float(np.min(f))
    assert float(np.max(x_data)) >= float(np.max(f))
    assert float(np.min(x_fit)) >= 10
    assert float(np.max(x_fit)) <= 20 * (1 + 1e-12)
