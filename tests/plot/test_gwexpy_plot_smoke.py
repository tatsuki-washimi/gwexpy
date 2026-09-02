import matplotlib

matplotlib.use("Agg")

import numpy as np


def test_plot_timeseries_no_matrix_does_not_error():
    from gwexpy.plot import Plot
    from gwexpy.timeseries import TimeSeries

    ts = TimeSeries(np.arange(10), dt=1.0)
    ts_cropped = TimeSeries(np.arange(5), dt=1.0)

    plot = Plot(ts, ts_cropped)

    import matplotlib.pyplot as plt

    plt.close(plot)


def test_timeseriesdict_plot_not_blank():
    import matplotlib.pyplot as plt
    from gwpy.timeseries import TimeSeries

    from gwexpy.timeseries import TimeSeriesDict

    ts = TimeSeries(np.ones(16), t0=0, dt=1 / 16)
    tsd = TimeSeriesDict({"A": ts, "B": ts.copy()})

    plot = tsd.plot()

    artist_count = sum(
        len(axis.lines) + len(axis.collections) + len(axis.images) for axis in plot.axes
    )
    assert artist_count >= 2

    plt.close(plot)
