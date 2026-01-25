import numpy as np


def test_plot_timeseries_no_matrix_does_not_error():
    from gwexpy.plot import Plot
    from gwexpy.timeseries import TimeSeries

    ts = TimeSeries(np.arange(10), dt=1.0)
    ts_cropped = TimeSeries(np.arange(5), dt=1.0)

    plot = Plot(ts, ts_cropped)

    import matplotlib.pyplot as plt

    plt.close(plot)
