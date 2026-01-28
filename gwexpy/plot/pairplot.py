"""
gwexpy.plot.pairplot
--------------------

Pair plot for Series collections.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

__all__ = ["PairPlot"]


def _get_min_sample_rate(series_list: list) -> float | None:
    """Get the minimum sample rate from a list of series."""
    rates = []
    for s in series_list:
        if hasattr(s, "sample_rate"):
            rates.append(float(s.sample_rate.value))
        elif hasattr(s, "df"):
            rates.append(float(s.df.value))
    if not rates:
        return None
    return min(rates)


def _get_common_span(series_list: list) -> tuple[float | None, float | None]:
    """Get common time/frequency span from a list of series."""
    starts = []
    ends = []
    for s in series_list:
        if hasattr(s, "t0") and hasattr(s, "duration"):
            starts.append(float(s.t0.value))
            ends.append(float(s.t0.value) + float(s.duration.value))
        elif hasattr(s, "f0") and hasattr(s, "df"):
            starts.append(float(s.f0.value))
            ends.append(float(s.f0.value) + float(s.df.value) * len(s))
        elif hasattr(s, "xindex"):
            starts.append(float(s.xindex[0]))
            ends.append(float(s.xindex[-1]))
    if not starts or not ends:
        return None, None
    return max(starts), min(ends)


def _align_series(series_list: list) -> list:
    """Align series to common sample rate and span."""
    if len(series_list) < 2:
        return series_list

    # Get minimum sample rate
    min_rate = _get_min_sample_rate(series_list)

    # Get common span
    start, end = _get_common_span(series_list)

    aligned = []
    for s in series_list:
        aligned_s = s

        # Resample if needed
        if min_rate is not None and hasattr(s, "sample_rate"):
            current_rate = float(s.sample_rate.value)
            if current_rate > min_rate:
                aligned_s = aligned_s.resample(min_rate)

        # Crop to common span
        if start is not None and end is not None:
            if hasattr(aligned_s, "crop"):
                aligned_s = aligned_s.crop(start, end)

        aligned.append(aligned_s)

    return aligned


def _normalize_input(data: Any) -> tuple[list, list[str]]:
    """
    Normalize input data to a list of series with labels.

    Accepts: list, tuple, dict, TimeSeriesDict, TimeSeriesList, etc.
    """
    series_list = []
    labels = []

    if isinstance(data, dict):
        for key, value in data.items():
            series_list.append(value)
            labels.append(str(key))
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            series_list.append(item)
            if hasattr(item, "name") and item.name:
                labels.append(str(item.name))
            else:
                labels.append(f"Series {i}")
    elif hasattr(data, "items"):
        # Dict-like (TimeSeriesDict, etc.)
        for key, value in data.items():
            series_list.append(value)
            labels.append(str(key))
    elif hasattr(data, "__iter__"):
        # List-like (TimeSeriesList, etc.)
        for i, item in enumerate(data):
            series_list.append(item)
            if hasattr(item, "name") and item.name:
                labels.append(str(item.name))
            else:
                labels.append(f"Series {i}")
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")

    return series_list, labels


class PairPlot:
    """
    Pair plot for Series collections.

    Visualizes pairwise relationships between series in a collection.

    Parameters
    ----------
    data : list, tuple, dict, TimeSeriesDict, TimeSeriesList, etc.
        Collection of series to plot.
    corner : bool, default True
        If True, show only lower triangle.
    diag : str, default 'hist'
        Diagonal plot type: 'hist', 'kde'.
    offdiag : str, default 'hist2d'
        Off-diagonal plot type: 'hist2d', 'scatter'.
    bins : int, default 50
        Number of bins for histograms.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    **kwargs
        Additional arguments.

    Examples
    --------
    >>> from gwexpy.plot import PairPlot
    >>> plot = PairPlot(timeseries_dict)
    >>> plot.show()
    """

    def __init__(
        self,
        data: Any,
        *,
        corner: bool = True,
        diag: str = "hist",
        offdiag: str = "hist2d",
        bins: int = 50,
        figsize: tuple | None = None,
        **kwargs,
    ):
        self.corner = corner
        self.diag = diag
        self.offdiag = offdiag
        self.bins = bins
        self.kwargs = kwargs

        # Normalize input
        series_list, labels = _normalize_input(data)

        # Align series
        self._series = _align_series(series_list)
        self._labels = labels
        self._n = len(self._series)

        # Calculate figsize
        if figsize is None:
            figsize = (2.5 * self._n, 2.5 * self._n)
        self._figsize = figsize

        # Create figure
        self._fig, self._axes = self._create_figure()
        self._plot()

    def _create_figure(self) -> tuple[Figure, np.ndarray]:
        """Create figure and axes grid."""
        fig, axes = plt.subplots(self._n, self._n, figsize=self._figsize, squeeze=False)
        return fig, axes

    def _plot(self):
        """Plot the pair plot."""
        for i in range(self._n):
            for j in range(self._n):
                ax = self._axes[i, j]

                if self.corner and j > i:
                    # Upper triangle: hide
                    ax.set_visible(False)
                    continue

                if i == j:
                    # Diagonal
                    self._plot_diagonal(ax, i)
                else:
                    # Off-diagonal
                    self._plot_offdiagonal(ax, i, j)

                # Labels
                if i == self._n - 1:
                    ax.set_xlabel(self._labels[j])
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(self._labels[i])
                else:
                    ax.set_yticklabels([])

        self._fig.tight_layout()

    def _plot_diagonal(self, ax, i: int):
        """Plot diagonal cell (histogram or kde)."""
        data = np.asarray(self._series[i].value).flatten()
        data = data[~np.isnan(data)]

        if self.diag == "hist":
            ax.hist(data, bins=self.bins, density=True, alpha=0.7)
        elif self.diag == "kde":
            from scipy import stats

            kde = stats.gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 200)
            ax.plot(x, kde(x))
            ax.fill_between(x, kde(x), alpha=0.3)

    def _plot_offdiagonal(self, ax, i: int, j: int):
        """Plot off-diagonal cell (2d histogram or scatter)."""
        data_i = np.asarray(self._series[i].value).flatten()
        data_j = np.asarray(self._series[j].value).flatten()

        # Remove NaN
        mask = ~(np.isnan(data_i) | np.isnan(data_j))
        data_i = data_i[mask]
        data_j = data_j[mask]

        # Truncate to same length
        min_len = min(len(data_i), len(data_j))
        data_i = data_i[:min_len]
        data_j = data_j[:min_len]

        if self.offdiag == "hist2d":
            ax.hist2d(data_j, data_i, bins=self.bins, cmap="Blues")
        elif self.offdiag == "scatter":
            ax.scatter(data_j, data_i, alpha=0.3, s=1)

    @property
    def figure(self) -> Figure:
        """Return the matplotlib Figure."""
        return self._fig

    @property
    def axes(self) -> np.ndarray:
        """Return the axes array."""
        return self._axes

    def show(self):
        """Show the plot."""
        plt.show()
        return self

    def savefig(self, *args, **kwargs):
        """Save the figure."""
        self._fig.savefig(*args, **kwargs)
        return self
