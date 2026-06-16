"""Minimal stand-in for the GWexpy public API.

This module exists only so the prototype documentation can generate real
:mod:`sphinx.ext.autodoc` / :mod:`sphinx.ext.autosummary` content without
installing the full ``gwexpy`` package. It mirrors a small, representative
slice of the container API.
"""

from __future__ import annotations

import numpy as np


class TimeSeriesMatrix:
    """A multi-channel time series.

    Wraps a 2-D array whose first axis indexes channels and whose second axis
    indexes time samples, together with a common sampling rate.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_channels, n_samples)``.
    sample_rate : float
        Sampling rate in hertz, shared by every channel.

    Attributes
    ----------
    data : numpy.ndarray
        The underlying sample array.
    sample_rate : float
        Sampling rate in hertz.

    Examples
    --------
    >>> import numpy as np
    >>> tsm = TimeSeriesMatrix(np.zeros((2, 8)), sample_rate=4.0)
    >>> tsm.duration
    2.0
    """

    def __init__(self, data: np.ndarray, sample_rate: float) -> None:
        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError("data must be 2-D (n_channels, n_samples)")
        self.sample_rate = float(sample_rate)

    @property
    def n_channels(self) -> int:
        """int: Number of channels (rows) in the matrix."""
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        """int: Number of time samples (columns) per channel."""
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        """float: Duration of the series in seconds."""
        return self.n_samples / self.sample_rate

    def psd(self, nperseg: int = 256) -> np.ndarray:
        """Estimate a one-sided power spectral density per channel.

        Parameters
        ----------
        nperseg : int, optional
            Length of each segment used in the estimate.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_channels, nperseg // 2 + 1)``.
        """
        n_bins = nperseg // 2 + 1
        return np.abs(np.fft.rfft(self.data[:, :nperseg], axis=1)) ** 2 / n_bins


class FrequencySeriesMatrix:
    """A multi-channel frequency series.

    The frequency-domain counterpart of :class:`TimeSeriesMatrix`.

    Parameters
    ----------
    data : numpy.ndarray
        Complex array of shape ``(n_channels, n_freqs)``.
    df : float
        Frequency resolution in hertz.

    Attributes
    ----------
    data : numpy.ndarray
        The underlying spectrum array.
    df : float
        Frequency spacing in hertz.
    """

    def __init__(self, data: np.ndarray, df: float) -> None:
        self.data = np.asarray(data)
        self.df = float(df)

    @property
    def frequencies(self) -> np.ndarray:
        """numpy.ndarray: Frequency axis in hertz."""
        return np.arange(self.data.shape[-1]) * self.df


def combine_channels(matrix: TimeSeriesMatrix) -> np.ndarray:
    """Combine channels of a matrix in quadrature.

    Parameters
    ----------
    matrix : TimeSeriesMatrix
        The multi-channel series to combine.

    Returns
    -------
    numpy.ndarray
        A 1-D array of length ``matrix.n_samples`` holding the
        root-sum-square across channels.
    """
    return np.sqrt(np.sum(np.square(matrix.data), axis=0))
