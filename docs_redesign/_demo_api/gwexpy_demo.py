"""Tiny stand-in for the GWexpy API, used only to render the docs prototype.

This module intentionally has no third-party dependencies so the
documentation can be built without installing GWexpy or GWpy. The classes and
functions below mirror the *shape* of the real API and carry NumPy-style
docstrings so that ``autosummary``/``autodoc`` have something to render.
"""

from __future__ import annotations


class TimeSeriesMatrix:
    """A 2-D stack of time series sharing a common time axis.

    Each row is a channel; all channels share the same sample rate and epoch.
    This container mirrors GWpy's ``TimeSeries`` API but adds a channel
    dimension for multi-sensor analysis.

    Parameters
    ----------
    data : array_like
        Two-dimensional array of shape ``(n_channels, n_samples)``.
    sample_rate : float, optional
        Sampling rate in hertz. Defaults to ``1.0``.
    names : list of str, optional
        Per-channel names. Defaults to ``None``.

    Attributes
    ----------
    shape : tuple of int
        ``(n_channels, n_samples)``.

    Examples
    --------
    >>> m = TimeSeriesMatrix([[0, 1, 2], [3, 4, 5]], sample_rate=16.0)
    >>> m.shape
    (2, 3)
    """

    def __init__(self, data, sample_rate: float = 1.0, names=None):
        self._data = data
        self.sample_rate = sample_rate
        self.names = names or []

    @property
    def shape(self):
        """tuple of int: The ``(n_channels, n_samples)`` shape."""
        n_rows = len(self._data)
        n_cols = len(self._data[0]) if n_rows else 0
        return (n_rows, n_cols)

    def transfer_function(self, reference: int = 0):
        """Estimate transfer functions relative to a reference channel.

        Parameters
        ----------
        reference : int, optional
            Index of the channel used as the denominator. Defaults to ``0``.

        Returns
        -------
        FrequencySeriesMatrix
            One complex transfer function per channel.
        """
        return FrequencySeriesMatrix([], reference=reference)


class FrequencySeriesMatrix:
    """A 2-D stack of frequency series sharing a common frequency axis.

    Parameters
    ----------
    data : array_like
        Two-dimensional array of shape ``(n_channels, n_freqs)``.
    reference : int, optional
        Index of the reference channel, if any. Defaults to ``None``.
    """

    def __init__(self, data, reference=None):
        self._data = data
        self.reference = reference

    def coherence(self):
        """Return the magnitude-squared coherence of each channel pair.

        Returns
        -------
        FrequencySeriesMatrix
            Real-valued coherence in ``[0, 1]``.
        """
        return FrequencySeriesMatrix(self._data)


def bruco(target, witnesses, fftlength: float = 1.0):
    """Brute-force coherence scan of a target against many witnesses.

    Ranks witness channels by their coherence with a target channel — a common
    first step in noise hunting.

    Parameters
    ----------
    target : TimeSeriesMatrix
        The channel to explain.
    witnesses : TimeSeriesMatrix
        Candidate witness channels.
    fftlength : float, optional
        FFT length in seconds. Defaults to ``1.0``.

    Returns
    -------
    list of tuple
        ``(channel_name, peak_coherence)`` pairs, sorted descending.

    Examples
    --------
    >>> ranking = bruco(target, witnesses, fftlength=2.0)
    """
    return []
