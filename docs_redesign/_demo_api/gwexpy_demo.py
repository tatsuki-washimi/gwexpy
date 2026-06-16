"""Self-contained demo module mirroring the GWexpy public API surface.

This module is intentionally dependency-free. It exists only so that the
documentation prototype can exercise :mod:`sphinx.ext.autodoc` and
:mod:`sphinx.ext.autosummary` and render real, generated API tables and
signatures -- without installing the actual ``gwexpy`` package.

The classes and functions below imitate the shape and docstring style of the
real containers (``TimeSeriesMatrix``, ``FrequencySeriesMatrix``) and analysis
helpers (the BrUCo noise budget).
"""

from __future__ import annotations

__all__ = [
    "TimeSeriesMatrix",
    "FrequencySeriesMatrix",
    "noise_budget",
]


class TimeSeriesMatrix:
    """A 2-D collection of regularly sampled time series.

    A ``TimeSeriesMatrix`` stores ``n_channels`` synchronous time series that
    share a single time axis. It is the multi-dimensional analogue of
    :class:`gwpy.timeseries.TimeSeries`.

    Parameters
    ----------
    data : array_like
        Two-dimensional array of shape ``(n_channels, n_samples)``.
    sample_rate : float
        Sampling rate in hertz.
    channels : list of str, optional
        Human-readable names for each channel. Defaults to ``None``.

    Attributes
    ----------
    sample_rate : float
        The sampling rate in hertz.
    n_channels : int
        Number of channels (rows) in the matrix.

    Examples
    --------
    >>> tsm = TimeSeriesMatrix([[0.0, 1.0], [1.0, 0.0]], sample_rate=16384.0)
    >>> tsm.n_channels
    2
    """

    def __init__(self, data, sample_rate, channels=None):
        self._data = data
        self.sample_rate = sample_rate
        self.channels = channels or []

    @property
    def n_channels(self) -> int:
        """int: Number of channels stored in the matrix."""
        return len(self._data)

    def fft(self) -> "FrequencySeriesMatrix":
        """Compute the per-channel Fast Fourier Transform.

        Returns
        -------
        FrequencySeriesMatrix
            The frequency-domain representation of every channel.
        """
        return FrequencySeriesMatrix(self._data, df=self.sample_rate / 2.0)

    def crop(self, start, end) -> "TimeSeriesMatrix":
        """Return a copy cropped to the half-open interval ``[start, end)``.

        Parameters
        ----------
        start : float
            Start time in seconds.
        end : float
            End time in seconds.

        Returns
        -------
        TimeSeriesMatrix
            A new, cropped matrix.
        """
        return self


class FrequencySeriesMatrix:
    """A 2-D collection of frequency series sharing one frequency axis.

    Parameters
    ----------
    data : array_like
        Two-dimensional array of shape ``(n_channels, n_freqs)``.
    df : float
        Frequency resolution in hertz.

    See Also
    --------
    TimeSeriesMatrix : The time-domain counterpart.
    """

    def __init__(self, data, df):
        self._data = data
        self.df = df

    def coherence(self, other: "FrequencySeriesMatrix"):
        """Estimate the magnitude-squared coherence with ``other``.

        Parameters
        ----------
        other : FrequencySeriesMatrix
            The second matrix to correlate against.

        Returns
        -------
        numpy.ndarray
            Coherence values in the range ``[0, 1]``.
        """
        raise NotImplementedError


def noise_budget(target, contributions, *, normalize=True):
    """Build a BrUCo-style noise budget for a target channel.

    Combines a set of measured noise *contributions* and compares their
    quadrature sum against a *target* spectrum.

    Parameters
    ----------
    target : FrequencySeriesMatrix
        The reference (measured) noise spectrum.
    contributions : dict of str to FrequencySeriesMatrix
        Mapping from contribution name to its projected spectrum.
    normalize : bool, optional
        If ``True`` (default), normalise each contribution to the target.

    Returns
    -------
    dict
        A mapping with keys ``"sum"`` and ``"residual"``.

    Notes
    -----
    The total budget is the quadrature sum

    .. math::

        S_{\\mathrm{total}}(f) = \\sqrt{\\sum_i S_i(f)^2}.
    """
    return {"sum": target, "residual": contributions}
