from __future__ import annotations

import numpy as np

from gwexpy.types.seriesmatrix import SeriesMatrix

from .collections import FrequencySeriesDict, FrequencySeriesList
from .frequencyseries import FrequencySeries, SeriesType
from .matrix_analysis import FrequencySeriesMatrixAnalysisMixin
from .matrix_core import FrequencySeriesMatrixCoreMixin


class FrequencySeriesMatrix(  # type: ignore[misc]
    FrequencySeriesMatrixCoreMixin, FrequencySeriesMatrixAnalysisMixin, SeriesMatrix
):
    """A 2D matrix of FrequencySeries objects sharing a common frequency axis.

    `FrequencySeriesMatrix` represents a 2-dimensional array (rows x columns)
    where each element is a `FrequencySeries`. All elements in the matrix
    must share the same frequency synchronization (same `f0`, `df`, and
    number of frequency bins).

    This class is typically used to represent multi-channel spectral data,
    such as Cross-Spectral Density (CSD) matrices, coherence matrices,
    or multi-channel Power Spectral Densities (PSDs).

    Parameters
    ----------
    data : array-like, optional
        The data values for the matrix. Should be of shape
        `(rows, columns, frequencies)`.

    frequencies : array-like, optional
        The frequency values corresponding to each bin. If provided,
        `df` and `f0` are ignored.

    df : `float`, `~astropy.units.Quantity`, optional
        The frequency resolution.

    f0 : `float`, `~astropy.units.Quantity`, optional
        The start frequency.

    **kwargs
        Additional keyword arguments:
        - `channel_names`: list of strings for channel labels.
        - `unit`: physical unit of the data.
        - `name`: descriptive title for the matrix.

    Notes
    -----
    `FrequencySeriesMatrix` supports element-wise spectral operations
    (e.g., `zpk`, `filter`, `smooth`) and statistical aggregations.

    Key methods:

    .. autosummary::

       ~FrequencySeriesMatrix.plot
       ~FrequencySeriesMatrix.smooth
       ~FrequencySeriesMatrix.to_dict

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeriesMatrix
    >>> import numpy as np
    >>> data = np.ones((2, 2, 100))
    >>> fsm = FrequencySeriesMatrix(data, df=1, unit='V/Hz')
    >>> fsm
    <SeriesMatrix shape=(2, 2, 100) rows=('row0', 'row1') cols=('col0', 'col1')>

    """

    series_class = FrequencySeries
    dict_class = FrequencySeriesDict
    list_class = FrequencySeriesList
    series_type = SeriesType.FREQ
    default_xunit = "Hz"
    default_yunit = None
    _default_plot_method = "plot"

    def __new__(cls, data=None, frequencies=None, df=None, f0=None, **kwargs):
        """Create a new FrequencySeriesMatrix.

        This constructor extends the standard `SeriesMatrix` by adding support 
        for frequency-domain metadata (`f0`, `df`) and automatic axis alignment.
        """
        channel_names = kwargs.pop("channel_names", None)

        # Map frequency-specific arguments to SeriesMatrix generic arguments
        if frequencies is not None:
            kwargs["xindex"] = frequencies
        if df is not None:
            kwargs["dx"] = df
        if f0 is not None:
            kwargs["x0"] = f0
        elif frequencies is None and df is not None and "x0" not in kwargs:
            # Default f0 to 0 if not specified but df is provided (requiring explicit axis)
            kwargs["x0"] = 0

        # Set default xunit to Hz if not specified
        if "xunit" not in kwargs:
            kwargs["xunit"] = cls.default_xunit

        # Map channel_names to names kwarg for SeriesMatrix
        if channel_names is not None:
            if "names" not in kwargs:
                cn = np.asarray(channel_names)

                # Intelligent reshaping based on data shape
                try:
                    if hasattr(data, "shape"):
                        dshape = data.shape
                    else:
                        dshape = np.shape(data)

                    if len(dshape) >= 2:
                        N, M = dshape[:2]
                        if cn.size == N * M:
                            kwargs["names"] = cn.reshape(N, M)
                        elif cn.size == N:
                            kwargs["names"] = cn.reshape(N, 1)
                        else:
                            # Default to 1D, which broadcasts to (..., M) if size matches M
                            kwargs["names"] = cn
                    else:
                        kwargs["names"] = cn
                except (ValueError, TypeError, AttributeError):
                    if cn.ndim == 1:
                        kwargs["names"] = cn.reshape(-1, 1)
                    else:
                        kwargs["names"] = cn
        obj = super().__new__(cls, data, **kwargs)
        return obj
