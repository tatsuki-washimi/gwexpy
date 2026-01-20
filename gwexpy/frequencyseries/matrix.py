from __future__ import annotations

from typing import Any

import numpy as np

from gwexpy.types.seriesmatrix import SeriesMatrix

from .collections import FrequencySeriesDict, FrequencySeriesList
from .frequencyseries import FrequencySeries, SeriesType
from .matrix_analysis import FrequencySeriesMatrixAnalysisMixin
from .matrix_core import FrequencySeriesMatrixCoreMixin


class FrequencySeriesMatrix(
    FrequencySeriesMatrixCoreMixin, FrequencySeriesMatrixAnalysisMixin, SeriesMatrix
):
    """
    2D Matrix container for multiple FrequencySeries objects sharing a common frequency axis.

    This class represents a 2-dimensional array (rows x columns) where each element
    is a `FrequencySeries`. All elements in the matrix share the same frequency array
    (same `f0`, `df`, and number of frequency bins). It is typically used to represent
    multi-channel spectral data, such as CSD matrices or multi-channel PSDs.

    Inherits from `SeriesMatrix` and returns `FrequencySeries` instances when indexed
    per-channel.
    """

    series_class = FrequencySeries
    dict_class = FrequencySeriesDict
    list_class = FrequencySeriesList
    series_type = SeriesType.FREQ
    default_xunit = "Hz"
    default_yunit = None
    _default_plot_method = "plot"

    def __new__(cls, data=None, frequencies=None, df=None, f0=None, **kwargs):
        """
        Create a new FrequencySeriesMatrix.

        Parameters
        ----------
        data : array_like, optional
            The data values for the matrix.
        frequencies : array_like, optional
            The frequency values corresponding to each bin.
        df : float, astropy.units.Quantity, optional
            The frequency resolution.
        f0 : float, astropy.units.Quantity, optional
            The start frequency.
        **kwargs
            Additional keyword arguments.
            Supported: `channel_names`, `xunit`, `unit`, `name`, `meta`.

        Returns
        -------
        FrequencySeriesMatrix
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

    def plot(self, **kwargs: Any) -> Any:
        """Plot FrequencySeriesMatrix."""
        from gwexpy.plot import Plot

        return Plot(self, **kwargs)
