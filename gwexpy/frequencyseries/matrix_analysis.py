from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from gwexpy.types.metadata import MetaDataDict, MetaDataMatrix


class _FrequencySeriesMatrixLike(Protocol):
    """Protocol defining the interface expected by FrequencySeriesMatrixAnalysisMixin."""

    value: np.ndarray
    meta: MetaDataMatrix
    shape: tuple[int, ...]
    df: u.Quantity | float
    epoch: Any
    rows: MetaDataDict
    cols: MetaDataDict

    def copy(self) -> _FrequencySeriesMatrixLike: ...


class FrequencySeriesMatrixAnalysisMixin:
    """Analysis methods for FrequencySeriesMatrix (FFT, Filtering, Smoothing)."""

    def ifft(self: _FrequencySeriesMatrixLike) -> Any:
        """
        Compute the inverse FFT of this frequency-domain matrix.

        Matches GWpy FrequencySeries.ifft normalization.

        Returns
        -------
        TimeSeriesMatrix
            The time-domain matrix resulting from the inverse FFT.
        """
        import numpy.fft as fft

        from gwexpy.timeseries import TimeSeriesMatrix

        n_freq = self.shape[-1]
        nout = (n_freq - 1) * 2

        # Undo normalization from TimeSeries.fft (GWpy logic):
        # the DC component does not have the factor of two applied.
        spectrum = self.value.copy()
        spectrum[..., 1:] /= 2.0
        time_data = fft.irfft(spectrum * nout, n=nout, axis=-1)

        # 4. Metadata
        # dt = 1 / (df * nout)
        if isinstance(self.df, u.Quantity):
            dt = (1 / (self.df * nout)).to("s")
        else:
            dt = u.Quantity(1.0 / (float(self.df) * nout), "s")

        return TimeSeriesMatrix(
            time_data,
            meta=self.meta,
            dt=dt,
            epoch=self.epoch,
            name=getattr(self, "name", ""),
            rows=self.rows,
            cols=self.cols,
            xunit="s",
        )

    def filter(self: _FrequencySeriesMatrixLike, *filt: Any, **kwargs: Any) -> Any:
        """
        Apply a filter to the FrequencySeriesMatrix.

        Matches GWpy FrequencySeries.filter behavior (magnitude-only response)
        by delegating to gwpy.frequencyseries._fdcommon.fdfilter.
        Use apply_response() for complex response application.

        Parameters
        ----------
        *filt : filter arguments
            Filter definition.
        inplace : bool, optional
            If True, modify in-place. Default False.
        **kwargs :
            Additional arguments passed to fdfilter (e.g. analog=True).

        Returns
        -------
        FrequencySeriesMatrix
            Filtered matrix.
        """
        from gwpy.frequencyseries._fdcommon import fdfilter

        return fdfilter(self, *filt, **kwargs)

    def apply_response(self: _FrequencySeriesMatrixLike, response: Any, inplace: bool = False) -> Any:
        """
        Apply a complex frequency response to the matrix.

        Extension method (not in GWpy) to support complex filtering/calibration.

        Parameters
        ----------
        response : array-like or Quantity
            Complex frequency response array aligned with self.frequencies.
        inplace : bool, optional
            If True, modify in-place.
        """
        if isinstance(response, u.Quantity):
            h = response
        else:
            h = u.Quantity(np.asarray(response), u.dimensionless_unscaled)

        if inplace:
            self *= h
            return self
        else:
            return self * h

    def smooth(
        self: _FrequencySeriesMatrixLike, width: int, method: str = "amplitude", ignore_nan: bool = True
    ) -> Any:
        """
        Smooth the frequency series matrix along the frequency axis.

        Parameters
        ----------
        width : int
            Full width of the smoothing window in samples.
        method : str, optional
            Smoothing method: 'amplitude', 'power', 'complex', 'db'.
            Default is 'amplitude'.
        ignore_nan : bool, optional
            If True, ignore NaNs during smoothing. Default is True.
        """
        from scipy.ndimage import uniform_filter1d

        val = self.value
        # FrequencySeriesMatrix elements can have different units.
        # We take the first one as representative for calculation, or handle carefully.
        unit = self.meta[0, 0].unit
        axis = -1  # Frequency axis is always last

        def _smooth_axis(x):
            if ignore_nan:
                import pandas as pd

                # Handle n-dimensional array by reshaping to 2D (batch, time), smoothing, and reshaping back.
                orig_shape = x.shape
                flat_x = x.reshape(-1, orig_shape[-1])
                # pandas rolling mean on DataFrame (axis=1)
                res = (
                    pd.DataFrame(flat_x)
                    .T.rolling(window=width, center=True, min_periods=1)
                    .mean()
                    .T.values
                )
                return res.reshape(orig_shape)
            else:
                return uniform_filter1d(x, size=width, axis=axis)

        if method == "complex":
            # Complex data: smooth real and imag parts separately
            re = _smooth_axis(val.real)
            im = _smooth_axis(val.imag)
            new_val = re + 1j * im
        elif method == "amplitude":
            new_val = _smooth_axis(np.abs(val))
        elif method == "power":
            new_val = _smooth_axis(np.abs(val) ** 2)
            unit = unit**2
        elif method == "db":
            # Convert to dB first
            with np.errstate(divide="ignore"):
                db = 20 * np.log10(np.abs(val))
            new_val = _smooth_axis(db)
            unit = u.Unit("dB")
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        new_mat = self.copy()
        new_mat.value[:] = new_val
        # Note: if unit changed (power, db), update all metadata.
        if unit != self.meta[0, 0].unit:
            for r in range(new_mat.shape[0]):
                for c in range(new_mat.shape[1]):
                    new_mat.meta[r, c].unit = unit

        return new_mat
