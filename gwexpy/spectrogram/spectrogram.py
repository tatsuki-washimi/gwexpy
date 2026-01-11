from __future__ import annotations

from typing import Any, Optional

import numpy as np

from gwpy.spectrogram import Spectrogram as BaseSpectrogram
from gwexpy.types.mixin import PhaseMethodsMixin, InteropMixin


class Spectrogram(PhaseMethodsMixin, InteropMixin, BaseSpectrogram):
    """
    Extends gwpy.spectrogram.Spectrogram with additional interop methods.
    """
    def plot(self, **kwargs: Any) -> Any:
        """Plot this Spectrogram. Delegates to gwexpy.plot.Plot."""
        from gwexpy.plot import Plot
        return Plot(self, **kwargs)

    def bootstrap(
        self,
        n_boot=1000,
        method="median",
        average=None,
        ci=0.68,
        window="hann",
        nperseg=None,
        noverlap=None,
        block_size=None,
        rebin_width=None,
        return_map=False,
        ignore_nan=True,
    ):
        """
        Estimate robust ASD from this spectrogram using bootstrap resampling.

        This is a convenience wrapper around `gwexpy.spectral.bootstrap_spectrogram`.
        """
        from gwexpy.spectral import bootstrap_spectrogram

        if average is not None:
            method = average

        return bootstrap_spectrogram(
            self,
            n_boot=n_boot,
            method=method,
            average=None,
            ci=ci,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            block_size=block_size,
            rebin_width=rebin_width,
            return_map=return_map,
            ignore_nan=ignore_nan,
        )

    def bootstrap_asd(
        self,
        n_boot=1000,
        average="median",
        ci=0.68,
        window="hann",
        nperseg=None,
        noverlap=None,
        block_size=None,
        rebin_width=None,
        return_map=False,
        ignore_nan=True,
    ):
        """
        Convenience wrapper for bootstrap ASD estimation.
        """
        return self.bootstrap(
            n_boot=n_boot,
            method=average,
            ci=ci,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            block_size=block_size,
            rebin_width=rebin_width,
            return_map=return_map,
            ignore_nan=ignore_nan,
        )

    def to_th2d(self, error=None):
        """
        Convert to ROOT TH2D.
        """
        from gwexpy.interop import to_th2d
        return to_th2d(self, error=error)

    def to_quantities(self, units=None):
        """
        Convert to quantities.Quantity (Elephant/Neo compatible).
        """
        from gwexpy.interop import to_quantity
        return to_quantity(self, units=units)

    @classmethod
    def from_quantities(cls, q, times, frequencies):
        """
        Create Spectrogram from quantities.Quantity.

        Parameters
        ----------
        q : quantities.Quantity
            Input data (Time x Frequency matrix).
        times : array-like
            Time axis.
        frequencies : array-like
            Frequency axis.
        """
        from gwexpy.interop import from_quantity
        return from_quantity(cls, q, times=times, frequencies=frequencies)

    @classmethod
    def from_root(cls, obj, return_error=False):
        """
        Create Spectrogram from ROOT TH2D.
        """
        from gwexpy.interop import from_root
        return from_root(cls, obj, return_error=return_error)

    def to_mne(self, info: Optional[Any] = None) -> Any:
        """
        Convert to MNE-Python object.

        Parameters
        ----------
        info : mne.Info, optional
            MNE Info object.

        Returns
        -------
        mne.time_frequency.EpochsTFRArray
        """
        from gwexpy.interop import to_mne
        return to_mne(self, info=info)

    @classmethod
    def from_mne(cls, tfr: Any, **kwargs: Any) -> Any:
        """
        Create Spectrogram from MNE-Python TFR object.

        Parameters
        ----------
        tfr : mne.time_frequency.EpochsTFR or AverageTFR
            Input TFR data.
        **kwargs
            Additional arguments passed to constructor.

        Returns
        -------
        Spectrogram or SpectrogramDict
        """
        from gwexpy.interop import from_mne
        return from_mne(cls, tfr, **kwargs)

    def to_obspy(self, **kwargs: Any) -> Any:
        """
        Convert to Obspy Stream.

        Returns
        -------
        obspy.Stream
        """
        from gwexpy.interop import to_obspy
        return to_obspy(self, **kwargs)

    @classmethod
    def from_obspy(cls, stream: Any, **kwargs: Any) -> Any:
        """
        Create Spectrogram from Obspy Stream.

        Parameters
        ----------
        stream : obspy.Stream
            Input stream.
        **kwargs
            Additional arguments.

        Returns
        -------
        Spectrogram
        """
        from gwexpy.interop import from_obspy
        return from_obspy(cls, stream, **kwargs)

    def imshow(self, **kwargs):
        """Plot using imshow. Inherited from gwpy."""
        return super().imshow(**kwargs)

    def pcolormesh(self, **kwargs):
        """Plot using pcolormesh. Inherited from gwpy."""
        return super().pcolormesh(**kwargs)

    def radian(self, unwrap: bool = False) -> "Spectrogram":
        """
        Calculate the phase of this Spectrogram in radians.
        """
        val = np.angle(self.value)
        if unwrap:
            # Unwrap along time axis
            val = np.unwrap(val, axis=0)

        name = self.name + "_phase" if self.name else "phase"

        return self.__class__(
            val,
            unit="rad",
            name=name,
        )

    def degree(self, unwrap: bool = False) -> "Spectrogram":
        """
        Calculate the phase of this Spectrogram in degrees.
        """
        p = self.radian(unwrap=unwrap)
        val = np.rad2deg(p.value)

        name = self.name + "_phase_deg" if self.name else "phase_deg"

        return self.__class__(
            val,
            unit="deg",
            name=name,
        )


# Import Matrix, List, and Dict to maintain backward compatibility if this file is imported
