from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
from astropy import units as u
from gwpy.spectrogram import Spectrogram as BaseSpectrogram

from gwexpy.types.mixin import InteropMixin, PhaseMethodsMixin

if TYPE_CHECKING:
    from astropy.units import Quantity

    from gwexpy.frequencyseries import FrequencySeriesList
    from gwexpy.timeseries import TimeSeriesList


class Spectrogram(PhaseMethodsMixin, InteropMixin, BaseSpectrogram):
    """
    Extends gwpy.spectrogram.Spectrogram with additional interop methods.
    """

    def plot(self, **kwargs: Any) -> Any:
        """Plot this Spectrogram. Delegates to gwexpy.plot.Plot."""
        from gwexpy.plot import Plot

        return Plot(self, **kwargs)

    def __getitem__(self, item: Any) -> Any:
        """
        Custom getitem to handle 1D views safely.

        In some environments (e.g. newer matplotlib/astropy), iterating over a
        2D Spectrogram yields 1D Spectrogram views. GWpy's Array2D.__getitem__
        implementation expects 2D data and fails with a ValueError when
        unpacking slices for ndim=1. This override detects ndim=1 and falls
        back to the parent (Series) implementation.
        """
        if self.ndim == 1:
            from gwpy.types.series import Series

            return Series.__getitem__(self, item)
        return super().__getitem__(item)

    def __reduce_ex__(self, protocol: SupportsIndex):
        from gwexpy.io.pickle_compat import spectrogram_reduce_args

        return spectrogram_reduce_args(self)

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

    def to_mne(self, info: Any | None = None) -> Any:
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

    def rebin(
        self, dt: float | u.Quantity | None = None, df: float | u.Quantity | None = None
    ) -> Spectrogram:
        """
        Rebin the spectrogram in time and/or frequency.

        Parameters
        ----------
        dt : float or Quantity, optional
            New time bin width in seconds.
        df : float or Quantity, optional
            New frequency bin width in Hz.

        Returns
        -------
        Spectrogram
            The rebinned spectrogram.
        """
        data = self.value
        times = self.times
        freqs = self.frequencies

        # Frequency rebinning
        if df is not None:
            if hasattr(df, "to"):
                df = df.to("Hz").value
            orig_df = self.df.to("Hz").value if hasattr(self.df, "to") else self.df
            bin_size = int(round(df / orig_df))
            if bin_size > 1:
                nt, nf = data.shape
                nf_new = nf // bin_size
                data = (
                    data[:, : nf_new * bin_size]
                    .reshape(nt, nf_new, bin_size)
                    .mean(axis=2)
                )
                freqs = (
                    freqs[: nf_new * bin_size].reshape(nf_new, bin_size).mean(axis=1)
                )

        # Time rebinning
        if dt is not None:
            if hasattr(dt, "to"):
                dt = dt.to("s").value
            orig_dt = self.dt.to("s").value if hasattr(self.dt, "to") else self.dt
            bin_size = int(round(dt / orig_dt))
            if bin_size > 1:
                nt, nf = data.shape
                nt_new = nt // bin_size
                data = (
                    data[: nt_new * bin_size, :]
                    .reshape(nt_new, bin_size, nf)
                    .mean(axis=1)
                )
                times = (
                    times[: nt_new * bin_size].reshape(nt_new, bin_size).mean(axis=1)
                )

        return self.__class__(
            data,
            times=times,
            frequencies=freqs,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
            epoch=self.epoch,
        )

    def imshow(self, **kwargs):
        """Plot using Matplotlib ``imshow`` (GWpy-compatible).

        This method is provided for convenience and forwards arguments to
        :meth:`gwpy.spectrogram.Spectrogram.imshow`.

        Common keyword arguments include ``ax``, ``cmap``, ``norm`` (or
        ``log=True`` in GWpy), and color scaling controls like ``vmin``/``vmax``.
        For the full set of supported keywords, see the GWpy documentation.
        """
        return super().imshow(**kwargs)

    def pcolormesh(self, **kwargs):
        """Plot using Matplotlib ``pcolormesh`` (GWpy-compatible).

        This method is provided for convenience and forwards arguments to
        :meth:`gwpy.spectrogram.Spectrogram.pcolormesh`.

        Common keyword arguments include ``ax``, ``cmap``, ``norm`` and
        ``vmin``/``vmax``. For the full set of supported keywords, see the GWpy
        documentation.
        """
        return super().pcolormesh(**kwargs)

    def radian(self, unwrap: bool = False) -> Spectrogram:
        """
        Calculate the phase of this Spectrogram in radians.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase to remove discontinuities along the time axis.
            Default is False.

        Returns
        -------
        Spectrogram
            A new Spectrogram containing the phase in radians.
            All other metadata (times, frequencies, channel, epoch, etc.) are preserved.
        """
        # Copy to preserve all metadata (times, frequencies, channel, epoch, metadata dict, etc.)
        new = self.copy()

        # Calculate phase values
        val = np.angle(self.view(np.ndarray))
        if unwrap:
            # Unwrap along time axis (axis 0 of (Time, Freq) Spectrogram)
            val = np.unwrap(val, axis=0)

        # If original was complex, Ensure new is real-valued to hold phase
        if np.iscomplexobj(new):
            new = new.real.copy()

        # Update data and metadata
        # Use raw ndarray view for assignment to bypass Astropy's unit check
        new.view(np.ndarray)[:] = val
        new.override_unit("rad")
        if self.name:
            new.name = self.name + "_phase"
        else:
            new.name = "phase"

        return new

    def degree(self, unwrap: bool = False) -> Spectrogram:
        """
        Calculate the phase of this Spectrogram in degrees.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase to remove discontinuities along the time axis.
            Default is False.

        Returns
        -------
        Spectrogram
            A new Spectrogram containing the phase in degrees.
            All other metadata (times, frequencies, channel, epoch, etc.) are preserved.
        """
        # Re-use radian() implementation which handles unwrap and metadata preservation
        p = self.radian(unwrap=unwrap)

        # Convert values to degrees
        val = np.rad2deg(p.view(np.ndarray))

        # Create final object (p already has correct metadata and is real-valued)
        new = p
        # Use raw ndarray view for assignment to bypass Astropy's unit check
        new.view(np.ndarray)[:] = val
        new.override_unit("deg")
        if self.name:
            new.name = self.name + "_phase_deg"
        else:
            new.name = "phase_deg"

        return new

    def to_timeseries_list(self) -> tuple[TimeSeriesList, Quantity]:
        """
        Convert this Spectrogram to a list of TimeSeries, one per frequency bin.

        For a Spectrogram with shape ``(ntimes, nfreqs)``, this extracts each
        column (frequency bin) as a TimeSeries with the same time axis.

        Returns
        -------
        ts_list : TimeSeriesList
            A list of TimeSeries, one for each frequency bin.
            Each TimeSeries has length ``ntimes``.
        frequencies : Quantity
            The frequency axis of this Spectrogram (length ``nfreqs``).

        Notes
        -----
        - Each TimeSeries inherits ``unit``, ``epoch``, ``channel`` from this
          Spectrogram.
        - ``name`` is set to ``"{original_name}_f{freq}"`` or ``"f{freq}"`` if
          the Spectrogram has no name, where ``freq`` is the frequency value.

        Examples
        --------
        >>> spec = Spectrogram(data, t0=0, dt=0.1, f0=10, df=5, name="test")
        >>> ts_list, freqs = spec.to_timeseries_list()
        >>> len(ts_list)  # equals nfreqs
        5
        >>> ts_list[0].name
        'test_f10.0Hz'
        """
        from gwexpy.timeseries import TimeSeries, TimeSeriesList

        ntimes, nfreqs = self.shape
        # Extract raw ndarray to avoid unit doubling
        data_raw = self.view(np.ndarray)
        unit = self.unit
        times = self.times
        frequencies = self.frequencies
        # Convert epoch to GPS float to avoid astropy.Time interpretation issues
        epoch = (
            float(self.epoch.gps) if hasattr(self.epoch, "gps") else float(self.epoch)
        )
        channel = self.channel
        base_name = self.name

        ts_list = TimeSeriesList()
        for i in range(nfreqs):
            freq_val = frequencies[i]
            # Create descriptive name
            if base_name:
                name = f"{base_name}_f{freq_val}"
            else:
                name = f"f{freq_val}"

            # Extract column i (all times, single frequency)
            ts_data = data_raw[:, i].copy()

            # Create TimeSeries with explicit unit
            ts = TimeSeries(
                ts_data,
                times=times,
                unit=unit,
                name=name,
                channel=channel,
                epoch=epoch,
            )
            # Bypass validation by using list.append
            list.append(ts_list, ts)

        return ts_list, frequencies

    def to_frequencyseries_list(self) -> tuple[FrequencySeriesList, Quantity]:
        """
        Convert this Spectrogram to a list of FrequencySeries, one per time bin.

        For a Spectrogram with shape ``(ntimes, nfreqs)``, this extracts each
        row (time bin) as a FrequencySeries with the same frequency axis.

        Returns
        -------
        fs_list : FrequencySeriesList
            A list of FrequencySeries, one for each time bin.
            Each FrequencySeries has length ``nfreqs``.
        times : Quantity
            The time axis of this Spectrogram (length ``ntimes``).

        Notes
        -----
        - Each FrequencySeries inherits ``unit``, ``epoch``, ``channel`` from
          this Spectrogram.
        - ``name`` is set to ``"{original_name}_t{time}"`` or ``"t{time}"`` if
          the Spectrogram has no name, where ``time`` is the time value.

        Examples
        --------
        >>> spec = Spectrogram(data, t0=0, dt=0.1, f0=10, df=5, name="test")
        >>> fs_list, times = spec.to_frequencyseries_list()
        >>> len(fs_list)  # equals ntimes
        10
        >>> fs_list[0].name
        'test_t0.0s'
        """
        from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList

        ntimes, nfreqs = self.shape
        # Extract raw ndarray to avoid unit doubling
        data_raw = self.view(np.ndarray)
        unit = self.unit
        times = self.times
        frequencies = self.frequencies
        # Convert epoch to GPS float to avoid astropy.Time interpretation issues
        epoch = (
            float(self.epoch.gps) if hasattr(self.epoch, "gps") else float(self.epoch)
        )
        channel = self.channel
        base_name = self.name

        fs_list = FrequencySeriesList()
        for j in range(ntimes):
            time_val = times[j]
            # Create descriptive name
            if base_name:
                name = f"{base_name}_t{time_val}"
            else:
                name = f"t{time_val}"

            # Extract row j (single time, all frequencies)
            fs_data = data_raw[j, :].copy()

            # Create FrequencySeries with explicit unit
            fs = FrequencySeries(
                fs_data,
                frequencies=frequencies,
                unit=unit,
                name=name,
                channel=channel,
                epoch=epoch,
            )
            # Bypass validation by using list.append
            list.append(fs_list, fs)

        return fs_list, times


# Import Matrix, List, and Dict to maintain backward compatibility if this file is imported
