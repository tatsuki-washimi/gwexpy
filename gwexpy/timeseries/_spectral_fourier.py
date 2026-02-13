"""
Standard Fourier-based spectral transform methods for TimeSeries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Union, cast

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
from astropy import units as u

from ._typing import TimeSeriesAttrs

if TYPE_CHECKING:
    from gwexpy.frequencyseries import FrequencySeries
    from gwexpy.spectrogram import Spectrogram

NumberLike: TypeAlias = Union[int, float, np.number]
WindowLike: TypeAlias = Union[str, tuple[Any, ...], npt.ArrayLike]


def _get_next_fast_len() -> Callable[[int], int]:
    """Return scipy's next_fast_len if available, else identity."""
    try:
        import scipy.fft

        return scipy.fft.next_fast_len
    except ImportError:
        try:
            from scipy.fftpack import next_fast_len

            return next_fast_len  # type: ignore[return-value]
        except ImportError:
            return lambda n: n


class TimeSeriesSpectralFourierMixin(TimeSeriesAttrs):
    """
    Mixin class providing standard Fourier-based spectral transform methods.
    """

    def _super_ts(self) -> TimeSeriesAttrs:
        return cast(TimeSeriesAttrs, super())

    def _prepare_data_for_transform(
        self,
        data: npt.ArrayLike | None = None,
        window: WindowLike | None = None,
        detrend: bool = False,
    ) -> npt.NDArray[Any]:
        """Helper to copy data and apply detrending and windowing."""
        try:
            import scipy.signal  # noqa: F401 - availability check
        except ImportError:
            raise ImportError("scipy is required for this transform")

        if data is None:
            data_array = self.value.copy()
        else:
            data_array = np.asarray(data).copy()

        if detrend:
            data_array = scipy.signal.detrend(data_array, type="linear")

        if window is not None:
            if isinstance(window, (str, tuple)):
                win = scipy.signal.get_window(window, len(data_array))
            else:
                win = np.asarray(window)
                if len(win) != len(data_array):
                    raise ValueError("Window length must match data length")
            data_array *= win
        return data_array

    def fft(
        self,
        nfft: NumberLike | None = None,
        *,
        mode: Literal["gwpy", "transient"] = "gwpy",
        pad_mode: Literal["zero", "reflect"] = "zero",
        pad_left: NumberLike = 0,
        pad_right: NumberLike = 0,
        nfft_mode: str | None = None,
        other_length: NumberLike | None = None,
        **kwargs: Any,
    ) -> FrequencySeries:
        """
        Compute the Discrete Fourier Transform (DFT).

        Parameters
        ----------
        nfft : `int`, optional
            Length of the FFT.
        mode : `str`, optional
            "gwpy": standard GWpy FFT (normalized).
            "transient": transient-restoration mode for round-trip IFFT.
        pad_mode : `str`, optional
            Padding mode ("zero", "reflect"). Only for "transient" mode.
        pad_left, pad_right : `int`, optional
            Padding length (samples or seconds).
        nfft_mode : `str`, optional
            "next_fast_len" for optimal performance.
        other_length : `int`, optional
            For convolution-like transforms.

        Returns
        -------
        `FrequencySeries`
        """
        self._check_regular("fft")

        if mode == "gwpy":
            return self._fft_gwpy(nfft=nfft, **kwargs)

        if mode == "transient":
            return self._fft_transient(
                nfft=nfft,
                pad_mode=pad_mode,
                pad_left=pad_left,
                pad_right=pad_right,
                nfft_mode=nfft_mode,
                other_length=other_length,
                **kwargs,
            )

        raise ValueError(f"Unknown mode: {mode}")

    def _fft_gwpy(
        self, nfft: NumberLike | None = None, **kwargs: Any
    ) -> FrequencySeries:
        """Internal method for standard GWpy FFT."""
        from gwexpy.frequencyseries import FrequencySeries as GWEXFrequencySeries

        base_fs = self._super_ts().fft(nfft=nfft, **kwargs)
        if isinstance(base_fs, GWEXFrequencySeries):
            return base_fs
        return GWEXFrequencySeries(
            base_fs.value,
            frequencies=base_fs.frequencies,
            unit=base_fs.unit,
            name=base_fs.name,
            channel=base_fs.channel,
            epoch=base_fs.epoch,
        )

    def _fft_transient(
        self,
        nfft: NumberLike | None = None,
        pad_mode: Literal["zero", "reflect"] = "zero",
        pad_left: NumberLike = 0,
        pad_right: NumberLike = 0,
        nfft_mode: str | None = None,
        other_length: NumberLike | None = None,
        **kwargs: Any,
    ) -> FrequencySeries:
        """Internal method for transient-restoration FFT.

        Returns an **amplitude spectrum** (not a density spectrum [V/√Hz]).
        The formula ``rfft(x) / N`` with one-sided values doubled allows
        direct reading of sinusoidal amplitudes.

        Notes
        -----
        This function returns an amplitude spectrum, not a power spectral
        density. The normalization ``rfft(x) / N`` with factor-of-2
        correction for one-sided spectrum (excluding DC and Nyquist)
        yields the peak amplitude of sinusoidal components.

        This convention has been validated through unit tests and independent technical review.
        The suggestion to multiply by ``dt`` (Gemini Web) applies to density
        spectra, not amplitude spectra used for transient analysis.

        References
        ----------
        .. [1] Oppenheim & Schafer, Discrete-Time Signal Processing
               (3rd ed., 2010), §8.6.2
        .. [2] SciPy rfft documentation
        """
        x = self.value

        if pad_left > 0 or pad_right > 0:
            if hasattr(pad_left, "to"):
                pad_left = pad_left.to("s").value * self.sample_rate.value
            elif isinstance(pad_left, float):
                pad_left = pad_left * self.sample_rate.value

            if hasattr(pad_right, "to"):
                pad_right = pad_right.to("s").value * self.sample_rate.value
            elif isinstance(pad_right, float):
                pad_right = pad_right * self.sample_rate.value

            pad_left = int(round(float(pad_left)))
            pad_right = int(round(float(pad_right)))

            if pad_left > 0 or pad_right > 0:
                if pad_mode == "zero":
                    x = np.pad(
                        x, (pad_left, pad_right), mode="constant", constant_values=0
                    )
                elif pad_mode == "reflect":
                    x = np.pad(x, (pad_left, pad_right), mode="reflect")
                else:
                    raise ValueError(f"Unknown pad_mode: {pad_mode}")

        len_x = x.shape[0]

        if nfft is not None:
            if nfft < len_x:
                raise ValueError(f"nfft={nfft} must be >= padded length {len_x}")
            target_nfft = int(nfft)
        else:
            if other_length is not None:
                target_len = len_x + int(other_length) - 1
            else:
                target_len = len_x

            if nfft_mode == "next_fast_len":
                next_fast_len_func = _get_next_fast_len()
                target_nfft = int(next_fast_len_func(target_len))
            else:
                target_nfft = target_len

        dft = np.fft.rfft(x, n=target_nfft) / target_nfft
        n_freq = dft.shape[0]
        if n_freq > 1:
            # One-sided spectrum amplitude correction:
            # - DC (index 0): no correction (unique)
            # - Nyquist (last index, even-length only): no correction (unique)
            # - Other frequencies: multiply by 2 (positive + negative pair)
            if target_nfft % 2 == 0:
                # Even length: DC=0 and Nyquist=n_freq-1 are unique
                dft[1:-1] *= 2.0
            else:
                # Odd length: only DC=0 is unique (no Nyquist)
                dft[1:] *= 2.0

        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries(
            dft,
            epoch=self.epoch,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
            **kwargs,
        )
        if self.dt is None:
            raise ValueError("FFT requires defined dt.")
        fs.frequencies = np.fft.rfftfreq(target_nfft, d=self.dt.value) * u.Hz
        fs._gwex_fft_mode = "transient"
        fs._gwex_pad_left = pad_left
        fs._gwex_pad_right = pad_right
        fs._gwex_pad_mode = pad_mode
        fs._gwex_target_nfft = target_nfft
        fs._gwex_original_n = len(self)
        fs._gwex_other_length = other_length

        return fs

    def rfft(self, *args: Any, **kwargs: Any) -> FrequencySeries:
        """Real-valued Fast Fourier Transform.

        Compute the one-dimensional discrete Fourier Transform for real input.
        This method delegates to GWpy's TimeSeries.rfft() and returns a
        FrequencySeries with Hermitian symmetry.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to `gwpy.timeseries.TimeSeries.rfft`
        **kwargs : dict
            Keyword arguments passed to `gwpy.timeseries.TimeSeries.rfft`

        Returns
        -------
        FrequencySeries
            The FFT of the input, with Hermitian symmetry (only positive
            frequencies). The output length is ``len(self) // 2 + 1``.

        Raises
        ------
        ValueError
            If the TimeSeries is not regularly sampled.

        See Also
        --------
        numpy.fft.rfft : NumPy's real FFT function
        TimeSeries.fft : Standard FFT with complex output (both pos/neg freqs)
        FrequencySeries.ifft : Inverse FFT
        gwpy.timeseries.TimeSeries.rfft : GWpy's rfft implementation

        Notes
        -----
        For real-valued input, the FFT is Hermitian-symmetric, so only the
        positive frequencies are stored to save memory. The output has length
        ``n//2 + 1`` for input length ``n``.

        Examples
        --------
        >>> from gwexpy.timeseries import TimeSeries
        >>> ts = TimeSeries([1.0, 2.0, 3.0, 4.0], sample_rate=1.0)
        >>> fft = ts.rfft()
        >>> print(fft.size)
        3
        """
        self._check_regular("rfft")
        return self._super_ts().rfft(*args, **kwargs)

    def psd(self, *args: Any, **kwargs: Any) -> FrequencySeries:
        self._check_regular("psd")
        from gwexpy.frequencyseries import FrequencySeries

        res = self._super_ts().psd(*args, **kwargs)
        return res.view(FrequencySeries)

    def asd(self, *args: Any, **kwargs: Any) -> FrequencySeries:
        self._check_regular("asd")
        from gwexpy.frequencyseries import FrequencySeries

        res = self._super_ts().asd(*args, **kwargs)
        return res.view(FrequencySeries)

    def csd(self, other: Any, *args: Any, **kwargs: Any) -> FrequencySeries:
        self._check_regular("csd")
        from gwexpy.frequencyseries import FrequencySeries

        res = self._super_ts().csd(other, *args, **kwargs)
        # GWpy's csd sometimes returns incorrect units when inputs have different units
        target_unit = (
            self.unit * getattr(other, "unit", u.dimensionless_unscaled) / u.Hz
        )
        res_fs = res.view(FrequencySeries)
        if res_fs.unit != target_unit:
            res_fs.override_unit(target_unit)
        return res_fs

    def coherence(self, *args: Any, **kwargs: Any) -> FrequencySeries:
        self._check_regular("coherence")
        from gwexpy.frequencyseries import FrequencySeries

        res = self._super_ts().coherence(*args, **kwargs)
        return res.view(FrequencySeries)

    def spectrogram(self, *args: Any, **kwargs: Any) -> Spectrogram:
        """
        Compute the average power spectrogram.

        This method overrides the base gwpy implementation to return
        gwexpy.spectrogram.Spectrogram instead of gwpy.spectrogram.Spectrogram.

        Returns
        -------
        Spectrogram
            gwexpy.spectrogram.Spectrogram instance
        """
        from gwexpy.spectrogram import Spectrogram

        res = self._super_ts().spectrogram(*args, **kwargs)
        return res.view(Spectrogram)

    def spectrogram2(self, *args: Any, **kwargs: Any) -> Spectrogram:
        """
        Compute an alternative spectrogram (spectrogram2).
        Returns Spectrogram.
        """
        return self.spectrogram(*args, **kwargs)

    def dct(
        self,
        type: int = 2,
        norm: str = "ortho",
        *,
        window: WindowLike | None = None,
        detrend: bool = False,
    ) -> FrequencySeries:
        """
        Compute the Discrete Cosine Transform (DCT).

        The DCT expresses the signal as a sum of cosine functions at
        different frequencies. It is widely used in signal compression
        and spectral analysis due to its energy compaction properties.

        Parameters
        ----------
        type : int, optional
            DCT type (1, 2, 3, or 4). Default is 2 (most common).
        norm : str, optional
            Normalization mode: 'ortho' for orthonormal, None for standard.
            Default is 'ortho'.
        window : str, tuple, or array-like, optional
            Window function to apply before the transform.
        detrend : bool, optional
            If True, remove linear trend before the transform.

        Returns
        -------
        FrequencySeries
            The DCT coefficients as a FrequencySeries. The frequencies
            represent DCT bin indices (k / 2*N*dt).

        Notes
        -----
        The DCT is useful for analyzing signals where edge effects are
        a concern, as it implicitly assumes even symmetry at boundaries.

        Examples
        --------
        >>> ts = TimeSeries(data, sample_rate=1024)
        >>> dct_coeffs = ts.dct()
        """
        self._check_regular("dct")

        try:
            import scipy.fft
        except ImportError:
            raise ImportError("scipy is required for dct.")

        data = self._prepare_data_for_transform(window=window, detrend=detrend)
        out_dct = scipy.fft.dct(data, type=type, norm=norm)
        N = len(data)
        if self.dt is None:
            raise ValueError(
                "TimeSeries must have a valid dt for DCT frequency calculation"
            )
        dt = self.dt.to("s").value
        k = np.arange(N)
        freqs_val = k / (2 * N * dt)
        frequencies = u.Quantity(freqs_val, "Hz")

        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries(
            out_dct,
            frequencies=frequencies,
            unit=self.unit,
            name=self.name + "_dct" if self.name else "dct",
            channel=self.channel,
            epoch=self.epoch,
        )
        fs.transform = "dct"
        fs.dct_type = type
        fs.dct_norm = norm
        fs.original_n = N
        fs.dt = self.dt
        return fs

    def cepstrum(
        self,
        kind: Literal["real", "complex", "power"] = "real",
        *,
        window: WindowLike | None = None,
        detrend: bool = False,
        eps: float | None = None,
        fft_mode: str = "gwpy",
    ) -> FrequencySeries:
        """
        Compute the cepstrum of the time series.

        The cepstrum is the inverse Fourier transform of the log spectrum.
        It is useful for detecting periodicity in the spectrum, such as
        for pitch detection, echo analysis, and deconvolution.

        Parameters
        ----------
        kind : {'real', 'complex', 'power'}, optional
            Type of cepstrum to compute:
            - 'real': IFFT of log magnitude spectrum (default)
            - 'complex': IFFT of log complex spectrum
            - 'power': IFFT of log power spectrum
        window : str, tuple, or array-like, optional
            Window function to apply before the transform.
        detrend : bool, optional
            If True, remove linear trend before the transform.
        eps : float, optional
            Small value to add to spectrum to avoid log(0).
        fft_mode : str, optional
            FFT mode (reserved for future use).

        Returns
        -------
        FrequencySeries
            The cepstrum with quefrency axis (in seconds).

        Notes
        -----
        The x-axis of the output is in "quefrency" (time units), which
        represents periodicity in the spectrum. Peaks in the cepstrum
        indicate harmonic spacing or echo delays.

        Examples
        --------
        >>> ts = TimeSeries(data, sample_rate=1024)
        >>> ceps = ts.cepstrum(kind='real')
        >>> # Find fundamental period from peak in cepstrum
        """
        try:
            import scipy.fft
        except ImportError:
            raise ImportError("scipy is required for cepstrum.")

        data = self._prepare_data_for_transform(window=window, detrend=detrend)
        if kind == "complex":
            spec = scipy.fft.rfft(data)
            if eps is not None:
                spec += eps
            with np.errstate(divide="ignore", invalid="ignore"):
                log_spec = np.log(spec)
            ceps = scipy.fft.irfft(log_spec, n=len(data))
        elif kind == "real":
            spec = scipy.fft.rfft(data)
            mag = np.abs(spec)
            if eps is not None:
                mag += eps
            with np.errstate(divide="ignore"):
                log_mag = np.log(mag)
            ceps = scipy.fft.irfft(log_mag, n=len(data))
        elif kind == "power":
            spec = scipy.fft.rfft(data)
            pwr = np.abs(spec) ** 2
            if eps is not None:
                pwr += eps
            with np.errstate(divide="ignore"):
                log_pwr = np.log(pwr)
            ceps = scipy.fft.irfft(log_pwr, n=len(data))
        else:
            raise ValueError(f"Unknown cepstrum kind: {kind}")

        if self.dt is None:
            dt = 1.0
        else:
            dt = self.dt.to("s").value

        n = len(ceps)
        quefrencies = np.arange(n) * dt
        quefrencies = u.Quantity(quefrencies, "s")

        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries(
            ceps,
            frequencies=quefrencies,
            unit=u.dimensionless_unscaled,
            name=self.name + "_cepstrum" if self.name else "cepstrum",
            channel=self.channel,
            epoch=self.epoch,
        )
        fs.axis_type = "quefrency"
        fs.transform = "cepstrum"
        fs.cepstrum_kind = kind
        fs.original_n = len(data)
        fs.dt = self.dt
        return fs
