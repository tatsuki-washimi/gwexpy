"""
Standard Fourier-based spectral transform methods for TimeSeries.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

class TimeSeriesSpectralFourierMixin:
    """
    Mixin class providing standard Fourier-based spectral transform methods.
    """

    def _prepare_data_for_transform(self, data=None, window=None, detrend=False):
        """Helper to copy data and apply detrending and windowing."""
        try:
            import scipy.signal  # noqa: F401 - availability check
        except ImportError:
            raise ImportError("scipy is required for this transform")

        if data is None:
            data = self.value.copy()
        else:
            data = data.copy()

        if detrend:
            data = scipy.signal.detrend(data, type='linear')

        if window is not None:
            if isinstance(window, (str, tuple)):
                win = scipy.signal.get_window(window, len(data))
            else:
                win = np.asarray(window)
                if len(win) != len(data):
                    raise ValueError("Window length must match data length")
            data *= win
        return data

    def fft(
        self,
        nfft: Optional[int] = None,
        *,
        mode: str = "gwpy",
        pad_mode: str = "zero",
        pad_left: int = 0,
        pad_right: int = 0,
        nfft_mode: Optional[str] = None,
        other_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
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
                **kwargs
            )

        raise ValueError(f"Unknown mode: {mode}")

    def _fft_gwpy(self, nfft: Optional[int] = None, **kwargs: Any) -> Any:
        """Internal method for standard GWpy FFT."""
        from gwexpy.frequencyseries import FrequencySeries as GWEXFrequencySeries
        base_fs = super().fft(nfft=nfft, **kwargs)
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
        nfft: Optional[int] = None,
        pad_mode: str = "zero",
        pad_left: int = 0,
        pad_right: int = 0,
        nfft_mode: Optional[str] = None,
        other_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Internal method for transient-restoration FFT."""
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
                    x = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=0)
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
                try:
                    import scipy.fft
                    def next_fast_len(n):
                        return scipy.fft.next_fast_len(n)
                except ImportError:
                    try:
                        from scipy.fftpack import next_fast_len
                    except ImportError:
                        def next_fast_len(n):
                            return n
                target_nfft = next_fast_len(target_len)
            else:
                target_nfft = target_len

        dft = np.fft.rfft(x, n=target_nfft) / target_nfft
        if dft.shape[0] > 1:
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
        fs.frequencies = np.fft.rfftfreq(target_nfft, d=self.dt.value) * u.Hz
        fs._gwex_fft_mode = "transient"
        fs._gwex_pad_left = pad_left
        fs._gwex_pad_right = pad_right
        fs._gwex_pad_mode = pad_mode
        fs._gwex_target_nfft = target_nfft
        fs._gwex_original_n = len(self)
        fs._gwex_other_length = other_length

        return fs

    def rfft(self, *args: Any, **kwargs: Any) -> Any:
        self._check_regular("rfft")
        return super().rfft(*args, **kwargs)

    def psd(self, *args: Any, **kwargs: Any) -> Any:
        self._check_regular("psd")
        from gwexpy.frequencyseries import FrequencySeries
        res = super().psd(*args, **kwargs)
        return res.view(FrequencySeries)

    def asd(self, *args: Any, **kwargs: Any) -> Any:
        self._check_regular("asd")
        from gwexpy.frequencyseries import FrequencySeries
        res = super().asd(*args, **kwargs)
        return res.view(FrequencySeries)

    def csd(self, other: Any, *args: Any, **kwargs: Any) -> Any:
        self._check_regular("csd")
        from gwexpy.frequencyseries import FrequencySeries
        res = super().csd(other, *args, **kwargs)
        # GWpy's csd sometimes returns incorrect units when inputs have different units
        target_unit = self.unit * getattr(other, "unit", u.dimensionless_unscaled) / u.Hz
        res_fs = res.view(FrequencySeries)
        if res_fs.unit != target_unit:
             res_fs.override_unit(target_unit)
        return res_fs

    def coherence(self, *args: Any, **kwargs: Any) -> Any:
        self._check_regular("coherence")
        from gwexpy.frequencyseries import FrequencySeries
        res = super().coherence(*args, **kwargs)
        return res.view(FrequencySeries)

    def spectrogram2(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute an alternative spectrogram (spectrogram2).
        Returns Spectrogram.
        """
        from gwexpy.spectrogram import Spectrogram
        res = self.spectrogram(*args, **kwargs)
        return res.view(Spectrogram)

    def dct(
        self, type: int = 2, norm: str = "ortho", *, window: Any = None, detrend: bool = False
    ) -> Any:
        self._check_regular("dct")
        try:
            import scipy.fft
        except ImportError:
            raise ImportError("scipy is required for dct.")

        data = self._prepare_data_for_transform(window=window, detrend=detrend)
        out_dct = scipy.fft.dct(data, type=type, norm=norm)
        N = len(data)
        if self.dt is None:
             raise ValueError("TimeSeries must have a valid dt for DCT frequency calculation")
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
            epoch=self.epoch
        )
        fs.transform = "dct"
        fs.dct_type = type
        fs.dct_norm = norm
        fs.original_n = N
        fs.dt = self.dt
        return fs

    def cepstrum(
        self,
        kind: str = "real",
        *,
        window: Any = None,
        detrend: bool = False,
        eps: Optional[float] = None,
        fft_mode: str = "gwpy",
    ) -> Any:
        try:
             import scipy.fft
        except ImportError:
            raise ImportError("scipy is required for cepstrum.")

        data = self._prepare_data_for_transform(window=window, detrend=detrend)
        if kind == "complex":
             spec = scipy.fft.rfft(data)
             if eps is not None:
                  spec += eps
             with np.errstate(divide='ignore', invalid='ignore'):
                  log_spec = np.log(spec)
             ceps = scipy.fft.irfft(log_spec, n=len(data))
        elif kind == "real":
             spec = scipy.fft.rfft(data)
             mag = np.abs(spec)
             if eps is not None:
                  mag += eps
             with np.errstate(divide='ignore'):
                  log_mag = np.log(mag)
             ceps = scipy.fft.irfft(log_mag, n=len(data))
        elif kind == "power":
             spec = scipy.fft.rfft(data)
             pwr = np.abs(spec)**2
             if eps is not None:
                  pwr += eps
             with np.errstate(divide='ignore'):
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
            epoch=self.epoch
        )
        fs.axis_type = "quefrency"
        fs.transform = "cepstrum"
        fs.cepstrum_kind = kind
        fs.original_n = len(data)
        fs.dt = self.dt
        return fs
