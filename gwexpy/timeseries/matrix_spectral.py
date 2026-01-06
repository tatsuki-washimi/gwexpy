from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from gwexpy.types.metadata import MetaData, MetaDataMatrix
from .utils import _extract_axis_info, _validate_common_axis


class TimeSeriesMatrixSpectralMixin:
    """Spectral analysis methods for TimeSeriesMatrix."""

    def _vectorized_fft(self, **kwargs: Any) -> Any:
        """
        Vectorized implementation of FFT.
        """
        from gwexpy.frequencyseries import FrequencySeriesMatrix
        
        # We assume regular sampling for vectorized FFT
        self._check_regular("Vectorized FFT")
        
        # Handle n-dimensional array (N, M, T)
        # np.fft.rfft handles axis
        data = np.asarray(self.value)
        n = data.shape[-1]
        
        # Pass kwargs to rfft (like n)
        rfft_len = kwargs.get("n", n)
        fs_data = np.fft.rfft(data, n=rfft_len, axis=-1)
        
        # Calculate frequencies
        df = 1.0 / (self.dt.value * rfft_len)
        freqs = np.fft.rfftfreq(rfft_len, d=self.dt.value) * u.Hz
        
        # Metadata logic: simplified for now, uses same meta for all
        return FrequencySeriesMatrix(
            fs_data,
            frequencies=freqs,
            meta=self.meta,
            rows=self.rows,
            cols=self.cols,
            name=self.name,
            epoch=self.epoch,
        )

    def _vectorized_psd(self, **kwargs: Any) -> Any:
        """
        Vectorized implementation of PSD (Welch).
        """
        from scipy.signal import welch
        from gwexpy.frequencyseries import FrequencySeriesMatrix
        
        self._check_regular("Vectorized PSD")
        
        data = np.asarray(self.value)
        fs = 1.0 / self.dt.value
        
        # Adjust kwargs to match scipy.signal.welch
        nperseg = kwargs.pop("fftlength", kwargs.pop("nperseg", None))
        noverlap = kwargs.pop("overlap", kwargs.pop("noverlap", None))
        
        freqs, psd_data = welch(
            data, 
            fs=fs, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            axis=-1, 
            **kwargs
        )
        
        return FrequencySeriesMatrix(
            psd_data,
            frequencies=freqs * u.Hz,
            meta=self.meta,
            rows=self.rows,
            cols=self.cols,
            name=self.name,
            epoch=self.epoch,
        )

    def _vectorized_asd(self, **kwargs: Any) -> Any:
        """
        Vectorized implementation of ASD.
        """
        psd = self._vectorized_psd(**kwargs)
        asd_data = np.sqrt(psd.value)
        psd.value[:] = asd_data
        return psd

        asd_data = np.sqrt(psd.value)
        psd.value[:] = asd_data
        return psd

    def _vectorized_csd(self, other: Any, **kwargs: Any) -> Any:
        """
        Vectorized implementation of CSD.
        """
        from scipy.signal import csd
        from gwexpy.frequencyseries import FrequencySeriesMatrix
        
        self._check_regular("Vectorized CSD")
        
        data = np.asarray(self.value)
        other_data = np.asarray(other.value)
        fs = 1.0 / self.dt.value
        
        nperseg = kwargs.pop("fftlength", kwargs.pop("nperseg", None))
        noverlap = kwargs.pop("overlap", kwargs.pop("noverlap", None))
        
        freqs, csd_data = csd(
            data, 
            other_data, 
            fs=fs, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            axis=-1, 
            **kwargs
        )
        
        return FrequencySeriesMatrix(
            csd_data,
            frequencies=freqs * u.Hz,
            meta=self.meta,
            rows=self.rows,
            cols=self.cols,
            name=self.name,
            epoch=self.epoch,
        )

    def _vectorized_coherence(self, other: Any, **kwargs: Any) -> Any:
        """
        Vectorized implementation of Coherence.
        """
        from scipy.signal import coherence
        from gwexpy.frequencyseries import FrequencySeriesMatrix
        
        self._check_regular("Vectorized Coherence")
        
        data = np.asarray(self.value)
        other_data = np.asarray(other.value)
        fs = 1.0 / self.dt.value
        
        nperseg = kwargs.pop("fftlength", kwargs.pop("nperseg", None))
        noverlap = kwargs.pop("overlap", kwargs.pop("noverlap", None))
        
        freqs, coh_data = coherence(
            data, 
            other_data, 
            fs=fs, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            axis=-1, 
            **kwargs
        )
        
        return FrequencySeriesMatrix(
            coh_data,
            frequencies=freqs * u.Hz,
            meta=self.meta,
            rows=self.rows,
            cols=self.cols,
            name=self.name,
            epoch=self.epoch,
        )

    def lock_in(self, **kwargs: Any) -> Any:
        """
        Apply lock-in amplification element-wise.

        Returns
        -------
        TimeSeriesMatrix or tuple of TimeSeriesMatrix
            If output='amp_phase' (default) or 'iq', returns (matrix1, matrix2).
            If output='complex', returns a single complex TimeSeriesMatrix.
        """
        output = kwargs.get("output", "amp_phase")
        expect_tuple = output in ["amp_phase", "iq"]

        N, M, _ = self.shape
        if N == 0 or M == 0:
            if expect_tuple:
                return self.copy(), self.copy()
            return self.copy()

        vals1 = [[None for _ in range(M)] for _ in range(N)]
        vals2 = [[None for _ in range(M)] for _ in range(N)] if expect_tuple else None

        meta1 = np.empty((N, M), dtype=object)
        meta2 = np.empty((N, M), dtype=object) if expect_tuple else None

        ax_infos = []
        method_name = "lock_in"
        dtype1 = None
        dtype2 = None

        for i in range(N):
            for j in range(M):
                ts = self[i, j]
                res = ts.lock_in(**kwargs)

                if expect_tuple:
                    r1, r2 = res
                    vals1[i][j] = np.asarray(r1.value)
                    meta1[i, j] = MetaData(unit=str(r1.unit), name=r1.name, channel=r1.channel)
                    dtype1 = np.result_type(dtype1, r1.value.dtype) if dtype1 else r1.value.dtype

                    vals2[i][j] = np.asarray(r2.value)
                    meta2[i, j] = MetaData(unit=str(r2.unit), name=r2.name, channel=r2.channel)
                    dtype2 = np.result_type(dtype2, r2.value.dtype) if dtype2 else r2.value.dtype

                    ax_infos.append(_extract_axis_info(r1))
                else:
                    # Single return
                    vals1[i][j] = np.asarray(res.value)
                    meta1[i, j] = MetaData(unit=str(res.unit), name=res.name, channel=res.channel)
                    dtype1 = np.result_type(dtype1, res.value.dtype) if dtype1 else res.value.dtype
                    ax_infos.append(_extract_axis_info(res))

        # Validate common axis
        common_axis, axis_len = _validate_common_axis(ax_infos, method_name)

        def _build(v, d, m):
            out_shape = (N, M, axis_len)
            out = np.empty(out_shape, dtype=d)
            for r in range(N):
                for c in range(M):
                    out[r, c, :] = v[r][c]
            new_mat = self.__class__(
                out,
                xindex=common_axis,
                xunit=common_axis.unit if isinstance(common_axis, u.Quantity) else None,
            )
            new_mat.meta = MetaDataMatrix(m)
            return new_mat

        m1 = _build(vals1, dtype1, meta1)
        if expect_tuple:
            m2 = _build(vals2, dtype2, meta2)
            return m1, m2
        return m1

    def fft(self, **kwargs: Any) -> Any:
        """
        Compute FFT of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("fft", **kwargs)

    def psd(self, **kwargs: Any) -> Any:
        """
        Compute PSD of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("psd", **kwargs)

    def asd(self, **kwargs: Any) -> Any:
        """
        Compute ASD of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("asd", **kwargs)

    def spectrogram(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute spectrogram of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("spectrogram", *args, **kwargs)

    def spectrogram2(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute spectrogram2 of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("spectrogram2", *args, **kwargs)

    def q_transform(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute Q-transform of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("q_transform", *args, **kwargs)

    def _run_spectral_method(self, method_name: str, **kwargs: Any) -> Any:
        """
        Helper for fft, psd, asd.
        """
        from gwexpy.frequencyseries import FrequencySeriesMatrix

        N, M, K = self.shape

        # Run first element to determine frequency axis and output properties
        ts0 = self[0, 0]
        method = getattr(ts0, method_name)
        fs0 = method(**kwargs)

        # Prepare output array
        n_freq = len(fs0)
        out_shape = (N, M, n_freq)
        out_data = np.empty(out_shape, dtype=fs0.dtype)

        # Attributes
        out_units = np.empty((N, M), dtype=object)
        out_names = np.empty((N, M), dtype=object)
        out_channels = np.empty((N, M), dtype=object)

        # Loop over all elements
        for i in range(N):
            for j in range(M):
                if i == 0 and j == 0:
                    fs = fs0
                else:
                    ts = self[i, j]
                    fs = getattr(ts, method_name)(**kwargs)

                out_data[i, j, :] = fs.value
                out_units[i, j] = fs.unit
                out_names[i, j] = fs.name
                out_channels[i, j] = fs.channel

        return FrequencySeriesMatrix(
            out_data,
            frequencies=fs0.frequencies,
            units=out_units,
            names=out_names,
            channels=out_channels,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=getattr(self, "epoch", None),
        )
