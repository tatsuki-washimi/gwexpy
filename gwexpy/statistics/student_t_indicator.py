"""gwexpy.statistics.student_t_indicator - Student-t indicator for non-Gaussianity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from ..spectrogram import Spectrogram

if TYPE_CHECKING:
    from ..timeseries import TimeSeries


def compute_student_t_nu(
    ts: TimeSeries,
    fftlength: float,
    stride: float | None = None,
    window: int = 40,
    overlap: float | None = None,
) -> Spectrogram:
    """
    Compute Student-t degree of freedom (nu) for non-Gaussianity detection.
    Nu -> infinity for Gaussian, small nu (e.g., < 10) for non-Gaussian.

    Parameters
    ----------
    ts : TimeSeries
    fftlength : float
    stride : float, optional
    window : int, default=40
    overlap : float, optional

    Returns
    -------
    Spectrogram (nu values)
    """
    if stride is None:
        if overlap is None:
            stride = fftlength
        else:
            stride = fftlength - overlap

    # 1. Compute FFT segments
    # Actually, we can use ts.spectrogram with 'complex' return if possible, 
    # but gwpy's spectrogram normally returns real PSD.
    # We need the complex FFT coefficients.
    
    fs = ts.sample_rate.value
    nfft = int(fftlength * fs)
    nstep = int(stride * fs)
    
    # Simple STFT to get complex values
    # shape (n_freqs, n_times)
    freqs, times, Zxx = stats.sigtools._segment_axis(ts.value, nfft, nfft - nstep, axis=-1)
    # Wait, sigtools is internal. Use scipy.signal.stft or manual rolling.
    from scipy.signal import stft
    f, t, Zxx = stft(ts.value, fs=fs, nperseg=nfft, noverlap=nfft-nstep, return_onesided=True)
    # Zxx shape: (n_freqs, n_times)
    
    n_freqs, n_times = Zxx.shape
    if n_times < window:
        raise ValueError(f"Too few segments ({n_times}) for window size {window}.")

    n_out = n_times - window + 1
    nu_map = np.zeros((n_out, n_freqs))
    
    for i in range(n_out):
        for j in range(n_freqs):
            # Complex FFT values for this frequency bin over the window
            # Zxx is (freq, time)
            segments = Zxx[j, i : i + window]
            
            # Use real and imaginary parts as independent samples
            samples = np.concatenate([np.real(segments), np.imag(segments)])
            
            # Fit Student-t distribution
            # scipy.stats.t.fit(data) returns (nu, loc, scale)
            # nu = degree of freedom
            try:
                nu, _, _ = stats.t.fit(samples)
                nu_map[i, j] = nu
            except Exception:
                nu_map[i, j] = np.nan
                
    # Center times
    out_times = t[window // 2 : window // 2 + n_out]
    
    return Spectrogram(
        nu_map,
        times=out_times,
        frequencies=f,
        unit="",
        name="student_t_nu",
    )
