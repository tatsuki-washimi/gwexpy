"""gwexpy.statistics.student_t_indicator - Student-t indicator for non-Gaussianity."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from scipy import stats
except ImportError as _exc:
    raise ImportError(
        "scipy is required for gwexpy.statistics. Install with: pip install scipy"
    ) from _exc

from ..spectrogram import Spectrogram
from ..spectrogram.provenance import analysis_provenance

if TYPE_CHECKING:
    from ..timeseries import TimeSeries


def _require_finite_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def compute_student_t_nu(
    ts: TimeSeries,
    fftlength: float,
    stride: float | None = None,
    window: int = 40,
    overlap: float | None = None,
    frange: tuple[float, float] | None = None,
) -> Spectrogram:
    """Compute Student-t degree of freedom (nu) for non-Gaussianity detection.

    Nu -> infinity for Gaussian, small nu (e.g., < 10) for non-Gaussian.

    Parameters
    ----------
    ts : TimeSeries
        Input time series to analyze.
    fftlength : float
        FFT segment length in seconds.
    stride : float, optional
        Step between adjacent FFT windows in seconds.
    window : int, default=40
        Number of STFT segments per Student-t fit window.
    overlap : float, optional
        Overlap between adjacent FFT windows in seconds.
    frange : (float, float), optional
        Frequency range (low, high) in Hz to limit computation.

    Returns
    -------
    Spectrogram
        Spectrogram of estimated Student-t ``nu`` values. ``times`` are GPS
        (``ts.t0`` plus the underlying `scipy.signal.stft` relative time),
        not relative-to-start seconds.

    Notes
    -----
    This Student-t non-Gaussianity indicator is a GWexpy-specific
    extension with no equivalent in GWpy (which does not expose a
    Student-t fit API); "DC/Nyquist real-only fit" is not a GWpy
    compatibility behavior to match, it is this indicator's own
    correctness fix. The reasoning follows the same real-input one-sided
    FFT convention SciPy (and GWpy's periodogram-based spectral estimates)
    already rely on elsewhere: a real signal's DC and (for even segment
    length) Nyquist bins have no distinct negative-frequency counterpart
    to fold in, so they carry half the effective degrees of freedom of an
    ordinary bin.

    Concretely: re/im independence is assumed for ``0 < f < Nyquist``. At
    DC and (when the segment length is even) Nyquist, a real-valued
    input's FFT coefficient is purely real, so only the real part is used
    for those bins -- their fit uses ``window`` samples instead of
    ``2 * window``, so their ``nu`` estimate has higher variance. ``ts``
    must be real-valued: `scipy.signal.stft` silently ignores
    ``return_onesided=True`` for complex input (switching to two-sided
    output), which breaks both the DC/Nyquist real-FFT assumption above
    and the frequency-bin count/`frange` handling below. There is
    currently no option to opt back into the old (re+im at DC/Nyquist)
    behavior.

    The underlying `scipy.signal.stft` call is made with explicit
    ``window="hann"``, ``detrend=False``, ``boundary="zeros"`` and
    ``padded=True``. ``boundary="zeros"`` is what makes ``stft``'s
    relative time origin (``t[0] == 0``) coincide with the first sample of
    ``ts``, which is the assumption the GPS time axis reconstruction below
    depends on. Segment center times carry the usual STFT systematic
    offset of about half a ``stride`` relative to each segment's start.
    Specifying ``overlap`` correlates adjacent STFT columns, reducing the
    effective number of independent samples per fit window below
    ``window`` (or ``2 * window``).

    """
    if not np.isrealobj(ts.value):
        raise ValueError(
            "compute_student_t_nu requires a real-valued TimeSeries; complex "
            "input breaks the real-FFT symmetry that DC/Nyquist bin handling "
            "relies on, and scipy.signal.stft silently switches to two-sided "
            "output (ignoring return_onesided=True) for complex input"
        )

    _require_finite_positive("fftlength", fftlength)

    stride_was_provided = stride is not None
    if stride is None:
        if overlap is None:
            stride = fftlength
        else:
            if not np.isfinite(overlap) or overlap < 0:
                raise ValueError(
                    f"overlap must be finite and non-negative, got {overlap!r}"
                )
            stride = fftlength - overlap
    _require_finite_positive("stride", stride)

    if stride > fftlength:
        raise ValueError(
            f"stride ({stride}) must not exceed fftlength ({fftlength}); "
            "scipy.signal.stft would otherwise silently skip samples "
            "between segments (a gapped analysis), not raise an error"
        )

    if not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}")

    if frange is not None:
        flo, fhi = frange
        if not np.isfinite(flo) or not np.isfinite(fhi):
            raise ValueError(f"frange must be finite, got {frange!r}")
        if flo > fhi:
            raise ValueError(f"frange low ({flo}) must be <= high ({fhi})")

    # 1. Compute FFT segments
    # Actually, we can use ts.spectrogram with 'complex' return if possible,
    # but gwpy's spectrogram normally returns real PSD.
    # We need the complex FFT coefficients.

    fs = ts.sample_rate.value
    _require_finite_positive("sample_rate", fs)
    nfft = int(fftlength * fs)
    nstep = int(stride * fs)
    if nfft < 1:
        raise ValueError(f"fftlength * sample_rate must be >= 1, got {nfft}")
    if nstep < 1:
        raise ValueError(
            f"stride * sample_rate must be >= 1 (got {nstep}); a too-small "
            "stride would silently collapse to nstep=0 (100% overlap)"
        )

    # Simple STFT to get complex values
    # shape (n_freqs, n_times)
    from scipy.signal import stft

    f, t, Zxx = stft(
        ts.value,
        fs=fs,
        nperseg=nfft,
        noverlap=nfft - nstep,
        window="hann",
        detrend=False,
        boundary="zeros",
        padded=True,
        return_onesided=True,
    )
    # Zxx shape: (n_freqs, n_times)

    n_freqs, n_times = Zxx.shape

    # scipy.signal.stft silently shrinks nperseg to len(ts) when the
    # requested nfft exceeds it (with a UserWarning), so the *effective*
    # segment length -- and therefore its parity -- can differ from the
    # requested nfft. Using the requested nfft's parity here would get the
    # Nyquist classification backwards for exactly the inputs short enough
    # to trigger that shrink.
    effective_nfft = min(nfft, len(ts.value))

    # DC (index 0, exactly f=0) and, when the effective segment length is
    # even, Nyquist (the last one-sided bin, exactly f=fs/2) bins of a
    # real-input FFT are purely real. Identified structurally on the
    # *full* one-sided output -- index 0 is always DC and, for an even
    # effective segment length, scipy's one-sided output always ends
    # exactly at the Nyquist bin -- rather than via a floating-point
    # frequency comparison (e.g. np.isclose(f, fs/2)), which has no natural
    # tolerance scale and, at very fine frequency resolution (nfft large,
    # fs small), risks matching several near-DC bins as if they were all
    # DC. An odd effective segment length (no exact Nyquist bin) or an
    # frange that excludes DC/Nyquist are both still handled correctly
    # since this mask is carried through the same frange filter as
    # `f`/`Zxx` below (#465).
    is_real_only_bin_full = np.zeros(n_freqs, dtype=bool)
    is_real_only_bin_full[0] = True
    if effective_nfft % 2 == 0:
        is_real_only_bin_full[-1] = True

    # Apply frequency range restriction to limit computation
    if frange is not None:
        flo, fhi = frange
        freq_mask = (f >= flo) & (f <= fhi)
        f = f[freq_mask]
        Zxx = Zxx[freq_mask, :]
        n_freqs = f.size
        is_real_only_bin = is_real_only_bin_full[freq_mask]
    else:
        is_real_only_bin = is_real_only_bin_full

    if n_times < window:
        raise ValueError(f"Too few segments ({n_times}) for window size {window}.")

    n_out = n_times - window + 1
    nu_map = np.zeros((n_out, n_freqs))
    n_fit_failures = 0

    for i in range(n_out):
        for j in range(n_freqs):
            # Complex FFT values for this frequency bin over the window
            # Zxx is (freq, time)
            segments = Zxx[j, i : i + window]

            if is_real_only_bin[j]:
                # DC/Nyquist: real part alone (see Notes above).
                samples = np.real(segments)
            else:
                # Use real and imaginary parts as independent samples
                samples = np.concatenate([np.real(segments), np.imag(segments)])

            # Fit Student-t distribution
            # scipy.stats.t.fit(data) returns (nu, loc, scale)
            # nu = degree of freedom
            try:
                nu, _, _ = stats.t.fit(samples)
                nu_map[i, j] = nu
            except Exception:  # noqa: BLE001 - scipy fit can fail many ways
                # Record the failure but defer to a single aggregate warning so
                # a long series with many failing bins is not silently filled
                # with NaN (issue #459; aggregate-once per #450/#452 convention).
                nu_map[i, j] = np.nan
                n_fit_failures += 1

    if n_fit_failures:
        warnings.warn(
            f"compute_student_t_nu: stats.t.fit failed for "
            f"{n_fit_failures}/{n_out * n_freqs} bins; those nu values are NaN",
            RuntimeWarning,
            stacklevel=2,
        )

    # Center times, shifted onto the GPS axis (relies on boundary="zeros"
    # above so t[0] == 0 corresponds to ts's first sample; see #465).
    out_times = t[window // 2 : window // 2 + n_out] + float(ts.t0.value)

    result = Spectrogram(
        nu_map,
        times=out_times,
        frequencies=f,
        unit="",
        name="student_t_nu",
    )
    provenance_parameters: dict[str, Any] = {
        "fftlength": fftlength,
        "stride": stride,
        "window": window,
        "frange": None if frange is None else list(frange),
    }
    if stride_was_provided and overlap is not None:
        provenance_parameters["overlap_ignored"] = True
    else:
        provenance_parameters["overlap"] = overlap
    result.provenance = analysis_provenance(
        "compute_student_t_nu",
        provenance_parameters,
    )
    return result
