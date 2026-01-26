# FFT Specifications and Conventions

This document details the mathematical conventions, normalization, and sign definitions used for Fast Fourier Transforms (FFT) in `gwexpy`.

## 1. Temporal FFT (`fft_time`)

The temporal FFT is applied to the time axis (axis 0). It is designed for compatibility with `gwpy` and standard gravitational-wave data analysis practices.

### Normalization
$1/N$ normalization is applied to the forward transform. This means the amplitude of a sine wave $A \sin(\omega t)$ appears as $A$ in the resulting frequency spectrum (after accounting for the one-sided factor).
The inverse transform (`ifft_time`) undoes this normalization.

### Spectral Definition
- **Type**: One-sided (`rfft`). Only non-negative frequencies are returned.
- **Correction**: To preserve total power, all non-DC and non-Nyquist bins are multiplied by 2.
- **Frequency Axis**: $f = [0, 1/2dt]$.

### Sign Convention
- **Forward**: $X[k] = \frac{1}{N} \sum_{n=0}^{N-1} x[n] e^{-i 2\pi k n / N}$
- **Inverse**: $x[n] = \sum_{k=0}^{N/2} X[k] e^{i 2\pi k n / N}$ (with appropriate bin weighting)

---

## 2. Spatial FFT (`fft_space`)

The spatial FFT is applied to spatial axes (axes 1, 2, 3). It is designed for physical field analysis across multi-dimensional grids.

### Normalization
**Unnormalized**. No factor is applied to the forward transform, following standard `numpy.fft.fftn` behavior.
The inverse transform (`ifft_space`) applies $1/N$ normalization.

### Spectral Definition
- **Type**: Two-sided. Returns both positive and negative wavenumbers.
- **Wavenumber Axis**: By default, `fft_space` produces **angular wavenumber** $k = 2\pi / \lambda$.
- **Ordering**: The zero-frequency (DC) component is at index 0 (not shifted). Use `numpy.fft.fftshift` externally if a centered spectrum is desired for visualization.

### Sign Convention
- **Forward**: $X[\mathbf{k}] = \sum_{\mathbf{n}} x[\mathbf{n}] e^{-i 2\pi \sum k_j n_j / N_j}$
- **Inverse**: $x[\mathbf{n}] = \frac{1}{\prod N_j} \sum_{\mathbf{k}} X[\mathbf{k}] e^{i 2\pi \sum k_j n_j / N_j}$

---

## 3. Spectral Density (`spectral_density`, `compute_psd`)

The `spectral_density` method provides power spectral density (PSD) estimation using the Welch method (averaged periodograms).

### Scaling
- **Density**: Returns power per unit frequency/wavenumber resolution ($V^2 / \text{Hz}$ or $V^2 / [\text{unit}^{-1}]$).
- **Spectrum**: Returns total power in each bin ($V^2$).

### Wavenumber Convention in Signal Utils
Note that while `fft_space` uses angular wavenumber ($k = 2\pi/\lambda$), `spectral_density(axis='x')` produces **non-angular wavenumber** ($f_k = 1/\lambda$) by default, consistent with `scipy.signal.welch`.
To convert to angular wavenumber, multiply the resulting axis by $2\pi$.

---

## 4. Normalization Summary Table

| Method | Normalization (Forward) | Domain | Sign ($ikx$) |
| :--- | :--- | :--- | :--- |
| `fft_time` | $1/N$ | One-sided | Negative |
| `ifft_time` | $N$ | One-sided | Positive |
| `fft_space` | $1$ | Two-sided | Negative |
| `ifft_space` | $1/N$ | Two-sided | Positive |
| `spectral_density` | Welch / $1/N^2$ | One-sided | N/A (Magnitude sq.) |
