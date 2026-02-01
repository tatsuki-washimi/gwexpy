Validated Algorithms
====================

The following algorithms have been validated through cross-verification
by 12 different AI models (2026-02-01). This page documents the verified
implementations with references.

k-space Computation
-------------------

**Function**: :meth:`gwexpy.fields.ScalarField.fft_space`

**Consensus**: 10/12 AI models confirmed correctness

The angular wavenumber computation uses the standard physics definition:

.. math::

    k = 2\pi \cdot \text{fftfreq}(n, d)

This satisfies :math:`k = 2\pi / \lambda` and is consistent with:

- Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2
- NumPy ``fftfreq`` documentation
- GWpy FrequencySeries (Duncan Macleod et al., SoftwareX 13, 2021)

The ``2π`` factor is correctly applied, and units are properly set as
``1/dx_unit`` (rad/length).


Amplitude Spectrum (Transient FFT)
----------------------------------

**Function**: ``TimeSeries._fft_transient``

**Consensus**: Validated with clear rebuttal of incorrect critiques

The transient FFT returns an **amplitude spectrum**, not a density spectrum:

.. math::

    \text{amplitude} = \text{rfft}(x) / N

with one-sided values (excluding DC and Nyquist) doubled.

This convention allows direct reading of sinusoidal peak amplitudes.
The suggestion to multiply by ``dt`` applies to density spectra
(V/√Hz), which is a different use case.

**References**:

- Oppenheim & Schafer, *Discrete-Time Signal Processing* (3rd ed., 2010), §8.6.2
- SciPy ``rfft`` documentation


VIF (Variance Inflation Factor)
-------------------------------

**Function**: :func:`gwexpy.spectral.estimation.calculate_correlation_factor`

**Consensus**: 8/12 AI models confirmed correctness

The VIF formula follows Percival & Walden (1993):

.. math::

    \text{VIF} = \sqrt{1 + 2 \sum_{k=1}^{M-1} \left(1 - \frac{k}{M}\right) |\rho(kS)|^2}

**Important**: This is NOT the regression VIF (1/(1-R²)) used for
multicollinearity diagnosis. The name collision caused confusion,
but the implementation is correct for spectral analysis.

**References**:

- Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications*
  (1993), Ch. 7.3.2, Eq.(56)
- Bendat, J.S. & Piersol, A.G., *Random Data* (4th ed., 2010)


Forecast Timestamp (ARIMA)
--------------------------

**Function**: :meth:`gwexpy.timeseries.arima.ArimaResult.forecast`

**Consensus**: Validated with rebuttal of incorrect concerns

The forecast start time is computed as:

.. math::

    t_{\text{forecast}} = t_0 + n_{\text{obs}} \times \Delta t

This assumes equally-spaced data without gaps. GPS times follow the
LIGO/GWpy convention using TAI continuous seconds.

The leap-second concern raised by some models does not apply to
GPS/TAI time systems used in gravitational wave data analysis.

**References**:

- GWpy TimeSeries.epoch documentation
- LIGO GPS time convention (LIGO-T980044)


About the Validation
--------------------

These validations were performed as part of a comprehensive algorithm
audit using 12 different AI models:

- ChatGPT 5.2 Pro (Deep Research)
- Claude Opus 4.5 (Antigravity, IDE)
- Copilot (IDE)
- Cursor
- Felo
- Gemini 3 Pro (Antigravity, CLI, Web)
- Grok
- NotebookLM
- Perplexity

The full validation report is available in the developer documentation.
