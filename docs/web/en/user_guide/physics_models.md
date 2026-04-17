# Physics Models and Analysis Theory

This page explains the advanced models and analytical theories implemented in `gwexpy` for handling specific physical phenomena and hardware responses.

## Response and Coupling Functions

### Automatic Excitation Detection

Extracts stable intervals for analysis from data containing injections (such as swept sine or stepped sine). By tracking power in specific frequency bands on a spectrogram, it identifies segments that exceed thresholds, eliminating the need for manual time-range specification.

### Coupling Function (:term:`Coupling Function`; CF)

Estimates coupling functions while accounting for background noise. By comparing power during injection and background periods for both the target and witness signals, it isolates the true coupling degree.

$$
\text{CF}(f) = \sqrt{\frac{P_{\text{tgt,inj}}(f) - P_{\text{tgt,bkg}}(f)}{P_{\text{wit,inj}}(f) - P_{\text{wit,bkg}}(f)}}
$$

| Variable | Definition | Physical Meaning |
| :--- | :--- | :--- |
| $f$ | Frequency | Frequency point for analysis |
| $P_{\text{tgt,inj}}(f)$ | Target signal power (during injection) | Power distribution of the main signal during excitation |
| $P_{\text{tgt,bkg}}(f)$ | Target signal power (during background) | Noise floor of the main signal without excitation |
| $P_{\text{wit,inj}}(f)$ | Witness signal power (during injection) | Power of the reference signal (e.g., environmental noise) |
| $P_{\text{wit,bkg}}(f)$ | Witness signal power (during background) | Noise floor of the reference signal without excitation |

- **Related API**: {doc}`../reference/api/timeseries` (`TimeSeriesDict.calculate_coupling`)

---

## Built-in Noise Models

Provides physically motivated noise generators for use as initial models in simulations or fitting.

### 1. Schumann Resonance (:term:`Schumann Resonance`)

Models magnetic noise corresponding to the resonance modes of the Earth-ionosphere cavity. It reproduces the low-frequency magnetic background by superimposing multiple independent Lorentzian profiles.

- **Related API**: {doc}`../reference/api/signal` (`generate_schumann_model`)

### 2. Voigt Profile

Generates peak shapes found in atomic physics or high-Q mechanical resonances, which combine Gaussian (Doppler broadening, etc.) and Lorentzian (collision/natural broadening, etc.) characteristics. It is calculated efficiently using the Faddeeva function.

- **Related API**: {doc}`../reference/api/signal` (`voigt_profile`)

---

## Advanced Analysis Engines and Algorithms

### 1. Independent and Principal Component Analysis (ICA/PCA)

The ICA/PCA implementation in `gwexpy` is optimized for physical data characteristics:

- **Unit Variance Standardization**: Standardizes data to unit variance internally to improve convergence, then restores (re-scales) the original physical scale after computation.
- **Spatio-temporal Metadata Inheritance**: Automatically inherits the GPS time conventions from the input data for each statistically extracted component.

- **Related API**: {doc}`../reference/api/signal` (ICA, PCA)

### 2. Fast Correlation Engine (:term:`Bruco`)

The `FastCoherenceEngine` scans thousands of auxiliary channels for contributions to a target signal with extreme speed.

- **FFT Caching**: Reuses FFT results for a common target signal in memory.
- **Sparse-like Computation**: Skips non-correlated channels early to focus resources on significant contributors.

- **Related API**: {doc}`../reference/api/timeseries` (`TimeSeriesDict.scan_coherence`)

### 3. Bayesian Inference and GLS Fitting

Handles parameter estimation for multidimensional data with complex error structures.

- **GLS (Generalized Least Squares)**: Applies statistically justified weighting when bins at different frequencies have correlated (non-diagonal) covariance.
- **MCMC Integration**: Uses `emcee` for posterior sampling, enabling robust fitting even for non-linear physical models.

- **Related API**: {doc}`../reference/api/stats`

## Related Documents

- {doc}`architecture` — Internal data structures and engine design
- {doc}`validated_algorithms` — Validation reports for mathematical formulas
