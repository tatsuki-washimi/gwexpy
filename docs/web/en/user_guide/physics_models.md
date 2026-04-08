# Physics Models and Analytical Theory

This section describes advanced models implemented in `gwexpy` for handling specific physical phenomena and hardware responses.

## Response and Coupling Functions

### Automatic Injection Segment Detection
Detects and extracts stable intervals suitable for analysis from data containing injections (e.g., swept-sines or step-sines). By tracking power in specific frequency bands on a spectrogram, it identifies segments where the excitation is active and stable, removing the need for manual timestamps.

### Coupling Function (CF) Estimation
Estimates coupling functions while accounting for background noise. By comparing signal power during injection and background periods for both target and witness channels, it derives a more accurate estimation of the true coupling strength.

$$
\text{CF}(f) = \sqrt{\frac{P_{\text{tgt,inj}}(f) - P_{\text{tgt,bkg}}(f)}{P_{\text{wit,inj}}(f) - P_{\text{wit,bkg}}(f)}}
$$

---

## Built-in Noise Models

`gwexpy` provides physically-grounded noise generators that can be used for simulations or as initial models for fitting.

### 1. Schumann Resonance
Models magnetic background noise corresponding to the Earth-ionosphere cavity modes. It reproduces the low-frequency geomagnetic background by superimposing multiple independent Lorentzian profiles.

### 2. Voigt Profile
Generates spectral peak shapes that combine Gaussian (e.g., Doppler broadening) and Lorentzian (e.g., natural or collisional broadening) characteristics, commonly found in atomic physics and high-Q mechanical resonances. Computed efficiently using the Faddeeva function.

---

## Related Documents
- {doc}`architecture` — System design and data flow
- {doc}`validated_algorithms` — Validation reports for numerical formulas
