# Architecture and Data Flow

This section details the design philosophy of `gwexpy` and the internal data handling logic.
`gwexpy` extends GWpy to provide intuitive multi-channel time-series matrix operations and 4D physical field handling.

## Design Philosophy

### 1. Matrix Object Flattening Flow
Classes like `TimeSeriesMatrix` and `FrequencySeriesMatrix` handle automatic conversion to formats compatible with machine learning libraries like scikit-learn.
Typically, 3D data (Channels $\times$ Samples $\times$ Columns) is temporarily flattened into a 2D feature matrix for computation, while metadata (GPS timestamps, units) is preserved and restored after processing.

### 2. 4D Field Model
`ScalarField` adopts a (Time, Frequency, x, y) 4D structure as its base unit.
By **maintaining all 4 dimensions** during indexing operations, the package ensures that grid information and axis metadata remain perfectly synchronized with the data.

---

## Core Numerical Engines

### Independent and Principal Component Analysis (ICA/PCA)
While utilizing scikit-learn's engine, `gwexpy` enhances the process by:
- **Amplitude Scaling**: Internally standardizing data to unit variance before/after ICA to improve convergence while maintaining physical scales.
- **Metadata Re-connection**: Automatically re-assigning GPS times and sampling rates to the resulting independent components.

### Fast Coherence Engine (:term:`BruCo`)

The `FastCoherenceEngine` is designed to accelerate noise auditing across thousands of channels.
By caching FFTs/PSDs for a common target signal and performing only minimal calculations for auxiliary channels, it achieves orders-of-magnitude faster performance than repeated standard `coherence()` calls.

### Bayesian Fitting (GLS/MCMC)
The framework supports a Generalized Least Squares (GLS) solver for handling correlated errors via known covariance matrices ($\Sigma$).
Integration with `emcee` allows for robust MCMC parameter estimation, correctly handling complex residuals for transfer function fitting.

---

## Related Documents
- {doc}`physics_models` — Details on Schumann resonances and response models
- {doc}`validated_algorithms` — Validation reports for numerical algorithms
- {doc}`scalarfield_slicing` — Details on 4D field operations
