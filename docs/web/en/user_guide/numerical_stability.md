# Numerical Stability and Precision

`gwexpy` is designed to handle the extreme dynamic ranges often found in gravitational-wave data analysis. 
Gravitational-wave strain signals are typically on the order of $10^{-21}$, while intermediate processing steps might involve values closer to unity.

To ensure scientific accuracy and prevent numerical artifacts (such as "Death Floats" or vanishing signals), `gwexpy` implements a robust numerical hardening strategy.

## Key Features

### 1. Adaptive Whitening
Standard whitening algorithms often use a fixed regularization parameter (`eps`) to prevent division by zero. If this parameter is too large (e.g., `1e-12`), it can swamp physically significant but numerically small signals (like $10^{-21}$ strain).

`gwexpy` automatically adapts this parameter to the scale of your data:
- **Auto-scaling `eps`**: The regularization floor is calculated relative to the variance of the input data.
- **Safety Floor**: A minimum floor (`SAFE_FLOOR = 1e-50`) ensures that even extremely quiet data (or zero-padded regions) does not cause singularities.

```python
from gwexpy.timeseries import TimeSeries
import numpy as np

# Create a very small signal (O(1e-21))
data = TimeSeries(np.random.randn(1000) * 1e-21, sample_rate=1024)

# Whiten the data
# eps="auto" (default) ensures the signal is not destroyed
whitened = data.whiten(eps="auto")
```

### 2. Robust Independent Component Analysis (ICA)
ICA separation is performed in a numerically stable manner:
- **Internal Standardization**: Data is standardized to unit variance internally before being passed to the FastICA algorithm, ensuring that convergence criteria are met regardless of the input amplitude.
- **Relative Tolerance**: Convergence tolerances are defined relative to the data scale.

### 3. Safe Logarithmic Scaling
When visualizing data with large dynamic ranges (e.g., spectrograms, PSDs), converting to decibels (dB) can be problematic if the data contains zeros or very small values.

`gwexpy` uses a "Safe Log" approach:
- **Dynamic Floor**: A floor is dynamically calculated based on the maximum value in the data and a specified dynamic range (default 200 dB).
- **No Artifacts**: This prevents `-inf` values and ensures that "silence" is rendered as the bottom of the color scale rather than as white holes or numerical errors.

### 4. Machine Precision Awareness
Numerical constants used throughout the package (for variance floors, coherence denominators, etc.) are derived from the machine precision (`epsilon`) of the floating-point type being used (float32 vs float64), ensuring optimal precision without unnecessary overhead.

## Best Practices for Users

- **Avoid Manual Offsets**: You do not need to add arbitrary small numbers (e.g., `data + 1e-20`) before plotting or processing. `gwexpy` functions handle this internally.
- **Trust Defaults**: The default parameters for functions like `whiten()` and `ica_fit()` are tuned for numerical safety. Use `eps="auto"` or `tol="auto"` whenever possible.
- **Check Warnings**: If you attempt operations that are numerically unstable (e.g., whitening a constant zero channel), `gwexpy` will issue informative warnings or errors guiding you to a solution.
