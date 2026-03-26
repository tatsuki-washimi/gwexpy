# Phase 1: Dangerous Defaults Analysis

**Generated:** 2026-02-03
**Source:** `scripts/audit_numerical_risks.py` (Deep Scan)

## 1. Overview

The specific issue of "Death Floats" (hardcoded small numbers) often manifests as **Default Argument Values** in function signatures. These values acts as implicit assumptions about the scale of the data. When the data is Gravitational Wave strain ($10^{-21}$), defaults like `1e-12` are **9 orders of magnitude too large**, acting as brick walls that block signals.

## 2. Identified "Death Defaults"

| File | Argument | Default Value | Why it is Fatal |
| :--- | :--- | :--- | :--- |
| `gwexpy/signal/preprocessing/whitening.py` | `eps` | `1e-12` | Regularization for variance. $10^{-24} \ll 10^{-12}$. **Whitening becomes Identity.** |
| `gwexpy/timeseries/pipeline.py` | `eps` | `1e-12` | Same as above. |
| `gwexpy/timeseries/matrix_analysis.py` | `eps` | `1e-12` | Matrix operations. |
| `gwexpy/timeseries/preprocess.py` | `eps` | `1e-12` | General preprocessing. |
| `gwexpy/timeseries/decomposition.py` | `tol` | `1e-4` | ICA convergence tolerance. $10^{-4}$ is huge compared to unscaled signal. |
| `gwexpy/types/time_plane_transform.py` | `eps` | `1e-30` | Better (`1e-30`), but still arbitrary hardcoding. |
| `gwexpy/noise/magnetic.py` | `amplitude` | `1e-11` | Mock data generation default. |

## 3.  `gwexpy.numerics` Module Design

To solve this permanently, we introduce a text-book "Single Source of Truth".

### 3.1 `constants.py`

Instead of magic numbers, we define semantic constants derived from machine precision.

```python
import numpy as np

# Machine precision
EPS_FLOAT64 = np.finfo(np.float64).eps  # ~2.22e-16
EPS_FLOAT32 = np.finfo(np.float32).eps  # ~1.19e-07

# Physical Limits (Safe Safe Floors)
# GW Strain power is ~10^-42. We need a floor below that.
SAFE_FLOOR_STRAIN = 1e-50
```

### 3.2 `scaling.py` (AutoScaler)

The `AutoScaler` handles the "Relative Limit" problem. `eps` should not be a constant; it should be relative to the data.

```python
def get_safe_epsilon(data, rel_tol=1e-6, abs_tol=1e-50):
    """
    Returns an epsilon appropriate for the data's scale.
    eps = max(abs_tol, std(data) * rel_tol)
    """
    # Logic ...
```

## 4. Migration Path

| Target Module | Action |
| :--- | :--- |
| **Whitening** | Replace `eps=1e-12` with `eps='auto'`. Inside: call `get_safe_epsilon(X)`. |
| **ICA** | Replace `tol=1e-4` with logic: `tol = relative_tol * scale_of_data`. |
| **Pipeline** | Import `get_safe_epsilon` from `gwexpy.numerics`. |

## 5. Implementation Steps

1. **Create** `gwexpy/numerics/` package.
2. **Edit** all 7 files listed above.
3. **Update Signature**: Change default to `None` or `'auto'` (where backward compatibility allows), or keep the float but add a DeprecationWarning if it's used blindly on small data.
