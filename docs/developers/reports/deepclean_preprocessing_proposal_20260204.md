# Proposal: DeepClean Preprocessing Pipeline Integration

**Date**: 2026-02-04
**Status**: Proposed

## 1. Overview

This document proposes integrating the preprocessing logic from DeepClean v2 into `gwexpy` as a generalized, reusable module. Currently, DeepClean's preprocessing (splitting, filtering, channel-wise scaling) is tightly coupled with its PyTorch training loop (`LightningDataModule`). By extracting this logic into `gwexpy`, we enable:

1. **Model Agnosticism**: Use the same preprocessing for non-DL models (RandomForest, XGBoost, etc.).
2. **Ease of Use**: Provide a "one-line" preparation step for noise subtraction tasks.
3. **Standardization**: Ensure consistent data treatment across different analysis pipelines.

## 2. Proposed Architecture

We propose adding a new module `gwexpy.signal.preprocessing.deepclean` containing the `DeepCleanPreprocessor` class.

### 2.1 `DeepCleanPreprocessor` Class

This class mimics the `scikit-learn` Transformer API (`fit`, `transform`) but is specialized for GW time series data.

```python
class DeepCleanPreprocessor:
    def __init__(
        self,
        sample_rate: Quantity,
        freq_low: list[float] | None = None,
        freq_high: list[float] | None = None,
        filt_order: int = 8,
        valid_frac: float = 0.0,
    ):
        ...
```

#### Key Methods

1. **`fit(X, y=None)`**:
    * Computes statistics (mean, std/median, mad) for each channel in `X` (Witnesses) and `y` (Strain).
    * Designs the bandpass filter coefficients if frequency bands are specified.

2. **`transform(X, y=None)`**:
    * Applies bandpass filtering (zero-phase `filtfilt`).
    * Applies standardization (Z-score or Robust).
    * Returns processed `TimeSeriesMatrix` and `TimeSeries`.

3. **`split(X, y)` -> `(X_train, y_train, X_valid, y_valid)`**:
    * Splits data chronologically based on `valid_frac`.
    * Ensures the split point respects integer sampling points.
    * Returns the split datasets, ready for `transform`.

### 2.2 Integration with `TimeSeriesWindowDataset`

The workflow for a user would be:

```python
# 1. Load Data
witnesses = TimeSeriesMatrix(...)
strain = TimeSeries(...)

# 2. Preprocess (Split -> Filter -> Scale)
preprocessor = DeepCleanPreprocessor(sample_rate=4096, valid_frac=0.2)
X_train, y_train, X_valid, y_valid = preprocessor.split(witnesses, strain)

# Learn scaling/filtering from Train only
preprocessor.fit(X_train, y_train)

# Apply to both
X_train_proc, y_train_proc = preprocessor.transform(X_train, y_train)
X_valid_proc, y_valid_proc = preprocessor.transform(X_valid, y_valid)

# 3. Create Datasets (for PyTorch)
train_ds = TimeSeriesWindowDataset(X_train_proc, labels=y_train_proc, ...)
valid_ds = TimeSeriesWindowDataset(X_valid_proc, labels=y_valid_proc, ...)
```

## 3. Implementation Details

* **Filtering**: Leverage `gwpy.signal.filter_design` for filter construction (`butter`).
* **Scaling**: Use existing `standardize_matrix` logic but wrap it to persist statistics (stateful).
* **Interop**: Ensure output types are compatible with `gwexpy`'s existing `TimeSeries` ecosystem.

## 4. Work Items

* [ ] Create `gwexpy/signal/preprocessing/deepclean.py`.
* [ ] Implement `DeepCleanPreprocessor` class.
* [ ] Add unit tests verifying numerical layout matches DeepClean original.
* [ ] Create tutorial notebook `tutorial_DeepClean_Preprocessing.ipynb`.
