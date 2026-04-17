# Architecture and Data Flow

This section details the design philosophy of `gwexpy` and the internal data handling logic.
`gwexpy` extends GWpy to provide intuitive multi-channel time-series matrix operations and 4D physical field handling.

## How to Read This Page

- Read this page first if you want the mental model behind `gwexpy` container design and internal reshaping.
- For the mathematical foundations of each algorithm, continue to [Physics Models and Analysis Theory](physics_models.md).
- For concrete callable surfaces, follow the API entry links in each section to the [matrix API](../reference/api/matrix.rst), [fields API](../reference/api/fields.rst), and [fitting API](../reference/api/fitting.rst).

## Design Philosophy

### 1. Matrix Object Flattening Flow
Classes like `TimeSeriesMatrix` and `FrequencySeriesMatrix` handle automatic conversion to formats compatible with machine learning libraries like scikit-learn.
Typically, 3D data (Channels $\times$ Samples $\times$ Columns) is temporarily flattened into a 2D feature matrix for computation, while metadata (GPS timestamps, units) is preserved and restored after processing.

API entry: [matrix API](../reference/api/matrix.rst), [timeseries API](../reference/api/timeseries.rst)

### 2. 4D Field Model
`ScalarField` adopts a (Time, Frequency, x, y) 4D structure as its base unit.
By **maintaining all 4 dimensions** during indexing operations, the package ensures that grid information and axis metadata remain perfectly synchronized with the data.

API entry: [fields API](../reference/api/fields.rst), [Scalar Field Slicing Guide](scalarfield_slicing.md)

---

## Core Analysis Components

`gwexpy` builds advanced analysis pipelines by combining the following core components. For detailed mathematical and physical foundations, refer to [Physics Models and Analysis Theory](physics_models.md).

- **Multi-channel Analysis Engine**: Implementation of ICA/PCA for environmental noise isolation.
- **Fast Correlation Framework**: Accelerated coherence calculations (Bruco) for large-scale observation data.
- **Statistical Inference & Fitting**: Parameter estimation using GLS and MCMC.

## Related Documents
- [Physics Models and Analysis Theory](physics_models.md) — Detailed analysis theory and physics models (ICA/Bruco/MCMC)
- [Validated Algorithms](validated_algorithms.md) — Validation reports for numerical algorithms
- [Scalar Field Slicing Guide](scalarfield_slicing.md) — Details on 4D field operations
