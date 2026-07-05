---
myst:
  html_meta:
    description: "Choose the right GWexpy onboarding path with quickstart, tutorial, GWpy migration, and learning-path links for the next steps in analysis."
---

# Getting Started

We provide a **systematic learning path** for GWexpy users.
Choose the best starting point based on your background and goals.

If you want the shared assumptions and FFT/time conventions first, use [Prerequisites and Conventions](../explanation/prerequisites_and_conventions.md) as the entry point.

## Choose Your Path

### 🚀 5-min Quick Start

[Quickstart](quickstart.md)

For those who want to see results immediately. Get a figure in just 3 lines of code.

### 📖 30-min Hands-on

[Tutorial index](index.md)

Learn the core data structures and GWexpy-specific matrix operations from scratch.

### 🔄 For GWpy Users

[GWexpy for GWpy Users](../explanation/gwexpy_for_gwpy_users.md)

For current GWpy users. Start with the migration recipes, then use the [GWpy Difference API Index](../reference/gwpy_added_api.md) when you need a difference-oriented lookup of added APIs.

(en-learning-path)=
## Learning Path

### 1. Preparation

Start with the [Installation Guide](installation.md) to prepare a Python 3.11+ environment.

### 2. Core Data Structures

We recommend learning the primary containers in the following order:

1. [Basic Time Series](intro_timeseries.ipynb)
2. [Basic Frequency Series](intro_frequencyseries.ipynb)
3. [Basic Spectrograms](intro_spectrogram.ipynb)
4. [Plot Customization](intro_plotting.ipynb)

### 3. Advanced Analysis

When you need a specific technique, see these how-to recipes:

* **Multi-channel & Matrix Processing**: [TimeSeriesMatrix basics](matrix_timeseries.ipynb) / [Matrix containers](../how-to/containers/index.md)
* **High-dimensional Data**: [Field API basics](../how-to/containers/field_scalar_intro.ipynb) / [ScalarField slicing guide](../how-to/scalarfield_slicing.md)
* **Signal Processing**: [Fitting](intro_fitting.ipynb) / [HHT](../how-to/spectral/advanced_hht.ipynb) / [ARIMA](../how-to/fitting/advanced_arima.ipynb)

### 4. Practical Applications

Explore real-world analysis workflows in our [Case Studies Gallery](../how-to/case-studies/index.md).

<a id="next-to-read"></a>
<a id="next-steps"></a>

## Next to Read

* [Case Studies Gallery](../how-to/case-studies/index.md) - Visual examples and practical workflows.
* [All Tutorials](index.md)
* [GWexpy for GWpy Users](../explanation/gwexpy_for_gwpy_users.md) - start from the difference-oriented migration recipes
* [GWpy Difference API Index](../reference/gwpy_added_api.md) - look up added APIs from a GWpy-difference view
* [Prerequisites and Conventions](../explanation/prerequisites_and_conventions.md) - entry point for environment assumptions, GPS time, and FFT conventions
* [API Reference](../reference/index.md)
* [Validated Algorithms](../explanation/validated_algorithms.md) - Verification reports for numerical accuracy.
