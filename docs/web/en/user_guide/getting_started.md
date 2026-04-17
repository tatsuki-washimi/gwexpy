# Getting Started

We provide a **systematic learning path** for GWexpy users.
Choose the best starting point based on your background and goals.

## Quick Summary

| Item | Details |
| --- | --- |
| **Target Audience** | Experimental data analysts, GWpy users, and anyone interested in signal processing with Python. |
| **Prerequisites** | Basics of Python 3.11+, NumPy array manipulation, (Recommended) Matplotlib. |
| **Time Required** | From 5 minutes (Quick Start) to 30 minutes (Basic Hands-on). |
| **Goal** | Load data, visualize results, and perform basic frequency-domain analysis. |

If you want the shared assumptions and FFT/time conventions first, use [Prerequisites and Conventions](prerequisites_and_conventions.md) as the entry point.

## Choose Your Path

### 🚀 5-min Quick Start

[Quickstart](quickstart.md)

For those who want to see results immediately. Get a figure in just 3 lines of code.

### 📖 30-min Hands-on

[Tutorial index](tutorials/index.rst)

Learn the core data structures and GWexpy-specific matrix operations from scratch.

### 🔄 For GWpy Users

[Migration from GWpy](gwexpy_for_gwpy_users_en.md)

For current GWpy users. Start with the migration recipes, then use the [GWpy Difference API Index](gwpy_added_api_index_en.md) when you need a difference-oriented lookup of added APIs.

## Learning Path

### 1. Preparation

Start with the [Installation Guide](installation.md) to prepare a Python 3.11+ environment.

### 2. Core Data Structures

We recommend learning the primary containers in the following order:

1. [Basic Time Series](tutorials/intro_timeseries.ipynb)
2. [Basic Frequency Series](tutorials/intro_frequencyseries.ipynb)
3. [Basic Spectrograms](tutorials/intro_spectrogram.ipynb)
4. [Plot Customization](tutorials/intro_plotting.ipynb)

### 3. Advanced Analysis

Refer to these guides based on your needs:

* **Multi-channel & Matrix Processing**: [Using Matrix Containers](tutorials/matrix_timeseries.ipynb)
* **High-dimensional Data**: [Field API Intro](tutorials/field_scalar_intro.ipynb) / [ScalarField Slicing Guide](scalarfield_slicing.md)
* **Signal Processing**: [Fitting](tutorials/advanced_fitting.ipynb) / [HHT](tutorials/advanced_hht.ipynb) / [ARIMA](tutorials/advanced_arima.ipynb)

### 4. Practical Applications

Explore real-world analysis workflows in our [Case Studies Gallery](../examples/index.rst).

## Next Steps

* [Case Studies Gallery](../examples/index.rst) - Visual examples and practical workflows.
* [All Tutorials](tutorials/index.rst)
* [Migration from GWpy](gwexpy_for_gwpy_users_en.md) - start from the difference-oriented migration recipes
* [GWpy Difference API Index](gwpy_added_api_index_en.md) - look up added APIs from a GWpy-difference view
* [Prerequisites and Conventions](prerequisites_and_conventions.md) - entry point for environment assumptions, GPS time, and FFT conventions
* [API Reference](../reference/index.rst)
* [Validated Algorithms](validated_algorithms.md) - Verification reports for numerical accuracy.
