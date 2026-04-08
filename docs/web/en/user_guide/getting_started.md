# Getting Started

We provide a **systematic learning path** for GWexpy users.
Choose the best starting point based on your background and goals.

## Quick Summary

.. list-table::
   :widths: 25 75

   * - **Target Audience**
     - Experimental data analysts, GWpy users, and anyone interested in signal processing with Python.
   * - **Prerequisites**
     - Basics of Python 3.11+, NumPy array manipulation, (Recommended) Matplotlib.
   * - **Time Required**
     - From 5 minutes (Quick Start) to 30 minutes (Basic Hands-on).
   * - **Goal**
     - Load data, visualize results, and perform basic frequency-domain analysis.

## Choose Your Path

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 🚀 5-min Quick Start
        :link: quickstart
        :link-type: doc

        For those who want to see results immediately. Get a figure in just 3 lines of code.

    .. grid-item-card:: 📖 30-min Hands-on
        :link: tutorials/index
        :link-type: doc

        Learn the core data structures and GWexpy-specific matrix operations from scratch.

    .. grid-item-card:: 🔄 For GWpy Users
        :link: gwexpy_for_gwpy_users_en
        :link-type: doc

        For current GWpy users. Learn the key differences and how to simplify your code with new features.

## Learning Path

### 1. Preparation

GWexpy requires **Python 3.11 or later**.
Copy the following code and run it in your terminal or Jupyter Notebook.

### 2. Core Data Structures

We recommend learning the primary containers in the following order:

1. [{doc}`tutorials/intro_timeseries <tutorials/intro_timeseries>`] - Basic Time Series
2. [{doc}`tutorials/intro_frequencyseries <tutorials/intro_frequencyseries>`] - Basic Frequency Series
3. [{doc}`tutorials/intro_spectrogram <tutorials/intro_spectrogram>`] - Basic Spectrograms
4. [{doc}`tutorials/intro_plotting <tutorials/intro_plotting>`] - Plot Customization

### 3. Advanced Analysis

Refer to these guides based on your needs:

* **Multi-channel & Matrix Processing**: :doc:`Using Matrix Containers <tutorials/matrix_timeseries>`
* **High-dimensional Data**: :doc:`Field API Intro <tutorials/field_scalar_intro>` / :doc:`ScalarField Slicing Guide <scalarfield_slicing>`
* **Signal Processing**: :doc:`Fitting <tutorials/advanced_fitting>` / :doc:`HHT <tutorials/advanced_hht>` / :doc:`ARIMA <tutorials/advanced_arima>`

### 4. Practical Applications

Explore real-world analysis workflows in our :doc:`Case Studies Gallery <../examples/index>`.

## Next Steps

* [{doc}`Case Studies Gallery <../examples/index>`] - Visual examples and practical workflows.
* All Tutorials: [{doc}`tutorials/index <tutorials/index>`]
* API Reference: [{doc}`Reference <../reference/index>`]
* [{doc}`Validated Algorithms <validated_algorithms>`] - Verification reports for numerical accuracy.
